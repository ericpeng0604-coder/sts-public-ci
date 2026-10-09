from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.sts1.sts1_g7_seed_ledger import (
    CONFIRMATION_POOL_SIZES,
    EXPLORATION_POOL_SIZES,
    MAX_SEED,
    REQUIRED_SOURCE_CATEGORIES,
    SeedLedgerError,
    extend_inventory_with_pools,
    generate_confirmation_trial,
    generate_exploration_round,
    sha256_json,
    validate_inventory,
    write_confirmation_trial,
    write_exploration_round,
)


def _inventory() -> dict[str, object]:
    return {
        "schema_version": "sts1-existing-seed-inventory-v1",
        "inventory_id": "fixture-inventory-v1",
        "complete": True,
        "source_manifests": [
            {
                "category": category,
                "source_ref": f"fixture/{category}.txt",
                "sha256": hashlib.sha256(category.encode("utf-8")).hexdigest(),
                "seed_ids": [index * 100 + 7] if index < 7 else [],
            }
            for index, category in enumerate(REQUIRED_SOURCE_CATEGORIES)
        ],
    }


class SeedLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inventory = _inventory()
        self.inventory_sha256 = sha256_json(self.inventory)

    def test_inventory_accepts_pinned_simulator_uint64_seed_domain(self) -> None:
        inventory = _inventory()
        upper = (1 << 64) - 1
        inventory["source_manifests"][0]["seed_ids"] = [(1 << 32) + 17, upper]

        excluded, _ = validate_inventory(inventory)
        self.assertIn((1 << 32) + 17, excluded)
        self.assertIn(upper, excluded)

        for invalid in (0, 1 << 64):
            inventory["source_manifests"][0]["seed_ids"] = [invalid]
            with self.subTest(invalid_seed_domain=True):
                with self.assertRaises(SeedLedgerError):
                    validate_inventory(inventory)

    def test_exploration_pools_are_deterministic_sized_and_disjoint(self) -> None:
        first = generate_exploration_round(
            self.inventory,
            inventory_sha256=self.inventory_sha256,
            round_id="round-001",
            generation_key="public-fixed-key-20261008",
        )
        second = generate_exploration_round(
            self.inventory,
            inventory_sha256=self.inventory_sha256,
            round_id="round-001",
            generation_key="public-fixed-key-20261008",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "GENERATED_NOT_RUN")
        excluded = {
            seed
            for source in self.inventory["source_manifests"]
            for seed in source["seed_ids"]
        }
        allocated: set[int] = set()
        for name, expected_count in EXPLORATION_POOL_SIZES.items():
            pool = first["pools"][name]
            seeds = pool["seed_ids"]
            expected_purpose = "train" if name.startswith("train_") else name
            self.assertEqual(pool["purpose"], expected_purpose)
            self.assertEqual(pool["role"], name)
            self.assertEqual(len(seeds), expected_count)
            self.assertEqual(len(seeds), len(set(seeds)))
            self.assertFalse(excluded.intersection(seeds))
            self.assertFalse(allocated.intersection(seeds))
            self.assertTrue(all(1 <= seed <= MAX_SEED for seed in seeds))
            self.assertEqual(pool["manifest_sha256"], sha256_json({k: v for k, v in pool.items() if k != "manifest_sha256"}))
            allocated.update(seeds)

    def test_incomplete_inventory_fails_closed(self) -> None:
        self.inventory["source_manifests"] = [
            source for source in self.inventory["source_manifests"] if source["category"] != "fresh"
        ]
        with self.assertRaisesRegex(SeedLedgerError, "missing required categories"):
            generate_exploration_round(
                self.inventory,
                inventory_sha256=self.inventory_sha256,
                round_id="round-001",
                generation_key="public-fixed-key-20261008",
            )

    def test_exhausted_seed_space_fails_without_sampling_forever(self) -> None:
        inventory = _inventory()
        inventory["source_manifests"] = [
            {
                "category": category,
                "source_ref": f"fixture/{category}.txt",
                "sha256": hashlib.sha256(category.encode("utf-8")).hexdigest(),
                "seed_ids": list(range(1, 32)) if index == 0 else [],
            }
            for index, category in enumerate(REQUIRED_SOURCE_CATEGORIES)
        ]
        with patch("scripts.sts1.sts1_g7_seed_ledger.MAX_SEED", 31):
            with self.assertRaisesRegex(SeedLedgerError, "insufficient unused legal"):
                generate_exploration_round(
                    inventory,
                    inventory_sha256=sha256_json(inventory),
                    round_id="round-exhausted",
                    generation_key="fixed-exhaustion-test-key",
                )

    def test_confirmation_trial_freezes_hashes_and_generates_disjoint_batches(self) -> None:
        exploration = generate_exploration_round(
            self.inventory,
            inventory_sha256=self.inventory_sha256,
            round_id="round-001",
            generation_key="explore-key-20261008",
        )
        extended_inventory = extend_inventory_with_pools(
            self.inventory,
            list(exploration["pools"].values()),
            inventory_id="fixture-inventory-after-round-001",
        )
        extended_inventory_sha256 = sha256_json(extended_inventory)
        trial = generate_confirmation_trial(
            extended_inventory,
            inventory_sha256=extended_inventory_sha256,
            trial_k=1,
            prior_trials=[],
            generation_key="confirm-key-20261008",
            candidate_sha256="a" * 64,
            config_sha256="b" * 64,
        )
        self.assertEqual(trial["alpha_exact"], "1/40")
        self.assertEqual(trial["candidate_sha256"], "a" * 64)
        self.assertEqual(trial["config_sha256"], "b" * 64)
        batches = trial["pools"]
        self.assertEqual(set(batches), set(CONFIRMATION_POOL_SIZES))
        first = set(batches["confirmation_a"]["seed_ids"])
        second = set(batches["confirmation_b"]["seed_ids"])
        self.assertEqual(len(first), 100)
        self.assertEqual(len(second), 100)
        self.assertFalse(first.intersection(second))
        self.assertEqual(batches["confirmation_a"]["candidate_sha256"], "a" * 64)
        exploration_ids = {
            seed
            for pool in exploration["pools"].values()
            for seed in pool["seed_ids"]
        }
        confirmation_ids = first | second
        self.assertFalse(exploration_ids.intersection(confirmation_ids))

    def test_confirmation_trial_requires_contiguous_prior_history_and_excludes_its_seeds(self) -> None:
        first = generate_confirmation_trial(
            self.inventory,
            inventory_sha256=self.inventory_sha256,
            trial_k=1,
            prior_trials=[],
            generation_key="confirm-key-20261008",
            candidate_sha256="a" * 64,
            config_sha256="b" * 64,
        )
        with self.assertRaisesRegex(SeedLedgerError, "earlier k"):
            generate_confirmation_trial(
                self.inventory,
                inventory_sha256=self.inventory_sha256,
                trial_k=2,
                prior_trials=[],
                generation_key="confirm-key-20261009",
                candidate_sha256="c" * 64,
                config_sha256="d" * 64,
            )

        prior = {**first, "result_status": "fail"}
        second = generate_confirmation_trial(
            self.inventory,
            inventory_sha256=self.inventory_sha256,
            trial_k=2,
            prior_trials=[prior],
            generation_key="confirm-key-20261009",
            candidate_sha256="c" * 64,
            config_sha256="d" * 64,
        )
        self.assertEqual(second["alpha_exact"], "1/80")
        prior_ids = {
            seed
            for pool in first["pools"].values()
            for seed in pool["seed_ids"]
        }
        second_ids = {
            seed
            for pool in second["pools"].values()
            for seed in pool["seed_ids"]
        }
        self.assertFalse(prior_ids.intersection(second_ids))

    def test_writer_stays_in_allowed_directory_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo_root = Path(temp) / "repo"
            repo_root.mkdir()
            inventory_path = repo_root / "inventory.json"
            inventory_bytes = json.dumps(self.inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
            inventory_path.write_bytes(inventory_bytes)
            output = repo_root / "evidence" / "sts1" / "g7-improvement" / "seeds" / "round-001"
            private_output = Path(temp) / "private-round-001"
            ledger = write_exploration_round(
                inventory_path=inventory_path,
                output_dir=output,
                private_output_dir=private_output,
                repo_root=repo_root,
                round_id="round-001",
                generation_key="public-fixed-key-20261008",
            )
            saved_private = json.loads((private_output / "ledger.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_private["ledger_sha256"], ledger["ledger_sha256"])
            public_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(public_summary["manifest_sha256"], ledger["ledger_sha256"])
            self.assertEqual(public_summary["pools"]["dev"]["count"], 30)
            self.assertEqual(public_summary["schema_version"], "sts1-g7-seed-public-summary-v2")
            self.assertNotIn("inventory_id", public_summary)
            self.assertNotIn("inventory_sha256", public_summary)
            self.assertNotIn("source_audit_sha256", public_summary)
            self.assertNotIn("generation_key", public_summary)
            self.assertNotIn("seed_ids", json.dumps(public_summary))
            with self.assertRaises(FileExistsError):
                write_exploration_round(
                    inventory_path=inventory_path,
                    output_dir=output,
                    private_output_dir=Path(temp) / "private-round-001-again",
                    repo_root=repo_root,
                    round_id="round-001",
                    generation_key="public-fixed-key-20261008",
                )
            with self.assertRaisesRegex(SeedLedgerError, "must stay under"):
                write_exploration_round(
                    inventory_path=inventory_path,
                    output_dir=repo_root / "outside",
                    private_output_dir=Path(temp) / "private-round-002",
                    repo_root=repo_root,
                    round_id="round-002",
                    generation_key="another-key",
                )
            with self.assertRaisesRegex(SeedLedgerError, "outside the repository"):
                write_exploration_round(
                    inventory_path=inventory_path,
                    output_dir=repo_root / "evidence" / "sts1" / "g7-improvement" / "seeds" / "round-003",
                    private_output_dir=repo_root / "private-round-003",
                    repo_root=repo_root,
                    round_id="round-003",
                    generation_key="another-key",
                )

    def test_confirmation_writer_records_frozen_candidate_and_trial_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo_root = Path(temp) / "repo"
            repo_root.mkdir()
            inventory_path = repo_root / "inventory.json"
            inventory_path.write_text(json.dumps(self.inventory), encoding="utf-8")
            prior_trials_path = repo_root / "prior-trials.json"
            prior_trials_path.write_text("[]", encoding="utf-8")
            output = repo_root / "evidence" / "sts1" / "g7-improvement" / "seeds" / "confirmation-k001"
            private_output = Path(temp) / "private-confirmation-k001"
            trial = write_confirmation_trial(
                inventory_path=inventory_path,
                prior_trials_path=prior_trials_path,
                output_dir=output,
                private_output_dir=private_output,
                repo_root=repo_root,
                trial_k=1,
                generation_key="confirm-key-20261008",
                candidate_sha256="a" * 64,
                config_sha256="b" * 64,
            )
            saved = json.loads((private_output / "trial.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["trial_manifest_sha256"], trial["trial_manifest_sha256"])
            self.assertEqual(saved["alpha_exact"], "1/40")
            self.assertEqual(saved["candidate_sha256"], "a" * 64)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["alpha_exact"], "1/40")
            self.assertEqual(summary["pools"]["confirmation_a"]["count"], 100)
            self.assertNotIn("inventory_id", summary)
            self.assertNotIn("inventory_sha256", summary)
            self.assertNotIn("source_audit_sha256", summary)
            self.assertNotIn("generation_key", summary)
            self.assertNotIn("seed_ids", json.dumps(summary))
            self.assertEqual(len(saved["pools"]["confirmation_a"]["seed_ids"]), 100)


if __name__ == "__main__":
    unittest.main()
