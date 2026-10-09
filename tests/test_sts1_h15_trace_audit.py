from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.sts1.sts1_g7_h15_train_trace_audit import (
    EvaluationIntegrityError,
    POOL_SIZES,
    _validate_private_paths,
    _validate_pool_seed_sets,
)


class H15PoolIntegrityTests(unittest.TestCase):
    def test_accepts_legal_disjoint_pool_shapes(self) -> None:
        pools: dict[str, list[int]] = {}
        start = 100
        for role, count in POOL_SIZES.items():
            pools[role] = list(range(start, start + count))
            start += count

        checked = _validate_pool_seed_sets(pools, {1, 2, 3})

        self.assertEqual({key: len(value) for key, value in checked.items()}, POOL_SIZES)

    def test_rejects_overlap_with_exclusion_inventory(self) -> None:
        pools = {role: list(range(100 + index * 50, 100 + index * 50 + count)) for index, (role, count) in enumerate(POOL_SIZES.items())}
        pools["train_hypothesis_1"][0] = 7

        with self.assertRaises(EvaluationIntegrityError):
            _validate_pool_seed_sets(pools, {7})

    def test_rejects_overlap_between_round008_pools(self) -> None:
        pools = {role: list(range(100 + index * 50, 100 + index * 50 + count)) for index, (role, count) in enumerate(POOL_SIZES.items())}
        pools["probe"][0] = pools["train_hypothesis_1"][0]

        with self.assertRaises(EvaluationIntegrityError):
            _validate_pool_seed_sets(pools, set())

    def test_rejects_boolean_or_out_of_range_seed(self) -> None:
        pools = {role: list(range(100 + index * 50, 100 + index * 50 + count)) for index, (role, count) in enumerate(POOL_SIZES.items())}
        pools["dev"][0] = True

        with self.assertRaises(EvaluationIntegrityError):
            _validate_pool_seed_sets(pools, set())

    def test_rejects_wrong_pool_count(self) -> None:
        pools = {role: list(range(100 + index * 50, 100 + index * 50 + count)) for index, (role, count) in enumerate(POOL_SIZES.items())}
        pools["dev"].pop()

        with self.assertRaises(EvaluationIntegrityError):
            _validate_pool_seed_sets(pools, set())

    def test_private_paths_are_canonical_and_run_once(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pools_dir = root / "round-008-seeds" / "generated-pools"
            pools_dir.mkdir(parents=True)
            output_dir = root / "round-008-h15-train-trace-20261009"
            usage_ledger = root / "round-008-seeds" / "h15-usage-private.jsonl"

            _validate_private_paths(pools_dir, output_dir, usage_ledger)
            with self.assertRaises(EvaluationIntegrityError):
                _validate_private_paths(pools_dir, root / "alternate-output", usage_ledger)

            output_dir.mkdir()
            with self.assertRaises(EvaluationIntegrityError):
                _validate_private_paths(pools_dir, output_dir, usage_ledger)


if __name__ == "__main__":
    unittest.main()
