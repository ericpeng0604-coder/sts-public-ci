"""Generate deterministic, disjoint STS1 G7 improvement seed manifests.

The generator requires a complete inventory of every prior seed purpose. It
does not read simulator or Gate/Fresh files itself; the caller supplies an
audited inventory with per-source references and SHA-256 hashes. Never run a
pool unless its inventory included every required source category.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA_VERSION = "sts1-g7-seed-ledger-v1"
INVENTORY_SCHEMA_VERSION = "sts1-existing-seed-inventory-v1"
GENERATOR_VERSION = "sha256-counter-rejection-v1"
MAX_SEED = 10**9
REQUIRED_SOURCE_CATEGORIES = (
    "mining",
    "dev",
    "probe",
    "gate",
    "fresh",
    "reserved",
    "unknown",
    "train_used",
)
EXPLORATION_POOL_SIZES = {
    "train_hypothesis_1": 10,
    "train_hypothesis_2": 10,
    "train_hypothesis_3": 10,
    "probe": 10,
    "dev": 30,
}
CONFIRMATION_POOL_SIZES = {
    "confirmation_a": 100,
    "confirmation_b": 100,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SeedLedgerError(ValueError):
    """Raised when a seed inventory or generated pool is unsafe to use."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _seed_id(value: Any, *, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_SEED:
        raise SeedLedgerError(f"invalid simulator seed in {source}: expected integer 1..{MAX_SEED}")
    return value


def validate_inventory(inventory: Mapping[str, Any]) -> tuple[set[int], list[dict[str, Any]]]:
    if inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise SeedLedgerError("unsupported existing-seed inventory schema")
    if inventory.get("complete") is not True:
        raise SeedLedgerError("seed inventory is not marked complete")
    inventory_id = inventory.get("inventory_id")
    if not isinstance(inventory_id, str) or not inventory_id.strip():
        raise SeedLedgerError("seed inventory_id is required")
    sources = inventory.get("source_manifests")
    if not isinstance(sources, list):
        raise SeedLedgerError("source_manifests must be a list")

    covered_categories: set[str] = set()
    source_refs: set[str] = set()
    excluded: set[int] = set()
    source_audit: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        label = f"source_manifests[{index}]"
        if not isinstance(source, Mapping):
            raise SeedLedgerError(f"{label} must be an object")
        category = source.get("category")
        if category not in REQUIRED_SOURCE_CATEGORIES:
            raise SeedLedgerError(f"{label} has unsupported category")
        source_ref = source.get("source_ref")
        if not isinstance(source_ref, str) or not source_ref.strip() or source_ref in source_refs:
            raise SeedLedgerError(f"{label} requires a unique non-empty source_ref")
        source_refs.add(source_ref)
        digest = source.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise SeedLedgerError(f"{label} requires a lowercase source SHA-256")
        seeds = source.get("seed_ids")
        if not isinstance(seeds, list):
            raise SeedLedgerError(f"{label}.seed_ids must be a list")
        normalized = [_seed_id(seed, source=source_ref) for seed in seeds]
        excluded.update(normalized)
        covered_categories.add(category)
        source_audit.append({
            "category": category,
            "source_ref": source_ref,
            "sha256": digest,
            "seed_count": len(normalized),
        })

    missing = sorted(set(REQUIRED_SOURCE_CATEGORIES) - covered_categories)
    if missing:
        raise SeedLedgerError(f"seed inventory missing required categories: {', '.join(missing)}")
    return excluded, sorted(source_audit, key=lambda item: (item["category"], item["source_ref"]))


def extend_inventory_with_pools(
    inventory: Mapping[str, Any],
    pool_manifests: list[Mapping[str, Any]],
    *,
    inventory_id: str,
) -> dict[str, Any]:
    """Append generated pools to the reserved inventory for later rounds."""

    existing, _ = validate_inventory(inventory)
    if not isinstance(inventory_id, str) or not inventory_id.strip() or inventory_id == inventory.get("inventory_id"):
        raise SeedLedgerError("extended inventory requires a new non-empty inventory_id")
    source_manifests = list(inventory["source_manifests"])
    known_refs = {record["source_ref"] for record in source_manifests}
    appended: list[dict[str, Any]] = []
    blocked = set(existing)
    for index, pool in enumerate(pool_manifests):
        if not isinstance(pool, Mapping):
            raise SeedLedgerError(f"pool_manifests[{index}] must be an object")
        pool_id = pool.get("pool_id")
        digest = pool.get("manifest_sha256")
        if not isinstance(pool_id, str) or not pool_id.strip():
            raise SeedLedgerError(f"pool_manifests[{index}] requires pool_id")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise SeedLedgerError(f"pool_manifests[{index}] requires manifest_sha256")
        payload = {key: value for key, value in pool.items() if key != "manifest_sha256"}
        if sha256_json(payload) != digest:
            raise SeedLedgerError(f"pool manifest hash mismatch: {pool_id}")
        source_ref = f"generated-pool:{pool_id}"
        if source_ref in known_refs:
            raise SeedLedgerError(f"pool was already appended to this inventory: {pool_id}")
        seeds = pool.get("seed_ids")
        if not isinstance(seeds, list):
            raise SeedLedgerError(f"pool {pool_id} must contain seed_ids")
        normalized = [_seed_id(seed, source=pool_id) for seed in seeds]
        if len(normalized) != len(set(normalized)) or blocked.intersection(normalized):
            raise SeedLedgerError(f"pool overlaps a prior seed source or contains duplicates: {pool_id}")
        blocked.update(normalized)
        known_refs.add(source_ref)
        appended.append({
            "category": "reserved",
            "source_ref": source_ref,
            "sha256": digest,
            "seed_ids": normalized,
        })

    result = dict(inventory)
    result["inventory_id"] = inventory_id
    result["source_manifests"] = source_manifests + appended
    result["lineage"] = {
        "parent_inventory_id": inventory["inventory_id"],
        "parent_inventory_sha256": sha256_json(inventory),
        "appended_pool_manifests": [record["source_ref"] for record in appended],
    }
    validate_inventory(result)
    return result


def _candidate(seed_key: str, counter: int) -> tuple[int | None, int]:
    digest = hashlib.sha256(f"{GENERATOR_VERSION}\0{seed_key}\0{counter}".encode("utf-8")).digest()
    sample = int.from_bytes(digest[:8], "big")
    limit = (1 << 64) - ((1 << 64) % MAX_SEED)
    if sample >= limit:
        return None, counter + 1
    return sample % MAX_SEED + 1, counter + 1


def _allocate(
    *,
    seed_key: str,
    count: int,
    counter: int,
    excluded: set[int],
) -> tuple[list[int], int]:
    if count < 0 or count > MAX_SEED - len(excluded):
        raise SeedLedgerError("insufficient unused legal simulator seeds for requested pool")
    values: list[int] = []
    allocated: set[int] = set()
    while len(values) < count:
        candidate, counter = _candidate(seed_key, counter)
        if candidate is None or candidate in excluded or candidate in allocated:
            continue
        allocated.add(candidate)
        values.append(candidate)
    return values, counter


def generate_exploration_round(
    inventory: Mapping[str, Any],
    *,
    inventory_sha256: str,
    round_id: str,
    generation_key: str,
) -> dict[str, Any]:
    excluded, source_audit = validate_inventory(inventory)
    if not _SHA256_RE.fullmatch(inventory_sha256):
        raise SeedLedgerError("inventory_sha256 must be lowercase SHA-256")
    if not isinstance(round_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", round_id):
        raise SeedLedgerError("round_id must be a lowercase stable identifier")
    if not isinstance(generation_key, str) or not generation_key.strip():
        raise SeedLedgerError("generation_key is required")

    source_audit_hash = sha256_json(source_audit)
    counter = 0
    blocked = set(excluded)
    pools: dict[str, Any] = {}
    for name, size in EXPLORATION_POOL_SIZES.items():
        seeds, counter = _allocate(
            seed_key=f"{generation_key}\0{round_id}\0{name}",
            count=size,
            counter=counter,
            excluded=blocked,
        )
        blocked.update(seeds)
        purpose = "train" if name.startswith("train_") else name
        payload = {
            "schema_version": "sts1-g7-seed-pool-v1",
            "pool_id": f"{round_id}-{name}",
            "purpose": purpose,
            "role": name,
            "round_id": round_id,
            "generator": GENERATOR_VERSION,
            "generation_key": generation_key,
            "counter_end_exclusive": counter,
            "inventory_id": inventory["inventory_id"],
            "inventory_sha256": inventory_sha256,
            "source_audit_sha256": source_audit_hash,
            "source_audit": source_audit,
            "seed_ids": seeds,
            "status": "GENERATED_NOT_RUN",
        }
        payload["manifest_sha256"] = sha256_json(payload)
        pools[name] = payload

    result = {
        "schema_version": SCHEMA_VERSION,
        "round_id": round_id,
        "generation_key": generation_key,
        "generator": GENERATOR_VERSION,
        "inventory_id": inventory["inventory_id"],
        "inventory_sha256": inventory_sha256,
        "source_audit_sha256": source_audit_hash,
        "pools": pools,
        "status": "GENERATED_NOT_RUN",
    }
    result["ledger_sha256"] = sha256_json(result)
    _verify_generated(result, excluded)
    return result


def _verify_generated(ledger: Mapping[str, Any], excluded: set[int]) -> None:
    all_seeds: set[int] = set()
    for name, expected_count in EXPLORATION_POOL_SIZES.items():
        pool = ledger["pools"][name]
        seeds = [_seed_id(seed, source=pool["pool_id"]) for seed in pool["seed_ids"]]
        if len(seeds) != expected_count or len(seeds) != len(set(seeds)):
            raise SeedLedgerError(f"generated pool {name} has an invalid count or duplicate")
        if excluded.intersection(seeds):
            raise SeedLedgerError(f"generated pool {name} overlaps an existing seed source")
        if all_seeds.intersection(seeds):
            raise SeedLedgerError(f"generated pool {name} overlaps another new pool")
        all_seeds.update(seeds)


def generate_confirmation_trial(
    inventory: Mapping[str, Any],
    *,
    inventory_sha256: str,
    trial_k: int,
    prior_trials: list[Mapping[str, Any]],
    generation_key: str,
    candidate_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    excluded, source_audit = validate_inventory(inventory)
    if not _SHA256_RE.fullmatch(inventory_sha256):
        raise SeedLedgerError("inventory_sha256 must be lowercase SHA-256")
    if not isinstance(trial_k, int) or isinstance(trial_k, bool) or trial_k < 1:
        raise SeedLedgerError("confirmation trial k must be a positive integer")
    if not isinstance(prior_trials, list):
        raise SeedLedgerError("prior_trials must include every previous confirmation attempt")
    prior_ks: list[int] = []
    prior_seed_ids: set[int] = set()
    for index, record in enumerate(prior_trials):
        if not isinstance(record, Mapping):
            raise SeedLedgerError(f"prior_trials[{index}] must be an object")
        prior_k = record.get("confirmation_trial_k", record.get("trial_k"))
        result_status = record.get("result_status")
        digest = record.get("trial_manifest_sha256")
        if not isinstance(prior_k, int) or isinstance(prior_k, bool) or prior_k < 1:
            raise SeedLedgerError(f"prior_trials[{index}] has invalid k")
        if result_status not in {"pass", "fail", "invalid", "not_run"}:
            raise SeedLedgerError(f"prior_trials[{index}] has invalid result_status")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise SeedLedgerError(f"prior_trials[{index}] requires trial_manifest_sha256")
        digest_payload = {
            key: value
            for key, value in record.items()
            if key not in {"trial_manifest_sha256", "result_status"}
        }
        if sha256_json(digest_payload) != digest:
            raise SeedLedgerError(f"prior trial manifest hash mismatch for k={prior_k}")
        batches = record.get("pools")
        if not isinstance(batches, Mapping):
            raise SeedLedgerError(f"prior_trials[{index}] must retain both seed batches")
        trial_seeds: set[int] = set()
        for batch_name, expected_count in CONFIRMATION_POOL_SIZES.items():
            batch = batches.get(batch_name)
            if not isinstance(batch, Mapping) or not isinstance(batch.get("seed_ids"), list):
                raise SeedLedgerError(f"prior_trials[{index}] is missing {batch_name} seed IDs")
            pool_digest = batch.get("manifest_sha256")
            if not isinstance(pool_digest, str) or not _SHA256_RE.fullmatch(pool_digest):
                raise SeedLedgerError(f"prior trial {prior_k}/{batch_name} requires manifest_sha256")
            if sha256_json({key: value for key, value in batch.items() if key != "manifest_sha256"}) != pool_digest:
                raise SeedLedgerError(f"prior pool manifest hash mismatch: {prior_k}/{batch_name}")
            batch_seeds = [
                _seed_id(seed, source=f"prior trial {prior_k}/{batch_name}")
                for seed in batch["seed_ids"]
            ]
            if len(batch_seeds) != expected_count or len(batch_seeds) != len(set(batch_seeds)):
                raise SeedLedgerError(f"prior trial {prior_k}/{batch_name} has invalid seed count or duplicates")
            if trial_seeds.intersection(batch_seeds):
                raise SeedLedgerError(f"prior trial {prior_k} reuses seeds between batches")
            trial_seeds.update(batch_seeds)
        if prior_seed_ids.intersection(trial_seeds):
            raise SeedLedgerError("prior confirmation trials reuse seed IDs")
        prior_seed_ids.update(trial_seeds)
        prior_ks.append(prior_k)
    if prior_ks != list(range(1, trial_k)):
        raise SeedLedgerError("prior_trials must preserve every earlier k without gaps or resets")
    if not isinstance(generation_key, str) or not generation_key.strip():
        raise SeedLedgerError("generation_key is required")
    if not _SHA256_RE.fullmatch(candidate_sha256) or not _SHA256_RE.fullmatch(config_sha256):
        raise SeedLedgerError("frozen candidate and config SHA-256 values are required")

    source_audit_hash = sha256_json(source_audit)
    counter = 0
    prior_blocked = set(excluded) | prior_seed_ids
    blocked = set(prior_blocked)
    pools: dict[str, Any] = {}
    for name, size in CONFIRMATION_POOL_SIZES.items():
        seeds, counter = _allocate(
            seed_key=f"{generation_key}\0confirmation-k{trial_k}\0{name}",
            count=size,
            counter=counter,
            excluded=blocked,
        )
        blocked.update(seeds)
        payload = {
            "schema_version": "sts1-g7-seed-pool-v1",
            "pool_id": f"confirmation-k{trial_k}-{name}",
            "purpose": "confirmation",
            "role": name,
            "confirmation_trial_k": trial_k,
            "candidate_sha256": candidate_sha256,
            "config_sha256": config_sha256,
            "generator": GENERATOR_VERSION,
            "generation_key": generation_key,
            "counter_end_exclusive": counter,
            "inventory_id": inventory["inventory_id"],
            "inventory_sha256": inventory_sha256,
            "source_audit_sha256": source_audit_hash,
            "source_audit": source_audit,
            "seed_ids": seeds,
            "status": "GENERATED_NOT_RUN",
        }
        payload["manifest_sha256"] = sha256_json(payload)
        pools[name] = payload

    alpha = Fraction(1, 20 * 2**trial_k)
    result = {
        "schema_version": "sts1-g7-confirmation-trial-v1",
        "confirmation_trial_k": trial_k,
        "alpha_formula": f"0.05 / 2^{trial_k}",
        "alpha_exact": f"{alpha.numerator}/{alpha.denominator}",
        "generation_key": generation_key,
        "generator": GENERATOR_VERSION,
        "inventory_id": inventory["inventory_id"],
        "inventory_sha256": inventory_sha256,
        "source_audit_sha256": source_audit_hash,
        "candidate_sha256": candidate_sha256,
        "config_sha256": config_sha256,
        "prior_trials": [dict(record) for record in prior_trials],
        "pools": pools,
        "status": "GENERATED_NOT_RUN",
    }
    result["trial_manifest_sha256"] = sha256_json(result)
    all_seeds: set[int] = set()
    for name, expected_count in CONFIRMATION_POOL_SIZES.items():
        seeds = pools[name]["seed_ids"]
        if len(seeds) != expected_count or len(seeds) != len(set(seeds)):
            raise SeedLedgerError(f"generated confirmation batch {name} has an invalid count or duplicate")
        if prior_blocked.intersection(seeds) or all_seeds.intersection(seeds):
            raise SeedLedgerError(f"generated confirmation batch {name} overlaps an existing seed source")
        all_seeds.update(seeds)
    return result


def write_exploration_round(
    *,
    inventory_path: Path,
    output_dir: Path,
    repo_root: Path,
    round_id: str,
    generation_key: str,
) -> dict[str, Any]:
    inventory, inventory_sha256 = _read_inventory(inventory_path)
    ledger = generate_exploration_round(
        inventory,
        inventory_sha256=inventory_sha256,
        round_id=round_id,
        generation_key=generation_key,
    )
    destination = _create_output_dir(output_dir, repo_root)
    for name, pool in ledger["pools"].items():
        (destination / f"{name}.json").write_text(
            json.dumps(pool, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    (destination / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return ledger


def write_confirmation_trial(
    *,
    inventory_path: Path,
    prior_trials_path: Path,
    output_dir: Path,
    repo_root: Path,
    trial_k: int,
    generation_key: str,
    candidate_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    inventory, inventory_sha256 = _read_inventory(inventory_path)
    prior_trials = json.loads(prior_trials_path.read_text(encoding="utf-8"))
    trial = generate_confirmation_trial(
        inventory,
        inventory_sha256=inventory_sha256,
        trial_k=trial_k,
        prior_trials=prior_trials,
        generation_key=generation_key,
        candidate_sha256=candidate_sha256,
        config_sha256=config_sha256,
    )
    destination = _create_output_dir(output_dir, repo_root)
    for name, pool in trial["pools"].items():
        (destination / f"{name}.json").write_text(
            json.dumps(pool, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    (destination / "trial.json").write_text(
        json.dumps(trial, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return trial


def _read_inventory(path: Path) -> tuple[Mapping[str, Any], str]:
    raw = path.read_bytes()
    inventory = json.loads(raw.decode("utf-8"))
    if not isinstance(inventory, Mapping):
        raise SeedLedgerError("inventory JSON root must be an object")
    return inventory, hashlib.sha256(raw).hexdigest()


def _create_output_dir(output_dir: Path, repo_root: Path) -> Path:
    allowed_root = (repo_root / "evidence" / "sts1" / "g7-improvement" / "seeds").resolve()
    destination = output_dir.resolve()
    if destination != allowed_root and allowed_root not in destination.parents:
        raise SeedLedgerError("output directory must stay under evidence/sts1/g7-improvement/seeds")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    exploration = subparsers.add_parser("exploration", help="generate non-overlapping train/probe/dev pools")
    confirmation = subparsers.add_parser("confirmation", help="generate two frozen-candidate confirmation batches")
    for subparser in (exploration, confirmation):
        subparser.add_argument("--inventory", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--generation-key", required=True)
    exploration.add_argument("--round-id", required=True)
    confirmation.add_argument("--trial-k", type=int, required=True)
    confirmation.add_argument("--prior-trials", type=Path, required=True)
    confirmation.add_argument("--candidate-sha256", required=True)
    confirmation.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    try:
        if args.mode == "exploration":
            ledger = write_exploration_round(
                inventory_path=args.inventory,
                output_dir=output_dir,
                repo_root=repo_root,
                round_id=args.round_id,
                generation_key=args.generation_key,
            )
            summary_key = "ledger_sha256"
        else:
            ledger = write_confirmation_trial(
                inventory_path=args.inventory,
                prior_trials_path=args.prior_trials,
                output_dir=output_dir,
                repo_root=repo_root,
                trial_k=args.trial_k,
                generation_key=args.generation_key,
                candidate_sha256=args.candidate_sha256,
                config_sha256=args.config_sha256,
            )
            summary_key = "trial_manifest_sha256"
    except (OSError, json.JSONDecodeError, SeedLedgerError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "round_id": ledger.get("round_id"),
        "confirmation_trial_k": ledger.get("confirmation_trial_k"),
        "manifest_sha256": ledger[summary_key],
        "pools": {
            name: {"count": len(pool["seed_ids"]), "manifest_sha256": pool["manifest_sha256"]}
            for name, pool in ledger["pools"].items()
        },
        "status": ledger["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
