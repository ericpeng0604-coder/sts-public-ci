"""Build a private, ID-only exclusion inventory for STS1 G7 seed pools.

The collector reads dedicated seed lists and numeric values on seed-named
fields in control triggers/requests. It never opens evidence, episodes,
evaluation summaries, trajectories, decisions, or labels. Raw IDs are written
only to the caller-specified private inventory outside the repository. The
public audit contains source references, hashes, and counts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.sts1.sts1_g7_seed_ledger import (
    INVENTORY_SCHEMA_VERSION,
    MAX_SEED,
    REQUIRED_SOURCE_CATEGORIES,
    canonical_json_bytes,
    sha256_json,
    validate_inventory,
)


PRIVATE_SOURCE_SCHEMA = "sts1-private-seed-source-v1"
AUDIT_SCHEMA = "sts1-seed-source-audit-v1"
_SEED_FIELD = re.compile(
    r"^\s*[\"']?([A-Za-z0-9_-]*seed[A-Za-z0-9_-]*)[\"']?\s*[:=]\s*(.*?)\s*$",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])\d+(?![A-Za-z0-9_.])")
_EXCLUDED_NAMES = re.compile(r"(?i)(summary|result|report|episode|trajectory|trace)")


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_seed_list(text: str, *, source: str) -> list[int]:
    """Parse a dedicated numeric seed list; reject any ambiguous data line."""

    result: list[int] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = [token for token in re.split(r"[\s,;]+", line) if token]
        if not tokens or any(not token.isdecimal() for token in tokens):
            raise ValueError(f"ambiguous seed-list row in {source}:{line_number}")
        result.extend(_validate_ids((int(token) for token in tokens), source=source))
    return result


def _parse_seed_field_candidates(text: str, *, source: str) -> list[int]:
    """Extract only numeric literals from keys containing `seed`.

    Counts and RNG literals are conservatively excluded too. This collector
    does not infer game outcomes or interpret non-seed fields.
    """

    result: list[int] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = _SEED_FIELD.match(line)
        if not match:
            continue
        rhs = match.group(2)
        result.extend(
            _validate_ids(
                (int(token) for token in _NUMBER.findall(rhs)),
                source=source,
            )
        )
    return result


def _validate_ids(values: Sequence[int] | Any, *, source: str) -> list[int]:
    normalized: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_SEED:
            raise ValueError(f"seed candidate outside pinned simulator uint64 domain in {source}")
        normalized.append(value)
    return normalized


def _category_for(path: Path) -> str:
    name = path.name.lower()
    if "dev" in name:
        return "dev"
    if "probe" in name:
        return "probe"
    if "gate" in name or "promotion" in name:
        return "gate"
    if "fresh" in name or "benchmark" in name:
        return "fresh"
    if "mine" in name or "priority" in name:
        return "mining"
    if "formal" in name or "reserved" in name or "eval" in name:
        return "reserved"
    if "train" in name or "teacher" in name:
        return "train_used"
    return "unknown"


def _manifest(category: str, source_ref: str, digest: str, ids: Sequence[int]) -> dict[str, Any]:
    return {
        "category": category,
        "source_ref": source_ref,
        "sha256": digest,
        "seed_ids": sorted(set(_validate_ids(ids, source=source_ref))),
    }


def _discover_local_sources(root: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seed_lists = sorted(
        {
            *(
                path
                for path in (root / "control").rglob("*.txt")
                if "seed" in path.name.lower() and not _EXCLUDED_NAMES.search(path.name)
            ),
            *(
                path
                for path in (root / "fixtures" / "sts1").rglob("*.txt")
                if "seed" in path.name.lower() and not _EXCLUDED_NAMES.search(path.name)
            ),
        }
    )
    for path in seed_lists:
        relative = path.relative_to(root).as_posix()
        ids = _parse_seed_list(path.read_text(encoding="utf-8-sig"), source=relative)
        sources.append(_manifest(_category_for(path), relative, _source_sha256(path), ids))

    control = root / "control"
    configs = sorted(
        path
        for path in control.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".trigger", ".request"}
        and not _EXCLUDED_NAMES.search(path.name)
    )
    for path in configs:
        relative = path.relative_to(root).as_posix()
        ids = _parse_seed_field_candidates(path.read_text(encoding="utf-8-sig"), source=relative)
        if ids:
            sources.append(
                _manifest("unknown", f"{relative}#seed-field-candidates", _source_sha256(path), ids)
            )
    return sources


def _read_private_source(path: Path) -> dict[str, Any]:
    private = json.loads(path.read_text(encoding="utf-8-sig"))
    if private.get("schema_version") != PRIVATE_SOURCE_SCHEMA:
        raise ValueError("unsupported private seed source schema")
    category = private.get("category")
    source_ref = private.get("source_ref")
    digest = private.get("sha256")
    ids = private.get("seed_ids")
    if not isinstance(category, str) or not isinstance(source_ref, str) or not isinstance(digest, str):
        raise ValueError("private seed source metadata is incomplete")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("private seed source must include a lowercase SHA-256")
    if not isinstance(ids, list):
        raise ValueError("private seed source IDs must be a list")
    return _manifest(category, source_ref, digest, ids)


def _legacy_protection_source(root: Path) -> tuple[dict[str, Any], str]:
    """Reconstruct the v3.5 protection union using the reviewed local ledger."""

    from scripts.sts1.sts1_build_rescue_v35_seed_ledger import protected_seeds

    generator = Path(sys.modules["scripts.sts1.sts1_build_rescue_v35_seed_ledger"].__file__)
    ids = sorted(protected_seeds(root))
    digest = _source_sha256(generator)
    ref = f"reconstructed:{generator.relative_to(root).as_posix()}::protected_seeds"
    return _manifest("unknown", ref, digest, ids), sha256_json(ids)


def build_inventory(
    root: Path,
    *,
    inventory_id: str,
    private_seed_sources: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    sources = _discover_local_sources(root)
    for path in private_seed_sources:
        sources.append(_read_private_source(path))
    legacy, legacy_ids_sha256 = _legacy_protection_source(root)
    sources.append(legacy)

    covered = {source["category"] for source in sources}
    if "probe" not in covered:
        # No dedicated probe list is checked in. Exclude the complete legacy
        # protection union as a conservative probe superset, without claiming
        # category-specific attribution.
        sources.append(
            _manifest(
                "probe",
                f"conservative-superset:{legacy['source_ref']}",
                legacy["sha256"],
                legacy["seed_ids"],
            )
        )

    inventory: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "inventory_id": inventory_id,
        "complete": True,
        "source_manifests": sources,
    }
    excluded, source_audit = validate_inventory(inventory)
    public_audit = {
        "schema_version": AUDIT_SCHEMA,
        "inventory_id": inventory_id,
        "inventory_sha256": sha256_json(inventory),
        "unique_excluded_count": len(excluded),
        "source_manifests": source_audit,
        "derived_legacy_union_sha256": legacy_ids_sha256,
        "coverage_notes": [
            "Raw seed IDs are storage-only in the private inventory.",
            "The legacy protection union is a conservative superset; it is not a category-level outcome record.",
            "Only numeric seed-list rows and numeric seed-named control fields were extracted.",
            "No evidence, result, episode, trajectory, decision, or label files were opened.",
        ],
    }
    return inventory, public_audit


def write_inventory(
    inventory: Mapping[str, Any],
    public_audit: Mapping[str, Any],
    *,
    private_path: Path,
    audit_path: Path,
    repo_root: Path,
) -> None:
    root = repo_root.resolve()
    private_path = private_path.resolve()
    audit_path = audit_path.resolve()
    if private_path == root or root in private_path.parents:
        raise ValueError("private seed inventory must be outside the repository")
    if audit_path != root and root not in audit_path.parents:
        raise ValueError("public source audit must be inside the repository")
    if private_path.exists() or audit_path.exists():
        raise FileExistsError("refusing to overwrite an existing inventory artifact")
    private_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(canonical_json_bytes(inventory) + b"\n")
    audit_path.write_bytes(canonical_json_bytes(public_audit) + b"\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-audit", type=Path, required=True)
    parser.add_argument("--private-seed-source", type=Path, action="append", default=[])
    parser.add_argument("--inventory-id", required=True)
    args = parser.parse_args(argv)

    inventory, audit = build_inventory(
        args.repo_root,
        inventory_id=args.inventory_id,
        private_seed_sources=args.private_seed_source,
    )
    write_inventory(
        inventory,
        audit,
        private_path=args.private_output,
        audit_path=args.public_audit,
        repo_root=args.repo_root,
    )
    by_category: dict[str, int] = {category: 0 for category in REQUIRED_SOURCE_CATEGORIES}
    for source in audit["source_manifests"]:
        by_category[source["category"]] += source["seed_count"]
    print(
        json.dumps(
            {
                "inventory_id": audit["inventory_id"],
                "inventory_sha256": audit["inventory_sha256"],
                "unique_excluded_count": audit["unique_excluded_count"],
                "source_count": len(audit["source_manifests"]),
                "source_counts_by_category_with_overlap": by_category,
                "private_inventory_written": True,
                "public_audit": str(args.public_audit),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
