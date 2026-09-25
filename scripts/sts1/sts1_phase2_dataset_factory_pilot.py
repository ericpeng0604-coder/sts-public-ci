"""Formal bounded STS1 Phase-2 Dataset Factory pilot from frozen Phase-1 artifacts.

The pilot consumes only already-PASS public-state Search evidence.  It never
reruns the frozen Teacher, never reads private BattleContext/RNG state, and
keeps the two explicitly-held-out enemy suites out of the training dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any
import zipfile

from roguelike_ai.sts1_training import DatasetBuildConfig, build_dataset
from roguelike_ai.sts1_training.teacher_capture import (
    FORMAL_SIMULATOR_SHA,
    FORMAL_TEACHER_CONFIG_HASH,
    FORMAL_TEACHER_SHA,
    capture_frozen_search_record,
    formal_teacher_provenance,
)


FORMAL_CAPTURE_CODE_SHA = "775dd9baa3b1c8a4841af490b7fdfeb3f384fc47"
FORMAL_TEACHER_ID = "public-state-search-v1"
PILOT_SPLIT_SALT = "sts1-phase2-dataset-factory-pilot-v1"
PILOT_SHARD_SIZE = 16


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: int
    name: str
    workflow_run_id: int
    head_sha: str
    zip_sha256: str
    suite_path: str
    decisions_path: str
    summary_path: str
    suite_digest: str
    rows_digest: str
    source_seed_file_sha256: str
    train_eligible: bool


ARTIFACTS = (
    ArtifactSpec(
        artifact_id=9710934904,
        name="sts1-phase1-public-search-v1-quality-evidence",
        workflow_run_id=33239485496,
        head_sha="ed5a906a082e8c01f8c3c62263fbe5fdd7b2e33a",
        zip_sha256="45009ad33ecb789ed07cac9da81d2c1c8f33e6d05e0a951abac24897e09e9e31",
        suite_path="suite.json",
        decisions_path="decisions.ndjson",
        summary_path="summary.json",
        suite_digest="d4df9693b2705a2c238807bf39e2ee19fa4f0e69088b2bc1965e91772a334baf",
        rows_digest="bc9b097d08dcb7cd21ed4909787adc833e468f6dfed9fe066e0ec8d6606400d6",
        source_seed_file_sha256="4cf653503c6c8f170ee0ca65c53a9809e635dc71d511433365ae9d76ffc332d8",
        train_eligible=True,
    ),
    ArtifactSpec(
        artifact_id=9711504622,
        name="sts1-phase1-public-search-v1-blue-slaver-quality-evidence",
        workflow_run_id=33241524526,
        head_sha="f5c0209690372402ddf2f8ab67356e2370312061",
        zip_sha256="9a50e4f578862edf17fb3ea2271b0be94aee879caf934443250178bf5ed99d9e",
        suite_path="suite.json",
        decisions_path="decisions.ndjson",
        summary_path="summary.json",
        suite_digest="44d8cfb732cee1996d89b441383449d096a3a655588d1b0c5510d1c57bac2b88",
        rows_digest="2b5be3ac622f72bcf26ef54c6708f8d046106c9d4c5b806b36ac483d705e21f0",
        source_seed_file_sha256="57967af095d099f74969ec7f37758d8d1f96f9074c6640b8e73d098aa8f8693d",
        train_eligible=True,
    ),
    ArtifactSpec(
        artifact_id=9710933572,
        name="sts1-phase1-public-search-v1-cultist-quality-evidence",
        workflow_run_id=33239485496,
        head_sha="ed5a906a082e8c01f8c3c62263fbe5fdd7b2e33a",
        zip_sha256="f37019daee1671fc376c91cb78790b97fce16f39b24ebc683c032064778d3b25",
        suite_path="suite.json",
        decisions_path="decisions.ndjson",
        summary_path="summary.json",
        suite_digest="d2978c1e5cb7ff6f62c95b19aa39147b811095fea2f4ff6cce189c1e1b2ef0e9",
        rows_digest="c4364804ff005f05df103b78bb182331c16961fb08108103afbd0faf567fec15",
        source_seed_file_sha256="d51d53c976ab1b03935203a808bde29297a4cb81ba472429248c22ee7d214142",
        train_eligible=True,
    ),
    ArtifactSpec(
        artifact_id=9712420709,
        name="sts1-phase1-public-search-v1-jaw-worm-turn-one-quality-evidence",
        workflow_run_id=33244551094,
        head_sha="4980ae1588170c952de131d0c9d67a54a33762f3",
        zip_sha256="91455b7e47a28de365cd6dfba893ea3cfc0fea96a45b7981154d76fd4656eb8f",
        suite_path="suite.json",
        decisions_path="decisions.ndjson",
        summary_path="summary.json",
        suite_digest="0fcf59f0b8936c5f80e8cc4d5b9cefce3799ea21835425a06a25f9d98839906f",
        rows_digest="14e3840f7dacef844bbef2f4486bf401399b43caeebd37db0a1ebdcb8142a2a6",
        source_seed_file_sha256="27b9ea4b5b1319d91cbb5c304b4de123da92f885363f375b8e7df357e1783b8d",
        train_eligible=True,
    ),
    ArtifactSpec(
        artifact_id=9727266093,
        name="sts1-phase1-red-slaver-fresh-holdout-evidence",
        workflow_run_id=33295427938,
        head_sha="ee447942e59c16c356268f427cda4651bc6b9091",
        zip_sha256="d815192f73927fd6cfffa7da3871d18d1501e2397f7adc014eac4fa81b13ff68",
        suite_path="sts1/red-slaver-holdout/suite.json",
        decisions_path="sts1/red-slaver-holdout/decisions.ndjson",
        summary_path="sts1/red-slaver-holdout/summary.json",
        suite_digest="5631a2352dea6cbed958f5b4552195ecc0c170fa7aa41b107e1b41e162ef868e",
        rows_digest="944adcdb93fae9738118befec86392fcfb94a6f297ef7a43bb749a3ff852853a",
        source_seed_file_sha256="bbd413a5621bd9cac89a36521261b420d1204d542dc953bd606de9ac8f9b592b",
        train_eligible=False,
    ),
    ArtifactSpec(
        artifact_id=9731656268,
        name="sts1-phase1-looter-v2-fresh-holdout-evidence",
        workflow_run_id=33309874208,
        head_sha="7a152f46e0e198ffc35e2a27e623f0fb6c4edffb",
        zip_sha256="60cfe47fb6d1c44f7c48cfa59f1c1ed5bde40a3d021c96524b2d389967411f9a",
        suite_path="suite.json",
        decisions_path="decisions.ndjson",
        summary_path="summary.json",
        suite_digest="8e550f1f016daaddaa1fceed3e5f19027fc8f63f50f3e70894839b2beb77fd2f",
        rows_digest="c34c7fcc91b4bf13e4045f78dd55c939345732ec6fac13c4bdbdf3b745095ba6",
        source_seed_file_sha256="5d7d538a1f1c38881d41e6e6f3e3a0b2a8fbd4534fe5393d070a2848588d359c",
        train_eligible=False,
    ),
    ArtifactSpec(
        artifact_id=9735798757,
        name="sts1-phase1-cultist-turn1-fresh-holdout-evidence",
        workflow_run_id=33324349048,
        head_sha="54b4d8f0e244fa29cbf6b5315bd55fb1f0148ed7",
        zip_sha256="f10887573c42660f2248adeb6688369c3a0fcc7c19f3fe7e6b4eb8894d42b8be",
        suite_path="suite.json",
        decisions_path="decisions.ndjson",
        summary_path="summary.json",
        suite_digest="76cd841039fae2e3e26410057a8bc1e36eb54b2d412b7b51da0ad9d4b0645d28",
        rows_digest="21b4dab38a597d8650dfba228a5d545bf09c3cc5958eb694fe1f48bf15a2e99d",
        source_seed_file_sha256="ba9580dfb7e2516fa62fc5ee0c13107cdc0af7631eac0a7eafde344d92a339bd",
        train_eligible=False,
    ),
    ArtifactSpec(
        artifact_id=9735057343,
        name="sts1-phase1-looter-turn1-fresh-holdout-evidence",
        workflow_run_id=33321649403,
        head_sha="464979b20d2b705338ffc4498cbd7b7cf572450b",
        zip_sha256="0a1f607ccfce83f00c3e6f04cad44449f7dd2ea1725883bf91ef046f5e4a292f",
        suite_path="suite.json",
        decisions_path="decisions.ndjson",
        summary_path="summary.json",
        suite_digest="1ad72a4a2d868ca040a358bf1be75fa383321e6cedf5208841fc2814a85af0b0",
        rows_digest="10a15240f0a46bcda9f1fd3ba21aa8ea407d3fbd695a0b32b435e99699525d57",
        source_seed_file_sha256="37f4470b5e8a45eb1143366f8d0e8d94636a9ea47aaf24c61e7ea6e2cb1e18e6",
        train_eligible=False,
    ),
)


def _run_checked(args: list[str], *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(args, check=True, capture_output=True, text=not binary)
    return completed.stdout


def _git_head() -> str:
    return str(_run_checked(["git", "rev-parse", "HEAD"])).strip()


def _artifact_metadata(spec: ArtifactSpec) -> dict[str, Any]:
    raw = _run_checked(
        ["gh", "api", f"repos/ericpeng0604-coder/sts-public-ci/actions/artifacts/{spec.artifact_id}"]
    )
    payload = json.loads(str(raw))
    if payload.get("name") != spec.name:
        raise RuntimeError(f"artifact {spec.artifact_id} name drift")
    workflow_run = payload.get("workflow_run")
    if not isinstance(workflow_run, dict):
        raise RuntimeError(f"artifact {spec.artifact_id} missing workflow_run metadata")
    if workflow_run.get("id") != spec.workflow_run_id or workflow_run.get("head_sha") != spec.head_sha:
        raise RuntimeError(f"artifact {spec.artifact_id} workflow identity drift")
    if payload.get("expired") is not False:
        raise RuntimeError(f"artifact {spec.artifact_id} is expired")
    return payload


def _artifact_zip(spec: ArtifactSpec) -> bytes:
    raw = _run_checked(
        ["gh", "api", f"repos/ericpeng0604-coder/sts-public-ci/actions/artifacts/{spec.artifact_id}/zip"],
        binary=True,
    )
    assert isinstance(raw, bytes)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec.zip_sha256:
        raise RuntimeError(f"artifact {spec.artifact_id} ZIP SHA256 drift: {digest}")
    return raw


def _load_artifact(spec: ArtifactSpec) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    _artifact_metadata(spec)
    archive = _artifact_zip(spec)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = set(zipped.namelist())
        expected_names = {spec.suite_path, spec.decisions_path, spec.summary_path}
        if not expected_names.issubset(names):
            raise RuntimeError(f"artifact {spec.artifact_id} missing expected files")
        suite = json.loads(zipped.read(spec.suite_path))
        summary = json.loads(zipped.read(spec.summary_path))
        raw_rows = zipped.read(spec.decisions_path)

    if hashlib.sha256(raw_rows).hexdigest() != spec.rows_digest:
        raise RuntimeError(f"artifact {spec.artifact_id} decisions digest drift")
    if suite.get("suite_digest") != spec.suite_digest or summary.get("suite_digest") != spec.suite_digest:
        raise RuntimeError(f"artifact {spec.artifact_id} suite digest drift")
    if summary.get("rows_digest") != spec.rows_digest:
        raise RuntimeError(f"artifact {spec.artifact_id} summary rows digest drift")
    source_seed_sha = suite.get("source_seed_file_sha256") or summary.get("source_seed_file_sha256")
    if source_seed_sha != spec.source_seed_file_sha256:
        raise RuntimeError(f"artifact {spec.artifact_id} source seed digest drift")
    if summary.get("quality_verdict") != "PASS" or summary.get("deterministic") is not True:
        raise RuntimeError(f"artifact {spec.artifact_id} is not an admitted deterministic PASS slice")
    if summary.get("search_unresolved") != 0:
        raise RuntimeError(f"artifact {spec.artifact_id} reports unresolved Search rows")
    if not spec.train_eligible and summary.get("suite_kind") != "holdout":
        raise RuntimeError(f"artifact {spec.artifact_id} lost holdout identity")

    rows = [json.loads(line) for line in raw_rows.decode("utf-8").splitlines() if line.strip()]
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) != len(rows) or summary.get("case_count") != len(rows):
        raise RuntimeError(f"artifact {spec.artifact_id} case-count mismatch")
    case_map: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise RuntimeError(f"artifact {spec.artifact_id} malformed suite case")
        case_id = case["case_id"]
        if case_id in case_map:
            raise RuntimeError(f"artifact {spec.artifact_id} duplicate suite case {case_id}")
        case_map[case_id] = case
    return rows, case_map, summary


def _serialized(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _duplicate_audit(records: list[dict[str, Any]]) -> dict[str, int]:
    seen: dict[tuple[str, str, int, str], str] = {}
    identical = 0
    contradictory = 0
    for record in records:
        provenance = record["provenance"]
        key = (
            str(provenance["seed"]),
            str(provenance["run_id"]),
            int(provenance["decision_index"]),
            str(record["decision_signature"]),
        )
        payload = _serialized(record)
        previous = seen.get(key)
        if previous is None:
            seen[key] = payload
        elif previous == payload:
            identical += 1
        else:
            contradictory += 1
    return {
        "unique_records": len(seen),
        "identical_duplicates": identical,
        "contradictory_duplicates": contradictory,
    }


def _capture_all() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    producer = formal_teacher_provenance(
        teacher_id=FORMAL_TEACHER_ID,
        capture_code_sha=FORMAL_CAPTURE_CODE_SHA,
        worker_generation=1,
    )
    train_eligible: list[dict[str, Any]] = []
    external_holdout: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []

    for spec in ARTIFACTS:
        rows, case_map, summary = _load_artifact(spec)
        captured: list[dict[str, Any]] = []
        for index, teacher_decision in enumerate(rows):
            case_id = teacher_decision.get("case_id")
            if not isinstance(case_id, str) or case_id not in case_map:
                raise RuntimeError(f"artifact {spec.artifact_id} decision has no matching suite case")
            case = case_map[case_id]
            public_state = case.get("public_state")
            if not isinstance(public_state, dict):
                raise RuntimeError(f"artifact {spec.artifact_id} case {case_id} missing public_state")
            if case.get("decision_signature") != teacher_decision.get("decision_signature"):
                raise RuntimeError(f"artifact {spec.artifact_id} case {case_id} signature mismatch")
            if case.get("source_seed_provenance_only") != teacher_decision.get("source_seed_provenance_only"):
                raise RuntimeError(f"artifact {spec.artifact_id} case {case_id} seed mismatch")

            record = capture_frozen_search_record(
                public_state=public_state,
                teacher_decision=teacher_decision,
                seed=teacher_decision["source_seed_provenance_only"],
                run_id=case_id,
                decision_index=0,
                provenance=producer,
                worker_id="sts1-phase2-dataset-factory-pilot",
                floor=public_state.get("floor") if isinstance(public_state.get("floor"), int) else None,
                combat_id=f"{case_id}:combat:1",
            )
            serialized = _serialized(record)
            if record["teacher"]["selected_action_id"] not in record["teacher"]["tie_action_ids"]:
                raise RuntimeError(f"artifact {spec.artifact_id} selected label escaped Teacher tie set")
            if any(key in record["observation"] for key in ("source", "reconstruction", "decision_signature")):
                raise RuntimeError(f"artifact {spec.artifact_id} leaked capture diagnostics into policy observation")
            if any(token in serialized for token in ("oracle_scores", "oracle_tie_ids", "simple_action_id", "random_action_id")):
                raise RuntimeError(f"artifact {spec.artifact_id} leaked diagnostic labels")
            captured.append(record)

        target = train_eligible if spec.train_eligible else external_holdout
        target.extend(captured)
        inventory.append(
            {
                "artifact_id": spec.artifact_id,
                "artifact_name": spec.name,
                "workflow_run_id": spec.workflow_run_id,
                "workflow_head_sha": spec.head_sha,
                "zip_sha256": spec.zip_sha256,
                "suite_digest": spec.suite_digest,
                "rows_digest": spec.rows_digest,
                "source_seed_file_sha256": spec.source_seed_file_sha256,
                "case_count": len(captured),
                "quality_verdict": summary.get("quality_verdict"),
                "train_eligible": spec.train_eligible,
                "external_holdout": not spec.train_eligible,
            }
        )

    return train_eligible, external_holdout, inventory


def _run(output_root: Path) -> dict[str, Any]:
    head = _git_head()
    if head != FORMAL_CAPTURE_CODE_SHA:
        raise RuntimeError(f"capture code HEAD drift: expected {FORMAL_CAPTURE_CODE_SHA}, got {head}")

    train_records, external_holdout, inventory = _capture_all()
    accepted_records = train_records + external_holdout
    duplicate_audit = _duplicate_audit(accepted_records)
    if duplicate_audit["contradictory_duplicates"] != 0:
        raise RuntimeError("contradictory duplicate audit failed")

    config = DatasetBuildConfig(
        split_salt=PILOT_SPLIT_SALT,
        shard_size=PILOT_SHARD_SIZE,
        require_all_splits=True,
    )
    producer = formal_teacher_provenance(
        teacher_id=FORMAL_TEACHER_ID,
        capture_code_sha=FORMAL_CAPTURE_CODE_SHA,
        worker_generation=1,
    )
    first = build_dataset(train_records, output_root / "first", expected_provenance=producer, config=config)
    second = build_dataset(
        list(reversed(train_records)),
        output_root / "second",
        expected_provenance=producer,
        config=config,
    )
    if first["dataset_hash"] != second["dataset_hash"] or first["shards"] != second["shards"]:
        raise RuntimeError("Dataset Factory output is not deterministic across input ordering")

    split_seed_sets = {
        split: set(first["split_seed_hashes"][split]) for split in ("train", "validation", "holdout")
    }
    seed_leakage = sum(
        len(split_seed_sets[left].intersection(split_seed_sets[right]))
        for left, right in (("train", "validation"), ("train", "holdout"), ("validation", "holdout"))
    )
    if seed_leakage != 0:
        raise RuntimeError("whole-seed split leakage detected")

    external_holdout_seed_hashes = {
        hashlib.sha256(str(record["provenance"]["seed"]).encode("utf-8")).hexdigest()
        for record in external_holdout
    }
    training_dataset_seed_hashes = set().union(*split_seed_sets.values())
    external_holdout_seed_overlap = len(external_holdout_seed_hashes.intersection(training_dataset_seed_hashes))
    if external_holdout_seed_overlap != 0:
        raise RuntimeError("formal external holdout seeds entered the training dataset")

    tie_sizes = Counter(len(record["teacher"]["tie_action_ids"]) for record in accepted_records)
    unique_signatures = len({record["decision_signature"] for record in accepted_records})
    return {
        "result": "STS1_DATASET_FACTORY_PILOT_PASS",
        "capture_code_sha": head,
        "teacher_sha": FORMAL_TEACHER_SHA,
        "teacher_config_hash": FORMAL_TEACHER_CONFIG_HASH,
        "simulator_sha": FORMAL_SIMULATOR_SHA,
        "source_artifacts": inventory,
        "input_decisions": len(accepted_records),
        "accepted_decisions": len(accepted_records),
        "rejected_decisions": 0,
        "rejection_reasons": {},
        "train_eligible_decisions": len(train_records),
        "external_holdout_decisions": len(external_holdout),
        "external_holdout_untouched": True,
        "formal_source_inventory_supports_1000": len(accepted_records) >= 1000,
        "unique_decision_signatures": unique_signatures,
        "tie_size_distribution": {str(size): count for size, count in sorted(tie_sizes.items())},
        "selected_label_outside_teacher_tie_set": 0,
        "legal_mismatch": 0,
        "hidden_leakage": 0,
        "timeout_or_unresolved": 0,
        "stale_provenance": 0,
        "duplicate_audit": duplicate_audit,
        "split_contract": config.to_dict(),
        "split_counts": first["split_counts"],
        "unique_seed_counts": first["unique_seed_counts"],
        "seed_leakage": seed_leakage,
        "external_holdout_seed_overlap": external_holdout_seed_overlap,
        "shards": first["shards"],
        "dataset_hash_first": first["dataset_hash"],
        "dataset_hash_second": second["dataset_hash"],
        "dataset_hash_identical": True,
        "policy_observation_contains_seed": first["policy_observation_contains_seed"],
        "provenance_is_not_policy_input": first["provenance_is_not_policy_input"],
        "holdout_training_allowed": first["holdout_training_allowed"],
        "student_training": "LOCKED",
        "ppo": "LOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Persist the two deterministic pilot dataset builds here")
    args = parser.parse_args()

    if args.output is not None:
        result = _run(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="sts1-phase2-dataset-factory-pilot-") as directory:
            result = _run(Path(directory))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
