"""Human path preference prior for STS1 Ironclad.

Run History does not contain the full map candidate set for each decision, so
this module deliberately learns a bounded *preference prior*, not behavior
cloning labels.  Promotion must still be decided by simulator gates.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping

from roguelike_ai.sts1_phase3.human_expert import act_bucket, iter_run_files


SCHEMA_VERSION = "sts1-human-path-prior-v1"
ROOMS = ("MONSTER", "ELITE", "REST", "SHOP", "EVENT", "TREASURE", "BOSS")
ROOM_ALIASES = {
    "M": "MONSTER",
    "MONSTER": "MONSTER",
    "E": "ELITE",
    "ELITE": "ELITE",
    "R": "REST",
    "REST": "REST",
    "$": "SHOP",
    "SHOP": "SHOP",
    "?": "EVENT",
    "EVENT": "EVENT",
    "T": "TREASURE",
    "TREASURE": "TREASURE",
    "B": "BOSS",
    "BOSS": "BOSS",
}


def normalize_room(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return ROOM_ALIASES.get(text)


def hp_bucket(hp: Any, max_hp: Any) -> str:
    if not isinstance(hp, (int, float)) or not isinstance(max_hp, (int, float)) or max_hp <= 0:
        return "unknown"
    ratio = max(0.0, min(1.0, float(hp) / float(max_hp)))
    if ratio < 0.35:
        return "critical"
    if ratio < 0.60:
        return "low"
    if ratio < 0.80:
        return "medium"
    return "high"


def gold_bucket(gold: Any) -> str:
    if not isinstance(gold, (int, float)):
        return "unknown"
    value = float(gold)
    if value < 75:
        return "low"
    if value < 150:
        return "medium"
    if value < 250:
        return "high"
    return "rich"


def _floor_value(values: Any, floor: int) -> Any:
    rows = list(values or [])
    idx = int(floor) - 1
    return rows[idx] if 0 <= idx < len(rows) else None


def _accepted_run(run: Mapping[str, Any], min_ascension: int) -> bool:
    return (
        str(run.get("character_chosen", "")).upper() == "IRONCLAD"
        and not bool(run.get("is_daily"))
        and not bool(run.get("is_trial"))
        and not bool(run.get("is_endless"))
        and int(run.get("ascension_level", 0) or 0) >= int(min_ascension)
    )


def _weight(run: Mapping[str, Any], path: Path) -> float:
    source = 3.0 if "panacea" in str(path).lower() else 1.0
    asc = max(0, min(20, int(run.get("ascension_level", 0) or 0)))
    asc_weight = 1.0 + asc / 40.0
    outcome = 1.25 if bool(run.get("victory")) else 0.75
    return source * asc_weight * outcome


def _log_probs(counts: Mapping[str, float], *, alpha: float) -> dict[str, float]:
    total = sum(float(counts.get(room, 0.0)) for room in ROOMS)
    denom = total + alpha * len(ROOMS)
    return {
        room: math.log((float(counts.get(room, 0.0)) + alpha) / denom)
        for room in ROOMS
    }


def build_path_prior(
    data_root: Path,
    output_path: Path,
    *,
    min_ascension: int = 15,
    alpha: float = 3.0,
) -> dict[str, Any]:
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("path prior alpha must be finite and positive")

    global_counts: dict[str, float] = defaultdict(float)
    act_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    context_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    accepted_runs = 0
    example_count = 0
    ignored_nodes = 0

    for path in iter_run_files(data_root):
        run = json.loads(path.read_text(encoding="utf-8"))
        if not _accepted_run(run, min_ascension):
            continue
        accepted_runs += 1
        weight = _weight(run, path)
        for floor, raw_room in enumerate(list(run.get("path_taken") or []), start=1):
            room = normalize_room(raw_room)
            if room is None:
                ignored_nodes += 1
                continue
            hp = _floor_value(run.get("current_hp_per_floor"), floor)
            max_hp = _floor_value(run.get("max_hp_per_floor"), floor)
            gold = _floor_value(run.get("gold_per_floor"), floor)
            act = act_bucket(floor)
            context = f"{act}|{hp_bucket(hp, max_hp)}|{gold_bucket(gold)}"
            global_counts[room] += weight
            act_counts[str(act)][room] += weight
            context_counts[context][room] += weight
            example_count += 1

    if example_count < 1:
        raise RuntimeError("human path corpus produced no usable examples")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "character": "IRONCLAD",
        "min_ascension": int(min_ascension),
        "alpha": float(alpha),
        "accepted_runs": accepted_runs,
        "example_count": example_count,
        "ignored_nodes": ignored_nodes,
        "global_log_probs": _log_probs(global_counts, alpha=alpha),
        "act_log_probs": {
            key: _log_probs(value, alpha=alpha)
            for key, value in sorted(act_counts.items())
        },
        "context_log_probs": {
            key: _log_probs(value, alpha=alpha)
            for key, value in sorted(context_counts.items())
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def path_prior_score(
    prior: Mapping[str, Any],
    room: str,
    *,
    floor: int,
    hp: Any,
    max_hp: Any,
    gold: Any,
) -> float:
    canonical = normalize_room(room)
    if canonical is None:
        return 0.0
    act = act_bucket(floor)
    global_score = float((prior.get("global_log_probs") or {}).get(canonical, 0.0) or 0.0)
    act_score = float(
        ((prior.get("act_log_probs") or {}).get(str(act), {}) or {}).get(canonical, global_score)
        or global_score
    )
    context_key = f"{act}|{hp_bucket(hp, max_hp)}|{gold_bucket(gold)}"
    context_row = (prior.get("context_log_probs") or {}).get(context_key, {}) or {}
    context_score = float(context_row.get(canonical, act_score) or act_score)
    # Only relative values matter when reranking candidates.  Context receives
    # the largest weight; act and global terms keep sparse buckets stable.
    return 0.15 * global_score + 0.25 * act_score + 0.60 * context_score


__all__ = [
    "ROOMS",
    "SCHEMA_VERSION",
    "build_path_prior",
    "gold_bucket",
    "hp_bucket",
    "normalize_room",
    "path_prior_score",
]
