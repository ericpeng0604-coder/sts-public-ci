"""Normalize public STS1 Run History into Human Strategy examples.

Exact-policy rows are emitted only when the run records the full offered set.
Partial observations are explicitly marked preference_only so they cannot be
mistaken for behavior-cloning labels.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from roguelike_ai.sts1_phase3.human_expert import (
    SKIP_TOKEN,
    act_bucket,
    iter_run_files,
    normalize_card_name,
)


SCHEMA_VERSION = "sts1-human-strategy-dataset-v1"


def _floor_value(values: Sequence[Any] | None, floor: int) -> Any:
    rows = list(values or [])
    index = int(floor) - 1
    if 0 <= index < len(rows):
        return rows[index]
    return None


def _state_context(run: Mapping[str, Any], floor: int) -> dict[str, Any]:
    hp = _floor_value(run.get("current_hp_per_floor"), floor)
    max_hp = _floor_value(run.get("max_hp_per_floor"), floor)
    gold = _floor_value(run.get("gold_per_floor"), floor)
    path = _floor_value(run.get("path_per_floor"), floor)
    hp_fraction = None
    if isinstance(hp, (int, float)) and isinstance(max_hp, (int, float)) and max_hp:
        hp_fraction = max(0.0, min(1.0, float(hp) / float(max_hp)))
    return {
        "floor": int(floor),
        "act": act_bucket(int(floor)),
        "hp": hp,
        "max_hp": max_hp,
        "hp_fraction": hp_fraction,
        "gold": gold,
        "path_node": path,
    }


def _accepted_run(run: Mapping[str, Any], *, min_ascension: int) -> bool:
    return (
        str(run.get("character_chosen", "")).upper() == "IRONCLAD"
        and not bool(run.get("is_daily"))
        and not bool(run.get("is_trial"))
        and not bool(run.get("is_endless"))
        and int(run.get("ascension_level", 0) or 0) >= int(min_ascension)
    )


def normalize_run(
    run: Mapping[str, Any],
    *,
    source: str,
    min_ascension: int = 15,
) -> list[dict[str, Any]]:
    if not _accepted_run(run, min_ascension=min_ascension):
        return []

    rows: list[dict[str, Any]] = []
    play_id = str(run.get("play_id") or "")
    base = {
        "play_id": play_id,
        "source": source,
        "ascension": int(run.get("ascension_level", 0) or 0),
        "victory": bool(run.get("victory")),
    }

    for choice in list(run.get("card_choices") or []):
        floor = int(choice.get("floor", 0) or 0)
        if floor <= 0:
            continue
        picked = normalize_card_name(choice.get("picked"))
        offered = [picked]
        offered.extend(normalize_card_name(v) for v in list(choice.get("not_picked") or []))
        if picked != SKIP_TOKEN and SKIP_TOKEN not in offered:
            offered.append(SKIP_TOKEN)
        offered = list(dict.fromkeys(offered))
        rows.append({
            **base,
            **_state_context(run, floor),
            "kind": "card_reward",
            "label_mode": "exact_policy",
            "choice": picked,
            "candidates": offered,
        })

    for index, choice in enumerate(list(run.get("boss_relics") or [])):
        picked = str(choice.get("picked") or "").strip()
        alternatives = [str(v).strip() for v in list(choice.get("not_picked") or []) if str(v).strip()]
        candidates = list(dict.fromkeys([picked, *alternatives]))
        if not picked or len(candidates) < 2:
            continue
        floor = 16 if index == 0 else 33 if index == 1 else 50
        rows.append({
            **base,
            **_state_context(run, floor),
            "kind": "boss_relic",
            "label_mode": "exact_policy",
            "choice": picked,
            "candidates": candidates,
        })

    for choice in list(run.get("campfire_choices") or []):
        floor = int(choice.get("floor", 0) or 0)
        action = str(choice.get("key") or "").strip().upper()
        if floor <= 0 or not action:
            continue
        rows.append({
            **base,
            **_state_context(run, floor),
            "kind": "campfire",
            "label_mode": "preference_only",
            "choice": action,
            "detail": choice.get("data"),
            "candidates": None,
        })

    for choice in list(run.get("event_choices") or []):
        floor = int(choice.get("floor", 0) or 0)
        event_name = str(choice.get("event_name") or "").strip()
        action = str(choice.get("player_choice") or "").strip()
        if floor <= 0 or not event_name or not action:
            continue
        rows.append({
            **base,
            **_state_context(run, floor),
            "kind": "event",
            "label_mode": "preference_only",
            "event_name": event_name,
            "choice": action,
            "candidates": None,
        })

    for item, floor_raw in zip(
        list(run.get("items_purchased") or []),
        list(run.get("item_purchase_floors") or []),
    ):
        floor = int(floor_raw or 0)
        item_name = str(item or "").strip()
        if floor <= 0 or not item_name:
            continue
        rows.append({
            **base,
            **_state_context(run, floor),
            "kind": "shop_purchase",
            "label_mode": "preference_only",
            "choice": item_name,
            "candidates": None,
        })

    for floor, node in enumerate(list(run.get("path_taken") or []), start=1):
        node = str(node or "").strip()
        if not node:
            continue
        rows.append({
            **base,
            **_state_context(run, floor),
            "kind": "path",
            "label_mode": "preference_only",
            "choice": node,
            "candidates": None,
        })

    return rows


def build_human_strategy_dataset(
    data_root: Path,
    output_path: Path,
    *,
    min_ascension: int = 15,
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    accepted_runs = 0
    for path in iter_run_files(data_root):
        run = json.loads(path.read_text(encoding="utf-8"))
        source = "panacea" if "panacea" in str(path).lower() else "rotating"
        rows = normalize_run(run, source=source, min_ascension=min_ascension)
        if not rows:
            continue
        accepted_runs += 1
        examples.extend(rows)

    if not examples:
        raise RuntimeError("human strategy corpus produced no usable examples")

    by_kind = Counter(str(row["kind"]) for row in examples)
    by_mode = Counter(str(row["label_mode"]) for row in examples)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "character": "IRONCLAD",
        "min_ascension": int(min_ascension),
        "accepted_runs": accepted_runs,
        "example_count": len(examples),
        "by_kind": dict(sorted(by_kind.items())),
        "by_label_mode": dict(sorted(by_mode.items())),
        "examples": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "build_human_strategy_dataset",
    "normalize_run",
]
