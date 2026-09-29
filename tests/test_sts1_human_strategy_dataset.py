from __future__ import annotations

import json
from pathlib import Path

from roguelike_ai.sts1_phase3.human_strategy_dataset import (
    build_human_strategy_dataset,
    normalize_run,
)


def _run() -> dict:
    return {
        "play_id": "expert-1",
        "character_chosen": "IRONCLAD",
        "ascension_level": 20,
        "victory": True,
        "is_daily": False,
        "is_trial": False,
        "is_endless": False,
        "current_hp_per_floor": [80] * 60,
        "max_hp_per_floor": [80] * 60,
        "gold_per_floor": [100] * 60,
        "path_per_floor": ["M"] * 60,
        "path_taken": ["M", "?", "$", "R"],
        "card_choices": [
            {"floor": 2, "picked": "Inflame", "not_picked": ["Flex", "Clash"]},
        ],
        "boss_relics": [
            {"picked": "Snecko Eye", "not_picked": ["Fusion Hammer", "Runic Dome"]},
        ],
        "campfire_choices": [{"floor": 15, "key": "REST"}],
        "event_choices": [
            {"floor": 4, "event_name": "Lab", "player_choice": "Got Potions"},
        ],
        "items_purchased": ["Inflame"],
        "item_purchase_floors": [5],
    }


def test_normalize_run_keeps_exact_and_partial_labels_separate() -> None:
    rows = normalize_run(_run(), source="panacea", min_ascension=15)
    exact = [row for row in rows if row["label_mode"] == "exact_policy"]
    partial = [row for row in rows if row["label_mode"] == "preference_only"]

    assert {row["kind"] for row in exact} == {"card_reward", "boss_relic"}
    assert {"campfire", "event", "shop_purchase", "path"} <= {
        row["kind"] for row in partial
    }
    assert all(row["candidates"] for row in exact)
    assert all(row["candidates"] is None for row in partial)


def test_build_dataset_reports_decision_coverage(tmp_path: Path) -> None:
    root = tmp_path / "data"
    directory = root / "runs" / "panacea-ironclad-sample"
    directory.mkdir(parents=True)
    (directory / "one.run").write_text(json.dumps(_run()), encoding="utf-8")

    out = tmp_path / "human-strategy.json"
    report = build_human_strategy_dataset(root, out, min_ascension=15)

    assert report["accepted_runs"] == 1
    assert report["by_kind"]["card_reward"] == 1
    assert report["by_kind"]["boss_relic"] == 1
    assert report["by_kind"]["campfire"] == 1
    assert report["by_kind"]["event"] == 1
    assert report["by_kind"]["shop_purchase"] == 1
    assert report["by_kind"]["path"] == 4
    assert report["by_label_mode"]["exact_policy"] == 2
    assert report["by_label_mode"]["preference_only"] == 7
    assert out.is_file()


def test_low_ascension_run_is_rejected() -> None:
    run = _run()
    run["ascension_level"] = 0
    assert normalize_run(run, source="panacea", min_ascension=15) == []
