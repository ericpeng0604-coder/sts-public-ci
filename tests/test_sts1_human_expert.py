from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from roguelike_ai.sts1_phase3.human_expert import (
    SKIP_TOKEN,
    build_card_prior,
    normalize_card_name,
    prior_card_score,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sts1"
    / "sts1_human_expert_loop.py"
)
SPEC = importlib.util.spec_from_file_location("sts1_human_expert_loop", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
loop = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = loop
SPEC.loader.exec_module(loop)


def _write_run(
    path: Path,
    *,
    play_id: str,
    picked: str = "Inflame",
    not_picked: list[str] | None = None,
    character: str = "IRONCLAD",
    ascension: int = 20,
    victory: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "play_id": play_id,
        "character_chosen": character,
        "ascension_level": ascension,
        "victory": victory,
        "is_daily": False,
        "is_trial": False,
        "is_endless": False,
        "card_choices": [
            {
                "floor": 3,
                "picked": picked,
                "not_picked": not_picked or ["Flex", "Clash"],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_normalize_card_name_handles_upgrade_and_skip() -> None:
    assert normalize_card_name("Inflame+1") == "Inflame"
    assert normalize_card_name("SKIP") == SKIP_TOKEN
    assert normalize_card_name("Singing Bowl") == SKIP_TOKEN


def test_build_prior_filters_to_high_ascension_ironclad(tmp_path: Path) -> None:
    data = tmp_path / "data"
    panacea = data / "runs" / "panacea-ironclad-sample"
    rotating = data / "runs" / "200-rotating-sample" / "IRONCLAD"

    for index in range(12):
        _write_run(
            panacea / f"p{index}.run",
            play_id=f"panacea-{index}",
            victory=index < 9,
        )
    for index in range(8):
        _write_run(
            rotating / f"r{index}.run",
            play_id=f"rotating-{index}",
            picked="Inflame",
            not_picked=["Flex", "Clash"],
            victory=index < 4,
        )

    # These must not enter the policy corpus.
    _write_run(
        panacea / "silent.run",
        play_id="silent",
        character="THE_SILENT",
    )
    _write_run(
        panacea / "low-asc.run",
        play_id="low-asc",
        ascension=0,
    )

    out = tmp_path / "prior.json"
    report = build_card_prior(data, out, min_ascension=15)
    assert report["accepted_runs"] == 20
    assert report["card_choice_examples"] == 20
    assert report["sources"]["panacea"] == 12
    assert report["sources"]["rotating"] == 8
    assert report["holdout_examples"] > 0
    assert report["train_examples"] > 0
    assert 0.0 <= report["baseline_holdout_top1"] <= 1.0
    assert report["holdout_top1"] >= report["baseline_holdout_top1"]
    assert 0.0 <= report["matchup_mix"] <= 1.0
    assert 0.0 <= report["context_mix"] <= 1.0
    assert report["observed_matchup_directions"] > 0
    assert prior_card_score(report, "Inflame", floor=3) > prior_card_score(
        report,
        "Flex",
        floor=3,
    )
    assert out.is_file()


def test_singing_bowl_teaches_skip(tmp_path: Path) -> None:
    data = tmp_path / "data"
    panacea = data / "runs" / "panacea-ironclad-sample"
    for index in range(10):
        _write_run(
            panacea / f"{index}.run",
            play_id=f"bowl-{index}",
            picked="Singing Bowl",
            not_picked=["Wild Strike", "Clash", "Flex"],
        )
    report = build_card_prior(data, tmp_path / "prior.json", min_ascension=15)
    assert prior_card_score(report, SKIP_TOKEN, floor=3) > prior_card_score(
        report,
        "Flex",
        floor=3,
    )


def test_candidate_strengths_start_conservative_and_change_after_promotion() -> None:
    assert loop._candidate_strengths(0.0) == (0.25, 0.5, 0.75)
    next_values = loop._candidate_strengths(0.5)
    assert 0.5 not in next_values
    assert all(0.0 < value <= 2.0 for value in next_values)


def test_fresh_seed_generator_is_disjoint_and_deterministic() -> None:
    a = loop._fresh_seeds(count=10, rng_seed=123, forbidden={1, 2, 3})
    b = loop._fresh_seeds(count=10, rng_seed=123, forbidden={1, 2, 3})
    assert a == b
    assert len(set(a)) == 10
    assert not (set(a) & {1, 2, 3})


def test_pairwise_offered_context_changes_score() -> None:
    prior = {
        "global_scores": {"A": 0.0, "B": 0.0, "C": 0.0},
        "act_scores": {"1": {"A": 0.0, "B": 0.0, "C": 0.0}},
        "pairwise_scores": {
            "A": {"B": 2.0, "C": -2.0},
            "B": {"A": -2.0},
            "C": {"A": 2.0},
        },
        "matchup_mix": 1.0,
    }
    assert prior_card_score(prior, "A", floor=3, offered=["A", "B"]) > 0.0
    assert prior_card_score(prior, "A", floor=3, offered=["A", "C"]) < 0.0


def test_deck_context_changes_score() -> None:
    prior = {
        "global_scores": {"Corruption": 0.0},
        "act_scores": {"2": {"Corruption": 0.0}},
        "pairwise_scores": {},
        "context_scores": {
            "Corruption": {
                "Feel No Pain": 2.0,
                "Clash": -2.0,
            }
        },
        "matchup_mix": 0.0,
        "context_mix": 1.0,
    }
    assert prior_card_score(
        prior,
        "Corruption",
        floor=20,
        deck_context=["Feel No Pain"],
    ) > 0.0
    assert prior_card_score(
        prior,
        "Corruption",
        floor=20,
        deck_context=["Clash"],
    ) < 0.0
