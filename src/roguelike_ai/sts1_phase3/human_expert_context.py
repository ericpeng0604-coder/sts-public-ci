"""Context-aware Human Expert prior for STS1 Ironclad card rewards.

v2 adds only information available before each card-reward decision:
- act / floor,
- current HP ratio,
- cards picked from earlier card rewards in the same run.

It deliberately avoids final-deck features and other future information.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from roguelike_ai.sts1_phase3.human_expert import (
    SKIP_TOKEN,
    _choice_candidates,
    _run_weight,
    _valid_run,
    act_bucket,
    iter_run_files,
    normalize_card_name,
)
from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, SimulatorRunError


CONTEXT_SCHEMA_VERSION = "sts1-human-expert-context-prior-v2"


def hp_bucket(current_hp: float, max_hp: float) -> str:
    maximum = float(max_hp)
    ratio = float(current_hp) / maximum if maximum > 0 else 1.0
    if ratio < 0.35:
        return "critical"
    if ratio < 0.65:
        return "medium"
    return "healthy"


def _run_hp_bucket(run: Mapping[str, Any], floor: int) -> str:
    current = list(run.get("current_hp_per_floor") or [])
    maximum = list(run.get("max_hp_per_floor") or [])
    index = max(0, int(floor) - 1)
    if index < len(current) and index < len(maximum):
        return hp_bucket(float(current[index] or 0), float(maximum[index] or 0))
    return "healthy"


def _split_name(play_id: str) -> str:
    digest = hashlib.sha256(play_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") % 10
    if value <= 1:
        return "test"
    if value <= 3:
        return "tune"
    return "train"


def _pairwise_score(
    wins: Mapping[str, float],
    losses: Mapping[str, float],
    name: str,
    alpha: float,
) -> float:
    return math.log(
        (float(wins.get(name, 0.0)) + alpha)
        / (float(losses.get(name, 0.0)) + alpha)
    )


def _fit_tables(
    examples: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    min_context_support: float,
) -> dict[str, Any]:
    wins: dict[str, float] = defaultdict(float)
    losses: dict[str, float] = defaultdict(float)
    act_wins: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    act_losses: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    hp_wins: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    hp_losses: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ctx_wins: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ctx_losses: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pair_count = 0

    for row in examples:
        picked = str(row["picked"])
        alternatives = [str(v) for v in row["alternatives"]]
        weight = float(row["weight"])
        act = str(row["act"])
        hp = str(row["hp_bucket"])
        context_cards = sorted(set(str(v) for v in row["prefix_cards"]))
        for other in alternatives:
            if other == picked:
                continue
            pair_count += 1
            wins[picked] += weight
            losses[other] += weight
            act_wins[act][picked] += weight
            act_losses[act][other] += weight
            hp_wins[hp][picked] += weight
            hp_losses[hp][other] += weight
            for existing in context_cards:
                ctx_wins[existing][picked] += weight
                ctx_losses[existing][other] += weight

    names = sorted(set(wins) | set(losses))
    global_scores = {
        name: _pairwise_score(wins, losses, name, alpha)
        for name in names
    }
    act_scores: dict[str, dict[str, float]] = {}
    for act in sorted(set(act_wins) | set(act_losses)):
        act_names = sorted(set(act_wins[act]) | set(act_losses[act]))
        act_scores[act] = {
            name: _pairwise_score(act_wins[act], act_losses[act], name, alpha)
            for name in act_names
        }
    hp_scores: dict[str, dict[str, float]] = {}
    for hp in sorted(set(hp_wins) | set(hp_losses)):
        hp_names = sorted(set(hp_wins[hp]) | set(hp_losses[hp]))
        hp_scores[hp] = {
            name: _pairwise_score(hp_wins[hp], hp_losses[hp], name, alpha)
            for name in hp_names
        }

    synergy_scores: dict[str, dict[str, float]] = {}
    synergy_support: dict[str, dict[str, float]] = {}
    for existing in sorted(set(ctx_wins) | set(ctx_losses)):
        values: dict[str, float] = {}
        supports: dict[str, float] = {}
        cards = sorted(set(ctx_wins[existing]) | set(ctx_losses[existing]))
        for name in cards:
            support = float(ctx_wins[existing].get(name, 0.0)) + float(
                ctx_losses[existing].get(name, 0.0)
            )
            if support < float(min_context_support):
                continue
            values[name] = _pairwise_score(
                ctx_wins[existing],
                ctx_losses[existing],
                name,
                alpha,
            )
            supports[name] = support
        if values:
            synergy_scores[existing] = values
            synergy_support[existing] = supports

    return {
        "global_scores": global_scores,
        "act_scores": act_scores,
        "hp_scores": hp_scores,
        "synergy_scores": synergy_scores,
        "synergy_support": synergy_support,
        "pairwise_examples": pair_count,
        "unique_cards": len(names),
    }


def _component_scores(
    tables: Mapping[str, Any],
    card_name: str,
    *,
    floor: int,
    hp: str,
    deck_cards: Sequence[str],
) -> tuple[float, float, float, float]:
    name = normalize_card_name(card_name)
    global_score = float((tables.get("global_scores") or {}).get(name, 0.0) or 0.0)
    act_score = float(
        ((tables.get("act_scores") or {}).get(str(act_bucket(floor)), {}) or {}).get(
            name,
            global_score,
        )
        or 0.0
    )
    hp_score = float(
        ((tables.get("hp_scores") or {}).get(str(hp), {}) or {}).get(name, global_score)
        or 0.0
    )

    synergy_table = tables.get("synergy_scores") or {}
    synergy_values: list[float] = []
    for existing in sorted(set(normalize_card_name(v) for v in deck_cards)):
        value = ((synergy_table.get(existing) or {}).get(name))
        if value is None:
            continue
        synergy_values.append(max(-2.0, min(2.0, float(value))))
    synergy_values.sort(key=lambda value: abs(value), reverse=True)
    synergy = (
        sum(synergy_values[:6]) / min(6, len(synergy_values))
        if synergy_values
        else global_score
    )
    return global_score, act_score, hp_score, synergy


def context_card_score(
    prior: Mapping[str, Any],
    card_name: str,
    *,
    floor: int,
    hp: str,
    deck_cards: Sequence[str],
) -> float:
    blend = prior.get("blend") or {
        "global": 0.15,
        "act": 0.45,
        "hp": 0.10,
        "synergy": 0.30,
    }
    g, a, h, s = _component_scores(
        prior,
        card_name,
        floor=floor,
        hp=hp,
        deck_cards=deck_cards,
    )
    return (
        float(blend.get("global", 0.0)) * g
        + float(blend.get("act", 0.0)) * a
        + float(blend.get("hp", 0.0)) * h
        + float(blend.get("synergy", 0.0)) * s
    )


def _accuracy(
    examples: Sequence[Mapping[str, Any]],
    tables: Mapping[str, Any],
    blend: Mapping[str, float],
) -> float:
    if not examples:
        return 0.0
    prior = dict(tables)
    prior["blend"] = dict(blend)
    correct = 0
    for row in examples:
        options = [str(row["picked"])] + [
            str(v) for v in row["alternatives"] if str(v) != str(row["picked"])
        ]
        scores = [
            context_card_score(
                prior,
                option,
                floor=int(row["floor"]),
                hp=str(row["hp_bucket"]),
                deck_cards=list(row["prefix_cards"]),
            )
            for option in options
        ]
        correct += int(options[max(range(len(options)), key=scores.__getitem__)] == row["picked"])
    return correct / len(examples)


def _baseline_accuracy(
    examples: Sequence[Mapping[str, Any]],
    tables: Mapping[str, Any],
) -> float:
    return _accuracy(
        examples,
        tables,
        {"global": 0.30, "act": 0.70, "hp": 0.0, "synergy": 0.0},
    )


def build_context_prior(
    data_root: Path,
    output_path: Path,
    *,
    min_ascension: int = 15,
    alpha: float = 2.0,
    min_context_support: float = 6.0,
) -> dict[str, Any]:
    if min_ascension < 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid context-prior settings")
    if not math.isfinite(min_context_support) or min_context_support <= 0:
        raise ValueError("min_context_support must be positive")

    files = iter_run_files(data_root)
    if not files:
        raise RuntimeError(f"no supported expert run files found under {data_root}")

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "tune": [],
        "test": [],
    }
    all_examples: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)
    accepted_runs = 0
    victory_runs = 0
    corpus = hashlib.sha256()

    for path in files:
        raw = path.read_bytes()
        corpus.update(str(path.relative_to(data_root)).encode("utf-8"))
        corpus.update(b"\0")
        corpus.update(hashlib.sha256(raw).digest())
        run = json.loads(raw)
        if not _valid_run(run, min_ascension=min_ascension):
            continue
        accepted_runs += 1
        victory_runs += int(bool(run.get("victory")))
        source_counts["panacea" if "panacea" in str(path).lower() else "rotating"] += 1
        play_id = str(run.get("play_id") or path.name)
        split = _split_name(play_id)
        weight = _run_weight(run, path)
        prefix_cards: list[str] = []
        choices = sorted(
            list(run.get("card_choices") or []),
            key=lambda row: int(row.get("floor", 0) or 0),
        )
        for choice in choices:
            floor = int(choice.get("floor", 0) or 0)
            if floor <= 0:
                continue
            picked, alternatives = _choice_candidates(choice)
            if not alternatives:
                continue
            row = {
                "floor": floor,
                "act": act_bucket(floor),
                "hp_bucket": _run_hp_bucket(run, floor),
                "picked": picked,
                "alternatives": alternatives,
                "prefix_cards": list(prefix_cards),
                "weight": weight,
            }
            splits[split].append(row)
            all_examples.append(row)
            if picked != SKIP_TOKEN:
                prefix_cards.append(picked)

    if not splits["train"] or not splits["tune"] or not splits["test"]:
        raise RuntimeError("context corpus must produce non-empty train/tune/test splits")

    train_tables = _fit_tables(
        splits["train"],
        alpha=alpha,
        min_context_support=min_context_support,
    )
    blends = [
        {"global": 0.30, "act": 0.70, "hp": 0.00, "synergy": 0.00},
        {"global": 0.20, "act": 0.45, "hp": 0.10, "synergy": 0.25},
        {"global": 0.15, "act": 0.45, "hp": 0.10, "synergy": 0.30},
        {"global": 0.15, "act": 0.40, "hp": 0.15, "synergy": 0.30},
        {"global": 0.10, "act": 0.40, "hp": 0.10, "synergy": 0.40},
        {"global": 0.10, "act": 0.35, "hp": 0.15, "synergy": 0.40},
    ]
    tune_rows = [
        {
            "blend": blend,
            "top1": _accuracy(splits["tune"], train_tables, blend),
        }
        for blend in blends
    ]
    selected = max(
        tune_rows,
        key=lambda row: (
            float(row["top1"]),
            float(row["blend"]["synergy"]),
            float(row["blend"]["hp"]),
        ),
    )
    selected_blend = dict(selected["blend"])
    test_top1 = _accuracy(splits["test"], train_tables, selected_blend)
    baseline_test_top1 = _baseline_accuracy(splits["test"], train_tables)

    full_tables = _fit_tables(
        all_examples,
        alpha=alpha,
        min_context_support=min_context_support,
    )
    payload = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "character": "IRONCLAD",
        "min_ascension": int(min_ascension),
        "alpha": float(alpha),
        "min_context_support": float(min_context_support),
        "corpus_sha256": corpus.hexdigest(),
        "sources": dict(sorted(source_counts.items())),
        "accepted_runs": accepted_runs,
        "victory_runs": victory_runs,
        "card_choice_examples": len(all_examples),
        "train_examples": len(splits["train"]),
        "tune_examples": len(splits["tune"]),
        "test_examples": len(splits["test"]),
        "baseline_test_top1": baseline_test_top1,
        "context_test_top1": test_top1,
        "context_test_delta": test_top1 - baseline_test_top1,
        "tune_results": tune_rows,
        "blend": selected_blend,
        **full_tables,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


class HumanExpertContextPolicy(ArmGNoncombatPolicy):
    """Frozen ArmG plus an expert prior conditioned on act, HP and deck context."""

    def __init__(
        self,
        *,
        expert_prior_path: Path,
        expert_strength: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not expert_prior_path.is_file():
            raise SimulatorRunError(f"human expert context prior missing: {expert_prior_path}")
        payload = json.loads(expert_prior_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CONTEXT_SCHEMA_VERSION:
            raise SimulatorRunError("human expert context prior schema mismatch")
        if str(payload.get("character")) != "IRONCLAD":
            raise SimulatorRunError("human expert context prior character mismatch")
        if not math.isfinite(float(expert_strength)) or float(expert_strength) < 0:
            raise SimulatorRunError("human expert strength must be finite and non-negative")
        self.expert_prior = payload
        self.expert_strength = float(expert_strength)
        self.last_expert_rerank: dict[str, Any] | None = None

    def decide(self, gc: Any, sts: Any) -> tuple[str, int, list[Any], list[Any], list[float]]:
        kind, descs, execs = self.choices(gc)
        if not descs:
            if gc.screen_state == sts.ScreenState.REWARDS:
                return "reward_empty", -1, [], [], []
            raise SimulatorRunError(f"ArmG produced no legal choice on screen: {gc.screen_state}")
        if len(descs) == 1:
            return kind, 0, descs, execs, [0.0]

        _, _, scores = self.score_choices(gc)
        raw = [float(value) for value in scores.tolist()]
        adjusted = list(raw)
        additions = [0.0] * len(descs)
        floor = int(getattr(gc, "floor_num", 0) or 0)
        current_hp = float(getattr(gc, "current_hp", getattr(gc, "hp", 0)) or 0)
        max_hp = float(getattr(gc, "max_hp", 0) or 0)
        hp = hp_bucket(current_hp, max_hp)
        deck_cards = [str(row["name"]) for row in self.deck_snapshot(gc)]

        if kind == "card" and self.expert_strength > 0:
            for index, desc in enumerate(descs):
                semantic = self.describe_choice("card", desc)
                card_name = (
                    SKIP_TOKEN
                    if semantic.get("choice") == "skip"
                    else normalize_card_name(semantic.get("card_name"))
                )
                additions[index] = self.expert_strength * context_card_score(
                    self.expert_prior,
                    card_name,
                    floor=floor,
                    hp=hp,
                    deck_cards=deck_cards,
                )
                adjusted[index] += additions[index]

        self.last_expert_rerank = {
            "kind": kind,
            "strength": self.expert_strength,
            "floor": floor,
            "hp_bucket": hp,
            "raw_scores": raw,
            "expert_additions": additions,
            "adjusted_scores": adjusted,
        }
        return kind, max(range(len(adjusted)), key=adjusted.__getitem__), descs, execs, adjusted
