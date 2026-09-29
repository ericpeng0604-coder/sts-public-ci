"""Human-expert priors for STS1 Ironclad non-combat decisions.

v1 intentionally uses only decisions that can be reconstructed reliably from
vanilla Run History files: card reward choices.  Other run fields are kept for
future Value-model work but never guessed into Policy labels.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from roguelike_ai.sts1_phase3.simulator import ArmGNoncombatPolicy, SimulatorRunError


PRIOR_SCHEMA_VERSION = "sts1-human-expert-card-prior-v3"
SKIP_TOKEN = "__SKIP__"
_UPGRADE_RE = re.compile(r"\+\d+$")


def normalize_card_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name:
        return SKIP_TOKEN
    if name.upper() == "SKIP" or name == "Singing Bowl":
        return SKIP_TOKEN
    return _UPGRADE_RE.sub("", name)


def act_bucket(floor: int) -> int:
    floor = int(floor)
    if floor <= 16:
        return 1
    if floor <= 33:
        return 2
    if floor <= 50:
        return 3
    return 4


def _stable_holdout(play_id: str, *, modulus: int = 5) -> bool:
    digest = hashlib.sha256(play_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulus == 0


def _source_weight(path: Path) -> float:
    text = str(path).lower()
    if "panacea" in text:
        return 3.0
    if "rotating" in text:
        return 1.0
    return 1.0


def _run_weight(run: Mapping[str, Any], path: Path) -> float:
    ascension = max(0, min(20, int(run.get("ascension_level", 0) or 0)))
    ascension_weight = 1.0 + ascension / 40.0
    outcome_weight = 1.25 if bool(run.get("victory")) else 0.75
    return _source_weight(path) * ascension_weight * outcome_weight


def iter_run_files(data_root: Path) -> list[Path]:
    wanted = (
        data_root / "runs" / "panacea-ironclad-sample",
        data_root / "runs" / "200-rotating-sample" / "IRONCLAD",
    )
    result: list[Path] = []
    for directory in wanted:
        if directory.is_dir():
            result.extend(sorted(directory.glob("*.run")))
    return result


def _valid_run(run: Mapping[str, Any], *, min_ascension: int) -> bool:
    if str(run.get("character_chosen", "")).upper() != "IRONCLAD":
        return False
    if bool(run.get("is_daily")) or bool(run.get("is_trial")) or bool(run.get("is_endless")):
        return False
    return int(run.get("ascension_level", 0) or 0) >= int(min_ascension)


def _choice_candidates(choice: Mapping[str, Any]) -> tuple[str, list[str]]:
    picked = normalize_card_name(choice.get("picked"))
    alternatives = [
        normalize_card_name(value)
        for value in list(choice.get("not_picked") or [])
    ]
    alternatives = [value for value in alternatives if value != picked]
    if picked != SKIP_TOKEN and SKIP_TOKEN not in alternatives:
        alternatives.append(SKIP_TOKEN)
    dedup: list[str] = []
    seen: set[str] = set()
    for value in alternatives:
        if value in seen:
            continue
        seen.add(value)
        dedup.append(value)
    return picked, dedup


def _fit_scores(
    examples: Sequence[tuple[int, str, list[str], float, tuple[str, ...]]],
    *,
    alpha: float,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    """Fit base utility, offered-set matchup, and existing-deck synergy."""
    wins: dict[str, float] = defaultdict(float)
    losses: dict[str, float] = defaultdict(float)
    act_wins: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    act_losses: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pair_wins: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    context_wins: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    context_losses: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pair_count = 0
    context_pair_count = 0

    for act, picked, alternatives, weight, context_cards in examples:
        context = tuple(sorted(set(normalize_card_name(x) for x in context_cards if x)))
        for other in alternatives:
            if other == picked:
                continue
            wins[picked] += weight
            losses[other] += weight
            act_wins[act][picked] += weight
            act_losses[act][other] += weight
            pair_wins[picked][other] += weight
            pair_count += 1
            for existing in context:
                context_wins[picked][existing] += weight
                context_losses[other][existing] += weight
                context_pair_count += 1

    names = sorted(set(wins) | set(losses))
    global_scores = {
        name: math.log((wins[name] + alpha) / (losses[name] + alpha))
        for name in names
    }

    by_act: dict[str, dict[str, float]] = {}
    for act in sorted(set(act_wins) | set(act_losses)):
        act_names = sorted(set(act_wins[act]) | set(act_losses[act]))
        by_act[str(act)] = {
            name: math.log(
                (act_wins[act][name] + alpha) / (act_losses[act][name] + alpha)
            )
            for name in act_names
        }

    pairwise_scores: dict[str, dict[str, float]] = {}
    matchup_pairs = 0
    for left in names:
        row: dict[str, float] = {}
        for right in names:
            if left == right:
                continue
            forward = float(pair_wins[left].get(right, 0.0))
            backward = float(pair_wins[right].get(left, 0.0))
            if forward <= 0.0 and backward <= 0.0:
                continue
            row[right] = math.log((forward + alpha) / (backward + alpha))
            matchup_pairs += 1
        if row:
            pairwise_scores[left] = row

    context_scores: dict[str, dict[str, float]] = {}
    context_directions = 0
    context_alpha = max(4.0, alpha * 2.0)
    for candidate in names:
        row: dict[str, float] = {}
        keys = sorted(set(context_wins[candidate]) | set(context_losses[candidate]))
        for existing in keys:
            positive = float(context_wins[candidate].get(existing, 0.0))
            negative = float(context_losses[candidate].get(existing, 0.0))
            row[existing] = math.log(
                (positive + context_alpha) / (negative + context_alpha)
            )
            context_directions += 1
        if row:
            context_scores[candidate] = row

    return global_scores, by_act, pairwise_scores, context_scores, {
        "pairwise_examples": pair_count,
        "context_pair_examples": context_pair_count,
        "unique_cards": len(names),
        "observed_matchup_directions": matchup_pairs,
        "observed_context_directions": context_directions,
    }


def _base_card_score(prior: Mapping[str, Any], card_name: str, *, floor: int) -> float:
    name = normalize_card_name(card_name)
    global_score = float((prior.get("global_scores") or {}).get(name, 0.0) or 0.0)
    act_score = float(
        ((prior.get("act_scores") or {}).get(str(act_bucket(floor)), {}) or {}).get(name, global_score)
        or 0.0
    )
    return 0.30 * global_score + 0.70 * act_score


def _mean_known(row: Mapping[str, Any], values: Sequence[str], *, exclude: str | None = None) -> float:
    known: list[float] = []
    for value in values:
        name = normalize_card_name(value)
        if exclude is not None and name == exclude:
            continue
        if name in row:
            known.append(float(row[name]))
    return sum(known) / len(known) if known else 0.0


def prior_card_score(
    prior: Mapping[str, Any],
    card_name: str,
    *,
    floor: int,
    offered: Sequence[str] | None = None,
    deck_context: Sequence[str] | None = None,
) -> float:
    """Score a card from Act, offered cards, and already-owned cards."""
    name = normalize_card_name(card_name)
    score = _base_card_score(prior, name, floor=floor)

    matchup_mix = float(prior.get("matchup_mix", 0.0) or 0.0)
    if offered and matchup_mix > 0.0:
        row = (prior.get("pairwise_scores") or {}).get(name, {}) or {}
        score += matchup_mix * _mean_known(row, offered, exclude=name)

    context_mix = float(prior.get("context_mix", 0.0) or 0.0)
    if deck_context and context_mix > 0.0:
        row = (prior.get("context_scores") or {}).get(name, {}) or {}
        score += context_mix * _mean_known(row, deck_context)

    return score


def _top1_accuracy(
    examples: Sequence[tuple[int, str, list[str], float, tuple[str, ...]]],
    prior: Mapping[str, Any],
) -> float:
    if not examples:
        return 0.0
    correct = 0
    total = 0
    for act, picked, alternatives, _weight, context_cards in examples:
        options = [picked] + [value for value in alternatives if value != picked]
        floor = 1 if act == 1 else 17 if act == 2 else 34 if act == 3 else 51
        scores = [
            prior_card_score(
                prior,
                value,
                floor=floor,
                offered=options,
                deck_context=context_cards,
            )
            for value in options
        ]
        best = max(range(len(options)), key=scores.__getitem__)
        correct += int(options[best] == picked)
        total += 1
    return correct / total if total else 0.0


def build_card_prior(
    data_root: Path,
    output_path: Path,
    *,
    min_ascension: int = 15,
    alpha: float = 2.0,
) -> dict[str, Any]:
    if min_ascension < 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid expert-prior training settings")

    files = iter_run_files(data_root)
    if not files:
        raise RuntimeError(f"no supported expert run files found under {data_root}")

    parsed: list[tuple[Path, dict[str, Any]]] = []
    known_cards: set[str] = set()
    corpus = hashlib.sha256()

    for path in files:
        raw = path.read_bytes()
        corpus.update(str(path.relative_to(data_root)).encode("utf-8"))
        corpus.update(b"\0")
        corpus.update(hashlib.sha256(raw).digest())
        try:
            run = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid run JSON: {path}: {exc}") from exc
        if not _valid_run(run, min_ascension=min_ascension):
            continue
        parsed.append((path, run))
        for choice in list(run.get("card_choices") or []):
            picked, alternatives = _choice_candidates(choice)
            if picked != SKIP_TOKEN:
                known_cards.add(picked)
            known_cards.update(x for x in alternatives if x != SKIP_TOKEN)

    train_examples: list[tuple[int, str, list[str], float, tuple[str, ...]]] = []
    holdout_examples: list[tuple[int, str, list[str], float, tuple[str, ...]]] = []
    all_examples: list[tuple[int, str, list[str], float, tuple[str, ...]]] = []
    accepted_runs = 0
    victory_runs = 0
    source_counts: dict[str, int] = defaultdict(int)

    for path, run in parsed:
        accepted_runs += 1
        victory_runs += int(bool(run.get("victory")))
        source = "panacea" if "panacea" in str(path).lower() else "rotating"
        source_counts[source] += 1
        play_id = str(run.get("play_id") or path.name)
        holdout = _stable_holdout(play_id)
        weight = _run_weight(run, path)

        purchases = sorted(
            (
                int(floor),
                normalize_card_name(item),
            )
            for item, floor in zip(
                list(run.get("items_purchased") or []),
                list(run.get("item_purchase_floors") or []),
            )
            if normalize_card_name(item) in known_cards
        )
        purges = sorted(
            (
                int(floor),
                normalize_card_name(item),
            )
            for item, floor in zip(
                list(run.get("items_purged") or []),
                list(run.get("items_purged_floors") or []),
            )
        )

        acquired: list[str] = []
        purchase_index = 0
        purge_index = 0
        choices = sorted(
            enumerate(list(run.get("card_choices") or [])),
            key=lambda row: (int(row[1].get("floor", 0) or 0), row[0]),
        )
        for _order, choice in choices:
            floor = int(choice.get("floor", 0) or 0)
            if floor <= 0:
                continue

            while purchase_index < len(purchases) and purchases[purchase_index][0] < floor:
                acquired.append(purchases[purchase_index][1])
                purchase_index += 1
            while purge_index < len(purges) and purges[purge_index][0] < floor:
                purge_name = purges[purge_index][1]
                try:
                    acquired.remove(purge_name)
                except ValueError:
                    pass
                purge_index += 1

            picked, alternatives = _choice_candidates(choice)
            if not alternatives:
                continue
            context = tuple(acquired)
            row = (act_bucket(floor), picked, alternatives, weight, context)
            all_examples.append(row)
            (holdout_examples if holdout else train_examples).append(row)

            if picked != SKIP_TOKEN:
                acquired.append(picked)

    if accepted_runs < 1 or not all_examples:
        raise RuntimeError("expert corpus produced no usable Ironclad card choices")
    if not train_examples:
        raise RuntimeError("expert corpus produced an empty training split")

    global_scores, act_scores, pairwise_scores, context_scores, stats = _fit_scores(
        train_examples,
        alpha=alpha,
    )
    baseline_prior = {
        "schema_version": PRIOR_SCHEMA_VERSION,
        "global_scores": global_scores,
        "act_scores": act_scores,
        "pairwise_scores": pairwise_scores,
        "context_scores": context_scores,
        "matchup_mix": 0.0,
        "context_mix": 0.0,
    }
    baseline_holdout_top1 = _top1_accuracy(holdout_examples, baseline_prior)

    search_results: list[dict[str, float]] = []
    for matchup_mix in (0.0, 0.25, 0.50):
        for context_mix in (0.0, 0.25, 0.50, 0.75, 1.0):
            candidate_prior = dict(baseline_prior)
            candidate_prior["matchup_mix"] = float(matchup_mix)
            candidate_prior["context_mix"] = float(context_mix)
            accuracy = _top1_accuracy(holdout_examples, candidate_prior)
            search_results.append(
                {
                    "matchup_mix": float(matchup_mix),
                    "context_mix": float(context_mix),
                    "holdout_top1": float(accuracy),
                }
            )
    selected = max(
        search_results,
        key=lambda row: (
            row["holdout_top1"],
            -(row["matchup_mix"] + row["context_mix"]),
        ),
    )
    matchup_mix = float(selected["matchup_mix"])
    context_mix = float(selected["context_mix"])
    holdout_top1 = float(selected["holdout_top1"])

    global_scores, act_scores, pairwise_scores, context_scores, full_stats = _fit_scores(
        all_examples,
        alpha=alpha,
    )
    payload = {
        "schema_version": PRIOR_SCHEMA_VERSION,
        "character": "IRONCLAD",
        "min_ascension": int(min_ascension),
        "alpha": float(alpha),
        "corpus_sha256": corpus.hexdigest(),
        "sources": dict(sorted(source_counts.items())),
        "accepted_runs": accepted_runs,
        "victory_runs": victory_runs,
        "card_choice_examples": len(all_examples),
        "train_examples": len(train_examples),
        "holdout_examples": len(holdout_examples),
        "baseline_holdout_top1": baseline_holdout_top1,
        "holdout_top1": holdout_top1,
        "matchup_mix": matchup_mix,
        "context_mix": context_mix,
        "mix_search": search_results,
        "pairwise_examples": full_stats["pairwise_examples"],
        "context_pair_examples": full_stats["context_pair_examples"],
        "observed_matchup_directions": full_stats["observed_matchup_directions"],
        "observed_context_directions": full_stats["observed_context_directions"],
        "unique_cards": full_stats["unique_cards"],
        "global_scores": global_scores,
        "act_scores": act_scores,
        "pairwise_scores": pairwise_scores,
        "context_scores": context_scores,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


class HumanExpertPolicy(ArmGNoncombatPolicy):
    """Frozen ArmG plus a learned human-expert card-reward prior."""

    def __init__(
        self,
        *,
        expert_prior_path: Path,
        expert_strength: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not expert_prior_path.is_file():
            raise SimulatorRunError(f"human expert prior missing: {expert_prior_path}")
        payload = json.loads(expert_prior_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != PRIOR_SCHEMA_VERSION:
            raise SimulatorRunError("human expert prior schema mismatch")
        if str(payload.get("character")) != "IRONCLAD":
            raise SimulatorRunError("human expert prior character mismatch")
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
        floor = int(getattr(gc, "floor_num", 0) or 0)
        additions = [0.0] * len(descs)

        if kind == "card" and self.expert_strength > 0:
            card_names: list[str] = []
            for desc in descs:
                semantic = self.describe_choice("card", desc)
                card_names.append(
                    SKIP_TOKEN
                    if semantic.get("choice") == "skip"
                    else normalize_card_name(semantic.get("card_name"))
                )
            deck_context = [
                normalize_card_name(self.module.card_name(card))
                for card in list(getattr(gc, "deck", []))
            ]
            for index, card_name in enumerate(card_names):
                additions[index] = self.expert_strength * prior_card_score(
                    self.expert_prior,
                    card_name,
                    floor=floor,
                    offered=card_names,
                    deck_context=deck_context,
                )
                adjusted[index] += additions[index]

        self.last_expert_rerank = {
            "kind": kind,
            "strength": self.expert_strength,
            "raw_scores": raw,
            "expert_additions": additions,
            "adjusted_scores": adjusted,
        }
        return kind, max(range(len(adjusted)), key=adjusted.__getitem__), descs, execs, adjusted
