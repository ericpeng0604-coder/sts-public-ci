"""Action-independent re-determinization plan for public-state STS1 rollouts.

A formal Phase-1 rollout must never continue from the simulator's real hidden
RNG streams or hidden previous monster move.  Instead we split one
``PublicSample`` into deterministic domain seeds.  The plan depends only on the
public decision signature and the sample identity, never on the candidate
action, so every candidate sees the same sampled future.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from .contract import DecisionContext
from .sampling import PublicSample

REDETERMINIZATION_SCHEMA = "sts1-public-redeterminization-v1"
_RNG_DOMAINS = (
    "ai",
    "card_random",
    "misc",
    "monster_hp",
    "potion",
    "shuffle",
    "draw_order",
)


def _derive_u64(sample: PublicSample, context: DecisionContext, domain: str) -> int:
    material = (
        f"{REDETERMINIZATION_SCHEMA}|{context.decision_signature}|"
        f"{sample.sample_key}|{domain}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class MonsterHistorySample:
    enemy_index: int
    current_public_intent: str | None
    previous_history_seed: int


@dataclass(frozen=True)
class PublicRedeterminizationPlan:
    schema_version: str
    sample_index: int
    sample_key: str
    rng_seeds: tuple[tuple[str, int], ...]
    monster_history: tuple[MonsterHistorySample, ...]

    def rng_seed(self, domain: str) -> int:
        for name, value in self.rng_seeds:
            if name == domain:
                return value
        raise KeyError(domain)


def build_redeterminization_plan(
    context: DecisionContext,
    sample: PublicSample,
) -> PublicRedeterminizationPlan:
    """Create one candidate-independent hidden-future sampling plan.

    ``current_public_intent`` is copied only from the public enemy observation.
    The older move-history slot is *not* read from the simulator.  Native
    reconstruction must derive/sample that older history from
    ``previous_history_seed`` or reject the enemy as unsupported.
    """

    rng_seeds = tuple(
        (domain, _derive_u64(sample, context, f"rng:{domain}"))
        for domain in _RNG_DOMAINS
    )

    histories: list[MonsterHistorySample] = []
    enemies = context.state.get("enemies", [])
    if isinstance(enemies, Iterable):
        for fallback_index, enemy in enumerate(enemies):
            if not isinstance(enemy, dict):
                continue
            raw_index = enemy.get("index", fallback_index)
            enemy_index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else fallback_index
            raw_intent = enemy.get("intent")
            intent = str(raw_intent) if raw_intent is not None else None
            histories.append(
                MonsterHistorySample(
                    enemy_index=enemy_index,
                    current_public_intent=intent,
                    previous_history_seed=_derive_u64(
                        sample,
                        context,
                        f"monster_history:{enemy_index}",
                    ),
                )
            )

    histories.sort(key=lambda item: item.enemy_index)
    return PublicRedeterminizationPlan(
        schema_version=REDETERMINIZATION_SCHEMA,
        sample_index=sample.sample_index,
        sample_key=sample.sample_key,
        rng_seeds=rng_seeds,
        monster_history=tuple(histories),
    )


__all__ = [
    "MonsterHistorySample",
    "PublicRedeterminizationPlan",
    "REDETERMINIZATION_SCHEMA",
    "build_redeterminization_plan",
]
