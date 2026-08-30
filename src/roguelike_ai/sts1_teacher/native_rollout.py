"""Native public-state rollout backends for the first audited STS1 V1 slices.

These backends never receive a source simulator BattleContext. Every rollout
creates a fresh native BattleContext from the already-admitted public state and
an action-independent ``PublicSample``. The deterministic follow-up policy is
intentionally simple; V1 proves plumbing and public-only search semantics, not
a final Phase-1 Teacher quality gate.

A caller may separately provide bounded reconstruction auxiliary counters from
the CommunicationMod command trace. They are not merged into policy state or
decision identity; native reconstruction validates them fail-closed.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .contract import ActionSpec, DecisionContext
from .reconstruction import require_public_reconstruction
from .redeterminization import build_redeterminization_plan
from .sampling import PublicSample
from .search import SearchConfig


class NativePublicRolloutError(RuntimeError):
    pass


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(value).rsplit(".", 1)[-1].upper()


class NativePublicJawWormRolloutV1:
    """Executable public-only rollout backend for the first Jaw Worm slice."""

    backend_id = "sts1-public-native-jaw-worm-rollout-v1"
    uses_hidden_information = False

    def __init__(
        self,
        public_state: dict[str, Any],
        native_module: Any,
        *,
        reconstruction_aux: Mapping[str, Any] | None = None,
    ) -> None:
        admitted = require_public_reconstruction(public_state)
        self._decision_signature = admitted.decision_signature
        self._public_state = deepcopy(public_state)
        self._native = native_module
        if reconstruction_aux is not None and not isinstance(reconstruction_aux, Mapping):
            raise NativePublicRolloutError("reconstruction_aux_must_be_mapping")
        self._reconstruction_aux = (
            None if reconstruction_aux is None else deepcopy(dict(reconstruction_aux))
        )

    def _build(self, context: DecisionContext, sample: PublicSample) -> Any:
        if context.decision_signature != self._decision_signature:
            raise NativePublicRolloutError(
                "rollout_context_signature_mismatch:"
                f"{context.decision_signature}!={self._decision_signature}"
            )
        plan = build_redeterminization_plan(context, sample)
        if len(plan.monster_history) != 1:
            raise NativePublicRolloutError("native_v1_requires_exactly_one_enemy")
        seeds = dict(plan.rng_seeds)
        seeds["previous_history"] = plan.monster_history[0].previous_history_seed
        if self._reconstruction_aux is None:
            return self._native.build_public_jaw_worm_context_v1(self._public_state, seeds)
        return self._native.build_public_jaw_worm_context_v1(
            self._public_state,
            seeds,
            self._reconstruction_aux,
        )

    def _native_action(self, bc: Any, action: ActionSpec) -> Any:
        legal = list(self._native.get_legal_actions(bc))
        kind = str(action.payload.get("kind", ""))

        if kind == "play_card":
            hand_index = action.payload.get("hand_index")
            if isinstance(hand_index, bool) or not isinstance(hand_index, int):
                raise NativePublicRolloutError("public_card_action_missing_hand_index")
            source_idx = hand_index - 1
            target = action.payload.get("target_index")
            for native_action in legal:
                if _enum_name(getattr(native_action, "action_type", "")) != "CARD":
                    continue
                if int(getattr(native_action, "source_idx")) != source_idx:
                    continue
                if target is not None and int(getattr(native_action, "target_idx")) != int(target):
                    continue
                return native_action

        elif kind == "end_turn":
            for native_action in legal:
                if _enum_name(getattr(native_action, "action_type", "")) == "END_TURN":
                    return native_action

        raise NativePublicRolloutError(
            f"public_native_legal_action_mismatch:{kind}:{action.action_id}"
        )

    @staticmethod
    def _followup_key(bc: Any, action: Any) -> tuple[int, int, int]:
        action_type = _enum_name(getattr(action, "action_type", ""))
        if action_type == "CARD":
            source_idx = int(getattr(action, "source_idx"))
            hand = list(getattr(bc, "hand"))
            card_name = str(getattr(hand[source_idx], "name", "")).upper()
            priority = {"BASH": 0, "STRIKE": 1, "POMMEL STRIKE": 1, "DEFEND": 2}.get(card_name, 50)
            return (0, priority, int(getattr(action, "target_idx", 0)))
        if action_type == "END_TURN":
            return (1, 0, 0)
        return (9, 0, 0)

    def _followup(self, bc: Any) -> bool:
        legal = list(self._native.get_legal_actions(bc))
        if not legal:
            return False
        chosen = min(legal, key=lambda action: self._followup_key(bc, action))
        chosen.execute(bc)
        return True

    @staticmethod
    def _score(bc: Any) -> float:
        outcome = _enum_name(getattr(bc, "outcome", "UNDECIDED"))
        if outcome == "PLAYER_VICTORY":
            return 1_000_000.0 + float(getattr(bc.player, "cur_hp", 0))
        if outcome == "PLAYER_LOSS":
            return -1_000_000.0

        player_hp = float(getattr(bc.player, "cur_hp", 0))
        player_block = float(getattr(bc.player, "block", 0))
        player_energy = float(getattr(bc.player, "energy", 0))
        enemy_burden = 0.0
        for enemy in list(getattr(bc, "monsters")):
            enemy_burden += float(getattr(enemy, "cur_hp", 0)) * 10.0
            enemy_burden += float(getattr(enemy, "block", 0)) * 2.0
        return player_hp * 10.0 + player_block * 2.0 + player_energy - enemy_burden

    def rollout(
        self,
        context: DecisionContext,
        action: ActionSpec,
        sample: PublicSample,
        *,
        config: SearchConfig,
    ) -> float | None:
        bc = self._build(context, sample)
        first = self._native_action(bc, action)
        first.execute(bc)

        steps = 1
        while steps < config.max_depth:
            if _enum_name(getattr(bc, "outcome", "UNDECIDED")) != "UNDECIDED":
                break
            if not self._followup(bc):
                break
            steps += 1
        return self._score(bc)


class NativePublicCultistRolloutV1(NativePublicJawWormRolloutV1):
    """Same public-only rollout policy, with a distinct Cultist evidence identity."""

    backend_id = "sts1-public-native-cultist-rollout-v1"


class NativePublicGremlinNobRolloutV1(NativePublicJawWormRolloutV1):
    """Same public-only rollout policy, with a distinct Gremlin Nob identity."""

    backend_id = "sts1-public-native-gremlin-nob-rollout-v1"


class NativePublicBlueSlaverRolloutV1(NativePublicJawWormRolloutV1):
    """Public-only rollout identity for the audited Blue Slaver opening slice."""

    backend_id = "sts1-public-native-blue-slaver-rollout-v1"


class NativePublicRedSlaverRolloutV1(NativePublicJawWormRolloutV1):
    """Public-only rollout identity for the audited Red Slaver turn-0 slice."""

    backend_id = "sts1-public-native-red-slaver-turn0-rollout-v1"


class NativePublicLooterRolloutV1(NativePublicJawWormRolloutV1):
    """Public-only rollout identity for the audited Looter turn-0 slice.

    Opening MUG and initial Thievery are reconstructed only from the public
    encounter, turn, and ascension. Future move rolls use fresh rollout RNG.
    The scoring function is intentionally unchanged from the already-established
    raw PublicStateSearch candidate for discovery; no Looter-specific tuning is
    performed before seeing discovery evidence.
    """

    backend_id = "sts1-public-native-looter-turn0-rollout-v1"


class NativePublicLooterRolloutV2(NativePublicLooterRolloutV1):
    """Looter V2: preserve combat score dominance and use gold only as a tie-break.

    The frozen quality candidate averages exactly eight rollout scores. The V1
    combat score is integer-valued per rollout, so two different averaged combat
    scores are separated by at least 1/8 = 0.125. This V2 adds at most 0.001 to
    any rollout, which is over 100x smaller than that minimum non-zero combat
    gap. One gold is worth 1e-7, above the frozen Search tie tolerance (1e-9),
    so equal combat outcomes can still be ordered by remaining public gold.

    The cap is structural, not fitted to any discovery or holdout row.
    """

    backend_id = "sts1-public-native-looter-turn0-rollout-v2"
    gold_tiebreak_per_gold = 1e-7
    gold_tiebreak_cap = 0.001

    @classmethod
    def gold_tiebreak(cls, gold: int) -> float:
        if isinstance(gold, bool) or not isinstance(gold, int):
            raise NativePublicRolloutError("looter_v2_gold_must_be_int")
        return min(max(gold, 0) * cls.gold_tiebreak_per_gold, cls.gold_tiebreak_cap)

    def _score(self, bc: Any) -> float:
        combat_score = NativePublicJawWormRolloutV1._score(bc)
        gold = self._native.get_public_player_gold_v1(bc)
        return combat_score + self.gold_tiebreak(gold)


__all__ = [
    "NativePublicJawWormRolloutV1",
    "NativePublicCultistRolloutV1",
    "NativePublicGremlinNobRolloutV1",
    "NativePublicBlueSlaverRolloutV1",
    "NativePublicRedSlaverRolloutV1",
    "NativePublicLooterRolloutV1",
    "NativePublicLooterRolloutV2",
    "NativePublicRolloutError",
]
