"""Native public-state rollout backends for the first audited STS1 V1 slices.

These backends never receive a source simulator BattleContext. Every rollout
creates a fresh native BattleContext from the already-admitted public state and
an action-independent ``PublicSample``. The deterministic follow-up policy is
intentionally simple; V1 proves plumbing and public-only search semantics, not
a final Phase-1 Teacher quality gate.
"""
from __future__ import annotations

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

    def __init__(self, public_state: dict[str, Any], native_module: Any) -> None:
        admitted = require_public_reconstruction(public_state)
        self._decision_signature = admitted.decision_signature
        self._public_state = deepcopy(public_state)
        self._native = native_module

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
        return self._native.build_public_jaw_worm_context_v1(self._public_state, seeds)

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
            priority = {"BASH": 0, "STRIKE": 1, "DEFEND": 2}.get(card_name, 50)
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
    """Same public-only rollout policy, with a distinct Cultist evidence identity.

    The shared native constructor is guarded by reconstruction admission and the
    native boundary, so Cultist V1 is still opening-turn INCANTATION only.
    """

    backend_id = "sts1-public-native-cultist-rollout-v1"


__all__ = [
    "NativePublicJawWormRolloutV1",
    "NativePublicCultistRolloutV1",
    "NativePublicRolloutError",
]
