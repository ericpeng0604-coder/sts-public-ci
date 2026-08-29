"""Conservative Arm B confidence fallback used only as a Phase-1 reference baseline.

This does not change the frozen Arm B model. When Arm B has one exact top
choice we keep it; when the frozen model is intrinsically ambiguous we fall
back to the existing deterministic simple public heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

ARMB_V2_POLICY_ID = "jialeiv-armb-v2-unique-else-simple-v1"


@dataclass(frozen=True)
class ArmBV2Decision:
    action_id: str | None
    source: str


def armb_v2_action_id(
    armb_tie_ids: Sequence[str],
    simple_action_id: str | None,
) -> ArmBV2Decision:
    """Return a deterministic public-only fallback decision.

    The rule is frozen before fresh-holdout results are observed:
    * exactly one Arm B top action -> use it;
    * otherwise -> use the deterministic simple-public action.
    """

    ties = tuple(str(action_id) for action_id in armb_tie_ids)
    if len(ties) == 1:
        return ArmBV2Decision(ties[0], "armb_unique")
    if simple_action_id is None:
        return ArmBV2Decision(None, "unresolved_no_simple_fallback")
    return ArmBV2Decision(str(simple_action_id), "simple_fallback_on_armb_ambiguity")


__all__ = ["ARMB_V2_POLICY_ID", "ArmBV2Decision", "armb_v2_action_id"]
