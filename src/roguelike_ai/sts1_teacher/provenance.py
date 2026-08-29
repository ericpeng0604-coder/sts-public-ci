"""Pinned external STS1 teacher/reference identities used by Phase 1."""

from __future__ import annotations

JIALEIV_STS_RL_AGENT_SHA = "b20eb2cac2f52b22fbb6c79900c309b51ea0a1db"
STS_LIGHTSPEED_SHA = "7476a81954020087da31d41d16fddf475746ec2d"
JIALEIV_ARMG_WEIGHT_BLOB_SHA = "6666fb690fcceb592a1aa8dc74fa27bfee4575f0"
JIALEIV_ARMB_WEIGHT_BLOB_SHA = "7b9db5598b1863b117d1ab8b9a80f4ed9056722c"
JIALEIV_CARD_VOCAB_BLOB_SHA = "30041ef13c199f7aa11847cef860777be52c537e"

# The upstream battle search clones BattleContext, and that clone explicitly
# includes RNG state.  Therefore it is valuable reference/oracle evidence but
# is not admissible as the formal public-state Teacher required by #345.
JIALEIV_BATTLE_SEARCH_ROLE = "oracle_reference"
JIALEIV_BATTLE_SEARCH_HIDDEN_INFORMATION = True
JIALEIV_NONCOMBAT_MODEL_ROLE = "external_public_candidate"
JIALEIV_BATTLE_POLICY_ROLE = "external_public_teacher_candidate"


__all__ = [
    "JIALEIV_ARMB_WEIGHT_BLOB_SHA",
    "JIALEIV_ARMG_WEIGHT_BLOB_SHA",
    "JIALEIV_BATTLE_POLICY_ROLE",
    "JIALEIV_CARD_VOCAB_BLOB_SHA",
    "JIALEIV_BATTLE_SEARCH_HIDDEN_INFORMATION",
    "JIALEIV_BATTLE_SEARCH_ROLE",
    "JIALEIV_NONCOMBAT_MODEL_ROLE",
    "JIALEIV_STS_RL_AGENT_SHA",
    "STS_LIGHTSPEED_SHA",
]
