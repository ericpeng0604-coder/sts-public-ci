from roguelike_ai.sts1_teacher.armb_v2 import ARMB_V2_POLICY_ID, armb_v2_action_id


def test_armb_v2_keeps_unique_armb_choice() -> None:
    decision = armb_v2_action_id(("armb",), "simple")
    assert decision.action_id == "armb"
    assert decision.source == "armb_unique"


def test_armb_v2_falls_back_when_armb_is_ambiguous() -> None:
    decision = armb_v2_action_id(("a", "b"), "simple")
    assert decision.action_id == "simple"
    assert decision.source == "simple_fallback_on_armb_ambiguity"


def test_armb_v2_fails_closed_without_fallback() -> None:
    decision = armb_v2_action_id(("a", "b"), None)
    assert decision.action_id is None
    assert decision.source == "unresolved_no_simple_fallback"


def test_armb_v2_policy_identity_is_frozen() -> None:
    assert ARMB_V2_POLICY_ID == "jialeiv-armb-v2-unique-else-simple-v1"
