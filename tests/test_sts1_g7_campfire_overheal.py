from types import SimpleNamespace

import pytest

from roguelike_ai.sts1_phase3.simulator import _apply_campfire_overheal_override


class _Policy:
    @staticmethod
    def describe_choice(kind, descriptor):
        assert kind == "rest"
        return {"kind": "rest", "option_index": descriptor["option_index"]}


class _GameContext:
    def __init__(self, hp, max_hp, *, pillow=False):
        self.cur_hp = hp
        self.max_hp = max_hp
        self._pillow = pillow

    def hasRelic(self, relic_id):
        assert relic_id == "REGAL_PILLOW"
        return self._pillow


class _Sts:
    RelicId = SimpleNamespace(REGAL_PILLOW="REGAL_PILLOW")


@pytest.mark.parametrize(
    ("hp", "max_hp", "pillow", "recommended", "expected_index", "expected_override"),
    [
        (75, 100, False, 0, 1, True),
        (70, 100, False, 0, 0, False),
        (60, 100, True, 0, 1, True),
        (75, 100, False, 1, 1, False),
    ],
)
def test_campfire_overheal_rule_uses_exact_legal_smith_choice(
    hp, max_hp, pillow, recommended, expected_index, expected_override
):
    selected, detail = _apply_campfire_overheal_override(
        recommended,
        "rest",
        [{"option_index": 0}, {"option_index": 1}],
        _GameContext(hp, max_hp, pillow=pillow),
        _Sts,
        _Policy(),
    )

    assert selected == expected_index
    assert detail["overridden"] is expected_override
    if expected_override:
        assert detail["selected_index"] == 1
        assert detail["reason"] == "rest_heal_would_exceed_max_hp"


def test_campfire_overheal_rule_fails_closed_without_legal_smith():
    selected, detail = _apply_campfire_overheal_override(
        0,
        "rest",
        [{"option_index": 0}, {"option_index": 2}],
        _GameContext(90, 100),
        _Sts,
        _Policy(),
    )

    assert selected == 0
    assert detail["overridden"] is False
    assert detail["reason"] == "rest_or_smith_not_uniquely_legal"


def test_campfire_overheal_rule_fails_closed_without_relic_state():
    context = SimpleNamespace(cur_hp=90, max_hp=100)
    selected, detail = _apply_campfire_overheal_override(
        0,
        "rest",
        [{"option_index": 0}, {"option_index": 1}],
        context,
        _Sts,
        _Policy(),
    )

    assert selected == 0
    assert detail["overridden"] is False
    assert detail["reason"] == "relic_state_unavailable"
