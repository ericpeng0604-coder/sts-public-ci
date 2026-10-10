"""Zero-episode native smoke for observation purity; emits booleans only."""
import argparse
import importlib.util
import json
from pathlib import Path


def smoke(module_dir: Path) -> dict:
    path = module_dir / "slaythespire.cp312-win_amd64.pyd"
    spec = importlib.util.spec_from_file_location("slaythespire", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("native module unavailable")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    gc = native.GameContext(native.CharacterClass.IRONCLAD, 0, 0)
    battle = native.BattleContext()
    battle.init_encounter(gc, native.MonsterEncounter.CULTIST)
    untouched = battle.clone()
    observed = battle.clone()
    before = repr(observed)
    legal_before = [action.bits for action in native.get_legal_actions(observed)]
    slots = list(observed.combat_potions)
    if slots != ["EMPTY_POTION_SLOT"] * 5:
        raise RuntimeError("incomplete live inventory")
    slots[0] = "FIRE_POTION"
    if list(observed.combat_potions) != ["EMPTY_POTION_SLOT"] * 5:
        raise RuntimeError("getter leaked native mutable state")
    if repr(observed) != before or legal_before != [action.bits for action in native.get_legal_actions(observed)]:
        raise RuntimeError("reading resources changed native state or legal actions")
    baseline = native.mcts_recommend(untouched, 2000)
    traced = native.mcts_recommend(observed, 2000)
    if baseline.bits != traced.bits:
        raise RuntimeError("reading resources changed MCTS2000 recommendation")
    return {"status": "PASS", "episodes_started": 0, "mcts_sims": 2000,
            "live_slots": 5, "getter_returns_copy": True, "native_state_unchanged": True,
            "legal_actions_unchanged": True, "mcts_recommendation_unchanged": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(smoke(args.module_dir), sort_keys=True))
    except Exception as error:
        print(json.dumps({"status": "NOT_VERIFIED", "error_class": type(error).__name__}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
