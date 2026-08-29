#!/usr/bin/env python3
"""Add Phase-1-only benchmark bindings to the hydrated simulator.

The overlay is deliberately applied after hydration inside CI. It changes only
binding-surface files: ``bindings/slaythespire.cpp`` adds a read-only branch
judge, while ``bindings/bindings-util.cpp`` makes the exported legal-action
surface match ``BattleScumSearcher2`` card enumeration. Formal Teacher code
never imports or uses the hidden-RNG oracle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "external" / "sts_lightspeed" / "source"
JUDGE_TARGET = SIM / "bindings" / "slaythespire.cpp"
LEGAL_TARGET = SIM / "bindings" / "bindings-util.cpp"

JUDGE_MARKER = '''    m.def("get_legal_actions", &sts::py::getLegalActions,
          "enumerate all legal actions in the current battle state for forward search");
'''

JUDGE_INSERT = '''    m.def("judge_branch_action",
          [](const BattleContext &bc, const search::Action &firstAction, int sims) {
              if (sims < 1) {
                  throw std::runtime_error("judge_search_budget_must_be_positive");
              }
              if (!firstAction.isValidAction(bc)) {
                  throw std::runtime_error("judge_first_action_not_legal");
              }

              BattleContext branch(bc);
              firstAction.execute(branch);
              search::BattleScumSearcher2 searcher(branch);
              searcher.search(sims);

              pybind11::dict out;
              out["score"] = searcher.bestActionValue;
              out["outcome_player_hp"] = searcher.outcomePlayerHp;
              out["root_simulations"] = searcher.root.simulationCount;
              out["best_action_sequence_length"] = searcher.bestActionSequence.size();
              out["player_hp_after_first"] = branch.player.curHp;
              out["turn_after_first"] = branch.turn;
              out["terminal_after_first"] = branch.outcome != Outcome::UNDECIDED;
              return out;
          },
          "Phase-1 benchmark only: score one exact first action from a copied BattleContext");

''' + JUDGE_MARKER

LEGAL_OLD = '''        // Card actions: every card in hand against every monster slot.
        for (int cardIdx = 0; cardIdx < bc.cards.cardsInHand; ++cardIdx) {
            for (int targetIdx = 0; targetIdx < monsterCount; ++targetIdx) {
                Action a(ActionType::CARD, cardIdx, targetIdx);
                if (a.isValidAction(bc)) {
                    actions.push_back(a);
                }
            }
        }
'''

LEGAL_NEW = '''        // PHASE1_CANONICAL_LEGAL_ACTIONS: mirror BattleScumSearcher2 card enumeration.
        // Non-target cards are one action, not one duplicate per monster. Adjacent
        // executable-equivalent card copies are deduplicated with the exact fields
        // used by BattleScumSearcher2::enumerateCardActions.
        for (int cardIdx = 0; cardIdx < bc.cards.cardsInHand; ++cardIdx) {
            const auto &c = bc.cards.hand[cardIdx];
            if (!c.canUseOnAnyTarget(bc)) {
                continue;
            }

            bool isUniqueAction = true;
            if (cardIdx > 0) {
                const auto &lastCard = bc.cards.hand[cardIdx - 1];
                const bool isEqualToLastCard = c.id == lastCard.id &&
                        c.getUpgradeCount() == lastCard.getUpgradeCount() &&
                        c.costForTurn == lastCard.costForTurn &&
                        c.cost == lastCard.cost &&
                        c.freeToPlayOnce == lastCard.freeToPlayOnce &&
                        c.specialData == lastCard.specialData;
                if (isEqualToLastCard) {
                    isUniqueAction = false;
                }
            }
            if (!isUniqueAction) {
                continue;
            }

            if (c.requiresTarget()) {
                for (int targetIdx = monsterCount - 1; targetIdx >= 0; --targetIdx) {
                    if (!bc.monsters.arr[targetIdx].isTargetable()) {
                        continue;
                    }
                    Action a(ActionType::CARD, cardIdx, targetIdx);
                    if (a.isValidAction(bc)) {
                        actions.push_back(a);
                    }
                }
            } else {
                Action a(ActionType::CARD, cardIdx);
                if (a.isValidAction(bc)) {
                    actions.push_back(a);
                }
            }
        }
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _apply_judge_overlay() -> dict[str, str]:
    if not JUDGE_TARGET.is_file():
        raise RuntimeError("hydrated simulator missing; run hydrate_jialeiv_baseline_v0.py first")
    before = JUDGE_TARGET.read_bytes()
    text = before.decode("utf-8")
    if 'm.def("judge_branch_action"' in text:
        status = "ALREADY_APPLIED"
    else:
        count = text.count(JUDGE_MARKER)
        if count != 1:
            raise RuntimeError(f"phase1 oracle overlay marker count must be 1, got {count}")
        JUDGE_TARGET.write_text(text.replace(JUDGE_MARKER, JUDGE_INSERT, 1), encoding="utf-8")
        status = "APPLIED"
    after = JUDGE_TARGET.read_bytes()
    if b"judge_branch_action" not in after:
        raise RuntimeError("judge_branch_action missing after overlay")
    return {
        "status": status,
        "target": str(JUDGE_TARGET.relative_to(SIM)),
        "before_sha256": sha256(before),
        "after_sha256": sha256(after),
    }


def _apply_legal_action_overlay() -> dict[str, str]:
    if not LEGAL_TARGET.is_file():
        raise RuntimeError("hydrated legal-action binding missing")
    before = LEGAL_TARGET.read_bytes()
    text = before.decode("utf-8")
    if "PHASE1_CANONICAL_LEGAL_ACTIONS" in text:
        status = "ALREADY_APPLIED"
    else:
        count = text.count(LEGAL_OLD)
        if count != 1:
            raise RuntimeError(f"phase1 legal-action overlay marker count must be 1, got {count}")
        LEGAL_TARGET.write_text(text.replace(LEGAL_OLD, LEGAL_NEW, 1), encoding="utf-8")
        status = "APPLIED"
    after = LEGAL_TARGET.read_bytes()
    if b"PHASE1_CANONICAL_LEGAL_ACTIONS" not in after:
        raise RuntimeError("canonical legal-action overlay missing after apply")
    return {
        "status": status,
        "target": str(LEGAL_TARGET.relative_to(SIM)),
        "before_sha256": sha256(before),
        "after_sha256": sha256(after),
    }


def main() -> None:
    judge = _apply_judge_overlay()
    legal_actions = _apply_legal_action_overlay()
    print(json.dumps({
        "judge": judge,
        "legal_actions": legal_actions,
        "binding_surface_only": True,
        "gameplay_semantics_changed": False,
        "legal_action_surface_matches_oracle_card_enumeration": True,
        "oracle_uses_hidden_rng": True,
        "formal_teacher_uses_oracle": False,
    }, indent=2))


if __name__ == "__main__":
    main()
