#!/usr/bin/env python3
"""Add a Phase-1-only native MCTS oracle binding to the hydrated simulator.

The overlay is deliberately applied after hydration inside CI.  It changes only
``bindings/slaythespire.cpp`` and exposes a read-only branch judge around the
pinned upstream ``BattleScumSearcher2``.  Formal Teacher code never imports or
uses this hidden-RNG oracle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "external" / "sts_lightspeed" / "source"
TARGET = SIM / "bindings" / "slaythespire.cpp"

MARKER = '''    m.def("get_legal_actions", &sts::py::getLegalActions,
          "enumerate all legal actions in the current battle state for forward search");
'''

INSERT = '''    m.def("judge_branch_action",
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

''' + MARKER


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if not TARGET.is_file():
        raise RuntimeError("hydrated simulator missing; run hydrate_jialeiv_baseline_v0.py first")
    before = TARGET.read_bytes()
    text = before.decode("utf-8")
    if 'm.def("judge_branch_action"' in text:
        status = "ALREADY_APPLIED"
    else:
        count = text.count(MARKER)
        if count != 1:
            raise RuntimeError(f"phase1 oracle overlay marker count must be 1, got {count}")
        TARGET.write_text(text.replace(MARKER, INSERT, 1), encoding="utf-8")
        status = "APPLIED"
    after = TARGET.read_bytes()
    if b"judge_branch_action" not in after:
        raise RuntimeError("judge_branch_action missing after overlay")
    print(json.dumps({
        "status": status,
        "target": str(TARGET.relative_to(SIM)),
        "before_sha256": sha256(before),
        "after_sha256": sha256(after),
        "binding_surface_only": True,
        "gameplay_semantics_changed": False,
        "oracle_uses_hidden_rng": True,
        "formal_teacher_uses_oracle": False,
    }, indent=2))


if __name__ == "__main__":
    main()
