from copy import deepcopy
import json

import pytest

from scripts.sts1.sts1_g7_train_resource_audit import AuditError, audit, consumption_diagnostic


def decisions():
    potions = ["ENERGY_POTION"] + ["EMPTY_POTION_SLOT"] * 4
    return [
        {"encounter_index": 1, "selected_action": {"kind": "use_potion", "potion_index": 0},
         "public_state": {"potions": list(potions)}},
        {"encounter_index": 1, "selected_action": {"kind": "end_turn"},
         "public_state": {"potions": list(potions)},
         "canonical_native_legal_actions": [{"kind": "end_turn"}]},
    ]


def test_consumption_diagnostic_preserves_raw_evidence_and_reports_suspicion():
    rows = decisions()
    original = deepcopy(rows)
    assert consumption_diagnostic(rows) == {"same_encounter_successors": 1, "retained_slot_without_legal_use": 1}
    assert rows == original


@pytest.mark.parametrize("change", ["new_encounter", "updated_slot", "still_legal"])
def test_does_not_infer_staleness_across_encounters_or_when_evidence_disagrees(change):
    rows = decisions()
    if change == "new_encounter":
        rows[1]["encounter_index"] = 2
    elif change == "updated_slot":
        rows[1]["public_state"]["potions"][0] = "EMPTY_POTION_SLOT"
    else:
        rows[1]["canonical_native_legal_actions"].append(rows[0]["selected_action"])
    assert consumption_diagnostic(rows)["retained_slot_without_legal_use"] == 0


def test_rejects_bool_ordinal_and_incomplete_resources():
    rows = decisions()
    rows[0]["selected_action"]["potion_index"] = True
    with pytest.raises(AuditError):
        consumption_diagnostic(rows)
    rows = decisions()
    rows[1]["public_state"]["potions"] = None
    with pytest.raises(AuditError):
        consumption_diagnostic(rows)


def test_audit_rejects_probe_before_considering_outcomes(tmp_path):
    for index in range(10):
        events = [{"type": "diagnostic_trace_header_v1", "run_metadata": {"stage": "probe"}},
                  {"type": "terminal_trace_v1"}]
        (tmp_path / f"pair-{index:02d}-parent.trace.ndjson").write_text(
            "\n".join(json.dumps(e) for e in events), encoding="utf-8"
        )
    with pytest.raises(AuditError, match="Train"):
        audit(tmp_path)
