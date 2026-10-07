from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github/workflows/sts1-v311-selective-residual.yml"
TRAINER = Path(__file__).parents[1] / "scripts/sts1/sts1_build_rescue_adapter_v35.py"
TEACHER_AUDIT = Path(__file__).parents[1] / "scripts/sts1/sts1_build_rescue_teacher_audit_v311.py"
RECOVERY = Path(__file__).parents[1] / ".github/workflows/sts1-v311-dev30-recovery.yml"


def test_v311_only_runs_training_audit_probe_and_conditional_dev30():
    text = WORKFLOW.read_text(encoding="utf-8")
    for forbidden in (
        "- name: Gate30 unseen precheck",
        "- name: Gate50 unseen promotion screen",
        "- name: Fresh100 formal paired exact-sign gate",
        "- name: Fresh500 formal paired exact-sign gate",
        "phaseb_gate50.txt",
        "phaseb_fresh100.txt",
        "phaseb_fresh500.txt",
    ):
        assert forbidden not in text
    assert "Audit all verified Teacher targets against v3.10 and retention guards" in text
    assert "Run activation-only Probe10 on untouched seeds" in text
    assert "- name: Conditional paired MCTS-2000 Dev30" in text
    assert "if: steps.probe.outputs.safe == 'true' && steps.probe.outputs.changed == 'true'" in text


def test_audit_and_trainer_load_the_pinned_simulator_binding():
    for path in (TRAINER, TEACHER_AUDIT):
        text = path.read_text(encoding="utf-8")
        assert '"sim"/"sts_lightspeed"/"build312"' in text
        assert "sys.path.insert(0,str(sim_dir))" in text
        assert "pinned slaythespire binding is missing" in text


def test_dev30_recovery_reuses_candidate_and_requires_saved_probe_activation():
    text = RECOVERY.read_text(encoding="utf-8")
    assert "SOURCE_RUN_ID: '37639357507'" in text
    assert "actions/download-artifact@v4" in text
    assert "probe.get('all_games_complete_and_safe')" in text
    assert "int(probe.get('top1_changed',0))<1" in text
    assert "steps.candidate.outputs.run_dev30 == 'true'" in text
    assert "sts1_build_rescue_adapter_v35.py" not in text
    assert "phaseb_gate50.txt" not in text
    assert "phaseb_fresh100.txt" not in text
    assert "phaseb_fresh500.txt" not in text
    assert "V311_PROBE_DECISION" not in WORKFLOW.read_text(encoding="utf-8")
