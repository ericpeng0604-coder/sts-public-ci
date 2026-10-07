from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

HELPER = Path(__file__).parents[1] / "src/roguelike_ai/sts1_phase3/residual_scoring.py"
SPEC = importlib.util.spec_from_file_location("residual_scoring_torch_v311", HELPER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
choice_relative_margin_loss = MODULE.choice_relative_margin_loss


def test_choice_relative_loss_gradient_changes_target_and_parent_top_logits_only():
    parent = torch.tensor([3.0, 1.0, -1.0])
    residual = torch.tensor([0.2, 0.0, -0.1], requires_grad=True)
    loss = choice_relative_margin_loss(parent, residual, 1, 0.5, torch.relu)
    assert loss is not None
    loss.backward()
    assert residual.grad.tolist() == [1.0, -1.0, 0.0]


def test_parent_top_teacher_has_no_contrastive_flip_gradient():
    parent = torch.tensor([3.0, 1.0, -1.0])
    residual = torch.zeros(3, requires_grad=True)
    assert choice_relative_margin_loss(parent, residual, 0, 0.5, torch.relu) is None
    assert residual.grad is None
