#!/usr/bin/env python3
"""Tail2 surgical New-Win trainer for STS1 ArmG.

v2.7 freezes the early ArmG backbone and updates only the final two Linear
layers. Verified New-Win examples use a pairwise margin objective:
the proven teacher action only needs to outrank G7's original action by a small
margin. G7 replay, distillation, preservation, and interpolation guards remain.

Compared with v2.1:
- uses a larger winner replay sample;
- distills the full parent action distribution, not just replay hard labels;
- guards parent top-1 agreement on a disjoint retention split;
- tightens the parent-KL limit;
- selects checkpoints only when retention gates pass;
- line-searches a parent<->candidate interpolation and keeps the safest useful
  update.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "sts1-armg-strategy-branch-dataset-v1"
PRESERVATION_SCHEMA = "sts1-build-preservation-v1"
PARENT_HIGH_CONFIDENCE_PROB_GAP = 0.05
MIN_HIGH_CONFIDENCE_DECISIONS = 256


def _load_rescues(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str, int, int]] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != SCHEMA:
            raise RuntimeError(f"rescue schema mismatch line {line_no}")
        obs = [float(v) for v in row["obs"]]
        descs = [[float(v) for v in d] for d in row["descs"]]
        probs = [float(v) for v in row["target_probs"]]
        cur = int(row["current_armg_index"])
        teacher = int(row["teacher_best_index"])
        if not descs or len(probs) != len(descs):
            raise RuntimeError(f"rescue candidate shape mismatch line {line_no}")
        if not 0 <= cur < len(descs) or not 0 <= teacher < len(descs):
            raise RuntimeError(f"rescue index mismatch line {line_no}")
        if cur == teacher:
            raise RuntimeError(f"rescue must be a disagreement line {line_no}")
        if abs(sum(probs) - 1.0) > 1e-6 or any(v < 0 for v in probs):
            raise RuntimeError(f"rescue target invalid line {line_no}")
        if float(row.get("confidence_weight", 0.0)) < 1.0:
            raise RuntimeError(f"rescue confidence below verified contract line {line_no}")
        if float(row.get("teacher_consensus_fraction", 0.0)) < 1.0:
            raise RuntimeError(f"rescue consensus below verified contract line {line_no}")
        key = (int(row["seed"]), int(row["floor"]), str(row["kind"]), cur, teacher)
        if key in seen:
            continue
        seen.add(key)
        rows.append({**row, "obs": obs, "descs": descs, "target_probs": probs})
    if not rows:
        raise RuntimeError("verified rescue dataset is empty")
    return rows


def _load_preservation(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str, int]] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != PRESERVATION_SCHEMA:
            raise RuntimeError(f"preservation schema mismatch line {line_no}")
        obs = [float(v) for v in row["obs"]]
        descs = [[float(v) for v in d] for d in row["descs"]]
        probs = [float(v) for v in row["target_probs"]]
        target = int(row["parent_selected_index"])
        if not descs or len(probs) != len(descs):
            raise RuntimeError(f"preservation candidate shape mismatch line {line_no}")
        if not 0 <= target < len(descs):
            raise RuntimeError(f"preservation target index mismatch line {line_no}")
        if abs(sum(probs) - 1.0) > 1e-6 or any(v < 0 for v in probs):
            raise RuntimeError(f"preservation target invalid line {line_no}")
        if int(np.argmax(np.asarray(probs))) != target:
            raise RuntimeError(f"preservation target distribution mismatch line {line_no}")
        key = (int(row["seed"]), int(row["floor"]), str(row["kind"]), target)
        if key in seen:
            continue
        seen.add(key)
        rows.append({**row, "obs": obs, "descs": descs, "target_probs": probs})
    if not rows:
        raise RuntimeError("preservation dataset is empty")
    return rows


def _load_elite(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        required = ("obs", "desc", "offsets", "action", "game_victory")
        missing = [k for k in required if k not in d]
        if missing:
            raise RuntimeError(f"elite replay missing keys: {missing}")
        out = {k: np.asarray(d[k]) for k in required}
    if len(out["offsets"]) != len(out["action"]) + 1:
        raise RuntimeError("elite offsets invalid")
    if len(out["action"]) < 2:
        raise RuntimeError("elite replay too small")
    if not np.all(out["game_victory"]):
        raise RuntimeError("elite replay contains non-victory decisions")
    return out


def _teacher_loss(actor, torch, row: dict[str, Any]):
    obs = torch.tensor(row["obs"], dtype=torch.float32)
    desc = torch.tensor(row["descs"], dtype=torch.float32)
    logits = actor.net(torch.cat([obs.repeat(len(desc), 1), desc], 1)).squeeze(1)
    target = torch.tensor(row["target_probs"], dtype=torch.float32)
    return -(target * torch.log_softmax(logits, 0)).sum(), logits


def _elite_terms(actor, parent, torch, elite: dict[str, np.ndarray], i: int):
    lo = int(elite["offsets"][i])
    hi = int(elite["offsets"][i + 1])
    if hi <= lo:
        raise RuntimeError(f"elite candidate span empty at {i}")
    obs = torch.tensor(elite["obs"][i], dtype=torch.float32)
    desc = torch.tensor(elite["desc"][lo:hi], dtype=torch.float32)
    x = torch.cat([obs.repeat(len(desc), 1), desc], 1)
    logits = actor.net(x).squeeze(1)
    with torch.no_grad():
        parent_logits = parent.net(x).squeeze(1)
        parent_probs = torch.softmax(parent_logits, 0)
        parent_top1 = int(torch.argmax(parent_logits))
        if len(parent_probs) >= 2:
            top2 = torch.topk(parent_probs, k=2).values
            parent_prob_gap = float(top2[0] - top2[1])
        else:
            parent_prob_gap = 1.0
    logp = torch.log_softmax(logits, 0)
    action = int(elite["action"][i])
    hard_ce = -logp[action]
    parent_kl = (parent_probs * (torch.log_softmax(parent_logits, 0) - logp)).sum()
    return hard_ce, parent_kl, logits, parent_top1, parent_prob_gap


def _eval_rescue(actor, torch, rows):
    losses: list[float] = []
    top1 = 0
    with torch.no_grad():
        for row in rows:
            loss, logits = _teacher_loss(actor, torch, row)
            losses.append(float(loss))
            top1 += int(int(torch.argmax(logits)) == int(row["teacher_best_index"]))
    return {"loss": float(np.mean(losses)), "top1": top1 / len(rows)}


def _teacher_margin_loss(actor, torch, row: dict[str, Any], margin: float):
    obs = torch.tensor(row["obs"], dtype=torch.float32)
    desc = torch.tensor(row["descs"], dtype=torch.float32)
    logits = actor.net(torch.cat([obs.repeat(len(desc), 1), desc], 1)).squeeze(1)
    target = int(row["teacher_best_index"])
    if len(logits) <= 1:
        return logits.sum() * 0.0, logits
    mask = torch.ones(len(logits), dtype=torch.bool)
    mask[target] = False
    best_other = torch.max(logits[mask])
    violation = best_other - logits[target] + float(margin)
    return torch.relu(violation), logits


def _eval_new_win(actor, torch, rows):
    if not rows:
        raise RuntimeError("new-win dataset is empty")
    losses: list[float] = []
    correct = 0
    by_seed: dict[int, list[bool]] = {}
    with torch.no_grad():
        for row in rows:
            loss, logits = _teacher_loss(actor, torch, row)
            hit = int(torch.argmax(logits)) == int(row["teacher_best_index"])
            losses.append(float(loss))
            correct += int(hit)
            by_seed.setdefault(int(row["seed"]), []).append(bool(hit))
    learned = sorted(seed for seed, hits in by_seed.items() if all(hits))
    return {
        "loss": float(np.mean(losses)),
        "top1": correct / len(rows),
        "top1_examples": int(correct),
        "examples": int(len(rows)),
        "fully_learned_seeds": int(len(learned)),
        "teacher_seeds_total": int(len(by_seed)),
        "fully_learned_seed_ids": learned,
    }


def _preservation_margin_loss(actor, torch, row, margin: float):
    obs = torch.tensor(row["obs"], dtype=torch.float32)
    desc = torch.tensor(row["descs"], dtype=torch.float32)
    logits = actor.net(torch.cat([obs.repeat(len(desc), 1), desc], 1)).squeeze(1)
    target = int(row["parent_selected_index"])
    if len(logits) <= 1:
        return logits.sum() * 0.0, logits
    mask = torch.ones(len(logits), dtype=torch.bool)
    mask[target] = False
    best_other = torch.max(logits[mask])
    violation = best_other - logits[target] + float(margin)
    return torch.relu(violation), logits


def _new_win_pairwise_margin_loss(actor, torch, row, margin: float):
    obs = torch.tensor(row["obs"], dtype=torch.float32)
    desc = torch.tensor(row["descs"], dtype=torch.float32)
    logits = actor.net(torch.cat([obs.repeat(len(desc), 1), desc], 1)).squeeze(1)
    teacher = int(row["teacher_best_index"])
    current = int(row["current_armg_index"])
    if not (0 <= teacher < len(logits) and 0 <= current < len(logits)):
        raise RuntimeError("new-win teacher/current index outside candidate range")
    if teacher == current:
        raise RuntimeError("new-win teacher must differ from current G7 choice")
    violation = logits[current] - logits[teacher] + float(margin)
    return torch.relu(violation), logits


def _eval_new_win_margin(actor, torch, rows, margin: float):
    losses = []
    top1 = 0
    margin_ok = 0
    raw_margins = []
    with torch.no_grad():
        for row in rows:
            loss, logits = _new_win_pairwise_margin_loss(actor, torch, row, margin)
            teacher = int(row["teacher_best_index"])
            current = int(row["current_armg_index"])
            raw = float(logits[teacher] - logits[current])
            raw_margins.append(raw)
            losses.append(float(loss))
            top1 += int(int(torch.argmax(logits)) == teacher)
            margin_ok += int(raw >= float(margin))
    return {
        "loss": float(np.mean(losses)),
        "top1": top1 / len(rows),
        "margin_pass": margin_ok / len(rows),
        "mean_teacher_minus_g7": float(np.mean(raw_margins)),
        "min_teacher_minus_g7": float(np.min(raw_margins)),
        "examples": len(rows),
    }


def _eval_preservation(actor, torch, rows):
    if not rows:
        return {"loss": 0.0, "top1": 1.0, "examples": 0}
    losses: list[float] = []
    correct = 0
    with torch.no_grad():
        for row in rows:
            loss, logits = _teacher_loss(actor, torch, row)
            losses.append(float(loss))
            correct += int(
                int(torch.argmax(logits)) == int(row["parent_selected_index"])
            )
    return {
        "loss": float(np.mean(losses)),
        "top1": correct / len(rows),
        "examples": len(rows),
    }


def _eval_retention(actor, parent, torch, elite, indices):
    if len(indices) == 0:
        raise RuntimeError("retention split is empty")
    winner_correct = 0
    parent_agree = 0
    high_conf_total = 0
    high_conf_agree = 0
    kls: list[float] = []
    gaps: list[float] = []
    with torch.no_grad():
        for raw in indices:
            i = int(raw)
            _, kl, logits, parent_top1, parent_prob_gap = _elite_terms(
                actor, parent, torch, elite, i
            )
            pred = int(torch.argmax(logits))
            winner_correct += int(pred == int(elite["action"][i]))
            parent_agree += int(pred == parent_top1)
            if parent_prob_gap >= PARENT_HIGH_CONFIDENCE_PROB_GAP:
                high_conf_total += 1
                high_conf_agree += int(pred == parent_top1)
            kls.append(float(kl))
            gaps.append(parent_prob_gap)
    high_conf_agreement = (
        high_conf_agree / high_conf_total if high_conf_total else 0.0
    )
    return {
        "winner_top1": winner_correct / len(indices),
        "parent_top1_agreement": parent_agree / len(indices),
        "parent_high_conf_top1_agreement": high_conf_agreement,
        "parent_high_conf_decisions": int(high_conf_total),
        "parent_high_conf_prob_gap": PARENT_HIGH_CONFIDENCE_PROB_GAP,
        "parent_prob_gap_mean": float(np.mean(gaps)),
        "parent_kl": float(np.mean(kls)),
        "decisions": int(len(indices)),
    }


def _blend_state(torch, parent_state, candidate_state, alpha: float):
    if not 0.0 <= alpha <= 1.0:
        raise RuntimeError("blend alpha outside [0,1]")
    return {
        key: parent_state[key] + (candidate_state[key] - parent_state[key]) * alpha
        for key in parent_state
    }


def _retention_ok(stats, before, *, min_parent_agreement, max_parent_kl, max_winner_drop):
    return (
        int(stats["parent_high_conf_decisions"]) >= MIN_HIGH_CONFIDENCE_DECISIONS
        and float(stats["parent_high_conf_top1_agreement"]) >= min_parent_agreement
        and float(stats["parent_kl"]) <= max_parent_kl
        and float(stats["winner_top1"]) + max_winner_drop >= float(before["winner_top1"])
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--armg-root", type=Path, required=True)
    p.add_argument("--module-dir", type=Path, required=True)
    p.add_argument("--base-weight", type=Path, required=True)
    p.add_argument("--elite-replay", type=Path, required=True)
    p.add_argument("--rescue-replay", type=Path, required=True)
    p.add_argument("--new-win-replay", type=Path, required=True)
    p.add_argument("--preservation-replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--elite-train-max", type=int, default=8192)
    p.add_argument("--retention-eval-max", type=int, default=2048)
    p.add_argument("--rescue-repeats", type=int, default=2)
    p.add_argument("--new-win-repeats", type=int, default=8)
    p.add_argument("--winner-coef", type=float, default=0.80)
    p.add_argument("--distill-coef", type=float, default=1.20)
    p.add_argument("--rescue-coef", type=float, default=0.05)
    p.add_argument("--new-win-coef", type=float, default=0.50)
    p.add_argument("--new-win-margin", type=float, default=0.01)
    p.add_argument("--new-win-margin-coef", type=float, default=0.50)
    p.add_argument("--min-fully-learned-seeds", type=int, default=1)
    p.add_argument("--preservation-coef", type=float, default=0.30)
    p.add_argument("--preservation-margin", type=float, default=0.001)
    p.add_argument("--param-anchor-coef", type=float, default=1e-5)
    p.add_argument("--min-parent-agreement", type=float, default=0.995)
    p.add_argument("--max-parent-kl", type=float, default=0.002)
    p.add_argument("--max-winner-drop", type=float, default=0.01)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    if not 1 <= a.epochs <= 20:
        raise RuntimeError("epochs outside 1..20")
    if not 0 < a.lr <= 2e-5:
        raise RuntimeError("lr outside safe bound")
    if not 128 <= a.elite_train_max <= 32768:
        raise RuntimeError("elite-train-max outside safe bound")
    if not 64 <= a.retention_eval_max <= 8192:
        raise RuntimeError("retention-eval-max outside safe bound")
    if not 1 <= a.rescue_repeats <= 16:
        raise RuntimeError("rescue-repeats outside safe bound")
    if not 1 <= a.new_win_repeats <= 24:
        raise RuntimeError("new-win-repeats outside safe bound")
    if not 0 < a.winner_coef <= 2 or not 0 < a.distill_coef <= 4 or not 0 < a.rescue_coef <= 2:
        raise RuntimeError("loss coefficient outside safe bound")
    if not 0 < a.new_win_coef <= 3:
        raise RuntimeError("new-win coefficient outside safe bound")
    if not 0 < a.new_win_margin <= 0.10:
        raise RuntimeError("new-win margin outside safe bound")
    if not 0 < a.new_win_margin <= 0.20:
        raise RuntimeError("new-win-margin outside safe bound")
    if not 0 < a.new_win_margin_coef <= 3:
        raise RuntimeError("new-win-margin-coef outside safe bound")
    if not 1 <= a.min_fully_learned_seeds <= 20:
        raise RuntimeError("min-fully-learned-seeds outside safe bound")
    if not 0 < a.preservation_coef <= 2:
        raise RuntimeError("preservation coefficient outside safe bound")
    if not 0 < a.preservation_margin <= 0.05:
        raise RuntimeError("preservation margin outside safe bound")
    if not 0 <= a.param_anchor_coef <= 1e-2:
        raise RuntimeError("param anchor outside safe bound")
    if not 0.95 <= a.min_parent_agreement <= 1.0:
        raise RuntimeError("min-parent-agreement outside safe bound")
    if not 0 < a.max_parent_kl <= 0.02:
        raise RuntimeError("max-parent-kl outside safe bound")
    if not 0 <= a.max_winner_drop <= 0.05:
        raise RuntimeError("max-winner-drop outside safe bound")

    os.environ["STS_BOT_DIR"] = str(a.armg_root)
    sys.path.insert(0, str(a.armg_root))
    sim_dir = a.armg_root / "sim" / "sts_lightspeed" / "build312"
    if sim_dir.exists():
        sys.path.insert(0, str(sim_dir))

    torch = importlib.import_module("torch")
    torch.set_num_threads(a.threads)
    torch.manual_seed(20260930)
    m = importlib.import_module("armG_train")
    m.card_idx = lambda n: m._vocab.get(n, m.VOCAB_CAP - 1)

    actor = m.Scorer((128, 128))
    parent_state = torch.load(a.base_weight, weights_only=True, map_location="cpu")
    actor.load_state_dict(parent_state)
    parent = m.Scorer((128, 128))
    parent.load_state_dict(parent_state)
    parent.eval()
    for param in parent.parameters():
        param.requires_grad_(False)

    rescues = _load_rescues(a.rescue_replay)
    new_wins = _load_rescues(a.new_win_replay)
    preservation = _load_preservation(a.preservation_replay)
    elite = _load_elite(a.elite_replay)
    if len(rescues) < 10:
        raise RuntimeError(f"v2.4 expects the frozen Rescue pool; got {len(rescues)}")
    if len(new_wins) < 3:
        raise RuntimeError(f"v2.4 requires at least 3 New-Win Teacher examples; got {len(new_wins)}")

    expected_obs = int(m.OBS_DIM)
    first_linear = next(layer for layer in actor.net if hasattr(layer, "in_features"))
    expected_desc = int(first_linear.in_features) - expected_obs
    for row in rescues:
        if len(row["obs"]) != expected_obs or any(len(d) != expected_desc for d in row["descs"]):
            raise RuntimeError("rescue feature dimension mismatch")
    for row in new_wins:
        if len(row["obs"]) != expected_obs or any(len(d) != expected_desc for d in row["descs"]):
            raise RuntimeError("new-win feature dimension mismatch")
    for row in preservation:
        if len(row["obs"]) != expected_obs or any(len(d) != expected_desc for d in row["descs"]):
            raise RuntimeError("preservation feature dimension mismatch")

    rng = np.random.default_rng(20260930)
    perm = rng.permutation(len(elite["action"]))
    eval_count = min(a.retention_eval_max, max(64, len(perm) // 5))
    if len(perm) - eval_count < 128:
        eval_count = max(1, len(perm) // 4)
    retention_eval = np.asarray(perm[:eval_count], dtype=np.int64)
    train_pool = np.asarray(perm[eval_count:], dtype=np.int64)
    elite_train = train_pool[: min(a.elite_train_max, len(train_pool))]
    if len(elite_train) < 128:
        raise RuntimeError("not enough disjoint elite winner decisions for anti-forgetting training")

    before_rescue = _eval_rescue(actor, torch, rescues)
    before_new_win = _eval_new_win(actor, torch, new_wins)
    before_new_win_margin = _eval_new_win_margin(
        actor, torch, new_wins, a.new_win_margin
    )
    before_preservation = _eval_preservation(actor, torch, preservation)
    before_retention = _eval_retention(actor, parent, torch, elite, retention_eval)
    if before_preservation["top1"] < 1.0:
        raise RuntimeError("frozen G7 does not satisfy its own preservation anchors")

    parent_params = {k: v.detach().clone() for k, v in parent.state_dict().items()}

    # v2.7 surgical Tail2: freeze everything, then unfreeze only the final
    # two Linear layers. This gives local nonlinear capacity without allowing
    # the full 780->128 backbone to drift.
    for param in actor.parameters():
        param.requires_grad_(False)
    linear_layers = [
        layer for layer in actor.net
        if hasattr(layer, "in_features") and hasattr(layer, "out_features")
    ]
    if len(linear_layers) < 2:
        raise RuntimeError("unable to locate final two ArmG Linear layers")
    tail_layers = linear_layers[-2:]
    for layer in tail_layers:
        for param in layer.parameters():
            param.requires_grad_(True)
    trainable = [p for p in actor.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("surgical Tail2 has no trainable parameters")
    opt = torch.optim.Adam(trainable, lr=a.lr)
    history: list[dict[str, Any]] = []
    trained_candidates: list[
        tuple[tuple[float, float, float], dict[str, Any], dict[str, Any]]
    ] = []

    for ep in range(a.epochs):
        actor.train()
        rescue_order = [i for i in range(len(rescues)) for _ in range(a.rescue_repeats)]
        new_win_order = [i for i in range(len(new_wins)) for _ in range(a.new_win_repeats)]
        rng.shuffle(rescue_order)
        rng.shuffle(new_win_order)
        elite_order = elite_train.copy()
        rng.shuffle(elite_order)
        # v2.5: elite replay defines epoch length. Small teacher pools are
        # deliberately cycled so their signal is present in every batch.
        steps = len(elite_order)
        losses: list[float] = []

        for start in range(0, steps, 32):
            rescue_terms = []
            new_win_terms = []
            new_win_margin_terms = []
            preservation_terms = []
            winner_terms = []
            distill_terms = []
            end = min(start + 32, steps)
            for j in range(start, end):
                if rescue_order:
                    row = rescues[int(rescue_order[j % len(rescue_order)])]
                    ce, _ = _teacher_loss(actor, torch, row)
                    priority = min(2.0, max(1.0, float(row.get("priority", 1.0))))
                    rescue_terms.append(priority * ce)
                if new_win_order:
                    row = new_wins[int(new_win_order[j % len(new_win_order)])]
                    ce, _ = _teacher_loss(actor, torch, row)
                    hinge, _ = _new_win_pairwise_margin_loss(
                        actor, torch, row, a.new_win_margin
                    )
                    priority = min(
                        2.0, max(1.0, float(row.get("priority", 1.0)) / 4.0)
                    )
                    new_win_terms.append(priority * ce)
                    new_win_margin_terms.append(hinge)
                if j < len(elite_order):
                    ce, kl, _, _, _ = _elite_terms(actor, parent, torch, elite, int(elite_order[j]))
                    winner_terms.append(ce)
                    distill_terms.append(kl)

            for row in preservation:
                hinge, _ = _preservation_margin_loss(
                    actor, torch, row, a.preservation_margin
                )
                preservation_terms.append(hinge)

            components = []
            if rescue_terms:
                components.append(a.rescue_coef * torch.stack(rescue_terms).mean())
            if new_win_terms:
                components.append(a.new_win_coef * torch.stack(new_win_terms).mean())
            if new_win_margin_terms:
                components.append(
                    a.new_win_margin_coef
                    * torch.stack(new_win_margin_terms).mean()
                )
            if preservation_terms:
                components.append(
                    a.preservation_coef * torch.stack(preservation_terms).mean()
                )
            if winner_terms:
                components.append(a.winner_coef * torch.stack(winner_terms).mean())
            if distill_terms:
                components.append(a.distill_coef * torch.stack(distill_terms).mean())

            l2 = None
            for name, param in actor.named_parameters():
                term = ((param - parent_params[name]) ** 2).mean()
                l2 = term if l2 is None else l2 + term
            if l2 is not None and a.param_anchor_coef > 0:
                components.append(a.param_anchor_coef * l2)
            if not components:
                continue

            loss = torch.stack(components).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()), 0.5)
            opt.step()
            losses.append(float(loss.detach()))

        actor.eval()
        rescue_eval = _eval_rescue(actor, torch, rescues)
        new_win_eval = _eval_new_win(actor, torch, new_wins)
        new_win_margin_eval = _eval_new_win_margin(
            actor, torch, new_wins, a.new_win_margin
        )
        preservation_eval = _eval_preservation(actor, torch, preservation)
        retention_eval_stats = _eval_retention(actor, parent, torch, elite, retention_eval)
        raw_retention_ok = _retention_ok(
            retention_eval_stats,
            before_retention,
            min_parent_agreement=a.min_parent_agreement,
            max_parent_kl=a.max_parent_kl,
            max_winner_drop=a.max_winner_drop,
        )
        new_win_improved = (
            new_win_eval["loss"] < before_new_win["loss"]
            or new_win_margin_eval["loss"] < before_new_win_margin["loss"]
        )
        new_win_top1_flipped = (
            int(new_win_eval["top1_examples"]) > int(before_new_win["top1_examples"])
        )
        learned_enough = (
            int(new_win_eval["fully_learned_seeds"])
            >= a.min_fully_learned_seeds
        )
        old_rescue_ok = rescue_eval["loss"] <= before_rescue["loss"] + 0.01
        improved = new_win_improved and learned_enough and old_rescue_ok
        row = {
            "epoch": ep + 1,
            "train_loss": float(np.mean(losses)) if losses else None,
            "rescue_loss": rescue_eval["loss"],
            "rescue_top1": rescue_eval["top1"],
            "new_win_loss": new_win_eval["loss"],
            "new_win_top1": new_win_eval["top1"],
            "new_win_top1_examples": new_win_eval["top1_examples"],
            "new_win_top1_flipped": bool(new_win_top1_flipped),
            "new_win_margin_loss": new_win_margin_eval["loss"],
            "new_win_margin_pass": new_win_margin_eval["margin_pass"],
            "new_win_mean_teacher_minus_g7": new_win_margin_eval["mean_teacher_minus_g7"],
            "fully_learned_seeds": new_win_eval["fully_learned_seeds"],
            "fully_learned_seed_ids": new_win_eval["fully_learned_seed_ids"],
            "preservation_loss": preservation_eval["loss"],
            "preservation_top1": preservation_eval["top1"],
            "winner_top1": retention_eval_stats["winner_top1"],
            "parent_top1_agreement": retention_eval_stats["parent_top1_agreement"],
            "parent_high_conf_top1_agreement": retention_eval_stats["parent_high_conf_top1_agreement"],
            "parent_high_conf_decisions": retention_eval_stats["parent_high_conf_decisions"],
            "parent_kl": retention_eval_stats["parent_kl"],
            "raw_retention_pass": bool(raw_retention_ok),
            "new_win_improved": bool(new_win_improved),
            "new_win_seed_coverage_pass": bool(learned_enough),
            "old_rescue_guard_pass": bool(old_rescue_ok),
            "rescue_improved": bool(improved),
        }
        history.append(row)
        print("V27_TAIL2_NEW_WIN_EPOCH", json.dumps(row, sort_keys=True), flush=True)

        # Important v2.2 rule: do NOT reject a useful Rescue checkpoint here
        # merely because the raw update changed too many G7 decisions.
        # The interpolation stage below exists specifically to shrink that update
        # back toward G7 before applying the hard anti-forgetting gate.
        if improved:
            score = (
                int(new_win_eval["fully_learned_seeds"]),
                float(new_win_eval["top1"]),
                -float(new_win_eval["loss"]),
                float(rescue_eval["top1"]),
                -float(rescue_eval["loss"]),
                -float(retention_eval_stats["parent_kl"]),
            )
            state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
            trained_candidates.append((score, state, row))

    if not trained_candidates:
        raise RuntimeError("no Tail2 checkpoint learned a complete New-Win seed while preserving guards")

    trained_candidates.sort(key=lambda x: x[0], reverse=True)

    blend_trials = []
    best = None
    alphas = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
    for candidate_rank, (_, trained_state, source_row) in enumerate(trained_candidates, 1):
        for alpha in alphas:
            blended = _blend_state(torch, parent_state, trained_state, alpha)
            actor.load_state_dict(blended)
            actor.eval()
            rescue_eval = _eval_rescue(actor, torch, rescues)
            new_win_eval = _eval_new_win(actor, torch, new_wins)
            new_win_margin_eval = _eval_new_win_margin(
                actor, torch, new_wins, a.new_win_margin
            )
            preservation_eval = _eval_preservation(actor, torch, preservation)
            retention_eval_stats = _eval_retention(actor, parent, torch, elite, retention_eval)
            retention_ok = _retention_ok(
                retention_eval_stats,
                before_retention,
                min_parent_agreement=a.min_parent_agreement,
                max_parent_kl=a.max_parent_kl,
                max_winner_drop=a.max_winner_drop,
            )
            new_win_improved = new_win_eval["loss"] < before_new_win["loss"]
            learned_enough = (
                int(new_win_eval["fully_learned_seeds"])
                >= a.min_fully_learned_seeds
            )
            old_rescue_ok = rescue_eval["loss"] <= before_rescue["loss"] + 0.01
            rescue_improved = new_win_improved and learned_enough and old_rescue_ok
            preservation_ok = preservation_eval["top1"] >= 1.0
            ok = rescue_improved and retention_ok and preservation_ok
            trial = {
                "candidate_rank": candidate_rank,
                "source_epoch": source_row["epoch"],
                "alpha": alpha,
                "rescue": rescue_eval,
                "new_win": new_win_eval,
                "new_win_margin": new_win_margin_eval,
                "retention": retention_eval_stats,
                "preservation": preservation_eval,
                "rescue_improved": bool(rescue_improved),
                "new_win_improved": bool(new_win_improved),
                "new_win_seed_coverage_pass": bool(learned_enough),
                "old_rescue_guard_pass": bool(old_rescue_ok),
                "preservation_pass": bool(preservation_ok),
                "retention_pass": bool(retention_ok),
                "pass": bool(ok),
            }
            blend_trials.append(trial)
            print("V27_TAIL2_BLEND_TRIAL", json.dumps(trial, sort_keys=True), flush=True)
            if ok:
                score = (
                    int(new_win_eval["fully_learned_seeds"]),
                    float(new_win_eval["top1"]),
                    -float(new_win_eval["loss"]),
                    float(rescue_eval["top1"]),
                    -float(rescue_eval["loss"]),
                    float(retention_eval_stats["parent_top1_agreement"]),
                    -float(retention_eval_stats["parent_kl"]),
                    -alpha,
                )
                if best is None or score > best[0]:
                    best = (
                        score,
                        {k: v.detach().clone() for k, v in blended.items()},
                        trial,
                    )

    if best is None:
        raise RuntimeError(
            "no Tail2 interpolation learned New-Win behavior while passing anti-forgetting guards"
        )

    selected_state = best[1]
    selected = best[2]
    actor.load_state_dict(selected_state)
    actor.eval()
    after_rescue = _eval_rescue(actor, torch, rescues)
    after_new_win = _eval_new_win(actor, torch, new_wins)
    after_new_win_margin = _eval_new_win_margin(
        actor, torch, new_wins, a.new_win_margin
    )
    after_preservation = _eval_preservation(actor, torch, preservation)
    after_retention = _eval_retention(actor, parent, torch, elite, retention_eval)

    if after_new_win["loss"] >= before_new_win["loss"]:
        raise RuntimeError("selected checkpoint did not improve New-Win Teacher loss")
    if int(after_new_win["fully_learned_seeds"]) < a.min_fully_learned_seeds:
        raise RuntimeError(
            "selected checkpoint did not fully learn a complete New-Win rescue seed"
        )
    if after_rescue["loss"] > before_rescue["loss"] + 0.01:
        raise RuntimeError("selected checkpoint regressed frozen Rescue loss beyond tolerance")
    if after_preservation["top1"] < 1.0:
        raise RuntimeError("selected checkpoint lost a proven parent-only win anchor")
    if not _retention_ok(
        after_retention,
        before_retention,
        min_parent_agreement=a.min_parent_agreement,
        max_parent_kl=a.max_parent_kl,
        max_winner_drop=a.max_winner_drop,
    ):
        raise RuntimeError("selected checkpoint failed final anti-forgetting guard")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), a.output)
    report = {
        "schema_version": "sts1-build-rescue-bc-v27-tail2-new-win",
        "verified_rescue_examples": len(rescues),
        "new_win_teacher_examples": len(new_wins),
        "new_win_teacher_seeds": len({int(r["seed"]) for r in new_wins}),
        "preservation_examples": len(preservation),
        "rescue_by_kind": {
            k: sum(str(r.get("kind")) == k for r in rescues)
            for k in sorted({str(r.get("kind")) for r in rescues})
        },
        "elite_decisions_total": int(len(elite["action"])),
        "elite_train_decisions": int(len(elite_train)),
        "retention_eval_decisions": int(len(retention_eval)),
        "new_win_margin": a.new_win_margin,
        "trainable_parameter_tensors": len(trainable),
        "trainable_tail_linear_layers": len(tail_layers),
        "preservation_margin": a.preservation_margin,
        "preservation_coef": a.preservation_coef,
        "new_win_margin": a.new_win_margin,
        "new_win_margin_coef": a.new_win_margin_coef,
        "min_fully_learned_seeds": a.min_fully_learned_seeds,
        "guards": {
            "min_parent_high_conf_agreement": a.min_parent_agreement,
            "parent_high_conf_prob_gap": PARENT_HIGH_CONFIDENCE_PROB_GAP,
            "min_high_conf_decisions": MIN_HIGH_CONFIDENCE_DECISIONS,
            "max_parent_kl": a.max_parent_kl,
            "max_winner_drop": a.max_winner_drop,
        },
        "before": {
            "rescue": before_rescue,
            "new_win": before_new_win,
            "new_win_margin": before_new_win_margin,
            "preservation": before_preservation,
            "retention": before_retention,
        },
        "after": {
            "rescue": after_rescue,
            "new_win": after_new_win,
            "new_win_margin": after_new_win_margin,
            "preservation": after_preservation,
            "retention": after_retention,
        },
        "selected_alpha": selected["alpha"],
        "blend_trials": blend_trials,
        "history": history,
        "base_weight": str(a.base_weight),
        "candidate_output": str(a.output),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "V27_TAIL2_NEW_WIN_TRAIN_PASS",
        json.dumps(
            {
                "verified_rescue_examples": len(rescues),
                "new_win_teacher_examples": len(new_wins),
                "new_win_teacher_seeds": len({int(r["seed"]) for r in new_wins}),
                "new_win_loss_before": before_new_win["loss"],
                "new_win_loss_after": after_new_win["loss"],
                "new_win_top1_after": after_new_win["top1"],
                "new_win_top1_examples_after": after_new_win["top1_examples"],
                "new_win_top1_flipped": int(after_new_win["top1_examples"]) > int(before_new_win["top1_examples"]),
                "new_win_margin_loss_before": before_new_win_margin["loss"],
                "new_win_margin_loss_after": after_new_win_margin["loss"],
                "new_win_margin_pass_after": after_new_win_margin["margin_pass"],
                "fully_learned_seeds_after": after_new_win["fully_learned_seeds"],
                "fully_learned_seed_ids_after": after_new_win["fully_learned_seed_ids"],
                "preservation_examples": len(preservation),
                "preservation_top1_after": after_preservation["top1"],
                "rescue_loss_before": before_rescue["loss"],
                "rescue_loss_after": after_rescue["loss"],
                "rescue_top1_after": after_rescue["top1"],
                "winner_top1_after": after_retention["winner_top1"],
                "parent_top1_agreement": after_retention["parent_top1_agreement"],
                "parent_high_conf_top1_agreement": after_retention["parent_high_conf_top1_agreement"],
                "parent_high_conf_decisions": after_retention["parent_high_conf_decisions"],
                "parent_kl": after_retention["parent_kl"],
                "selected_alpha": selected["alpha"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
