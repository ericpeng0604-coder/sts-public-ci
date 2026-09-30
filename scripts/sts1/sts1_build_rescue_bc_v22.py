#!/usr/bin/env python3
"""Anti-forgetting Build Rescue BC for STS1 ArmG.

v2.2 learns verified Build Rescue decisions while explicitly preserving the
frozen G7 parent's behavior on proven-win replay. Combat remains pure MCTS.

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
    logp = torch.log_softmax(logits, 0)
    action = int(elite["action"][i])
    hard_ce = -logp[action]
    parent_kl = (parent_probs * (torch.log_softmax(parent_logits, 0) - logp)).sum()
    return hard_ce, parent_kl, logits, parent_top1


def _eval_rescue(actor, torch, rows):
    losses: list[float] = []
    top1 = 0
    with torch.no_grad():
        for row in rows:
            loss, logits = _teacher_loss(actor, torch, row)
            losses.append(float(loss))
            top1 += int(int(torch.argmax(logits)) == int(row["teacher_best_index"]))
    return {"loss": float(np.mean(losses)), "top1": top1 / len(rows)}


def _eval_retention(actor, parent, torch, elite, indices):
    if len(indices) == 0:
        raise RuntimeError("retention split is empty")
    winner_correct = 0
    parent_agree = 0
    kls: list[float] = []
    with torch.no_grad():
        for raw in indices:
            i = int(raw)
            _, kl, logits, parent_top1 = _elite_terms(actor, parent, torch, elite, i)
            pred = int(torch.argmax(logits))
            winner_correct += int(pred == int(elite["action"][i]))
            parent_agree += int(pred == parent_top1)
            kls.append(float(kl))
    return {
        "winner_top1": winner_correct / len(indices),
        "parent_top1_agreement": parent_agree / len(indices),
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
        float(stats["parent_top1_agreement"]) >= min_parent_agreement
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--elite-train-max", type=int, default=8192)
    p.add_argument("--retention-eval-max", type=int, default=2048)
    p.add_argument("--rescue-repeats", type=int, default=4)
    p.add_argument("--winner-coef", type=float, default=0.80)
    p.add_argument("--distill-coef", type=float, default=1.20)
    p.add_argument("--rescue-coef", type=float, default=0.60)
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
    if not 0 < a.winner_coef <= 2 or not 0 < a.distill_coef <= 4 or not 0 < a.rescue_coef <= 2:
        raise RuntimeError("loss coefficient outside safe bound")
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
    elite = _load_elite(a.elite_replay)
    if len(rescues) < 10:
        raise RuntimeError(f"v2.2 expects the expanded verified pool; got {len(rescues)}")

    expected_obs = int(m.OBS_DIM)
    first_linear = next(layer for layer in actor.net if hasattr(layer, "in_features"))
    expected_desc = int(first_linear.in_features) - expected_obs
    for row in rescues:
        if len(row["obs"]) != expected_obs or any(len(d) != expected_desc for d in row["descs"]):
            raise RuntimeError("rescue feature dimension mismatch")

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
    before_retention = _eval_retention(actor, parent, torch, elite, retention_eval)

    parent_params = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    opt = torch.optim.Adam(actor.parameters(), lr=a.lr)
    history: list[dict[str, Any]] = []
    accepted: list[tuple[tuple[float, float, float], dict[str, Any], dict[str, Any]]] = []

    for ep in range(a.epochs):
        actor.train()
        rescue_order = [i for i in range(len(rescues)) for _ in range(a.rescue_repeats)]
        rng.shuffle(rescue_order)
        elite_order = elite_train.copy()
        rng.shuffle(elite_order)
        steps = max(len(rescue_order), len(elite_order))
        losses: list[float] = []

        for start in range(0, steps, 32):
            rescue_terms = []
            winner_terms = []
            distill_terms = []
            end = min(start + 32, steps)
            for j in range(start, end):
                if j < len(rescue_order):
                    row = rescues[int(rescue_order[j])]
                    ce, _ = _teacher_loss(actor, torch, row)
                    priority = min(2.0, max(1.0, float(row.get("priority", 1.0))))
                    rescue_terms.append(priority * ce)
                if j < len(elite_order):
                    ce, kl, _, _ = _elite_terms(actor, parent, torch, elite, int(elite_order[j]))
                    winner_terms.append(ce)
                    distill_terms.append(kl)

            components = []
            if rescue_terms:
                components.append(a.rescue_coef * torch.stack(rescue_terms).mean())
            if winner_terms:
                components.append(a.winner_coef * torch.stack(winner_terms).mean())
            if distill_terms:
                components.append(a.distill_coef * torch.stack(distill_terms).mean())

            l2 = None
            for name, param in actor.state_dict().items():
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
        retention_eval_stats = _eval_retention(actor, parent, torch, elite, retention_eval)
        ok = _retention_ok(
            retention_eval_stats,
            before_retention,
            min_parent_agreement=a.min_parent_agreement,
            max_parent_kl=a.max_parent_kl,
            max_winner_drop=a.max_winner_drop,
        )
        improved = rescue_eval["loss"] < before_rescue["loss"]
        row = {
            "epoch": ep + 1,
            "train_loss": float(np.mean(losses)) if losses else None,
            "rescue_loss": rescue_eval["loss"],
            "rescue_top1": rescue_eval["top1"],
            "winner_top1": retention_eval_stats["winner_top1"],
            "parent_top1_agreement": retention_eval_stats["parent_top1_agreement"],
            "parent_kl": retention_eval_stats["parent_kl"],
            "retention_pass": bool(ok),
            "rescue_improved": bool(improved),
        }
        history.append(row)
        print("V22_ANTI_FORGET_EPOCH", json.dumps(row, sort_keys=True), flush=True)

        if ok and improved:
            score = (
                float(rescue_eval["top1"]),
                -float(rescue_eval["loss"]),
                -float(retention_eval_stats["parent_kl"]),
            )
            state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
            accepted.append((score, state, row))

    if not accepted:
        raise RuntimeError("no training checkpoint improved Rescue while passing anti-forgetting guards")

    accepted.sort(key=lambda x: x[0], reverse=True)
    trained_state = accepted[0][1]

    blend_trials = []
    best = None
    for alpha in (1.0, 0.75, 0.5, 0.25, 0.125):
        blended = _blend_state(torch, parent_state, trained_state, alpha)
        actor.load_state_dict(blended)
        actor.eval()
        rescue_eval = _eval_rescue(actor, torch, rescues)
        retention_eval_stats = _eval_retention(actor, parent, torch, elite, retention_eval)
        ok = (
            rescue_eval["loss"] < before_rescue["loss"]
            and _retention_ok(
                retention_eval_stats,
                before_retention,
                min_parent_agreement=a.min_parent_agreement,
                max_parent_kl=a.max_parent_kl,
                max_winner_drop=a.max_winner_drop,
            )
        )
        trial = {
            "alpha": alpha,
            "rescue": rescue_eval,
            "retention": retention_eval_stats,
            "pass": bool(ok),
        }
        blend_trials.append(trial)
        print("V22_BLEND_TRIAL", json.dumps(trial, sort_keys=True), flush=True)
        if ok:
            score = (
                float(rescue_eval["top1"]),
                -float(rescue_eval["loss"]),
                -float(retention_eval_stats["parent_kl"]),
                -alpha,
            )
            if best is None or score > best[0]:
                best = (score, {k: v.detach().clone() for k, v in blended.items()}, trial)

    if best is None:
        raise RuntimeError("no interpolated checkpoint passed anti-forgetting guards")

    selected_state = best[1]
    selected = best[2]
    actor.load_state_dict(selected_state)
    actor.eval()
    after_rescue = _eval_rescue(actor, torch, rescues)
    after_retention = _eval_retention(actor, parent, torch, elite, retention_eval)

    if after_rescue["loss"] >= before_rescue["loss"]:
        raise RuntimeError("selected checkpoint did not improve verified Rescue loss")
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
        "schema_version": "sts1-build-rescue-bc-v22-antiforget",
        "verified_rescue_examples": len(rescues),
        "rescue_by_kind": {
            k: sum(str(r.get("kind")) == k for r in rescues)
            for k in sorted({str(r.get("kind")) for r in rescues})
        },
        "elite_decisions_total": int(len(elite["action"])),
        "elite_train_decisions": int(len(elite_train)),
        "retention_eval_decisions": int(len(retention_eval)),
        "guards": {
            "min_parent_agreement": a.min_parent_agreement,
            "max_parent_kl": a.max_parent_kl,
            "max_winner_drop": a.max_winner_drop,
        },
        "before": {"rescue": before_rescue, "retention": before_retention},
        "after": {"rescue": after_rescue, "retention": after_retention},
        "selected_alpha": selected["alpha"],
        "blend_trials": blend_trials,
        "history": history,
        "base_weight": str(a.base_weight),
        "candidate_output": str(a.output),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "V22_ANTI_FORGET_PASS",
        json.dumps(
            {
                "verified_rescue_examples": len(rescues),
                "rescue_loss_before": before_rescue["loss"],
                "rescue_loss_after": after_rescue["loss"],
                "rescue_top1_after": after_rescue["top1"],
                "winner_top1_after": after_retention["winner_top1"],
                "parent_top1_agreement": after_retention["parent_top1_agreement"],
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
