#!/usr/bin/env python3
"""Projected minimal last-head update for one verified STS1 New-Win decision.

The frozen G7 representation is kept intact.  We compute a tiny closed-form
update to the final scalar scoring head that makes the verified teacher action
outrank its strongest competitor.  The update direction is regularized against
G7 winner-replay decision-boundary directions and explicit regression anchors,
so the target can flip with minimal collateral policy drift.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

import sts1_build_rescue_bc_v28 as base


def _features(actor, torch, obs, descs):
    x = torch.cat(
        [
            torch.tensor(obs, dtype=torch.float32).repeat(len(descs), 1),
            torch.tensor(descs, dtype=torch.float32),
        ],
        1,
    )
    layers = list(actor.net)
    final_index = None
    for i in range(len(layers) - 1, -1, -1):
        layer = layers[i]
        if hasattr(layer, "out_features") and int(layer.out_features) == 1:
            final_index = i
            break
    if final_index is None:
        raise RuntimeError("unable to locate final scalar scoring head")
    h = x
    with torch.no_grad():
        for layer in layers[:final_index]:
            h = layer(h)
    return h, layers[final_index]


def _candidate_hidden(actor, torch, elite, i):
    lo = int(elite["offsets"][i])
    hi = int(elite["offsets"][i + 1])
    if hi <= lo:
        raise RuntimeError(f"empty elite candidate span at {i}")
    obs = elite["obs"][i]
    descs = elite["desc"][lo:hi]
    return _features(actor, torch, obs, descs)


def _build_covariance(
    actor,
    parent,
    torch,
    elite,
    indices,
    preservation,
    *,
    high_conf_gap: float,
):
    _, final = _candidate_hidden(actor, torch, elite, int(indices[0]))
    dim = int(final.in_features)
    cov = np.zeros((dim, dim), dtype=np.float64)
    rows = 0
    high_conf = 0

    with torch.no_grad():
        for raw in indices:
            i = int(raw)
            h, _ = _candidate_hidden(actor, torch, elite, i)
            logits = parent.net(
                torch.cat(
                    [
                        torch.tensor(elite["obs"][i], dtype=torch.float32).repeat(len(h), 1),
                        torch.tensor(
                            elite["desc"][
                                int(elite["offsets"][i]):int(elite["offsets"][i + 1])
                            ],
                            dtype=torch.float32,
                        ),
                    ],
                    1,
                )
            ).squeeze(1)
            probs = torch.softmax(logits, 0)
            top = int(torch.argmax(logits))
            if len(logits) < 2:
                continue
            order = torch.argsort(logits, descending=True)
            runner = int(order[1])
            prob_gap = float(probs[top] - probs[runner])
            weight = 4.0 if prob_gap >= high_conf_gap else 1.0
            high_conf += int(prob_gap >= high_conf_gap)
            d = (h[top] - h[runner]).detach().cpu().numpy().astype(np.float64)
            norm = float(np.linalg.norm(d))
            if norm <= 1e-12:
                continue
            d /= norm
            cov += weight * np.outer(d, d)
            rows += 1

        for row in preservation:
            h, _ = _features(actor, torch, row["obs"], row["descs"])
            target = int(row["parent_selected_index"])
            logits = parent.net(
                torch.cat(
                    [
                        torch.tensor(row["obs"], dtype=torch.float32).repeat(len(row["descs"]), 1),
                        torch.tensor(row["descs"], dtype=torch.float32),
                    ],
                    1,
                )
            ).squeeze(1)
            order = torch.argsort(logits, descending=True)
            competitor = next(int(x) for x in order if int(x) != target)
            d = (h[target] - h[competitor]).detach().cpu().numpy().astype(np.float64)
            norm = float(np.linalg.norm(d))
            if norm <= 1e-12:
                continue
            d /= norm
            cov += 25.0 * np.outer(d, d)
            rows += 1

    if rows == 0:
        raise RuntimeError("projected covariance has no usable protected directions")
    cov /= float(rows)
    return cov, {"protected_directions": rows, "high_conf_directions": high_conf}


def _solve_update(
    actor,
    torch,
    teacher_row,
    cov,
    *,
    target_margin: float,
    protection_scale: float,
    ridge: float,
):
    h, final = _features(actor, torch, teacher_row["obs"], teacher_row["descs"])
    w0 = final.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    b0 = float(final.bias.detach().cpu().numpy().reshape(-1)[0]) if final.bias is not None else 0.0
    h_np = h.detach().cpu().numpy().astype(np.float64)
    teacher = int(teacher_row["teacher_best_index"])

    delta = np.zeros_like(w0)
    eye = np.eye(len(w0), dtype=np.float64)
    metric = ridge * eye + protection_scale * cov
    steps = []

    for _ in range(12):
        logits = h_np @ (w0 + delta) + b0
        competitors = [i for i in range(len(logits)) if i != teacher]
        competitor = max(competitors, key=lambda i: float(logits[i]))
        raw_margin = float(logits[teacher] - logits[competitor])
        if raw_margin >= target_margin:
            break
        d = h_np[teacher] - h_np[competitor]
        v = np.linalg.solve(metric, d)
        denom = float(d @ v)
        if not np.isfinite(denom) or denom <= 1e-12:
            raise RuntimeError("projected update direction is singular")
        shortfall = float(target_margin - raw_margin)
        step = (shortfall / denom) * v
        delta += step
        steps.append(
            {
                "competitor": int(competitor),
                "margin_before": raw_margin,
                "shortfall": shortfall,
                "step_norm": float(np.linalg.norm(step)),
            }
        )

    logits = h_np @ (w0 + delta) + b0
    order = np.argsort(-logits)
    top = int(order[0])
    runner = int(order[1]) if len(order) > 1 else top
    return delta, {
        "teacher_index": teacher,
        "selected_index": top,
        "teacher_margin": float(logits[teacher] - logits[runner if top == teacher else top]),
        "delta_norm": float(np.linalg.norm(delta)),
        "solver_steps": steps,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--armg-root", type=Path, required=True)
    p.add_argument("--base-weight", type=Path, required=True)
    p.add_argument("--elite-replay", type=Path, required=True)
    p.add_argument("--new-win-replay", type=Path, required=True)
    p.add_argument("--preservation-replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--retention-eval-max", type=int, default=2048)
    p.add_argument("--protection-train-max", type=int, default=8192)
    p.add_argument("--target-margin", type=float, default=0.01)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--min-parent-agreement", type=float, default=0.995)
    p.add_argument("--max-parent-kl", type=float, default=0.002)
    p.add_argument("--max-winner-drop", type=float, default=0.01)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    if not 64 <= a.retention_eval_max <= 8192:
        raise RuntimeError("retention-eval-max outside safe bound")
    if not 128 <= a.protection_train_max <= 32768:
        raise RuntimeError("protection-train-max outside safe bound")
    if not 0 < a.target_margin <= 0.10:
        raise RuntimeError("target-margin outside safe bound")
    if not 1e-8 <= a.ridge <= 1.0:
        raise RuntimeError("ridge outside safe bound")

    os.environ["STS_BOT_DIR"] = str(a.armg_root)
    sys.path.insert(0, str(a.armg_root))
    torch = importlib.import_module("torch")
    torch.set_num_threads(a.threads)
    m = importlib.import_module("armG_train")
    m.card_idx = lambda n: m._vocab.get(n, m.VOCAB_CAP - 1)

    parent_state = torch.load(a.base_weight, weights_only=True, map_location="cpu")
    actor = m.Scorer((128, 128))
    actor.load_state_dict(parent_state)
    parent = m.Scorer((128, 128))
    parent.load_state_dict(parent_state)
    parent.eval()
    actor.eval()

    new_wins = base._load_rescues(a.new_win_replay)
    if len(new_wins) != 1:
        raise RuntimeError(f"v2.9 requires exactly one focused Teacher; got {len(new_wins)}")
    preservation = base._load_preservation(a.preservation_replay)
    elite = base._load_elite(a.elite_replay)

    rng = np.random.default_rng(20261001)
    perm = rng.permutation(len(elite["action"]))
    eval_count = min(a.retention_eval_max, max(64, len(perm) // 5))
    if len(perm) - eval_count < 128:
        eval_count = max(1, len(perm) // 4)
    retention_eval = np.asarray(perm[:eval_count], dtype=np.int64)
    protection_pool = np.asarray(
        perm[eval_count:eval_count + min(a.protection_train_max, len(perm) - eval_count)],
        dtype=np.int64,
    )
    if len(protection_pool) < 128:
        raise RuntimeError("not enough protected winner replay decisions")

    before_new_win = base._eval_new_win(actor, torch, new_wins)
    before_preservation = base._eval_preservation(actor, torch, preservation)
    before_retention = base._eval_retention(actor, parent, torch, elite, retention_eval)
    if before_new_win["top1"] != 0.0:
        raise RuntimeError("focused Teacher is already G7 top1; no projected update needed")
    if before_preservation["top1"] < 1.0:
        raise RuntimeError("G7 does not satisfy frozen preservation anchors")

    cov, cov_stats = _build_covariance(
        actor,
        parent,
        torch,
        elite,
        protection_pool,
        preservation,
        high_conf_gap=base.PARENT_HIGH_CONFIDENCE_PROB_GAP,
    )

    scales = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    trials = []
    best = None

    for scale in scales:
        actor.load_state_dict(parent_state)
        actor.eval()
        delta, solve = _solve_update(
            actor,
            torch,
            new_wins[0],
            cov,
            target_margin=a.target_margin,
            protection_scale=scale,
            ridge=a.ridge,
        )
        _, final = _features(actor, torch, new_wins[0]["obs"], new_wins[0]["descs"])
        with torch.no_grad():
            final.weight.add_(
                torch.tensor(delta, dtype=final.weight.dtype).reshape_as(final.weight)
            )
        actor.eval()

        new_eval = base._eval_new_win(actor, torch, new_wins)
        pres_eval = base._eval_preservation(actor, torch, preservation)
        retention = base._eval_retention(actor, parent, torch, elite, retention_eval)
        retention_ok = base._retention_ok(
            retention,
            before_retention,
            min_parent_agreement=a.min_parent_agreement,
            max_parent_kl=a.max_parent_kl,
            max_winner_drop=a.max_winner_drop,
        )
        target_ok = int(new_eval["fully_learned_seeds"]) == 1
        pres_ok = pres_eval["top1"] >= 1.0
        ok = target_ok and pres_ok and retention_ok
        trial = {
            "protection_scale": scale,
            "solve": solve,
            "new_win": new_eval,
            "preservation": pres_eval,
            "retention": retention,
            "target_pass": bool(target_ok),
            "preservation_pass": bool(pres_ok),
            "retention_pass": bool(retention_ok),
            "pass": bool(ok),
        }
        trials.append(trial)
        print("V29_PROJECTED_TRIAL", json.dumps(trial, sort_keys=True), flush=True)
        if ok:
            score = (
                float(retention["parent_high_conf_top1_agreement"]),
                -float(retention["parent_kl"]),
                float(retention["winner_top1"]),
                -float(solve["delta_norm"]),
                scale,
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    {k: v.detach().clone() for k, v in actor.state_dict().items()},
                    trial,
                )

    if best is None:
        raise RuntimeError(
            "no projected last-head update flipped the focused Teacher while passing G7 guards"
        )

    actor.load_state_dict(best[1])
    actor.eval()
    after_new_win = base._eval_new_win(actor, torch, new_wins)
    after_preservation = base._eval_preservation(actor, torch, preservation)
    after_retention = base._eval_retention(actor, parent, torch, elite, retention_eval)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), a.output)
    report = {
        "schema_version": "sts1-build-rescue-v29-projected-one",
        "teacher_seed": int(new_wins[0]["seed"]),
        "teacher_kind": str(new_wins[0]["kind"]),
        "teacher_floor": int(new_wins[0]["floor"]),
        "covariance": cov_stats,
        "target_margin": a.target_margin,
        "ridge": a.ridge,
        "before": {
            "new_win": before_new_win,
            "preservation": before_preservation,
            "retention": before_retention,
        },
        "after": {
            "new_win": after_new_win,
            "preservation": after_preservation,
            "retention": after_retention,
        },
        "selected_trial": best[2],
        "trials": trials,
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "V29_PROJECTED_ONE_PASS",
        json.dumps(
            {
                "teacher_seed": int(new_wins[0]["seed"]),
                "teacher_kind": str(new_wins[0]["kind"]),
                "teacher_floor": int(new_wins[0]["floor"]),
                "new_win_top1_after": after_new_win["top1"],
                "fully_learned_seeds_after": after_new_win["fully_learned_seeds"],
                "preservation_top1_after": after_preservation["top1"],
                "parent_high_conf_top1_agreement": after_retention["parent_high_conf_top1_agreement"],
                "parent_top1_agreement": after_retention["parent_top1_agreement"],
                "parent_kl": after_retention["parent_kl"],
                "winner_top1_after": after_retention["winner_top1"],
                "protection_scale": best[2]["protection_scale"],
                "delta_norm": best[2]["solve"]["delta_norm"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
