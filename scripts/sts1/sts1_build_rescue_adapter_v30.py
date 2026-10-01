#!/usr/bin/env python3
"""Train a frozen-G7 residual score adapter for STS1 ArmG.

The base ArmG checkpoint is never updated.  A small 780->H->1 residual network
is trained only for decision kinds present in verified New-Win Teacher data.
The adapter must learn at least one complete rescue seed while preserving
winner replay, explicit regression anchors, and tight parent KL.
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

import sts1_build_rescue_bc_v28 as base


SCHEMA = "sts1-armg-residual-adapter-v1"


def _x(torch, row: dict[str, Any]):
    obs = torch.tensor(row["obs"], dtype=torch.float32)
    desc = torch.tensor(row["descs"], dtype=torch.float32)
    return torch.cat([obs.repeat(len(desc), 1), desc], 1)


def _teacher_logits(parent, adapter, torch, row):
    x = _x(torch, row)
    with torch.no_grad():
        parent_logits = parent.net(x).squeeze(1)
    residual = adapter(x).squeeze(1)
    return parent_logits + residual, residual


def _teacher_eval(parent, adapter, torch, rows):
    correct = 0
    losses: list[float] = []
    margins: list[float] = []
    residual_abs: list[float] = []
    by_seed: dict[int, list[bool]] = {}
    with torch.no_grad():
        for row in rows:
            logits, residual = _teacher_logits(parent, adapter, torch, row)
            target = int(row["teacher_best_index"])
            pred = int(torch.argmax(logits))
            hit = pred == target
            by_seed.setdefault(int(row["seed"]), []).append(hit)
            correct += int(hit)
            target_logit = logits[target]
            mask = torch.ones(len(logits), dtype=torch.bool)
            mask[target] = False
            best_other = torch.max(logits[mask])
            margins.append(float(target_logit - best_other))
            losses.append(float(torch.logsumexp(logits, 0) - target_logit))
            residual_abs.extend(abs(float(x)) for x in residual.tolist())
    learned = sorted(seed for seed, hits in by_seed.items() if all(hits))
    return {
        "examples": len(rows),
        "teacher_seeds_total": len(by_seed),
        "top1_examples": correct,
        "top1": correct / len(rows),
        "fully_learned_seeds": len(learned),
        "fully_learned_seed_ids": learned,
        "mean_teacher_margin": float(np.mean(margins)),
        "min_teacher_margin": float(np.min(margins)),
        "loss": float(np.mean(losses)),
        "mean_abs_residual": float(np.mean(residual_abs)),
    }


def _preservation_eval(parent, adapter, torch, rows, enabled_kinds):
    if not rows:
        return {"examples": 0, "top1": 1.0}
    correct = 0
    margins = []
    with torch.no_grad():
        for row in rows:
            x = _x(torch, row)
            logits = parent.net(x).squeeze(1)
            if str(row["kind"]) in enabled_kinds:
                logits = logits + adapter(x).squeeze(1)
            target = int(row["parent_selected_index"])
            pred = int(torch.argmax(logits))
            correct += int(pred == target)
            mask = torch.ones(len(logits), dtype=torch.bool)
            mask[target] = False
            margins.append(float(logits[target] - torch.max(logits[mask])))
    return {
        "examples": len(rows),
        "top1": correct / len(rows),
        "min_margin": float(np.min(margins)),
        "mean_margin": float(np.mean(margins)),
    }


def _elite_logits(parent, adapter, torch, elite, i, apply_adapter: bool):
    lo = int(elite["offsets"][i])
    hi = int(elite["offsets"][i + 1])
    if hi <= lo:
        raise RuntimeError(f"empty elite candidate span at {i}")
    obs = torch.tensor(elite["obs"][i], dtype=torch.float32)
    desc = torch.tensor(elite["desc"][lo:hi], dtype=torch.float32)
    x = torch.cat([obs.repeat(len(desc), 1), desc], 1)
    with torch.no_grad():
        parent_logits = parent.net(x).squeeze(1)
    logits = parent_logits
    residual = torch.zeros_like(parent_logits)
    if apply_adapter:
        residual = adapter(x).squeeze(1)
        logits = parent_logits + residual
    return parent_logits, logits, residual


def _retention_eval(parent, adapter, torch, elite, indices):
    winner_correct = 0
    parent_agree = 0
    high_conf_total = 0
    high_conf_agree = 0
    kls: list[float] = []
    residual_sq: list[float] = []
    for raw in indices:
        i = int(raw)
        with torch.no_grad():
            parent_logits, logits, residual = _elite_logits(
                parent, adapter, torch, elite, i, True
            )
            parent_probs = torch.softmax(parent_logits, 0)
            cand_logp = torch.log_softmax(logits, 0)
            parent_logp = torch.log_softmax(parent_logits, 0)
            kl = (parent_probs * (parent_logp - cand_logp)).sum()
            pred = int(torch.argmax(logits))
            parent_top1 = int(torch.argmax(parent_logits))
            winner_correct += int(pred == int(elite["action"][i]))
            parent_agree += int(pred == parent_top1)
            if len(parent_probs) >= 2:
                top2 = torch.topk(parent_probs, k=2).values
                gap = float(top2[0] - top2[1])
            else:
                gap = 1.0
            if gap >= base.PARENT_HIGH_CONFIDENCE_PROB_GAP:
                high_conf_total += 1
                high_conf_agree += int(pred == parent_top1)
            kls.append(float(kl))
            residual_sq.extend(float(x * x) for x in residual.tolist())
    return {
        "winner_top1": winner_correct / len(indices),
        "parent_top1_agreement": parent_agree / len(indices),
        "parent_high_conf_top1_agreement": (
            high_conf_agree / high_conf_total if high_conf_total else 0.0
        ),
        "parent_high_conf_decisions": int(high_conf_total),
        "parent_kl": float(np.mean(kls)),
        "residual_rms": float(np.sqrt(np.mean(residual_sq))) if residual_sq else 0.0,
        "decisions": int(len(indices)),
    }


def _retention_ok(stats, before, *, min_parent_agreement, max_parent_kl, max_winner_drop):
    return (
        int(stats["parent_high_conf_decisions"]) >= base.MIN_HIGH_CONFIDENCE_DECISIONS
        and float(stats["parent_high_conf_top1_agreement"]) >= min_parent_agreement
        and float(stats["parent_kl"]) <= max_parent_kl
        and float(stats["winner_top1"]) + max_winner_drop >= float(before["winner_top1"])
    )


def _teacher_margin_loss(parent, adapter, torch, row, margin):
    logits, _ = _teacher_logits(parent, adapter, torch, row)
    target = int(row["teacher_best_index"])
    mask = torch.ones(len(logits), dtype=torch.bool)
    mask[target] = False
    best_other = torch.max(logits[mask])
    return torch.relu(best_other - logits[target] + margin)


def _preservation_loss(parent, adapter, torch, row, enabled_kinds, margin):
    if str(row["kind"]) not in enabled_kinds:
        return None
    x = _x(torch, row)
    with torch.no_grad():
        parent_logits = parent.net(x).squeeze(1)
    logits = parent_logits + adapter(x).squeeze(1)
    target = int(row["parent_selected_index"])
    mask = torch.ones(len(logits), dtype=torch.bool)
    mask[target] = False
    return torch.relu(torch.max(logits[mask]) - logits[target] + margin)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--armg-root", type=Path, required=True)
    p.add_argument("--base-weight", type=Path, required=True)
    p.add_argument("--elite-replay", type=Path, required=True)
    p.add_argument("--new-win-replay", type=Path, required=True)
    p.add_argument("--preservation-replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--elite-train-max", type=int, default=8192)
    p.add_argument("--retention-eval-max", type=int, default=2048)
    p.add_argument("--teacher-margin", type=float, default=0.05)
    p.add_argument("--teacher-coef", type=float, default=4.0)
    p.add_argument("--distill-coef", type=float, default=6.0)
    p.add_argument("--zero-residual-coef", type=float, default=0.50)
    p.add_argument("--preservation-coef", type=float, default=3.0)
    p.add_argument("--preservation-margin", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--min-fully-learned-seeds", type=int, default=1)
    p.add_argument("--min-parent-agreement", type=float, default=0.995)
    p.add_argument("--max-parent-kl", type=float, default=0.002)
    p.add_argument("--max-winner-drop", type=float, default=0.01)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    if not 4 <= a.hidden_dim <= 128:
        raise RuntimeError("hidden-dim outside 4..128")
    if not 1 <= a.epochs <= 500:
        raise RuntimeError("epochs outside 1..500")
    if not 1e-5 <= a.lr <= 1e-2:
        raise RuntimeError("lr outside safe bound")
    if not 128 <= a.elite_train_max <= 32768:
        raise RuntimeError("elite-train-max outside safe bound")
    if not 64 <= a.retention_eval_max <= 8192:
        raise RuntimeError("retention-eval-max outside safe bound")
    if not 0 < a.teacher_margin <= 1.0:
        raise RuntimeError("teacher-margin outside safe bound")
    if not 1 <= a.min_fully_learned_seeds <= 20:
        raise RuntimeError("min-fully-learned-seeds outside safe bound")

    os.environ["STS_BOT_DIR"] = str(a.armg_root)
    sys.path.insert(0, str(a.armg_root))
    torch = importlib.import_module("torch")
    torch.set_num_threads(a.threads)
    torch.manual_seed(20261001)
    m = importlib.import_module("armG_train")
    m.card_idx = lambda n: m._vocab.get(n, m.VOCAB_CAP - 1)

    parent_state = torch.load(a.base_weight, weights_only=True, map_location="cpu")
    parent = m.Scorer((128, 128))
    parent.load_state_dict(parent_state)
    parent.eval()
    for param in parent.parameters():
        param.requires_grad_(False)

    rows = base._load_rescues(a.new_win_replay)
    preservation = base._load_preservation(a.preservation_replay)
    elite = base._load_elite(a.elite_replay)
    kinds = sorted({str(r["kind"]) for r in rows})
    if not rows:
        raise RuntimeError("New-Win teacher pool is empty")

    expected_obs = int(m.OBS_DIM)
    first_linear = next(layer for layer in parent.net if hasattr(layer, "in_features"))
    expected_desc = int(first_linear.in_features) - expected_obs
    input_dim = expected_obs + expected_desc
    for row in rows:
        if len(row["obs"]) != expected_obs:
            raise RuntimeError("teacher obs feature dimension mismatch")
        if any(len(d) != expected_desc for d in row["descs"]):
            raise RuntimeError("teacher desc feature dimension mismatch")

    adapter = torch.nn.Sequential(
        torch.nn.Linear(input_dim, a.hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(a.hidden_dim, 1),
    )
    # Start with exactly zero residual so epoch zero is byte-for-byte G7 behavior.
    torch.nn.init.zeros_(adapter[2].weight)
    torch.nn.init.zeros_(adapter[2].bias)

    rng = np.random.default_rng(20261001)
    perm = rng.permutation(len(elite["action"]))
    eval_count = min(a.retention_eval_max, max(64, len(perm) // 5))
    if len(perm) - eval_count < 128:
        eval_count = max(1, len(perm) // 4)
    retention_eval = np.asarray(perm[:eval_count], dtype=np.int64)
    train_pool = np.asarray(perm[eval_count:], dtype=np.int64)
    elite_train = train_pool[: min(a.elite_train_max, len(train_pool))]
    if len(elite_train) < 128:
        raise RuntimeError("not enough disjoint winner replay for adapter training")

    before_new = _teacher_eval(parent, adapter, torch, rows)
    before_pres = _preservation_eval(parent, adapter, torch, preservation, set(kinds))
    before_ret = _retention_eval(parent, adapter, torch, elite, retention_eval)
    if before_pres["top1"] < 1.0:
        raise RuntimeError("zero adapter unexpectedly fails G7 preservation")

    opt = torch.optim.AdamW(adapter.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    history = []
    best = None

    batch = 64
    for epoch in range(1, a.epochs + 1):
        adapter.train()
        order = elite_train.copy()
        rng.shuffle(order)
        train_losses = []
        for start in range(0, len(order), batch):
            ids = order[start:start + batch]
            distill_terms = []
            zero_terms = []
            for raw in ids:
                i = int(raw)
                parent_logits, logits, residual = _elite_logits(
                    parent, adapter, torch, elite, i, True
                )
                with torch.no_grad():
                    parent_probs = torch.softmax(parent_logits, 0)
                    parent_logp = torch.log_softmax(parent_logits, 0)
                cand_logp = torch.log_softmax(logits, 0)
                distill_terms.append(
                    (parent_probs * (parent_logp - cand_logp)).sum()
                )
                zero_terms.append((residual ** 2).mean())

            teacher_terms = [
                _teacher_margin_loss(
                    parent, adapter, torch, row, a.teacher_margin
                )
                for row in rows
            ]
            pres_terms = []
            for row in preservation:
                term = _preservation_loss(
                    parent, adapter, torch, row, set(kinds), a.preservation_margin
                )
                if term is not None:
                    pres_terms.append(term)

            components = [
                a.distill_coef * torch.stack(distill_terms).mean(),
                a.zero_residual_coef * torch.stack(zero_terms).mean(),
                a.teacher_coef * torch.stack(teacher_terms).mean(),
            ]
            if pres_terms:
                components.append(
                    a.preservation_coef * torch.stack(pres_terms).mean()
                )
            loss = torch.stack(components).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach()))

        adapter.eval()
        new_eval = _teacher_eval(parent, adapter, torch, rows)
        pres_eval = _preservation_eval(
            parent, adapter, torch, preservation, set(kinds)
        )
        ret_eval = _retention_eval(parent, adapter, torch, elite, retention_eval)
        retention_ok = _retention_ok(
            ret_eval,
            before_ret,
            min_parent_agreement=a.min_parent_agreement,
            max_parent_kl=a.max_parent_kl,
            max_winner_drop=a.max_winner_drop,
        )
        learned = int(new_eval["fully_learned_seeds"]) >= a.min_fully_learned_seeds
        pres_ok = float(pres_eval["top1"]) >= 1.0
        passed = learned and pres_ok and retention_ok
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "new_win": new_eval,
            "preservation": pres_eval,
            "retention": ret_eval,
            "learned_enough": bool(learned),
            "preservation_pass": bool(pres_ok),
            "retention_pass": bool(retention_ok),
            "pass": bool(passed),
        }
        history.append(row)
        print("V30_ADAPTER_EPOCH", json.dumps(row, sort_keys=True), flush=True)

        if passed:
            score = (
                int(new_eval["fully_learned_seeds"]),
                int(new_eval["top1_examples"]),
                -float(ret_eval["parent_kl"]),
                float(ret_eval["parent_high_conf_top1_agreement"]),
                float(ret_eval["winner_top1"]),
                -float(ret_eval["residual_rms"]),
            )
            state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
            if best is None or score > best[0]:
                best = (score, state, row)

        # No need to keep training once a very safe useful adapter exists.
        if (
            best is not None
            and epoch >= 20
            and float(best[2]["retention"]["parent_kl"]) <= a.max_parent_kl * 0.25
            and int(best[2]["new_win"]["fully_learned_seeds"]) >= a.min_fully_learned_seeds
        ):
            break

    if best is None:
        raise RuntimeError(
            "no residual adapter learned a New-Win seed while passing G7 retention guards"
        )

    adapter.load_state_dict(best[1])
    adapter.eval()
    after_new = _teacher_eval(parent, adapter, torch, rows)
    after_pres = _preservation_eval(parent, adapter, torch, preservation, set(kinds))
    after_ret = _retention_eval(parent, adapter, torch, elite, retention_eval)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Candidate base stays G7; the sidecar is the only learned component.
    torch.save(parent_state, a.output)
    sidecar = Path(str(a.output) + ".adapter.pt")
    torch.save(
        {
            "schema_version": SCHEMA,
            "input_dim": int(input_dim),
            "hidden_dim": int(a.hidden_dim),
            "kinds": kinds,
            "state_dict": {k: v.detach().clone() for k, v in adapter.state_dict().items()},
        },
        sidecar,
    )

    report = {
        "schema_version": "sts1-build-rescue-v30-residual-adapter",
        "teacher_examples": len(rows),
        "teacher_seeds": sorted({int(r["seed"]) for r in rows}),
        "enabled_kinds": kinds,
        "hidden_dim": a.hidden_dim,
        "before": {
            "new_win": before_new,
            "preservation": before_pres,
            "retention": before_ret,
        },
        "after": {
            "new_win": after_new,
            "preservation": after_pres,
            "retention": after_ret,
        },
        "selected_epoch": int(best[2]["epoch"]),
        "history": history,
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "V30_ADAPTER_TRAIN_PASS",
        json.dumps(
            {
                "teacher_examples": len(rows),
                "teacher_seeds": len({int(r["seed"]) for r in rows}),
                "enabled_kinds": kinds,
                "selected_epoch": int(best[2]["epoch"]),
                "fully_learned_seeds": after_new["fully_learned_seeds"],
                "fully_learned_seed_ids": after_new["fully_learned_seed_ids"],
                "new_win_top1": after_new["top1"],
                "preservation_top1": after_pres["top1"],
                "parent_high_conf_top1_agreement": after_ret["parent_high_conf_top1_agreement"],
                "parent_top1_agreement": after_ret["parent_top1_agreement"],
                "parent_kl": after_ret["parent_kl"],
                "winner_top1": after_ret["winner_top1"],
                "residual_rms": after_ret["residual_rms"],
                "sidecar": str(sidecar),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
