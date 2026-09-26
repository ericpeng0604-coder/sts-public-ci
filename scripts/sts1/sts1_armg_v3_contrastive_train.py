#!/usr/bin/env python3
"""Outcome-aware ArmG v3: weight wins and near-wins as positive training evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path


def split(name: str) -> str:
    return "val" if int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"


def load(root: Path):
    games = []
    for path in sorted(root.rglob("seed-*.ndjson")):
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except Exception:
            continue
        summary = next((row for row in reversed(rows) if row.get("type") == "summary"), {})
        decisions = [
            row
            for row in rows
            if row.get("type") == "armg_noncombat_decision_v3"
            and row.get("selected_index", -1) >= 0
            and len(row.get("candidate_desc_368", [])) > 1
        ]
        if decisions and summary.get("outcome") in ("victory", "defeat"):
            games.append((path.stem, summary, decisions))
    return games


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("evidence", "armg-root", "base-weight", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--anchor", type=float, default=2e-4)
    parser.add_argument("--loss-weight", type=float, default=0.15)
    parser.add_argument("--win-weight", type=float, default=2.0)
    parser.add_argument("--near-win-weight", type=float, default=0.5)
    parser.add_argument("--near-win-floor", type=int, default=45)
    args = parser.parse_args()

    os.environ["STS_BOT_DIR"] = str(args.armg_root)
    sys.path.insert(0, str(args.armg_root))
    torch = importlib.import_module("torch")
    model_module = importlib.import_module("armG_train")
    model_module.card_idx = lambda name: model_module._vocab.get(name, model_module.VOCAB_CAP - 1)

    state = torch.load(args.base_weight, weights_only=True, map_location="cpu")
    net = model_module.Scorer((128, 128))
    net.load_state_dict(state)
    base = [parameter.detach().clone() for parameter in net.parameters()]

    games = load(args.evidence)
    train = [game for game in games if split(game[0]) == "train"]
    val = [game for game in games if split(game[0]) == "val"]
    if not train or not val:
        raise SystemExit(f"insufficient split games train={len(train)} val={len(val)}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()

    def category(summary: dict) -> str:
        if summary.get("outcome") == "victory":
            return "win"
        if int(summary.get("final_floor") or 0) >= args.near_win_floor:
            return "near_win"
        return "loss"

    def decision_loss(row: dict, summary: dict, grad: bool):
        obs = torch.tensor(row["obs_412"], dtype=torch.float32)
        descs = row["candidate_desc_368"]
        selected = int(row["selected_index"])
        kind = category(summary)
        with torch.set_grad_enabled(grad):
            logits = net.score(obs, descs).reshape(1, -1)
            if kind == "win":
                loss = args.win_weight * ce(logits, torch.tensor([selected]))
            elif kind == "near_win":
                loss = args.near_win_weight * ce(logits, torch.tensor([selected]))
            else:
                # Keep the old conservative treatment for ordinary failed runs.
                probs = torch.softmax(logits, 1)
                probability = probs[0, selected].clamp(max=0.999999)
                loss = args.loss_weight * (-torch.log1p(-probability))
            if grad:
                loss = loss + args.anchor * sum(
                    (parameter - reference).pow(2).mean()
                    for parameter, reference in zip(net.parameters(), base)
                )
        return loss

    best = None
    best_val_loss = 1e9
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        train_loss = 0.0
        train_count = 0
        for _, summary, decisions in train:
            for row in decisions:
                optimizer.zero_grad()
                loss = decision_loss(row, summary, True)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach())
                train_count += 1

        net.eval()
        val_loss = 0.0
        val_count = 0
        for _, summary, decisions in val:
            for row in decisions:
                loss = decision_loss(row, summary, False)
                val_loss += float(loss.detach())
                val_count += 1

        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_count),
            "val_loss": val_loss / max(1, val_count),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["val_loss"] < best_val_loss:
            best_val_loss = record["val_loss"]
            best = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            best_epoch = epoch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, args.output)

    counts = {
        "wins": sum(category(summary) == "win" for _, summary, _ in games),
        "near_wins": sum(category(summary) == "near_win" for _, summary, _ in games),
        "losses": sum(category(summary) == "loss" for _, summary, _ in games),
    }
    report = {
        "schema_version": "sts1-armg-v3-outcome-weighted-v2",
        "games": len(games),
        **counts,
        "train_games": len(train),
        "val_games": len(val),
        "win_weight": args.win_weight,
        "near_win_weight": args.near_win_weight,
        "near_win_floor": args.near_win_floor,
        "loss_weight": args.loss_weight,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("ARMG_V3_TRAIN_RESULT", json.dumps(report))


if __name__ == "__main__":
    main()
