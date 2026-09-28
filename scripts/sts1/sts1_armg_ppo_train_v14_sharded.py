#!/usr/bin/env python3
"""PPO v1.4: temperature-correct, critic-persistent, parent-anchored updates."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = float(np.var(y_true))
    if var_y < 1e-12:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("shards", "armg-root", "base-weight", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--base-critic", type=Path)
    p.add_argument("--critic-output", type=Path)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--clip", type=float, default=0.20)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--entropy", type=float, default=0.001)
    p.add_argument("--anchor-coef", type=float, default=0.02)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--behavior-temperature", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    if not 0.25 <= a.behavior_temperature <= 2.0:
        raise RuntimeError("behavior-temperature outside safe bounds")
    if not 0.0 <= a.entropy <= 0.005:
        raise RuntimeError("entropy outside v1.4 safety bounds")
    if not 0.0 <= a.anchor_coef <= 0.20:
        raise RuntimeError("anchor-coef outside safe bounds")

    os.environ["STS_BOT_DIR"] = str(a.armg_root)
    sys.path.insert(0, str(a.armg_root))
    torch = importlib.import_module("torch")
    torch.set_num_threads(a.threads)
    torch.manual_seed(20260928)
    m = importlib.import_module("armG_train")
    m.card_idx = lambda n: m._vocab.get(n, m.VOCAB_CAP - 1)

    actor = m.Scorer((128, 128))
    actor.load_state_dict(torch.load(a.base_weight, weights_only=True, map_location="cpu"))

    anchor = m.Scorer((128, 128))
    anchor.load_state_dict(torch.load(a.base_weight, weights_only=True, map_location="cpu"))
    anchor.eval()
    for param in anchor.parameters():
        param.requires_grad_(False)

    critic = torch.nn.Sequential(
        torch.nn.Linear(m.OBS_DIM, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 1),
    )
    critic_loaded = False
    if a.base_critic is not None and a.base_critic.is_file():
        critic.load_state_dict(
            torch.load(a.base_critic, weights_only=True, map_location="cpu")
        )
        critic_loaded = True

    opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=a.lr,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    critic_output = a.critic_output or a.output.with_name(a.output.stem + "_critic.pt")

    files = sorted(a.shards.glob("shard_*.npz"))
    if not files:
        raise RuntimeError("no shards")

    history: list[dict[str, float | int]] = []
    for ep in range(1, a.epochs + 1):
        losses: list[float] = []
        kls: list[float] = []
        entropies: list[float] = []
        anchor_kls: list[float] = []
        clip_fracs: list[float] = []
        explained: list[float] = []
        t0 = time.time()

        for si, f in enumerate(files):
            t = time.time()
            with np.load(f, allow_pickle=False) as d:
                if str(d["checkpoint_id"][0]) != a.checkpoint_id:
                    raise SystemExit("stale shard")

                obs = torch.from_numpy(d["obs"].astype(np.float32))
                desc = d["desc"]
                off = d["offsets"]
                act = torch.from_numpy(d["action"].astype(np.int64))
                rew = d["reward"].astype(np.float32)
                old = torch.from_numpy(d["old_logp"].astype(np.float32))
                done = d["done"].astype(bool)

                with torch.no_grad():
                    v = critic(obs).squeeze(1).numpy()

                adv = np.zeros(len(rew), np.float32)
                ret = np.zeros(len(rew), np.float32)
                gae = 0.0
                for i in range(len(rew) - 1, -1, -1):
                    nv = 0.0 if done[i] or i == len(rew) - 1 else v[i + 1]
                    delta = rew[i] + a.gamma * nv - v[i]
                    gae = delta + a.gamma * a.lam * (0.0 if done[i] else gae)
                    adv[i] = gae
                    ret[i] = gae + v[i]

                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                new_logp = []
                entropy_terms = []
                anchor_terms = []
                for i in range(len(rew)):
                    ds = torch.from_numpy(
                        desc[off[i] : off[i + 1]].astype(np.float32)
                    )
                    o = obs[i].repeat(len(ds), 1)
                    rows = torch.cat([o, ds], 1)

                    logits = actor.net(rows).squeeze(1) / a.behavior_temperature
                    lp = torch.log_softmax(logits, 0)
                    pr = torch.softmax(logits, 0)
                    new_logp.append(lp[act[i]])
                    entropy_terms.append(-(pr * lp).sum())

                    with torch.no_grad():
                        anchor_logits = (
                            anchor.net(rows).squeeze(1) / a.behavior_temperature
                        )
                        anchor_lp = torch.log_softmax(anchor_logits, 0)
                        anchor_pr = torch.softmax(anchor_logits, 0)
                    anchor_terms.append((anchor_pr * (anchor_lp - lp)).sum())

                new = torch.stack(new_logp)
                ent = torch.stack(entropy_terms)
                anchor_kl = torch.stack(anchor_terms)
                A = torch.from_numpy(adv)
                R = torch.from_numpy(ret)

                ratio = torch.exp(new - old)
                unclipped = ratio * A
                clipped = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A
                pg = -torch.minimum(unclipped, clipped).mean()

                values_now = critic(obs).squeeze(1)
                vl = torch.nn.functional.mse_loss(values_now, R)
                anchor_loss = anchor_kl.mean()
                loss = (
                    pg
                    + a.value_coef * vl
                    - a.entropy * ent.mean()
                    + a.anchor_coef * anchor_loss
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 1.0
                )
                opt.step()

                log_ratio = new.detach() - old
                kl = float(((torch.exp(log_ratio) - 1) - log_ratio).mean())
                clip_fraction = float(
                    ((ratio.detach() - 1.0).abs() > a.clip).float().mean()
                )
                with torch.no_grad():
                    v_after = critic(obs).squeeze(1).numpy()
                ev = explained_variance(v_after, ret)

                losses.append(float(loss.detach()))
                kls.append(kl)
                entropies.append(float(ent.mean().detach()))
                anchor_kls.append(float(anchor_loss.detach()))
                clip_fracs.append(clip_fraction)
                explained.append(ev)

                rec = {
                    "epoch": ep,
                    "shard": si,
                    "decisions": len(rew),
                    "loss": losses[-1],
                    "kl": kl,
                    "entropy": entropies[-1],
                    "anchor_kl": anchor_kls[-1],
                    "clip_fraction": clip_fraction,
                    "explained_variance": ev,
                    "seconds": time.time() - t,
                }
                if ep == 1 and si == 0:
                    rec["initial_ratio_mean"] = float(ratio.detach().mean())
                    rec["initial_ratio_abs_error"] = float(
                        (ratio.detach() - 1.0).abs().mean()
                    )
                print("TRAIN_SHARD_V14", json.dumps(rec), flush=True)

            torch.save(actor.state_dict(), a.output)
            torch.save(critic.state_dict(), critic_output)
            (a.output.parent / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "schema": "sts1-armg-ppo-v14-checkpoint",
                        "epoch": ep,
                        "completed_shard": si,
                        "critic_loaded": critic_loaded,
                        "behavior_temperature": a.behavior_temperature,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        epoch_rec = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "approx_kl": float(np.mean(kls)),
            "entropy": float(np.mean(entropies)),
            "anchor_kl": float(np.mean(anchor_kls)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "explained_variance": float(np.mean(explained)),
            "seconds": time.time() - t0,
        }
        history.append(epoch_rec)
        print("TRAIN_EPOCH_V14", json.dumps(epoch_rec), flush=True)
        if epoch_rec["approx_kl"] > a.target_kl:
            print(
                "PPO_V14_EARLY_STOP",
                json.dumps(
                    {
                        "epoch": ep,
                        "approx_kl": epoch_rec["approx_kl"],
                        "target_kl": a.target_kl,
                    }
                ),
                flush=True,
            )
            break

    report = {
        "schema": "sts1-armg-ppo-v14-sharded-train",
        "shards": len(files),
        "critic_loaded": critic_loaded,
        "behavior_temperature": a.behavior_temperature,
        "entropy_coef": a.entropy,
        "anchor_coef": a.anchor_coef,
        "history": history,
    }
    a.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SHARDED_TRAIN_V14_PASS", json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
