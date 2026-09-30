#!/usr/bin/env python3
"""PPO v1.4: temperature-correct, critic-persistent, parent-anchored updates.

The rollout buffer is immutable during PPO optimization: GAE advantages and
value targets are computed once from the starting Critic, then reused for every
epoch. The durable Critic checkpoint keeps only the best explained-variance
epoch so a bad optimization epoch cannot poison the next loop round.
"""

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


def gae_targets(
    reward: np.ndarray,
    done: np.ndarray,
    values: np.ndarray,
    *,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one immutable GAE/return target set for a rollout shard."""
    if not (len(reward) == len(done) == len(values)):
        raise RuntimeError("GAE input length mismatch")
    adv = np.zeros(len(reward), np.float32)
    ret = np.zeros(len(reward), np.float32)
    gae = 0.0
    for i in range(len(reward) - 1, -1, -1):
        next_value = 0.0 if done[i] or i == len(reward) - 1 else values[i + 1]
        delta = reward[i] + gamma * next_value - values[i]
        gae = delta + gamma * lam * (0.0 if done[i] else gae)
        adv[i] = gae
        ret[i] = gae + values[i]
    return adv, ret


def discounted_returns(
    reward: np.ndarray,
    done: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    """Critic-health target independent of the Critic itself."""
    out = np.zeros(len(reward), np.float32)
    running = 0.0
    for i in range(len(reward) - 1, -1, -1):
        running = float(reward[i]) + gamma * (
            0.0 if done[i] else running
        )
        out[i] = running
    return out


def should_reset_critic(
    *,
    critic_loaded: bool,
    health_explained_variance: float,
    threshold: float = -0.10,
) -> bool:
    return critic_loaded and health_explained_variance < threshold


def curriculum_floor_band(floor: int) -> str:
    """Map a terminal floor to a coarse act-sized training band."""
    if floor <= 16:
        return "act1"
    if floor <= 33:
        return "act2"
    return "act3"


def choose_curriculum_focus(
    floors: list[int],
    victories: list[bool],
) -> tuple[str, dict[str, int]]:
    """Pick the most common failure band, preferring later acts on ties."""
    if len(floors) != len(victories):
        raise RuntimeError("curriculum metadata length mismatch")
    counts = {"act1": 0, "act2": 0, "act3": 0}
    for floor, victory in zip(floors, victories):
        if victory:
            continue
        counts[curriculum_floor_band(int(floor))] += 1
    if sum(counts.values()) == 0:
        return "none", counts
    order = {"act1": 0, "act2": 1, "act3": 2}
    focus = max(counts, key=lambda k: (counts[k], order[k]))
    return focus, counts


STRATEGY_DATASET_SCHEMA_VERSION = "sts1-armg-strategy-branch-dataset-v1"


def load_strategy_teacher_examples(
    path: Path | None,
    *,
    max_examples: int,
    combat_policy: str,
    min_confidence_weight: float = 0.0,
    min_teacher_margin: float = 0.0,
    min_consensus_fraction: float = 0.0,
) -> list[dict[str, object]]:
    """Load only high-value, stable Strategy Teacher disagreements for BC."""
    if not 0.0 <= min_confidence_weight <= 1.0:
        raise RuntimeError("strategy teacher minimum confidence must be in [0, 1]")
    if min_teacher_margin < 0.0:
        raise RuntimeError("strategy teacher minimum margin must be non-negative")
    if not 0.0 <= min_consensus_fraction <= 1.0:
        raise RuntimeError("strategy teacher minimum consensus must be in [0, 1]")
    if path is None or not path.is_file() or max_examples <= 0:
        return []

    groups: dict[str, list[dict[str, object]]] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("schema_version") != STRATEGY_DATASET_SCHEMA_VERSION:
            raise RuntimeError(f"strategy replay schema mismatch at line {line_no}")
        if row.get("combat_policy") != combat_policy:
            raise RuntimeError(
                f"strategy replay combat policy mismatch at line {line_no}: "
                f"{row.get('combat_policy')!r} != {combat_policy!r}"
            )
        descs = row.get("descs")
        probs = row.get("target_probs")
        obs = row.get("obs")
        if not isinstance(obs, list) or not isinstance(descs, list) or len(descs) < 2:
            raise RuntimeError(f"strategy replay feature shape invalid at line {line_no}")
        if not isinstance(probs, list) or len(probs) != len(descs):
            raise RuntimeError(f"strategy replay target shape invalid at line {line_no}")
        current = int(row.get("current_armg_index", -1))
        teacher = int(row.get("teacher_best_index", -1))
        if not (0 <= current < len(descs) and 0 <= teacher < len(descs)):
            raise RuntimeError(f"strategy replay action index invalid at line {line_no}")
        # Only disagreements are imported. Agreement examples are already
        # represented by on-policy PPO and add little Teacher signal.
        if current == teacher:
            continue

        confidence = float(row.get("confidence_weight", 1.0))
        margin = float(row.get("teacher_margin", 0.0))
        consensus = float(row.get("teacher_consensus_fraction", 1.0))
        if confidence < min_confidence_weight:
            continue
        if margin < min_teacher_margin:
            continue
        if consensus < min_consensus_fraction:
            continue

        target = np.asarray(probs, np.float32)
        if not np.isfinite(target).all() or np.any(target < 0.0):
            raise RuntimeError(f"strategy replay target probabilities invalid at line {line_no}")
        total = float(target.sum())
        if total <= 0.0:
            raise RuntimeError(f"strategy replay target probabilities empty at line {line_no}")
        clean = dict(row)
        clean["target_probs"] = (target / total).tolist()
        kind = str(row.get("kind", "unknown"))
        groups.setdefault(kind, []).append(clean)

    for rows in groups.values():
        rows.sort(
            key=lambda row: (
                float(row.get("priority", 1.0)),
                float(row.get("teacher_margin", 0.0)),
                int(row.get("floor", 0)),
            ),
            reverse=True,
        )

    # Round-robin kinds so map/card/shop/rest/event all keep representation.
    kept: list[dict[str, object]] = []
    kinds = sorted(groups)
    cursor = 0
    while kinds and len(kept) < max_examples:
        kind = kinds[cursor % len(kinds)]
        bucket = groups[kind]
        if bucket:
            kept.append(bucket.pop(0))
        if not bucket:
            kinds.remove(kind)
            cursor = 0
        else:
            cursor += 1
    return kept


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("shards", "armg-root", "base-weight", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--base-critic", type=Path)
    p.add_argument("--critic-output", type=Path)
    p.add_argument("--replay-shards", type=Path)
    p.add_argument("--elite-replay", type=Path)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--critic-lr", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=0.20)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--entropy", type=float, default=0.001)
    p.add_argument("--anchor-coef", type=float, default=0.02)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--behavior-temperature", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bc-coef", type=float, default=0.01)
    p.add_argument("--bc-max-decisions", type=int, default=1024)
    p.add_argument("--curriculum-strength", type=float, default=0.0)
    p.add_argument("--strategy-replay", type=Path)
    p.add_argument("--strategy-bc-coef", type=float, default=0.0)
    p.add_argument("--strategy-bc-max-examples", type=int, default=512)
    p.add_argument("--strategy-combat-policy", default="mcts_2000")
    p.add_argument("--strategy-min-confidence-weight", type=float, default=0.0)
    p.add_argument("--strategy-min-teacher-margin", type=float, default=0.0)
    p.add_argument("--strategy-min-consensus-fraction", type=float, default=0.0)
    a = p.parse_args()

    if not 0.25 <= a.behavior_temperature <= 2.0:
        raise RuntimeError("behavior-temperature outside safe bounds")
    if not 0.0 <= a.entropy <= 0.005:
        raise RuntimeError("entropy outside v1.4 safety bounds")
    if not 0.0 <= a.anchor_coef <= 0.20:
        raise RuntimeError("anchor-coef outside safe bounds")
    if not 0.0 < a.critic_lr <= a.lr:
        raise RuntimeError("critic-lr must be positive and <= actor lr")
    if not 0.0 <= a.bc_coef <= 0.05:
        raise RuntimeError("bc-coef outside safe bounds")
    if not 0 <= a.bc_max_decisions <= 4096:
        raise RuntimeError("bc-max-decisions outside safe bounds")
    if not 0.0 <= a.curriculum_strength <= 1.5:
        raise RuntimeError("curriculum-strength outside safe bounds")
    if not 0.0 <= a.strategy_bc_coef <= 0.05:
        raise RuntimeError("strategy-bc-coef outside safe bounds")
    if not 0 <= a.strategy_bc_max_examples <= 4096:
        raise RuntimeError("strategy-bc-max-examples outside safe bounds")
    if not 0.0 <= a.strategy_min_confidence_weight <= 1.0:
        raise RuntimeError("strategy-min-confidence-weight outside safe bounds")
    if a.strategy_min_teacher_margin < 0.0:
        raise RuntimeError("strategy-min-teacher-margin outside safe bounds")
    if not 0.0 <= a.strategy_min_consensus_fraction <= 1.0:
        raise RuntimeError("strategy-min-consensus-fraction outside safe bounds")

    os.environ["STS_BOT_DIR"] = str(a.armg_root)
    sys.path.insert(0, str(a.armg_root))
    torch = importlib.import_module("torch")
    torch.set_num_threads(a.threads)
    torch.manual_seed(20260928)
    m = importlib.import_module("armG_train")
    m.card_idx = lambda n: m._vocab.get(n, m.VOCAB_CAP - 1)

    actor = m.Scorer((128, 128))
    actor.load_state_dict(
        torch.load(a.base_weight, weights_only=True, map_location="cpu")
    )

    anchor = m.Scorer((128, 128))
    anchor.load_state_dict(
        torch.load(a.base_weight, weights_only=True, map_location="cpu")
    )
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

    actor_opt = torch.optim.Adam(actor.parameters(), lr=a.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    critic_output = (
        a.critic_output
        or a.output.with_name(a.output.stem + "_critic.pt")
    )
    critic_output.parent.mkdir(parents=True, exist_ok=True)

    fresh_files = sorted(a.shards.glob("shard_*.npz"))
    replay_files = (
        sorted(a.replay_shards.glob("replay_*.npz"))
        if a.replay_shards is not None and a.replay_shards.is_dir()
        else []
    )
    files = fresh_files + replay_files
    if not fresh_files:
        raise RuntimeError("no fresh shards")
    if not files:
        raise RuntimeError("no shards")

    fresh_decisions = 0
    replay_decisions = 0
    for path in fresh_files:
        with np.load(path, allow_pickle=False) as d:
            fresh_decisions += len(d["action"])
    for path in replay_files:
        with np.load(path, allow_pickle=False) as d:
            replay_decisions += len(d["action"])
    if replay_decisions > fresh_decisions:
        raise RuntimeError("replay decisions must not exceed fresh decisions")

    curriculum_floors: list[int] = []
    curriculum_victories: list[bool] = []
    seen_curriculum_seeds: set[int] = set()
    for path in fresh_files:
        with np.load(path, allow_pickle=False) as d:
            if "game_seed" not in d or "game_final_floor" not in d or "game_victory" not in d:
                if a.curriculum_strength > 0.0:
                    raise RuntimeError("curriculum requires game floor/victory metadata")
                continue
            seeds = np.asarray(d["game_seed"], np.int64)
            floors = np.asarray(d["game_final_floor"], np.int16)
            victories = np.asarray(d["game_victory"], np.bool_)
            if not (len(seeds) == len(floors) == len(victories)):
                raise RuntimeError("curriculum metadata row mismatch")
            for raw_seed in np.unique(seeds):
                seed = int(raw_seed)
                if seed in seen_curriculum_seeds:
                    continue
                idx = int(np.flatnonzero(seeds == seed)[0])
                seen_curriculum_seeds.add(seed)
                curriculum_floors.append(int(floors[idx]))
                curriculum_victories.append(bool(victories[idx]))
    curriculum_focus, curriculum_failure_counts = choose_curriculum_focus(
        curriculum_floors,
        curriculum_victories,
    )
    fresh_file_set = set(fresh_files)
    print(
        "PPO_V15_CURRICULUM_FOCUS",
        json.dumps(
            {
                "focus": curriculum_focus,
                "failure_counts": curriculum_failure_counts,
                "strength": a.curriculum_strength,
                "games": len(curriculum_floors),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # Diagnose a persisted Critic against discounted returns that do not
    # depend on that Critic. A negative EV means it is worse than a constant
    # baseline and should not seed the next round's GAE.
    health_predictions: list[np.ndarray] = []
    health_returns: list[np.ndarray] = []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            if str(d["checkpoint_id"][0]) != a.checkpoint_id:
                raise SystemExit("stale shard")
            obs = torch.from_numpy(d["obs"].astype(np.float32))
            reward = d["reward"].astype(np.float32)
            done = d["done"].astype(bool)
            with torch.no_grad():
                health_predictions.append(
                    critic(obs).squeeze(1).numpy()
                )
            health_returns.append(
                discounted_returns(reward, done, gamma=a.gamma)
            )

    critic_health_ev = explained_variance(
        np.concatenate(health_predictions),
        np.concatenate(health_returns),
    )
    critic_reset = should_reset_critic(
        critic_loaded=critic_loaded,
        health_explained_variance=critic_health_ev,
    )
    if critic_reset:
        # Zero only the value head. Hidden features stay intact while the
        # output becomes a safe constant baseline with learnable gradients.
        value_head = critic[-1]
        with torch.no_grad():
            value_head.weight.zero_()
            value_head.bias.zero_()
        print(
            "PPO_V14_CRITIC_RESET",
            json.dumps(
                {
                    "health_explained_variance": critic_health_ev,
                    "threshold": -0.10,
                    "reason": "persisted_critic_worse_than_constant_baseline",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # PPO rollout targets must stay fixed for all optimization epochs.
    raw_advantages: list[np.ndarray] = []
    fixed_returns: list[np.ndarray] = []
    initial_value_predictions: list[np.ndarray] = []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            obs = torch.from_numpy(d["obs"].astype(np.float32))
            reward = d["reward"].astype(np.float32)
            done = d["done"].astype(bool)
            with torch.no_grad():
                values = critic(obs).squeeze(1).numpy()
            adv, ret = gae_targets(
                reward,
                done,
                values,
                gamma=a.gamma,
                lam=a.lam,
            )
            raw_advantages.append(adv)
            fixed_returns.append(ret)
            initial_value_predictions.append(values)

    all_adv = np.concatenate(raw_advantages)
    adv_mean = float(all_adv.mean())
    adv_std = float(all_adv.std() + 1e-8)
    fixed_advantages = [
        ((adv - adv_mean) / adv_std).astype(np.float32)
        for adv in raw_advantages
    ]
    initial_ev = explained_variance(
        np.concatenate(initial_value_predictions),
        np.concatenate(fixed_returns),
    )

    # Always leave a valid Critic checkpoint. Later epochs replace it only when
    # they improve explained variance on the immutable rollout targets.
    torch.save(critic.state_dict(), critic_output)
    best_critic_ev = initial_ev
    best_critic_epoch = 0

    print(
        "PPO_V14_FIXED_TARGETS",
        json.dumps(
            {
                "decisions": int(sum(len(x) for x in fixed_returns)),
                "advantage_mean": adv_mean,
                "advantage_std": adv_std,
                "initial_explained_variance": initial_ev,
                "critic_loaded": critic_loaded,
                "critic_reset": critic_reset,
                "critic_health_explained_variance_before_reset": critic_health_ev,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    elite_data: dict[str, np.ndarray] | None = None
    if a.elite_replay is not None and a.elite_replay.is_file():
        with np.load(a.elite_replay, allow_pickle=False) as d:
            elite_data = {
                "obs": np.asarray(d["obs"], np.float32),
                "desc": np.asarray(d["desc"], np.float32),
                "offsets": np.asarray(d["offsets"], np.int64),
                "action": np.asarray(d["action"], np.int64),
            }
        if len(elite_data["action"]) + 1 != len(elite_data["offsets"]):
            raise RuntimeError("elite replay offsets invalid")

    strategy_examples = load_strategy_teacher_examples(
        a.strategy_replay,
        max_examples=a.strategy_bc_max_examples,
        combat_policy=a.strategy_combat_policy,
        min_confidence_weight=a.strategy_min_confidence_weight,
        min_teacher_margin=a.strategy_min_teacher_margin,
        min_consensus_fraction=a.strategy_min_consensus_fraction,
    )
    if strategy_examples:
        expected_obs = int(m.OBS_DIM)
        first_linear = next(
            layer for layer in actor.net
            if hasattr(layer, "in_features")
        )
        expected_desc = int(first_linear.in_features) - expected_obs
        for row in strategy_examples:
            obs_row = row["obs"]
            desc_rows = row["descs"]
            if len(obs_row) != expected_obs:
                raise RuntimeError("strategy replay obs dimension mismatch")
            if any(len(desc) != expected_desc for desc in desc_rows):
                raise RuntimeError("strategy replay desc dimension mismatch")
    strategy_kind_counts: dict[str, int] = {}
    for row in strategy_examples:
        kind = str(row.get("kind", "unknown"))
        strategy_kind_counts[kind] = strategy_kind_counts.get(kind, 0) + 1
    print(
        "PPO_V15_STRATEGY_TEACHER",
        json.dumps(
            {
                "examples": len(strategy_examples),
                "examples_by_kind": dict(sorted(strategy_kind_counts.items())),
                "coef": a.strategy_bc_coef,
                "combat_policy": a.strategy_combat_policy,
                "mistakes_only": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    history: list[dict[str, float | int]] = []
    first_ratio_mean: float | None = None
    first_ratio_abs_error: float | None = None
    total_bc_updates = 0
    total_bc_decisions = 0
    total_strategy_bc_updates = 0
    total_strategy_bc_examples = 0

    for ep in range(1, a.epochs + 1):
        losses: list[float] = []
        kls: list[float] = []
        entropies: list[float] = []
        anchor_kls: list[float] = []
        clip_fracs: list[float] = []
        shard_evs: list[float] = []
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
                old = torch.from_numpy(d["old_logp"].astype(np.float32))
                A = torch.from_numpy(fixed_advantages[si])
                R = torch.from_numpy(fixed_returns[si])
                behavior_temperatures = (
                    np.asarray(d["behavior_temperature"], np.float32)
                    if "behavior_temperature" in d
                    else np.full(len(A), a.behavior_temperature, np.float32)
                )
                if len(behavior_temperatures) != len(A):
                    raise RuntimeError("behavior temperature length mismatch")
                if np.any(behavior_temperatures < 0.25) or np.any(behavior_temperatures > 2.0):
                    raise RuntimeError("replay behavior temperature outside safe bounds")

                curriculum_weights = np.ones(len(A), np.float32)
                if (
                    a.curriculum_strength > 0.0
                    and f in fresh_file_set
                    and curriculum_focus != "none"
                ):
                    floors = np.asarray(d["game_final_floor"], np.int16)
                    victories = np.asarray(d["game_victory"], np.bool_)
                    if not (len(floors) == len(victories) == len(A)):
                        raise RuntimeError("curriculum shard metadata mismatch")
                    focused = np.asarray(
                        [
                            (not bool(victories[i]))
                            and curriculum_floor_band(int(floors[i])) == curriculum_focus
                            for i in range(len(A))
                        ],
                        np.bool_,
                    )
                    curriculum_weights[focused] = 1.0 + a.curriculum_strength
                W = torch.from_numpy(curriculum_weights)

                new_logp = []
                entropy_terms = []
                anchor_terms = []
                for i in range(len(A)):
                    ds = torch.from_numpy(
                        desc[off[i] : off[i + 1]].astype(np.float32)
                    )
                    o = obs[i].repeat(len(ds), 1)
                    rows = torch.cat([o, ds], 1)

                    sample_temperature = float(behavior_temperatures[i])
                    logits = (
                        actor.net(rows).squeeze(1)
                        / sample_temperature
                    )
                    lp = torch.log_softmax(logits, 0)
                    pr = torch.softmax(logits, 0)
                    new_logp.append(lp[act[i]])
                    entropy_terms.append(-(pr * lp).sum())

                    with torch.no_grad():
                        anchor_logits = (
                            anchor.net(rows).squeeze(1)
                            / sample_temperature
                        )
                        anchor_lp = torch.log_softmax(anchor_logits, 0)
                        anchor_pr = torch.softmax(anchor_logits, 0)
                    anchor_terms.append(
                        (anchor_pr * (anchor_lp - lp)).sum()
                    )

                new = torch.stack(new_logp)
                ent = torch.stack(entropy_terms)
                anchor_kl = torch.stack(anchor_terms)

                ratio = torch.exp(new - old)
                unclipped = ratio * A
                clipped = (
                    torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A
                )
                objective = torch.minimum(unclipped, clipped)
                pg = -(objective * W).sum() / W.sum().clamp_min(1.0)
                anchor_loss = anchor_kl.mean()
                actor_loss = (
                    pg
                    - a.entropy * ent.mean()
                    + a.anchor_coef * anchor_loss
                )

                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()), 1.0
                )
                actor_opt.step()

                values_now = critic(obs).squeeze(1)
                value_loss = torch.nn.functional.mse_loss(values_now, R)
                critic_loss = a.value_coef * value_loss
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(critic.parameters()), 1.0
                )
                critic_opt.step()

                total_loss = actor_loss.detach() + critic_loss.detach()
                log_ratio = new.detach() - old
                kl = max(
                    0.0,
                    float(
                        (
                            (torch.exp(log_ratio) - 1)
                            - log_ratio
                        ).mean()
                    ),
                )
                clip_fraction = float(
                    (
                        (ratio.detach() - 1.0).abs() > a.clip
                    ).float().mean()
                )
                with torch.no_grad():
                    value_after = critic(obs).squeeze(1).numpy()
                shard_ev = explained_variance(
                    value_after,
                    fixed_returns[si],
                )

                losses.append(float(total_loss))
                kls.append(kl)
                entropies.append(float(ent.mean().detach()))
                anchor_kls.append(float(anchor_loss.detach()))
                clip_fracs.append(clip_fraction)
                shard_evs.append(shard_ev)

                rec = {
                    "epoch": ep,
                    "shard": si,
                    "decisions": len(A),
                    "loss": losses[-1],
                    "kl": kl,
                    "entropy": entropies[-1],
                    "anchor_kl": anchor_kls[-1],
                    "clip_fraction": clip_fraction,
                    "explained_variance": shard_ev,
                    "curriculum_weighted_decisions": int((curriculum_weights > 1.0).sum()),
                    "curriculum_focus": curriculum_focus,
                    "seconds": time.time() - t,
                }
                if ep == 1 and si == 0:
                    first_ratio_mean = float(ratio.detach().mean())
                    first_ratio_abs_error = float(
                        (ratio.detach() - 1.0).abs().mean()
                    )
                    rec["initial_ratio_mean"] = first_ratio_mean
                    rec["initial_ratio_abs_error"] = (
                        first_ratio_abs_error
                    )
                print(
                    "TRAIN_SHARD_V14",
                    json.dumps(rec),
                    flush=True,
                )

            # Actor checkpoint can follow the newest PPO update.
            torch.save(actor.state_dict(), a.output)

        bc_mean_loss = 0.0
        bc_used = 0
        if (
            elite_data is not None
            and a.bc_coef > 0.0
            and a.bc_max_decisions > 0
            and len(elite_data["action"]) > 0
        ):
            rng = np.random.default_rng(20260928 + ep)
            take = min(a.bc_max_decisions, len(elite_data["action"]))
            selected = rng.choice(
                len(elite_data["action"]),
                size=take,
                replace=False,
            )
            batch_losses: list[float] = []
            for start in range(0, len(selected), 64):
                chunk = selected[start : start + 64]
                bc_terms = []
                for raw_i in chunk:
                    i = int(raw_i)
                    begin = int(elite_data["offsets"][i])
                    end = int(elite_data["offsets"][i + 1])
                    ds = torch.from_numpy(elite_data["desc"][begin:end])
                    o = torch.from_numpy(elite_data["obs"][i]).repeat(len(ds), 1)
                    rows = torch.cat([o, ds], 1)
                    lp = torch.log_softmax(actor.net(rows).squeeze(1), 0)
                    action_i = int(elite_data["action"][i])
                    if action_i < 0 or action_i >= len(ds):
                        raise RuntimeError("elite replay contains illegal action")
                    bc_terms.append(-lp[action_i])
                if bc_terms:
                    bc_loss = torch.stack(bc_terms).mean() * a.bc_coef
                    actor_opt.zero_grad()
                    bc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(actor.parameters()), 1.0
                    )
                    actor_opt.step()
                    batch_losses.append(float(bc_loss.detach()))
                    total_bc_updates += 1
                    total_bc_decisions += len(bc_terms)
                    bc_used += len(bc_terms)
            if batch_losses:
                bc_mean_loss = float(np.mean(batch_losses))
            torch.save(actor.state_dict(), a.output)

        strategy_bc_mean_loss = 0.0
        strategy_bc_used = 0
        if (
            strategy_examples
            and a.strategy_bc_coef > 0.0
            and a.strategy_bc_max_examples > 0
        ):
            rng = np.random.default_rng(20260929 + ep)
            order = rng.permutation(len(strategy_examples))
            kind_total = max(1, len(strategy_kind_counts))
            kind_weights = {
                kind: min(
                    3.0,
                    len(strategy_examples) / float(kind_total * count),
                )
                for kind, count in strategy_kind_counts.items()
            }
            batch_losses: list[float] = []
            for start in range(0, len(order), 64):
                chunk = order[start : start + 64]
                teacher_terms = []
                for raw_i in chunk:
                    row = strategy_examples[int(raw_i)]
                    obs_row = torch.tensor(row["obs"], dtype=torch.float32)
                    ds = torch.tensor(row["descs"], dtype=torch.float32)
                    o = obs_row.repeat(len(ds), 1)
                    logits = actor.net(torch.cat([o, ds], 1)).squeeze(1)
                    lp = torch.log_softmax(logits, 0)
                    target = torch.tensor(
                        row["target_probs"],
                        dtype=torch.float32,
                    )
                    ce = -(target * lp).sum()
                    priority = min(3.0, max(1.0, float(row.get("priority", 1.0))))
                    balance = kind_weights[str(row.get("kind", "unknown"))]
                    teacher_terms.append(priority * balance * ce)
                if teacher_terms:
                    teacher_loss = (
                        torch.stack(teacher_terms).mean()
                        * a.strategy_bc_coef
                    )
                    actor_opt.zero_grad()
                    teacher_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(actor.parameters()), 1.0
                    )
                    actor_opt.step()
                    batch_losses.append(float(teacher_loss.detach()))
                    total_strategy_bc_updates += 1
                    total_strategy_bc_examples += len(teacher_terms)
                    strategy_bc_used += len(teacher_terms)
            if batch_losses:
                strategy_bc_mean_loss = float(np.mean(batch_losses))
            torch.save(actor.state_dict(), a.output)
            print(
                "PPO_V15_STRATEGY_BC",
                json.dumps(
                    {
                        "epoch": ep,
                        "examples": strategy_bc_used,
                        "mean_loss": strategy_bc_mean_loss,
                        "combat_policy": a.strategy_combat_policy,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        # Evaluate Critic globally against the same fixed rollout targets.
        epoch_value_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for f in files:
                with np.load(f, allow_pickle=False) as d:
                    obs = torch.from_numpy(
                        d["obs"].astype(np.float32)
                    )
                    epoch_value_predictions.append(
                        critic(obs).squeeze(1).numpy()
                    )
        epoch_ev = explained_variance(
            np.concatenate(epoch_value_predictions),
            np.concatenate(fixed_returns),
        )
        critic_improved = epoch_ev > best_critic_ev + 1e-9
        if critic_improved:
            best_critic_ev = epoch_ev
            best_critic_epoch = ep
            torch.save(critic.state_dict(), critic_output)

        epoch_rec = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "approx_kl": float(np.mean(kls)),
            "entropy": float(np.mean(entropies)),
            "anchor_kl": float(np.mean(anchor_kls)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "explained_variance": epoch_ev,
            "mean_shard_explained_variance": float(
                np.mean(shard_evs)
            ),
            "critic_checkpoint_improved": critic_improved,
            "best_critic_explained_variance": best_critic_ev,
            "elite_bc_decisions": bc_used,
            "elite_bc_mean_loss": bc_mean_loss,
            "strategy_bc_examples": strategy_bc_used,
            "strategy_bc_mean_loss": strategy_bc_mean_loss,
            "seconds": time.time() - t0,
        }
        history.append(epoch_rec)
        print(
            "TRAIN_EPOCH_V14",
            json.dumps(epoch_rec),
            flush=True,
        )

        (a.output.parent / "checkpoint.json").write_text(
            json.dumps(
                {
                    "schema": "sts1-armg-ppo-v14-checkpoint",
                    "epoch": ep,
                    "critic_loaded": critic_loaded,
                    "behavior_temperature": a.behavior_temperature,
                    "fixed_rollout_targets": True,
                    "best_critic_epoch": best_critic_epoch,
                    "best_critic_explained_variance": best_critic_ev,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

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
        "fresh_shards": len(fresh_files),
        "replay_shards": len(replay_files),
        "fresh_decisions": fresh_decisions,
        "replay_decisions": replay_decisions,
        "critic_loaded": critic_loaded,
        "critic_reset": critic_reset,
        "critic_health_explained_variance_before_reset": critic_health_ev,
        "behavior_temperature": a.behavior_temperature,
        "entropy_coef": a.entropy,
        "anchor_coef": a.anchor_coef,
        "bc_coef": a.bc_coef,
        "curriculum_strength": a.curriculum_strength,
        "curriculum_focus": curriculum_focus,
        "curriculum_failure_counts": curriculum_failure_counts,
        "curriculum_games": len(curriculum_floors),
        "elite_bc_total_updates": total_bc_updates,
        "elite_bc_total_decisions": total_bc_decisions,
        "strategy_bc_coef": a.strategy_bc_coef,
        "strategy_bc_source_examples": len(strategy_examples),
        "strategy_bc_examples_by_kind": dict(sorted(strategy_kind_counts.items())),
        "strategy_bc_total_updates": total_strategy_bc_updates,
        "strategy_bc_total_examples": total_strategy_bc_examples,
        "strategy_combat_policy": a.strategy_combat_policy,
        "actor_lr": a.lr,
        "critic_lr": a.critic_lr,
        "fixed_rollout_targets": True,
        "initial_ratio_mean": first_ratio_mean,
        "initial_ratio_abs_error": first_ratio_abs_error,
        "initial_critic_explained_variance": initial_ev,
        "best_critic_epoch": best_critic_epoch,
        "best_critic_explained_variance": best_critic_ev,
        "history": history,
    }
    a.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "SHARDED_TRAIN_V14_PASS",
        json.dumps(report),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
