from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
REPLAY = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v14_replay.py"


def write_shard(
    path: Path,
    *,
    checkpoint: str,
    seeds: list[int],
    temperature: float,
    floors: list[int],
    victories: list[bool],
) -> None:
    obs = []
    desc = []
    counts = []
    action = []
    reward = []
    old_logp = []
    done = []
    game_seed = []
    temps = []
    wins = []
    final_floor = []
    for gi, seed in enumerate(seeds):
        for di in range(2):
            obs.append(np.full(412, seed + di, np.float32))
            candidates = np.stack(
                [
                    np.full(368, seed + di + 0.1, np.float32),
                    np.full(368, seed + di + 0.2, np.float32),
                ]
            )
            desc.append(candidates)
            counts.append(2)
            action.append(di % 2)
            reward.append(0.1 if di == 0 else floors[gi] / 50.0)
            old_logp.append(-0.69314718)
            done.append(di == 1)
            game_seed.append(seed)
            temps.append(temperature)
            wins.append(victories[gi])
            final_floor.append(floors[gi])
    offsets = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(np.asarray(counts, np.int32), out=offsets[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=np.asarray(obs, np.float32),
        desc=np.concatenate(desc, axis=0).astype(np.float32),
        offsets=offsets,
        counts=np.asarray(counts, np.int32),
        action=np.asarray(action, np.int64),
        reward=np.asarray(reward, np.float32),
        old_logp=np.asarray(old_logp, np.float32),
        done=np.asarray(done, np.bool_),
        game_seed=np.asarray(game_seed, np.int64),
        behavior_temperature=np.asarray(temps, np.float32),
        game_victory=np.asarray(wins, np.bool_),
        game_final_floor=np.asarray(final_floor, np.int16),
        checkpoint_id=np.asarray([checkpoint]),
    )


def run_builder(
    fresh: Path,
    state: Path,
    out: Path,
    *,
    checkpoint: str,
    round_index: int,
) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(REPLAY),
            "--fresh-dir",
            str(fresh),
            "--state-dir",
            str(state),
            "--output-dir",
            str(out),
            "--checkpoint-id",
            checkpoint,
            "--round-index",
            str(round_index),
            "--max-ppo-games",
            "20",
            "--replay-fraction",
            "0.5",
            "--max-elite-games",
            "20",
            "--elite-min-floor",
            "45",
        ],
        check=True,
    )
    return json.loads((out / "replay-manifest.json").read_text())


def test_same_parent_replay_is_bounded_and_preserves_temperature(tmp_path: Path):
    state = tmp_path / "state"

    first = tmp_path / "fresh1"
    write_shard(
        first / "shard_00.npz",
        checkpoint="parent-A",
        seeds=[1, 2, 3, 4],
        temperature=1.12,
        floors=[30, 46, 50, 20],
        victories=[False, False, True, False],
    )
    m1 = run_builder(first, state, tmp_path / "out1", checkpoint="parent-A", round_index=1)
    assert m1["train_replay_games"] == 0
    assert m1["ppo_pool_games"] == 4
    assert m1["elite_pool_games"] == 2

    second = tmp_path / "fresh2"
    write_shard(
        second / "shard_00.npz",
        checkpoint="parent-A",
        seeds=[5, 6, 7, 8],
        temperature=0.98,
        floors=[35, 44, 45, 10],
        victories=[False, False, False, False],
    )
    m2 = run_builder(second, state, tmp_path / "out2", checkpoint="parent-A", round_index=2)
    assert m2["train_replay_games"] == 2
    assert m2["train_replay_fraction_of_fresh_games"] == 0.5
    with np.load(tmp_path / "out2" / "replay_00.npz", allow_pickle=False) as d:
        assert set(np.round(d["behavior_temperature"], 2)) == {1.12}
        assert set(map(int, d["game_seed"])).issubset({1, 2, 3, 4})


def test_parent_change_rejects_ppo_replay_but_keeps_elite_archive(tmp_path: Path):
    state = tmp_path / "state"

    first = tmp_path / "fresh1"
    write_shard(
        first / "shard_00.npz",
        checkpoint="parent-A",
        seeds=[10, 11],
        temperature=1.0,
        floors=[50, 48],
        victories=[True, False],
    )
    run_builder(first, state, tmp_path / "out1", checkpoint="parent-A", round_index=1)

    second = tmp_path / "fresh2"
    write_shard(
        second / "shard_00.npz",
        checkpoint="parent-B",
        seeds=[20, 21],
        temperature=1.0,
        floors=[20, 25],
        victories=[False, False],
    )
    m2 = run_builder(second, state, tmp_path / "out2", checkpoint="parent-B", round_index=2)
    assert m2["stale_ppo_games_rejected"] == 2
    assert m2["train_replay_games"] == 0
    assert m2["elite_pool_games"] == 2
    assert not (tmp_path / "out2" / "replay_00.npz").exists()
    with np.load(state / "elite-replay.npz", allow_pickle=False) as d:
        assert "parent-A" in set(map(str, d["source_checkpoint_id"]))


def test_replay_pool_keeps_complete_episode_boundaries(tmp_path: Path):
    state = tmp_path / "state"
    fresh = tmp_path / "fresh"
    write_shard(
        fresh / "shard_00.npz",
        checkpoint="parent-A",
        seeds=[31, 32, 33],
        temperature=1.04,
        floors=[40, 41, 42],
        victories=[False, False, False],
    )
    run_builder(fresh, state, tmp_path / "out", checkpoint="parent-A", round_index=3)
    with np.load(state / "ppo-replay.npz", allow_pickle=False) as d:
        for seed in set(map(int, d["game_seed"])):
            idx = np.flatnonzero(d["game_seed"] == seed)
            assert int(d["done"][idx].sum()) == 1
            assert bool(d["done"][idx[-1]])
