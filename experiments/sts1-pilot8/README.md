# Eight-world noncombat pilot on public GitHub Actions

This isolated research workflow runs the user-approved 240-state pilot, not the full 2,068-state labelling task. Eight balanced shards each use two CPU workers. Combat stays MCTS 2000; G7 handles every continuation noncombat decision. No production weights are replaced.

The data cover 112 source runs, acts 1–3, and low/medium/high health. Run groups remain train 144 / validation 80 / development 16. Every action is compared in the same eight future worlds, including the G7 action. The public runner recomputes eight-world labels from transparent simulator JSON.

The teacher is a public-state-compatible stochastic continuation approximation. It preserves the current visible map, boss, rewards and encounter history, but resamples future RNG streams, unseen encounter suffixes and remaining relic order. It is NOT an exact posterior over original game seeds. Raw relic counters are captured for audit but excluded from model features. Raw human `.run` files are not published; source-balanced human training preferences appear only as an aggregate prior.

Preflight checks package hashes, G7 and native compatibility, exact source-run replay, public-state/action invariance after sampling, and finite nonlinear inputs. All eight shards must complete with 240 unique identity-matched labels before training. The ArmG-sized nonlinear predictor learns paired terminal win advantage over G7. Calibration retains the original coverage/positive-bound criteria. Only a calibrated candidate is frozen and tested on 300 fresh paired A0 seeds in the normal simulator; no stochastic-world mutation is used during the final evaluation. No calibrated gate means HOLD_G7, not a claim of improvement.

Known historical seeds and world IDs are excluded; complete private historical exclusions cannot be proven. These remain research metrics, not formal promotion or real-client verification. Each shard is capped at three hours, preflight at 20 minutes, and final training/evaluation at 150 minutes; Actions queue delays are outside those caps. A failed/timed-out shard retains available evidence and blocks downstream training.

Runtime: Python 3.12, Torch 2.14.1+cpu, native simulator upstream 7476a81954020087da31d41d16fddf475746ec2d with existing binding hooks plus independent-map clone and public-state/stochastic-future bindings. G7 SHA256: 8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0. GitHub compiles the extension from pinned public sources and the auditable binding overlay. G7 is read from an already-public Actions artifact (run 36674639174, artifact 11079234863) and checked by SHA256; no opaque checkpoint or extension is committed to Git. Compatibility is explicitly tested on ubuntu-24.04 before shards start. Source overlay and patches are included for audit.

Upstream credits: [sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed), [sts-rl-agent](https://github.com/Jialeiv/sts-rl-agent), public human preference records from [MaT1g3R/Slay-the-Spire-data](https://github.com/MaT1g3R/Slay-the-Spire-data) and [Baalorlord](https://baalorlord.tv/). Native and ArmG upstream code use MIT licenses.

## Local result before public execution

The local 240-state pilot has completed and its advantage calibration failed: HOLD_G7_NO_CALIBRATED_ADVANTAGE. Public execution reproduces the same frozen data/protocol on parallel runners; it is not an independent performance test or a reason to relax thresholds. Full-scale labelling is not started by this workflow. See local-pilot-result.json.

Training and calibration traverse labels in stable case-ID order for reproducibility across parallel workers. The local result used its original file order; this ordering fix is not a new independent performance holdout.


Replay recovery (2026-10-13): original Actions run 38266942068 completed six shards,
but shards 5 and 7 stopped on historical map argmax differences. Local replay of
all 28 affected source runs recovered all 63 selected card states. Six map choices
had identical model descriptors and score gaps of at most 9.54e-7. Historical
reconstruction now accepts the recorded legal map index only for identical
descriptors and finite scores within 1e-5, logging every exception. Other journal
mismatches still stop the run; card states, action sets, teacher choices, observation
vectors and descriptors remain exact checks. This exception reconstructs history;
it does not change G7 decisions in future rollouts or evaluation.

Recovery reuses successful original shards 0, 1, 2, 3, 4 and 6, recomputes 5 and 7,
and requires all 240 states and eight complete shard summaries before training.
The architecture, calibration thresholds, paired worlds, and evaluation criteria
are unchanged. Replay diagnosis is diagnostic evidence, not a fresh win-rate test.
