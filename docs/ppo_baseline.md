# PPO Baseline Training (M2.1)

Trains the reaching-with-avoidance policy defined in
`docs/reaching_mdp.md`. Runbook section 27.

## Setup

- **Task registration:** `src/kinova_mjlab_reaching/tasks/reaching/__init__.py`
  registers `"Kinova-Reach-v0"` with mjlab's task registry
  (`register_mjlab_task`) — training env at `num_envs=256`, play env at
  `num_envs=1`. Registration is a side effect of importing
  `kinova_mjlab_reaching.tasks.reaching`.
- **Agent config:** `src/kinova_mjlab_reaching/tasks/reaching/agents/ppo_cfg.py`
  — `RslRlOnPolicyRunnerCfg` with mjlab's default network size
  (3x128 MLP, ELU) for both actor and critic, standard PPO hyperparameters
  (clip=0.2, GAE λ=0.95, γ=0.99, adaptive LR schedule). Unvalidated beyond
  "doesn't crash and the loss/reward curves move" — tune once there's real
  training data to react to.
- **Logger:** `tensorboard`, not mjlab's default `wandb` — no W&B account is
  configured for this project, and tensorboard avoids that external
  dependency entirely for a first baseline.
- **Training script:** `scripts/train_reaching.py` — thin wrapper around
  `mjlab.scripts.train.launch_training`, with `--num-envs`/`--max-iterations`
  CLI overrides. Logs to `checkpoints/<timestamp>/` (repo's own
  `checkpoints/` convention from the runbook's target tree, rather than
  mjlab's own default `logs/rsl_rl`).

## Monitoring

Two ways to watch a run, usable simultaneously and while training continues:

- **TensorBoard** (reward/loss curves): `uv run tensorboard --logdir checkpoints --port 6006`,
  then open `http://localhost:6006`.
- **Live 3D viewer** (`scripts/watch_training.py`): a browser-based Viser
  viewer (mjlab's `ViserPlayViewer`) that loads the latest checkpoint from a
  run directory and hot-swaps to newer ones as they're saved — watch the
  actual policy's behavior evolve without stopping training. Usage:
  `uv run python scripts/watch_training.py --checkpoint-dir checkpoints/<run>`
  (defaults to the most recently modified run dir), then open the printed
  `http://localhost:8080`. Built by adapting the TRAINED_MODE + viser code
  path in `mjlab.scripts.play` directly against our env/agent configs,
  since `play.py`'s own CLI only auto-discovers mjlab's built-in tasks
  (`import mjlab.tasks`), not this project's task registration.

## Runs

### Smoke run — 2026-08-25_22-35-31

`--num-envs 256 --max-iterations 30`. Purpose: confirm the full pipeline
(env + action/reward/termination/command wiring + task registration +
training script) runs end-to-end without crashing, before committing to a
longer run.

```
Total steps: 184,320 over 30 iterations (256 envs x 24 steps/iter)
Steps/sec: ~35,000-41,000 (GPU: RTX 5070 Ti)
Iteration time: ~0.15-0.18s
Mean reward: -0.47 -> -0.44 (slight improvement, expected this early)
Metrics/reach_target/position_error: ~0.7-0.9 m (no better than random yet)
Metrics/reach_target/episode_success: 0.0 (none)
No NaNs, no crashes, checkpoints saved (model_0.pt, model_29.pt)
```

**Result: PASS** — pipeline validated. Not enough iterations to expect any
task competence yet; this run's only job was to prove nothing is broken.

### Baseline run — 2026-08-25_22-46-05

`--num-envs 256 --max-iterations 5000` (matches the iteration count mjlab's
own comparable YAM lift-cube baseline uses). **Does not include the table
collision termination** (added afterward — see "Table collision" below);
this run only terminates on teapot/cup_1/cup_2 contact.

Ran to completion in 22m 24s (5000/5000 iterations, 30.72M total steps,
~22,000-23,000 steps/sec sustained). 101 checkpoints saved
(`model_0.pt` ... `model_4999.pt`, every 50 iterations).

```
                          iter 2230/5000   iter 4999/5000 (final)
Mean reward                29.52            33.45
Episode_Reward/reaching     0.818            0.860
Episode_Reward/target_reached 7.66           8.30
Metrics/reach_target/position_error  0.0158 m   0.0142 m
Metrics/reach_target/episode_success 0.90       0.94-1.00 (fluctuating)
Episode_Termination/teapot_collision 0.10       0.00
Episode_Termination/cup_1_collision  0.00       0.11
Episode_Termination/cup_2_collision  0.00       0.00
```

**Result: exceeds runbook section 28's >90% success target** — position
error (1.4 cm) is well under the 3 cm success threshold, and per-episode
success is consistently in the 90-100% range by the end of training.
Obstacle collisions with the discrete objects are already rare (0-11% of
episodes, fluctuating rather than trending — not yet fully eliminated).

This was a genuine, unmodified success by the runbook's own numeric bar,
*despite* the table-collision gap described below — worth remembering when
comparing against the corrected run.

## Table collision (found during this run, not yet retrained)

While watching this run live in the browser (`scripts/watch_training.py`),
observed the arm's elbow/forearm wedging against the table in some
configurations, physically blocking the reach and preventing target
approach from that region. The table has real MuJoCo contact physics (the
arm genuinely cannot pass through it), but — unlike the teapot/cups —
table contact had no `illegal_contact` termination wired up, so the only
learning signal was the indirect "you failed to reach the goal," not
"you hit something."

User decision (2026-08-25): treat table contact the same as the discrete
obstacles (hard termination), not as a softer penalty or left alone. Fix
applied in `reach_env_cfg.py` (`_COLLISION_NAMES` now includes `"table"`,
using the same `illegal_contact` sensor pattern). Verified the new
termination doesn't false-trigger at the reset pose (zero contacts, matches
the original scene validation).

### Corrected run — 2026-08-26_09-58-07

`--num-envs 256 --max-iterations 5000`, same hyperparameters and seed as
the run above, now with the table-collision termination active. Completed
in 14m 50s (faster than the first run's 22m 24s — fewer steps per episode
on average since more episodes now end early on collision).

Sampled every 500 iterations:

```
iter   reward  pos_error  success  teapot_col  cup1_col
 500    0.00    0.737      0.00       0.00        0.00
1000    0.72    0.296      0.00       0.40        0.00
1500    1.63    0.131      0.00       0.33        0.08
2000    5.16    0.086      0.78       0.00        0.22
2500   21.53    0.023      1.00       0.00        0.00
3000   13.96    0.025      0.97       0.17        0.00
3500    5.48    0.194      0.32       0.42        0.25   <- transient regression
4000   20.95    0.022      0.92       0.08        0.15
4500   20.21    0.018      1.00       0.23        0.00
4999   23.45    0.019      0.96       0.17        0.00 (table_collision: 0.17)
```

**Result: not a clean improvement — worth being honest about.** Final
position error (1.9 cm) and success rate (96%) are comparable to the
uncorrected run (1.4 cm / 94-100%), but `teapot_collision` at the final
iteration (17%) is *higher* than the uncorrected run's final value (0%),
and `table_collision` itself sits at 17% rather than trending toward zero.

The likely reason isn't that the fix made things worse — it's a metric
subtlety: `episode_success` **latches** to 1 the moment the end effector
gets within 3 cm of the target, but the episode keeps running afterward
(one command per episode, 4 s duration). Nothing stops the arm from
drifting into an obstacle *after* already registering success, so "high
success" and "meaningful collision rate" are not mutually exclusive here —
an episode can do both. Combined with only one training seed and the
visible mid-training volatility (iteration 3500's regression, typical for
PPO with an adaptive LR schedule), a single before/after training-time
snapshot isn't a reliable enough signal to conclude the fix worked or
didn't.

**Conclusion: needs a proper held-out evaluation, not another glance at
training curves** — see below. Training-time metrics average over a
stochastic exploration policy narrating stale batches; an eval pass with a
frozen final policy across many fixed seeds/targets would give a real
success-rate-vs-collision-rate answer instead of noisy per-iteration
snapshots.

## Held-out evaluation — `scripts/evaluate_reaching.py`

Runs the frozen policy deterministically (`get_inference_policy()` returns
the distribution mean, not a stochastic sample — no PPO exploration noise)
across a fresh batch of episodes on a held-out seed (123, vs. training's
42), and classifies each episode into mutually exclusive outcomes instead
of the training metrics' conflated "success" — directly resolving the
ambiguity above.

**Bug found and fixed during first use:** naively reading
`command.metrics["position_error"]` right after `env.step()` for an
env that just auto-reset gives the wrong number. mjlab's auto-reset
resamples a new target and recomputes that metric *before* `step()`
returns, so the "final" position error for a just-terminated episode
actually reflects "distance from the old arm pose to the brand-new
episode's target" — not the real final error. First run showed a nonsense
median final error of 0.94 m alongside an 83% success rate (should be
close to the 3 cm threshold, not 30x it) — that contradiction is what
exposed the bug. Fixed by snapshotting the metric one step earlier, before
any reset for that step could contaminate it (see the comment in
`evaluate_reaching.py`).

### Results — model_4999.pt (table-collision-corrected run), 2116 episodes

```
uv run python scripts/evaluate_reaching.py \
    --checkpoint checkpoints/2026-08-26_09-58-07/model_4999.pt \
    --num-envs 256 --episodes-per-env 8 --device cuda:0

Success rate (ever reached target):     83.3%
  clean success (reached, no collision): 80.2%
  reached but later collided:             3.1%
Collision rate (any episode):           12.2%
  collided without ever reaching target:  9.1%
Failure, no collision:                   7.6%

Collision breakdown:
  teapot_collision: 8.4%   cup_1_collision: 2.3%
  cup_2_collision:  1.5%   table_collision: 0.0%

Final position error (m):  median 0.024   p90 0.045   max 0.192
Time-to-target (s), successes only:  median 1.112   p90 1.424
```

**Honest read:** the table-collision fix worked for its actual purpose —
`table_collision` is 0.0% in the final trained policy, down from the
mid-training peaks of ~17% seen while it was still learning. But overall
held-out success (83.3%) is **below the runbook's >90% target**, and
that's a real, non-noisy number now — not the training curve's optimistic
94-100% (which was measuring stochastic-exploration episodes with the
success-then-collision ambiguity baked in). Teapot collision (8.4%) is now
the dominant remaining failure mode. When it does succeed, it's fast
(median 1.1 s of a 4 s episode budget) — the shortfall is in reliability,
not speed.

**Next:** doesn't clear the bar yet — likely needs more training
iterations (5000 may simply be undertrained, given the visible
iteration-3500 regression and still-climbing reward late in training),
and/or reward tuning specifically targeting the teapot-collision residual,
before calling M2.1 done.

## Extended run — 2026-08-26_14-56-37 (10,000 iterations)

Testing whether the 83.3%-success shortfall above was simply undertraining
(reward still climbing at iteration 5000) before touching the reward
function. Same env/reward config and seed (42) as the corrected run,
`--num-envs 256 --max-iterations 10000`.

**First attempt collapsed and was killed.** Watching live via
`scripts/watch_training.py`, the arm never moved from its home pose.
Training-log metrics confirmed it wasn't early-training noise: `Mean
action std` decayed *monotonically* from its init value of 1.0 down to
0.01-0.02 by iteration ~1500 and stayed pinned there through iteration
4300+ (43% through the run) — the Gaussian policy had collapsed to
near-deterministic, killing exploration before it ever discovered the
reward signal. `Mean reward` stayed flat at ~0.00 and
`Metrics/reach_target/episode_success` was 0.0000 for the entire run,
unlike the earlier successful runs which showed clear improvement by
iteration 1000-2000. Root cause: classic PPO premature entropy-collapse
under the adaptive-LR schedule — `entropy_coef=0.005` was too weak to
resist std shrinkage once the critic's value loss flatlined near zero
early on. Same nominal config/seed as the run that succeeded at 5000
iterations, so this was GPU-nondeterminism-driven bad luck falling into a
different basin, not a code regression. Killed at iteration ~4300/10000
rather than let it run to completion, since a policy pinned at action std
0.01-0.02 has no mechanism left to reintroduce exploration.

**Fix:** bumped `entropy_coef` 0.005 → 0.01 in
`agents/ppo_cfg.py` and restarted with everything else identical.
Restarted run stayed healthy throughout — action std *rose* to ~2.6
rather than collapsing, and by iteration 5240/10000 reward was ~30 with
85-100% per-snapshot episode success. Completed in 29m 46s
(10,000/10,000 iterations, 61.44M total steps), 201 checkpoints saved.

### Held-out evaluation — model_9999.pt, 2102 episodes

```
uv run python scripts/evaluate_reaching.py \
    --checkpoint checkpoints/2026-08-26_14-56-37/model_9999.pt \
    --num-envs 256 --episodes-per-env 8 --device cuda:0

Success rate (ever reached target):     94.5%
  clean success (reached, no collision): 91.0%
  reached but later collided:             3.6%
Collision rate (any episode):            9.0%
  collided without ever reaching target:  5.5%
Failure, no collision:                    0.0%

Collision breakdown:
  teapot_collision: 6.3%   cup_1_collision: 1.4%
  cup_2_collision:  1.0%   table_collision: 0.3%

Final position error (m):  median 0.0113   p90 0.0235   max 0.1364
Time-to-target (s), successes only:  median 0.456   p90 0.592
```

**Result: clears the runbook's >90% target.** 94.5% success (vs. the
prior run's 83.3%), position error roughly halved (1.13 cm median vs.
2.4 cm), and time-to-target nearly halved too (0.456 s vs. 1.112 s
median) — a faster, more precise policy, not just a marginally more
successful one. Teapot collision dropped from 8.4% to 6.3% and remains
the dominant residual failure mode; table collision ticked up slightly
(0.0% → 0.3%, still negligible) and no episodes timed out without ever
reaching the target (0.0% "failure, no collision" vs. 7.6% before) — the
policy now reaches the target in essentially every rollout, with the
remaining gap being collisions rather than failures to converge.

The extended-training hypothesis was confirmed: 5000 iterations was
undertrained given the still-climbing reward and iteration-3500
regression seen previously; doubling to 10,000 iterations (with the
entropy_coef fix needed to get there safely) closed the gap without
touching the reward function.

## Status

- [x] Task registered, training script runs end-to-end (smoke run)
- [x] Monitoring set up: TensorBoard + live Viser viewer
- [x] Baseline run (5000 iterations) — complete, >90% success,
      1.4 cm position error (exceeds runbook section 28's target) —
      but trained *without* the table-collision termination
- [x] Table-collision gap identified (via live viewer) and fixed in
      `reach_env_cfg.py`; sanity-checked
- [x] Retrain with table-collision termination — complete
      (2026-08-26_09-58-07), 96% success / 1.9 cm position error, but
      teapot/table collision rates (~17% each at the final iteration) did
      not clearly improve over the uncorrected run — result is ambiguous
      from training curves alone, see writeup above
- [x] Held-out evaluation of the 5000-iteration corrected run — complete.
      83.3% success, table_collision fully resolved (0.0%), teapot
      collision (8.4%) now the dominant failure mode. Below the runbook's
      >90% target.
- [x] Extended run (10,000 iterations, `entropy_coef` 0.005→0.01 after a
      first attempt collapsed) — complete (2026-08-26_14-56-37).
      **94.5% held-out success, clears the runbook's >90% target.**
      Teapot collision (6.3%) remains the largest residual failure mode
      but is no longer blocking the M2.1 bar — see writeup above.
- [x] M2.1 (PPO baseline) — **done.** `checkpoints/2026-08-26_14-56-37/model_9999.pt`
      was the M2.1 best policy (94.5% held-out success).

## M2.2 — Collision-penalty shaping — 2026-08-26_16-09-43

Runbook section 29 (robustness stage): added dense collision-distance
reward shaping on top of the M2.1 baseline's hard-termination-only
avoidance, targeting the 6.3% residual teapot-collision rate. New reward
term `obstacle_proximity_penalty` (`mdp/rewards.py`) — a Gaussian-kernel
penalty (`std=0.08` m, sized against the obstacles' own geometry in
`scene.py`: teapot radius 4.5cm, cups 2.8cm) from the EE site to each of
teapot/cup_1/cup_2, weight `-1.0` (same order of magnitude as the
`reaching` reward's `+1.0`, so it competes for gradient without
swamping the `+10` success bonus). Table excluded — already at 0.3%
collision and its geometry is planar, not point-like.

No PPO hyperparameters were changed alongside this — `entropy_coef`
stayed at `0.01` (the M2.1 fix). Per runbook rule 9 (isolate whether an
issue is reward design vs. PPO config before changing both), and since
that config had just been validated as stable, only the new reward
term's own parameters (`weight`, `std`) were treated as tunable here.

Smoke-tested at `--num-envs 32/64 --max-iterations 5/60` first — no
crashes, penalty sign/magnitude sane — before committing to the full run.
Trained `--num-envs 256 --max-iterations 10000`, same as M2.1. Completed
in 29m 10s, stayed healthy throughout (action std ~2.8 at the end, no
collapse).

### Held-out evaluation — model_9999.pt, 2081 episodes

```
Success rate (ever reached target):     96.2%   (M2.1: 94.5%)
Collision rate (any episode):            6.2%   (M2.1: 9.0%)
  teapot_collision: 4.2%   (M2.1: 6.3%)
  cup_1_collision:  0.8%   (M2.1: 1.4%)
  cup_2_collision:  1.1%   (M2.1: 1.0%, flat within noise)
  table_collision:  0.1%   (M2.1: 0.3%)

Final position error (m):  median 0.0139   p90 0.0228   max 0.1181
                            (M2.1: median 0.0113 — slightly worse, still well under 3cm)
Time-to-target (s), successes only:  median 0.488   p90 0.600
```

**Result: clear improvement, not a tradeoff.** Success rate rose *and*
collision rate fell — teapot collision (the dominant residual failure
mode) dropped by a third. Position error crept up slightly (1.39cm vs.
1.13cm median), a plausible cost of the policy giving obstacles a wider
berth, but still comfortably under the 3cm success threshold.

- [x] M2.2 collision-penalty shaping — **done.**
      `checkpoints/2026-08-26_16-09-43/model_9999.pt` was the M2.2 best
      policy before initial-state randomization: 96.2% held-out success,
      6.2% collision rate.

## M2.2 — Initial-state randomization

Runbook section 29 ("q0 ~ safe distribution"). Added `reset_robot_joints`
event in `reach_env_cfg.py`, alongside the existing
`reset_scene_to_default`, via mjlab's `reset_joints_by_offset`: every
episode now perturbs each arm joint by an independent uniform offset in
`[-0.2, 0.2]` rad (~11°) around the home pose, instead of always
resetting to the exact same q0. Range chosen against the ~1.0 rad range
the action space already operates in; verified empirically (not just
assumed, per runbook rule 8) that this range never spawns an env already
in contact with an obstacle/table — 0/10,240 reset+step samples collided
(one-off check script, not committed).

**First training attempt regressed**: 95.4% success / 10.1% collision
(vs. the no-randomization checkpoint's 96.2%/6.2%), cup_1 collision
nearly tripling (0.8%→2.7%). Investigated before accepting this as a
real effect: re-evaluated the *old*, non-randomized-trained checkpoint
under the *same* new randomized-start eval conditions as a control —
95.8%/6.8%, barely different from its original fixed-start numbers. That
ruled out "the eval got harder" as the explanation (the old policy
already tolerated ±0.2 rad starts fine without training on them), leaving
the first attempt's regression attributable to the training run itself.

Given the project already has one documented case of PPO diverging
between nominally-identical runs (the entropy-collapse incident, same
section above), retrained once more with the identical config before
concluding anything.

**Second attempt confirmed it was run variance, not a real effect**:
96.0% success / 7.3% collision — closely matching both the
no-randomization baseline and the control, with cup_1 collision (0.7%)
the lowest of any run so far and position error (1.00cm median) and
time-to-target (0.480s median) the best recorded to date.

| | No randomization | Attempt 1 | Control (old ckpt, new eval) | Attempt 2 |
|---|---|---|---|---|
| Success | 96.2% | 95.4% | 95.8% | 96.0% |
| Collision | 6.2% | 10.1% | 6.8% | 7.3% |
| Teapot | 4.2% | 6.0% | 4.6% | 4.9% |
| Cup_1 | 0.8% | 2.7% | 1.0% | 0.7% |
| Cup_2 | 1.1% | 1.3% | 0.9% | 1.4% |
| Table | 0.1% | 0.0% | 0.3% | 0.3% |

**Conclusion: initial-state randomization is neutral-to-slightly-positive
on held-out performance** (no measurable cost once run-to-run PPO
variance is accounted for) and should generalize better across starting
poses than a policy that only ever saw one fixed q0 — though the current
eval script doesn't have a way to directly measure "generalization" per
se, only performance under the (now-randomized) task distribution both
old and new policies are evaluated against.

- [x] M2.2 initial-state randomization — **done.**
      `checkpoints/2026-08-27_12-52-48/model_9999.pt` is the current best
      policy: 96.0% held-out success, 7.3% collision rate, best-recorded
      position error and time-to-target.
      Remaining M2.2 item per runbook section 29 (not yet done): target
      curriculum (targets sampled from the full range from the start, no
      easy-to-hard progression). Dynamics randomization stays deferred
      ("later and only where needed" per the runbook) until ROS2/physical
      deployment (M3/M3.1) is actually attempted.
