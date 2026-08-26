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
- [x] Held-out evaluation (`scripts/evaluate_reaching.py`) — complete.
      **83.3% success, table_collision fully resolved (0.0%), teapot
      collision (8.4%) now the dominant failure mode. Below the runbook's
      >90% target** — M2.1 is not done; see writeup above for next steps
      (more iterations and/or reward tuning targeting teapot avoidance).
