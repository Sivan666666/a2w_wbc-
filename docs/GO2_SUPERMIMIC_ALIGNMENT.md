# Go2 SuperMimic learning configuration → A2W+Z1

2026-11-04 status override: the user stopped PS1 training before40K and selected
the25K checkpoint as the migration baseline. No A100 training is currently
authorized; see [current release](SUPERMIMIC_25K_RELEASE.md). Launch commands
and production status below are historical, not instructions to restart.

2026-11-03. User cancelled the old independent BPTT experiment and requested
Go2-aligned fresh training. No old actor, critic, dynamics or optimizer is loaded.
The old run and partial evaluations remain intact; they are not resumed.

## Reference and scope

Reference: https://github.com/Zhengtianjin/go2-supermimic at
`d9eee9b3f248e5300e636c433255db2c1c3e086a` (the previously reviewed pinned version).

This aligns the **learner**, not the Go2 chase/dodge/terrain task. Replacing A2W
with Go2, removing Z1, or replacing EE tracking with chasing balls is not this
request. This remains Isaac Lab/PhysX WheelV2 Precision, not a reproduction of
the paper or a claim that the simulators/tasks are identical.

`scripts/go2_aligned/upstream.py` contains verbatim TrainCfg, Actor, Critic,
clip_grads and lr_at. `world_model.py` contains verbatim RunningNorm, mlp and
WorldModel, with only its imports/dimension provider adapted. An AST test checks
these against the pinned checkout. No algorithm source from the old learner is
modified; the new learner only reuses the audited robot adapter/FK/history and
TD(lambda) utilities.

## Matched learning settings

| Setting | Go2 / new A2W |
|---|---|
| Seed | 0 (upstream default; historical PPO was seed42) |
| Environments | 4096 |
| Training iterations | 40000, one collection control step per iteration |
| Initial replay fill | 10 steps; eight separately counted A2W adapter validation steps |
| Model-only warmup | 300 iterations |
| Replay | 1000 steps/environment |
| WM network | 1024×1024×1024, SiLU |
| WM fitting | 2 updates/iteration, batch2048, autoregressive8 steps |
| WM samples | 25% prioritized, 25% archive if nonempty, remaining uniform |
| Rare archive | 100000 windows, keep probability .02 |
| Actor / Critic | 512/256/128 ELU + RunningNorm |
| Actor distribution | Deterministic during imagination; no learned std or entropy |
| Collection noise | Gaussian std .1; first25% envs OU std .3, theta2 |
| Imagination | batch8192, horizon32, gamma.99, lambda.95 |
| Critic fitting | 5 epochs × 4 shuffled minibatches |
| Target critic | None; same critic supplies bootstrap, as in Go2 |
| Learning rates | 2e-4; cosine to .03× over30000 iterations |
| Actor LR ramp | 1000 iterations after model warmup; scale1 |
| Adam | Actor beta(.9,.95), WM/Critic default(.9,.999) |
| Gradient clipping | Global norm1; Actor elementwise .1 additionally |
| Compilation | torch.compile imagined step, dynamic=False |
| Logging / save | Every50 /1000 iterations |
| Collapse guard | Two logged fall averages >3: restore Actor/Critic, reset their optimizer moments, halve LR scale, reramp; KEEP WM/replay |

No hidden automatic reduction of batch, environments, horizon or network size.
GPU-memory execution chunks accumulate the **full** Actor8192 gradient before
one Adam step. Critic still performs exactly20 updates; each minibatch may be
executed in memory chunks. A CPU test compares chunked and unchunked Actor
updates. Chunking changes reduction/RNG ordering, not effective batch/objective.

Production CLI asserts4096env/40000iterations/compile enabled. Smaller validation
runs are explicitly marked smoke and never used as experiment results.

## Necessary robot/task changes (not hidden as exact reproduction)

1. A2W22 actions: 12 leg positions,6 Z1 positions,4 wheel velocities. Preserve
   original action scales, clips, gains, delays and WheelV2 limits. Actor input
   remains234, three history frames, so deployment consumes the original interface.
   The exported actor includes its new RunningNorm; old bare-MLP weights are incompatible.
2. WM state86: height/gravity/yaw-frame linear velocity/body angular velocity,
   18 relative positions/22 velocities,4 hips/4 knees/4 wheel-center positions.
   Predict state increments + planar pose delta; no torque, acceleration, contact
   or termination prediction heads. Flat terrain gives77 zero-relief scan points.
3. Physics context766 retains A2W masses/inertias/COM/materials/gains/delays.
   Startup-static718 values are indexed by environment; reset-varying48 values
   are stored per transition. Do not reuse current gains for historical replay.
   Critic input884 includes physical state,10 commands,context and previous action
   (required by delayed actuators), instead of the old114-dimensional PPO critic.
4. Preserve actual Precision task rewards and EE ranges, not Go2 chase/dodge/hit
   reward weights. Imagined torque is an analytic, limited endpoint PD surrogate;
   acceleration is control-period velocity difference, as in Go2. These are NOT
   guaranteed equal to measured final-physics-substep torque/acceleration. The
   exact physical reward remains the collected/evaluated reward.
5. Analytic differentiable termination uses actual A2W tilt limit1rad with Go2
   softness5degrees, plus base-ground box-clearance sigmoid width.03m for A2W
   base_contact. Geometry is read from the URDF. It is a smooth approximation,
   not a force-sensor predictor. No learned survival head can override tilt.
6. Rare events follow Go2 contact changes/flight/vertical speed/fall rules. Wheel
   radius .09486137m is measured from collision STL radial vertices (x,z), replacing
   Go2 foot-site geometry. Terminal windows are masked beginning with the done
   transition as in Go2; pre-reset targets are explicitly stored in this adapter.
7. EE/velocity command samples remain fixed within each short imagination rollout;
   seeds are resampled until8192 have enough time before command/episode boundaries.
   No Go2 ball generator, driver/expert policy, dodge shaping or terrain curriculum
   applies to this EE task. No new EE curriculum/reward tuning is introduced.
8. Yaw-frame velocity conversion uses Euler R_y(pitch)R_x(roll), not shortest
   gravity-aligning rotation. Roundtrip and real-state parity are tested.

## Storage / execution

- Local source: `/home/sivan/whole_body/wbc/a2w_wbc_supermimic_20261102`.
- New PS1 source: `/mnt/sda/ps/leggedmanip-lab/supermimic_go2_aligned_20261103`.
- New entrypoint: `scripts/train_go2_aligned.py`.
- Runtime: `reproduction/go2_aligned_runtime.sh`, non-root read-only container,
  private caches; no shared CUDA/Python/environment changes.
- Only physical GPU3 is used; no unrelated tasks stopped.
- Old containers stopped by explicit user request: training
  `a2w-bptt-ppo-pair-seed42-20261102` and90K evaluation
  `a2w-bptt-stage90000-eval-20261103`. SIGTERM did not complete within30s;
  Docker stopped them with137. Existing saved checkpoints are retained; do not
  claim a final checkpoint was created on shutdown.

## Validation / known limits

Seven CPU tests pass: pinned AST parity, defaults/schedule, normalized TorchScript
export, chunked Actor-gradient equivalence, replay masks/context, yaw-frame
roundtrip, analytic torque/termination.

Full-size compiled smoke `runs/smoke_compiled_4096` completed with exit0:
4096 environments, WM batch2048, full Actor batch8192 (execution chunk8192),
horizon32, 305 iterations and5 actual Actor updates. Physical transitions1323008;
imagined transitions1310720; nonfinite_skipped0. Peak PyTorch allocation was
18115270656 bytes. Startup real-state/observation parity and normalized actor
export checks passed. This is only a pipeline test, not a trained-policy result.

The exported305 actor also completed a physical64env ×10s interface test
(`runs/smoke_export_eval`,32000 samples,exit0), all reported metrics finite,
zero failure resets. EE error remains49.55cm at this nearly untrained stage;
this must not be presented as a quality milestone or PPO comparison.

## Replacement production launch

Started2026-11-03 07:28:59UTC (15:28:59Asia/Shanghai):

- Container: `a2w-go2-aligned-train-seed0-20261103`.
- Run: `/mnt/sda/ps/leggedmanip-lab/supermimic_go2_aligned_20261103/runs/go2_aligned_seed0_20261103`.
- Seed0,4096env,40000 iterations,compiled execution chunk8192.
- Fresh random Actor/Critic/WM/optimizers; no smoke or historical weights loaded.
- Full budget is40018 physical control steps/env including validation/prefill,
  or163913728 transitions, not40000 PPO iterations.
- Budget is not automatically extended. No new PPO worker is launched.

Verified live production iteration301:319 physical control steps/env,
1306624 physical transitions,262144 imagined transitions, Actor updates1,
WM loss2.835516, Actor gradient norm.331016, Critic loss.183586,
nonfinite_skipped0; container running and OOMKilled=false. This confirms the
actual production Actor/Critic update path after300 WM-only iterations.
The30-minute monitor now points to this replacement run, not the cancelled one.

```bash
cd /mnt/sda/ps/leggedmanip-lab/supermimic_go2_aligned_20261103
bash scripts/repro_container.sh 3 a2w-go2-aligned-train-seed0-20261103 \
  bash /mnt/sda/ps/leggedmanip-lab/supermimic_go2_aligned_20261103/reproduction/go2_aligned_runtime.sh \
  scripts/train_go2_aligned.py \
  --output /mnt/sda/ps/leggedmanip-lab/supermimic_go2_aligned_20261103/runs/go2_aligned_seed0_20261103 \
  --num_envs 4096 --iterations 40000 --imag_chunk 8192 --headless --device cuda:0
```

Do not rerun this command while the named container/run exists. Inspect
`progress.jsonl`, `latest_checkpoint.json`, `complete.json`/`FAILED.json` first.

Validation runs are kept, including failed preflights: wrong container cwd,
unrelocated mesh paths (invalid USD quarantined), yaw conversion mismatch, and
missing non-root username for Inductor. These are not training results. Asset
relocation asserts that only mesh filenames change; no geometry changes.

Live simulation/full-size validation and production launch results are recorded
in the run manifests/status, not inferred from these unit tests. Per-run files
include config.json, source hashes, env.yaml, adapter validation, progress.jsonl,
TensorBoard, normalized actor exports, full optimizer checkpoints, rollback
events and COMPLETE/FAILED status. No automatic PPO or real-robot deployment.

Physical samples include startup/fill, even after any policy rollback. Imagined
samples are separate. Historical PPO10K remains the existing A100 evaluation;
different hardware, training seed and any shared evaluation interference must be
disclosed. Neither matching configuration nor passing tests proves convergence.
