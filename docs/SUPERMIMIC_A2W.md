# A2W + Z1: from-scratch learned-dynamics BPTT vs PPO

Date: 2026-11-02. Experimental research code, **not a deployable robot controller**.
No Precision 50K, expert policy, demonstrations or pretrained world model is loaded.
Precision names the existing **task configuration**, not policy initialization.

## Sources / scope

Base repository snapshot: `a320da79159c2b67e096511b2b126e41e7007d0a`.
Read-only reference checkouts inspected:

| Reference | Commit | Relevant design |
|---|---|---|
| https://github.com/zita-ch/supermimic_vibed | `0525a23a2dcfa61f8543f98b84c075fe66d9d286` | Learned dynamics, differentiable reward/FK, BPTT, value bootstrap |
| https://github.com/Zhengtianjin/go2-supermimic | `d9eee9b3f248e5300e636c433255db2c1c3e086a` | Online non-imitation task learning from scratch |
| https://github.com/uwvwko-zzz/go2w-supermimic | `a299ed5f5571da4492b8a286d94acfe4a86128e1` | Mixed wheel/leg control, replay windows, frozen model / input gradients |

The first reference explicitly calls itself an unofficial reproduction. These
repositories differ in tasks, simulator bindings, observation conventions,
rewards and network sizes. This is an independent A2W adaptation of their common
learned-world-model/BPTT idea, **not identical code or a paper reproduction claim**.
No third-party source tree or simulator was copied into the project.

## Unchanged task and deployment interface

Original Isaac Sim 5.1 / Isaac Lab / PhysX task, asset, reward functions, command
sampling, randomizations and actuator limits remain unchanged. PPO source files
are untouched. The task is `A2W-Z1-WBC-WheelV2-Precision`:

- Actor: 234 values, three term-wise history frames; MLP 512/256/128, ELU.
- Critic: original privileged 114 values, same hidden sizes.
- Actions: FR/FL/RR/RL leg position (12), Z1 position (6), wheel velocity (4).
- Position: `clip(default_q + 0.25*a, -10, 10)`.
- Wheel: `clip(5*a, -100, 100)`, physical speed 50 rad/s, Kp/Kd 0/0.6, 20 Nm.
- Control 50 Hz, physics 200 Hz, delayed PD with randomized 0..4 physics-step delay.
- EE x 0.10..0.85 m; retain native mixed-frame XY/link0 and world-height Z convention.
- Actor inference is a single MLP; world model and privileged parameters are training-only.

No image encoder, RNN, door interaction, new height command, or real robot bridge
is introduced. Gripper remains fixed. No real-robot traffic is sent.

## Adaptation implementation

`scripts/supermimic/core.py`:

- Heading-invariant physical state: height 1, gravity 3, body linear/angular
  velocity 6, leg/arm angles 18, all joint velocities 22 = 50.
- Output-only auxiliary targets: measured reward-boundary torque 22,
  acceleration 22, wheel contact magnitudes 4; total target dimension 98.
- Residual world model 3x512 SiLU; normalized inputs and output increments;
  normalized gravity; learned failure-logit head. Auxiliary predictions receive
  supervised losses, not endpoint-PD approximations.
- Inputs include current/previous effective actions, episode age, randomized
  masses/inertias/COM/materials, gains and per-actuator delays (766 context values
  for this asset). Actual action target clipping is differentiable in the model
  input path. The raw action is retained for the original action-rate cost.
- Previous control action plus age and delay suffice for this control-period
  boundary because max actuator delay <= four physics substeps. Longer configured
  delays fail fast rather than silently discarding their queue.
- 8-step autoregressive world-model loss. Include the first terminal transition;
  exclude all transitions after that reset from the window. Finite masks are applied
  before evaluating invalid data, not by multiplying NaN by zero.
- Replay is a time-major GPU ring. Last 10% of environment IDs are held out from
  world-model fitting, normalization statistics and policy seed sampling.

`scripts/supermimic/adapter.py`:

- Instrument this environment instance's reward computation to capture actual
  post-physics state **before** automatic reset and command resampling.
- XML-derived differentiable FK includes all fixed mounts and the actual EE link.
- Construct a differentiable state proxy and call **the original 24 reward
  functions** and observation functions. No reward coefficient is copied by hand.
- Keep original boolean reward gates/clips. Learned failure continuation is a
  differentiable training approximation; real PhysX termination is unchanged.
- Imagined rollouts do not cross a command resampling or episode timeout boundary.

`scripts/supermimic/learner.py`:

- Random Actor/Critic/world model; Gaussian exploration starts with std=1.
- Reparameterized Gaussian actions during imagination and collection, unsquashed
  means; learned std bounded to exp([-3,1]). The small entropy regularizer and
  gradient clipping are explicit algorithm settings, not a PPO update.
- Freeze dynamics/target-critic weights, **not** their action/state derivatives.
- Full discounted imagined return + terminal value updates Actor. Detached
  TD(lambda) targets update Critic; EMA target critic (0.02).
- This is pure learned-dynamics BPTT policy optimization, **not PPO+BPTT blending**.

## Initial training configuration

| Setting | BPTT | PPO control |
|---|---|---|
| Training seed | 42 | 42 |
| Environments | 4096 | 4096 |
| Initial Actor/Critic | Random | Exact copies of BPTT step-zero random weights |
| Physical control steps/env | 120000 including warmup | 5000 iterations x 24 = 120000 |
| Physical transition budget | 491,520,000 + validation | Same |
| Additional startup validation | 8 steps/env, counted | Same |
| World-model warmup | 512 physical steps/env, counted | N/A |
| WM update | Every 4 physical steps, 2 updates, batch 256, window 8 | N/A |
| Imagined rollout | Batch 256, horizon 4->8->12->16 | N/A |
| Actor LR | 1e-4 | Original adaptive PPO, initial 1e-3 |
| WM / Critic LR | 1e-3 / 1e-3 | Original PPO |
| Discount / lambda | .99 / .95 | .99 / .95 |

First actor update is gated by finite held-out model loss below the configured
normalized-loss ceiling (5), after warmup. This is a numerical gate, **not proof
that model predictions are accurate enough for final policy success**. Log 1/8/16/32
step physical/FK prediction errors separately. If the gate never passes the run
fails, rather than claiming that a dynamics-only run trained a policy.

One seed is the initial implementation/learning-curve experiment, not statistical
proof of superiority. Extend to three training seeds only after this experiment
is stable. Compare held-out physical tracking error against BOTH real transition
count and measured wall time; do not compare unrelated iteration counts. BPTT's
WM validation environment interactions still count in its total sample budget.

The PS1 GPU 3 sequential queue avoids BPTT/PPO competing for the same card's
compute. Other GPUs and users' jobs are not modified. A100 access is currently
blocked by an unverified changed host key; no verification bypass is used.

## Entrypoints and results

- `scripts/train_supermimic.py`: fresh BPTT or explicitly requested resume.
- `scripts/train_supermimic_ppo_control.py`: existing RSL-RL PPO with step-zero
  initialization hash check. No old trained checkpoint accepted.
- `scripts/eval_supermimic.py`: shared PhysX evaluator for exported actors.
- `scripts/eval_supermimic_curve.py`: six saved checkpoints per method, one held-out
  seed x 64 env x 20 s diagnostic; actual transition count/wall-time axes are retained.
- `reproduction/supermimic_runtime.sh`: existing dependencies, private HOME/cache,
  source precedence; no package/system installs.
- `reproduction/run_supermimic_pair.sh`: sequential paired training and final
  held-out 3 seeds x 64 environments x 200 seconds.
- Per run: env.yaml, args/manifest, adapter validation, progress.jsonl, TensorBoard,
  checkpoints, actor TorchScript exports, completion/failure status.

Checkpoints contain Actor, noise distribution, Critic, target critic, world model,
normalizers, optimizer states and Torch RNG. Replay and simulator state are **not**
serialized: resume restarts physical episodes and mandates fresh replay/model
warmup. This is continuation, not bit-exact resume. No automatic policy deployment.

Native EE error, actual world-target distance, orientation, XY/yaw velocity error
and failure resets must all be reported; lower native EE error alone is not a
success claim. Model-predicted reward is never used as the final benchmark score.

## Verification

Run independent tests with `PYTHONPATH=scripts python -m unittest supermimic.test_core -v`.
Tests cover FK gradients, gravity transform, term-wise history, replay wrap/held-out
separation, terminal return masking, invalid-window NaNs, frozen-model BPTT gradients.
Simulator startup verifies FK, every active reward, policy and critic observations.
First 32 training steps verify actual noisy Isaac history order bit-for-bit.
Further checks at random-action steps 32/64/128 recheck FK/rewards/observations.
Short training must perform nonzero Actor updates, export/load actor, and support
the same real-physics evaluation path as PPO before a long run is launched.

Completed preflight on PS1 GPU 3:

- 7 simulator-independent unit tests passed.
- 32/64-env adapter/BPTT checks passed, including 16-step BPTT.
- 4096-env full-network test: 256 control steps, 33 Actor updates, finite gradients,
  1,081,344 total validation+collection transitions; ~99 s including startup.
- Full-size startup FK max position difference ~2.3e-5 m (large scene origin
  float32 precision); same-state all 24 reward and both observation groups passed.
- 2-iteration RSL-RL PPO paired-initialization smoke passed. The installed
  Isaac Lab deprecated-config migration helper is required, just as in the original
  PPO entrypoint; no PPO algorithm change was made.
- Both exported actors ran a 64-env / 4-second physical evaluation.
- Full checkpoint resumed from step 256 to 320 and continued Actor updates.

These are engineering checks, not trained performance or evidence of a speedup.
Logs before the resume-accounting fix count only the original eight validation
steps; the final source additionally counts new validation steps/time on resume.
