# Adapted from Zhengtianjin/go2-supermimic d9eee9b3f248e5300e636c433255db2c1c3e086a.
# TrainCfg, Actor, Critic, clip_grads and lr_at below are verbatim.
from dataclasses import dataclass
import torch
from torch import nn
from .world_model import RunningNorm, mlp

@dataclass(kw_only=True)
class TrainCfg:
  task: str = "Go2-Chase-Terrain"
  num_envs: int = 4096
  buffer_len: int = 1000
  iterations: int = 40_000
  warmup_iterations: int = 300
  """World-model-only iterations before policy updates start."""
  collect_steps: int = 1
  action_noise: float = 0.1
  explore_frac: float = 0.25
  """Fraction of envs collecting with larger, temporally correlated (OU) action
  noise, so the world model sees more varied motion than the current policy."""
  explore_noise: float = 0.3
  """Stationary std of the exploration OU noise (normalized action units)."""
  explore_theta: float = 2.0
  """Mean reversion of the exploration OU noise [1/s]."""
  wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
  wm_window: int = 8
  wm_batch: int = 2048
  wm_updates: int = 2
  wm_prioritized_frac: float = 0.25
  """Fraction of each world-model batch drawn from current windows with contact
  changes (touchdown / liftoff), flight or falls, where the model is least accurate."""
  wm_archive_frac: float = 0.25
  """Fraction of each world-model batch drawn from the rare-event archive."""
  archive_size: int = 100_000
  """Windows kept in the rare-event archive (jumps, landings, falls)."""
  archive_keep_prob: float = 0.02
  """Fraction of the rare candidate windows stored, so the archive turns over
  slowly and keeps old rare events."""
  wm_rewarmup_iterations: int = 1500
  """When resuming into a world model of a different shape, fit it this long
  before the policy moves again."""
  lr: float = 2e-4
  actor_lr_scale: float = 1.0
  """The actor's learning rate relative to lr. Fine-tuning on terrain wants smaller
  actor steps: at the full rate the actor drifts into states the world model
  predicts badly (falls rise once the ramp-up completes)."""
  lr_final_frac: float = 0.03
  lr_decay_iterations: int = 30_000
  horizon: int = 32
  imag_batch: int = 8192
  actor_hidden: tuple[int, ...] = (512, 256, 128)
  critic_hidden: tuple[int, ...] = (512, 256, 128)
  critic_epochs: int = 5
  critic_minibatches: int = 4
  grad_clip: float = 1.0
  grad_clip_elem: float = 0.1
  compile: bool = True
  """torch.compile one imagined step (cuts Python kernel-launch overhead ~1.5x)."""
  dodge_shaping: float = 20.0
  """Initial weight of the jump/duck hint (training wheels for the bars)."""
  dodge_shaping_decay_iterations: int = 20_000
  """The hint weight decays linearly to 0 over this many iterations."""
  target_curriculum_iterations: int = 15_000
  """Iterations over which the targets reach full difficulty. The env-side
  curriculum counts env steps, and this loop takes one env step per iteration
  (PPO takes 24), so it is set here explicitly."""
  terrain_curriculum_iterations: int = 12_000
  """Iterations over which the harder terrain rows open up (terrain task)."""
  sensor_grad: bool = True
  """See ImagineCfg.sensor_grad."""
  driver_policy: str = ""
  """MoE-CTS TorchScript policy that drives a fraction of the envs while
  collecting, so the world model sees terrain from the start (supermimic/driver.py)."""
  driver_frac: float = 0.25
  """Fraction of envs (the last ones; exploration noise uses the first) it drives."""
  expert_policy: str = ""
  """One of our checkpoints (e.g. a flat-ground chase-and-dodge policy) that drives
  another fraction of the envs, so the world model sees its skills (jumps, ducks)
  in this task. Its observation must be a prefix of this task's."""
  expert_frac: float = 0.0
  driver_fade: tuple[int, int] = (1_000_000_000, 1_000_000_000)
  """Both driven fractions fade linearly to zero between these iterations."""
  driver_vmin: float = 0.5
  driver_vmax: float = 3.0
  """Per-env top speed of the driver's commands is drawn from [vmin, vmax] [m/s]."""
  critic_warmup_iterations: int = 500
  """After resuming with a freshly initialised critic, train only the critic for
  this many policy iterations before the actor moves again."""
  actor_ramp_iterations: int = 1000
  """Whenever the actor starts (or restarts after a rollback) moving, its learning
  rate ramps up linearly over this many iterations, so the world model sees the
  new behaviour before the actor can exploit its errors."""
  reset_critic: bool = False
  """With --resume: start the critic afresh (e.g. after changing reward weights) and
  keep the actor still until it has warmed up."""
  continue_run: bool = False
  """With --resume: carry on that run's iteration count, so the target and terrain
  curricula, the hint decay and the learning-rate schedule pick up where it stopped
  (--iterations is then the total). Without it a resume starts them afresh."""
  resume: str = ""
  """Checkpoint to warm-start actor, critic and world model from (and optimizers if saved)."""
  collapse_falls: float = 3.0
  """Collapse guard: if the logged fall count exceeds this twice in a row, reload
  the last checkpoint and halve the learning rate."""
  log_interval: int = 50
  save_interval: int = 1000
  seed: int = 0
  run_name: str = "supermimic"
  log_root: str = "logs/supermimic"


class Actor(nn.Module):
  def __init__(self, obs_dim: int, hidden: tuple[int, ...], act_dim: int):
    super().__init__()
    self.norm = RunningNorm(obs_dim)
    self.net = mlp(obs_dim, hidden, act_dim, act=nn.ELU)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(obs))


class Critic(nn.Module):
  def __init__(self, in_dim: int, hidden: tuple[int, ...]):
    super().__init__()
    self.norm = RunningNorm(in_dim)
    self.net = mlp(in_dim, hidden, 1, act=nn.ELU)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(x))


def clip_grads(params, global_norm: float, elem: float) -> torch.Tensor:
  params = [p for p in params if p.grad is not None]
  norm = torch.nn.utils.clip_grad_norm_(params, global_norm)
  for p in params:
    p.grad.clamp_(-elem, elem)
  return norm


def lr_at(cfg: TrainCfg, it: int) -> float:
  import math

  t = min(1.0, it / cfg.lr_decay_iterations)
  frac = cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * t))
  return cfg.lr * frac



