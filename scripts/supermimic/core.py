"""Simulator-independent differentiable dynamics, replay and return utilities.

Algorithm references (independent implementation, not a vendored simulator):
zita-ch/supermimic_vibed, Zhengtianjin/go2-supermimic,
uwvwko-zzz/go2w-supermimic. See docs/SUPERMIMIC_A2W.md for pinned commits.
"""
from contextlib import contextmanager
import math
import xml.etree.ElementTree as ET
import torch
from torch import nn
from torch.nn import functional as F

# Heading-invariant state. Wheel phase is absent; auxiliary targets are measured
# at the reward computation boundary, NOT estimated from endpoint PD forces.
Z = slice(0, 1)
G = slice(1, 4)
V = slice(4, 7)
W = slice(7, 10)
Q = slice(10, 28)
QD = slice(28, 50)
TAU = slice(50, 72)
ACC = slice(72, 94)
CONTACT = slice(94, 98)
PHYSICAL = 50
STATE = 98
TERMS = (3, 3, 18, 22, 22, 3, 7)


def qmul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)), -1)


def qapply(q, v):
    t = 2 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1]*t + torch.cross(q[..., 1:], t, dim=-1)


def qinv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def tilt_quat(gravity):
    """Rotate body gravity to world -Z; world yaw is irrelevant to this task."""
    g = F.normalize(gravity, dim=-1, eps=1e-6)
    down = torch.zeros_like(g)
    down[..., 2] = -1
    q = torch.cat((1+(g*down).sum(-1, keepdim=True), torch.cross(g, down, dim=-1)), -1)
    # Deterministic antiparallel fallback, outside the environment's tilt limit.
    fallback = torch.zeros_like(q)
    fallback[..., 1] = 1
    return F.normalize(torch.where(q.norm(dim=-1, keepdim=True) < 1e-6, fallback, q), dim=-1)


class ArmFK(nn.Module):
    """URDF-exact root->link0/end_effector chain, including fixed mounts/tool."""
    def __init__(self, path, position_names):
        super().__init__()
        urdf = ET.parse(path).getroot()
        by_child = {j.find('child').get('link'): j for j in urdf.findall('joint')}
        chain, link = [], 'end_effector'
        while link != 'base':
            j = by_child[link]
            chain.append(j)
            link = j.find('parent').get('link')
        self.chain = []
        for i, j in enumerate(reversed(chain)):
            origin = j.find('origin')
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0]*3
            rpy = [float(x)/2 for x in origin.get('rpy', '0 0 0').split()] if origin is not None else [0]*3
            r, p, y = rpy
            cr, cp, cy, sr, sp, sy = math.cos(r), math.cos(p), math.cos(y), math.sin(r), math.sin(p), math.sin(y)
            quat = [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
            axis = j.find('axis')
            axis = [float(x) for x in axis.get('xyz', '1 0 0').split()] if axis is not None else [1, 0, 0]
            self.register_buffer(f'p{i}', torch.tensor(xyz, dtype=torch.float32))
            self.register_buffer(f'r{i}', torch.tensor(quat, dtype=torch.float32))
            self.register_buffer(f'a{i}', F.normalize(torch.tensor(axis, dtype=torch.float32), dim=0))
            idx = position_names.index(j.get('name')) if j.get('type') != 'fixed' else None
            self.chain.append((idx, j.find('child').get('link')))

    def forward(self, q):
        p = torch.zeros_like(q[:, :3])
        r = torch.zeros_like(q[:, :4]); r[:, 0] = 1
        poses = {}
        for i, (idx, child) in enumerate(self.chain):
            p = p + qapply(r, getattr(self, f'p{i}').expand_as(p))
            r = qmul(r, getattr(self, f'r{i}').expand_as(r))
            if idx is not None:
                theta = q[:, idx:idx+1] / 2
                rotation = torch.cat((theta.cos(), theta.sin()*getattr(self, f'a{i}')), -1)
                r = qmul(r, rotation)
            if child in ('link0', 'end_effector'):
                poses[child] = (p, r)
        return poses


def push_history(history, latest):
    parts, hi, fi = [], 0, 0
    for width in TERMS:
        parts.append(torch.cat((history[:, hi+width:hi+3*width], latest[:, fi:fi+width]), -1))
        hi += width*3; fi += width
    return torch.cat(parts, -1)


def mlp(inputs, outputs, hidden=(512, 256, 128), activation=nn.ELU):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(inputs, width), activation()))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class Norm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('var', torch.ones(size))
        self.register_buffer('count', torch.tensor(0.))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        n = len(x); total = self.count+n; delta = x.mean(0)-self.mean
        self.var.copy_((self.var*self.count + x.var(0, unbiased=False)*n + delta.square()*self.count*n/total)/total)
        self.mean.add_(delta*n/total); self.count.copy_(total)

    @property
    def scale(self):
        return self.var.clamp_min(1e-4).sqrt()

    def forward(self, x):
        return (x-self.mean)/self.scale


class Dynamics(nn.Module):
    def __init__(self, context_dim, hidden=512):
        super().__init__()
        self.register_buffer('default_q', torch.tensor([0., .8, -1.5]*4+[0.]*6))
        # Only physical state drives dynamics. No invented acceleration/force
        # feedback loop; these are output-only auxiliary quantities.
        self.input_norm = Norm(PHYSICAL+22+22+context_dim+1)
        self.output_norm = Norm(STATE)
        self.net = mlp(self.input_norm.mean.numel(), STATE+1, (hidden,)*3, nn.SiLU)

    def inputs(self, state, action, previous, context, age):
        def effective(raw):
            # Exact scale/offset THEN target clipping. Physical saturation is
            # not left to an MLP to extrapolate; action-rate cost still uses raw.
            q = (self.default_q+.25*raw[:, :18]).clamp(-10,10)
            wheels = (5*raw[:,18:]).clamp(-100,100)
            return torch.cat(((q-self.default_q)/.25, wheels/5), -1)
        return torch.cat((state[:, :PHYSICAL], effective(action), effective(previous), context, (age*.02).clamp_max(1)), -1)

    @torch.no_grad()
    def update_stats(self, state, action, previous, context, age, target):
        self.input_norm.update(self.inputs(state, action, previous, context, age))
        self.output_norm.update(torch.cat((target[:, :PHYSICAL]-state[:, :PHYSICAL], target[:, PHYSICAL:]), -1))

    def forward(self, state, action, previous, context, age):
        pred = self.net(self.input_norm(self.inputs(state, action, previous, context, age)))
        delta = pred[:, :STATE]*self.output_norm.scale+self.output_norm.mean
        core = state[:, :PHYSICAL]+delta[:, :PHYSICAL]
        core = torch.cat((core[:, :1], F.normalize(core[:, G], dim=-1, eps=1e-6), core[:, 4:]), -1)
        aux = torch.cat((delta[:, PHYSICAL:94], delta[:, CONTACT].clamp_min(0)), -1)
        return torch.cat((core, aux), -1), pred[:, -1]

    def window_loss(self, batch):
        state = batch['state'][:, 0]
        alive = torch.ones(len(state), dtype=torch.bool, device=state.device)
        total, count = state.sum()*0, 0
        for t in range(batch['action'].shape[1]):
            keep = alive & torch.isfinite(batch['target'][:, t]).all(-1)
            if not keep.any():
                break
            pred, terminal = self(state[keep], batch['action'][keep, t], batch['previous'][keep, t],
                                   batch['context'][keep, t], batch['age'][keep, t])
            error = ((pred-batch['target'][keep, t])/self.output_norm.scale).square().mean(-1)
            error += .1*F.binary_cross_entropy_with_logits(terminal, batch['terminated'][keep, t].float(), reduction='none')
            total = total+error.sum(); count += keep.sum()
            state = state.clone(); state[keep] = pred
            # Include terminal transition; never learn the post-reset jump.
            alive = keep & ~batch['done'][:, t]
        return total/torch.as_tensor(count, device=state.device).clamp_min(1)


class Replay:
    """Time-major GPU ring; held-out environment IDs never used for training."""
    def __init__(self, capacity, envs, device, holdout=0.1):
        self.capacity, self.envs, self.device = capacity, envs, device
        self.split = max(1, int(envs*(1-holdout)))
        self.data, self.cursor, self.size = {}, 0, 0

    def add(self, **values):
        for k, v in values.items():
            if k not in self.data:
                self.data[k] = torch.empty((self.capacity, *v.shape), device=self.device, dtype=v.dtype)
            self.data[k][self.cursor].copy_(v.detach())
        self.cursor = (self.cursor+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch, window=1, validation=False):
        if self.size < window:
            raise ValueError('Replay shorter than requested window')
        lo, hi = (self.split, self.envs) if validation else (0, self.split)
        env = torch.randint(lo, hi, (batch,), device=self.device)
        start = torch.randint(self.size-window+1, (batch,), device=self.device)
        times = ((self.cursor-self.size)+start[:, None]+torch.arange(window, device=self.device)) % self.capacity
        return {k: v[times, env[:, None]] for k, v in self.data.items()}


@contextmanager
def frozen(*modules):
    params = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for p, _ in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in params:
            p.requires_grad_(flag)


def lambda_returns(rewards, continuations, next_values, gamma=.99, lam=.95):
    carry = next_values[-1]
    targets = []
    for r, c, v in zip(reversed(rewards), reversed(continuations), reversed(next_values)):
        carry = r+gamma*c*((1-lam)*v+lam*carry)
        targets.append(carry)
    return torch.stack(targets[::-1])


def update(optimizer, loss, parameters, clip=1.):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss; refusing to corrupt checkpoint')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(list(parameters), clip, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)
