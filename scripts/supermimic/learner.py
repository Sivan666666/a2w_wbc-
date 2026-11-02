"""Reparameterized BPTT actor + detached TD(lambda) value learning."""
import copy
import torch
from torch import nn
from .core import *


class Learner:
    def __init__(self, adapter, args):
        self.bridge, self.args = adapter, args
        device = adapter.env.device
        # Exactly the same hidden sizes, activation, default Linear initialization
        # and unsquashed mean action interface as this repository's RSL-RL PPO.
        torch.manual_seed(args.seed)
        self.actor = mlp(234, 22).to(device)
        self.log_std = nn.Parameter(torch.zeros(22, device=device))
        self.critic = mlp(114, 1).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.model = Dynamics(adapter.context().shape[-1], args.wm_hidden).to(device)
        with torch.no_grad():
            self.model.default_q.copy_(adapter.default[adapter.ids[:18]])
        self.actor_params = list(self.actor.parameters())+[self.log_std]
        self.optimizers = [torch.optim.Adam(self.model.parameters(), lr=args.wm_lr),
                           torch.optim.Adam(self.actor_params, lr=args.actor_lr),
                           torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)]
        self.actor_updates = 0

    def action(self, obs, stochastic=True):
        mean = self.actor(obs)
        return mean + torch.randn_like(mean)*self.log_std.exp() if stochastic else mean

    def fit(self, batch):
        self.model.zero_grad(set_to_none=True)
        loss = self.model.window_loss(batch)
        grad = update(self.optimizers[0], loss, self.model.parameters())
        return dict(wm_loss=float(loss), wm_grad=grad)

    def improve(self, batch, horizon):
        b, a = self.bridge, self.args
        seed = {k:v[:,0] for k,v in batch.items()}
        keep = seed['remaining'][:,0] > (horizon+1)*b.env.step_dt
        keep &= seed['age'][:,0] > 0  # normal filled delay/history buffers
        if keep.sum() < 8:
            return {'actor_skipped_boundary': 1}
        seed = {k:v[keep][:a.imag_batch] for k,v in seed.items()}
        state, previous, history = seed['state'], seed['previous'], seed['history']
        commands, context, age = seed['commands'], seed['context'], seed['age']
        rewards, continuations, values, inputs = [], [], [], []
        alive = torch.ones(len(state), device=state.device)
        objective = torch.zeros_like(alive)
        # Clear model grads after fitting, so the frozen-gradient assertion is meaningful.
        self.model.zero_grad(set_to_none=True)
        with frozen(self.model, self.target):
            for t in range(horizon):
                current = b.proxy(state, previous, previous, commands)
                inputs.append(b.observation(current, 'critic').detach())
                action = self.action(history)
                nxt, terminated = self.model(state, action, previous, context, age+t)
                proxy = b.proxy(nxt, action, previous, commands)
                reward = b.reward(proxy)
                continuation = 1-terminated.sigmoid()
                objective = objective + (.99**t)*alive*reward
                alive = alive*continuation
                rewards.append(reward)
                continuations.append(continuation)
                next_input = b.observation(proxy, 'critic')
                values.append(self.target(next_input).squeeze(-1))
                history = push_history(history, b.observation(proxy, 'policy', noise=True))
                state, previous = nxt, action
            objective = objective + (.99**horizon)*alive*values[-1]
            loss = -objective.mean() - .005*self.log_std.mean()
            grad = update(self.optimizers[1], loss, self.actor_params)
        assert all(p.grad is None for p in self.model.parameters()), 'World model accidentally optimized by policy'
        with torch.no_grad():
            self.log_std.clamp_(-3, 1)
            targets = lambda_returns(torch.stack(rewards).detach(), torch.stack(continuations).detach(),
                                     torch.stack(values).detach()).flatten()
        inputs = torch.stack(inputs).flatten(0,1)
        for _ in range(a.critic_epochs):
            value_loss = (self.critic(inputs).squeeze(-1)-targets).square().mean()
            update(self.optimizers[2], value_loss, self.critic.parameters())
        with torch.no_grad():
            for target, source in zip(self.target.parameters(), self.critic.parameters()):
                target.lerp_(source, .02)
        self.actor_updates += 1
        return dict(actor_loss=float(loss), actor_grad=grad, critic_loss=float(value_loss),
                    action_std=float(self.log_std.exp().mean()), horizon=horizon,
                    actor_updates=self.actor_updates)

    @torch.no_grad()
    def validate_model(self, batch):
        loss = float(self.model.window_loss(batch))
        state = batch['state'][:,0]
        alive = torch.ones(len(state), dtype=torch.bool, device=state.device)
        result = {'wm_holdout_loss': loss}
        for t in range(batch['action'].shape[1]):
            state, terminal = self.model(state, batch['action'][:,t], batch['previous'][:,t],
                                         batch['context'][:,t], batch['age'][:,t])
            if t+1 in (1, 8, 16, 32) and alive.any():
                target = batch['target'][:,t]
                pp = self.bridge.fk(state[:,Q])['end_effector'][0]
                tp = self.bridge.fk(target[:,Q])['end_effector'][0]
                result[f'wm_holdout_{t+1}_ee_local_m'] = float((pp[alive]-tp[alive]).norm(dim=-1).mean())
                result[f'wm_holdout_{t+1}_velocity_mps'] = float((state[alive,V]-target[alive,V]).norm(dim=-1).mean())
                result[f'wm_holdout_{t+1}_samples'] = int(alive.sum())
            alive &= ~batch['done'][:,t]
        return result

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), log_std=self.log_std.detach(), critic=self.critic.state_dict(),
                    target=self.target.state_dict(), model=self.model.state_dict(),
                    optimizers=[o.state_dict() for o in self.optimizers], actor_updates=self.actor_updates)

    def load(self, data):
        for name in ('actor', 'critic', 'target', 'model'):
            getattr(self, name).load_state_dict(data[name], strict=True)
        with torch.no_grad():
            self.log_std.copy_(data['log_std'])
        for opt, state in zip(self.optimizers, data['optimizers']):
            opt.load_state_dict(state)
        self.actor_updates = data['actor_updates']
