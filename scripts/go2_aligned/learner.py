"""Pinned Go2 optimizer/rollout semantics, robot-specific bridge.

Memory-only chunking accumulates the gradient of the FULL 8192-seed objective
before ONE actor step. No reduced effective batch, no target-critic/entropy head.
"""
import torch
from supermimic.core import frozen, lambda_returns
from .upstream import Actor, Critic, clip_grads, lr_at
from .world_model import WorldModel


class Learner:
    def __init__(self, bridge, cfg, chunk=1024):
        self.b,self.cfg,self.chunk=bridge,cfg,chunk
        dev=bridge.env.device
        self.actor=Actor(234,cfg.actor_hidden,22).to(dev)
        self.critic=Critic(884,cfg.critic_hidden).to(dev)
        self.wm=WorldModel(cfg.wm_hidden).to(dev)
        self.opt_wm=torch.optim.Adam(self.wm.parameters(),lr=cfg.lr)
        self.opt_actor=torch.optim.Adam(self.actor.parameters(),lr=cfg.lr,betas=(.9,.95))
        self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=cfg.lr)
        self.actor_updates=0;self.nonfinite_steps=0;self.lr_scale=1.;self.ramp_start=cfg.warmup_iterations
        self.step=torch.compile(bridge.imagined_step,dynamic=False) if cfg.compile else bridge.imagined_step

    def schedule(self,it):
        c=self.cfg
        lr=lr_at(c,it)*self.lr_scale
        ramp=min(1.,max(0.,it-self.ramp_start)/max(1,c.actor_ramp_iterations))
        for opt,scale in ((self.opt_wm,1.),(self.opt_actor,ramp*c.actor_lr_scale),(self.opt_critic,1.)):
            for g in opt.param_groups:g['lr']=lr*scale
        return lr,lr*ramp*c.actor_lr_scale

    def fit(self,w):
        loss=self.wm.loss(w['s'][:,0],w['a'],w['previous'],w['context'],
                          torch.zeros((*w['a'].shape[:2],77),device=w['a'].device),
                          w['s'][:,1:],w['dpose'],w['valid'])
        self.opt_wm.zero_grad(set_to_none=True);loss.backward()
        norm=clip_grads(self.wm.parameters(),self.cfg.grad_clip,float('inf'))
        if torch.isfinite(norm) and torch.isfinite(loss):self.opt_wm.step()
        else:self.nonfinite_steps+=1
        return float(loss.detach()),float(norm)

    def improve(self,seed):
        c,b=self.cfg,self.b
        assert len(seed['s'])==c.imag_batch
        xs=[];ys=[];objectives=[];reward_means=[];survival=[]
        self.wm.zero_grad(set_to_none=True)
        self.opt_actor.zero_grad(set_to_none=True)
        with frozen(self.wm,self.critic):
            for offset in range(0,c.imag_batch,self.chunk):
                q={k:v[offset:offset+self.chunk] for k,v in seed.items()}
                s,previous,hist=q['s'],q['previous'],q['history']
                cmds,ctx=q['commands'],q['context']
                alive=torch.ones(len(s),device=s.device);ret=torch.zeros_like(alive)
                inputs=[];rewards=[];continuations=[]
                for t in range(c.horizon):
                    s,previous,hist,r,d,x=self.step(s,previous,hist,cmds,ctx,self.actor,self.wm)
                    inputs.append(x.detach());rewards.append(r.detach());continuations.append((1-d).detach())
                    ret=ret+(.99**t)*alive*r
                    alive=alive*(1-d)
                terminal=b.critic_input(s,cmds,ctx,previous)
                v=self.critic(terminal).squeeze(-1)
                objective=ret+.99**c.horizon*alive*v
                (-objective.sum()/c.imag_batch).backward()
                with torch.no_grad():
                    x=torch.stack(inputs)
                    next_x=torch.cat((x[1:],terminal.detach()[None]),0)
                    next_v=self.critic(next_x).squeeze(-1)
                    y=lambda_returns(torch.stack(rewards),torch.stack(continuations),next_v)
                    xs.append(x.flatten(0,1));ys.append(y.flatten())
                    objectives.append(objective.detach().sum());reward_means.append(torch.stack(rewards).sum())
                    survival.append(alive.detach().sum())
                del objective,ret,alive,s,hist,terminal,v,next_x,next_v
        norm=clip_grads(self.actor.parameters(),c.grad_clip,c.grad_clip_elem)
        finite=bool(torch.isfinite(norm)) and all(bool(torch.isfinite(v)) for v in objectives)
        if finite:self.opt_actor.step();self.actor_updates+=1
        else:self.nonfinite_steps+=1
        assert all(p.grad is None for p in self.wm.parameters())
        x,y=torch.cat(xs),torch.cat(ys)
        self.critic.norm.update(x)
        losses=[]
        for _ in range(c.critic_epochs):
            order=torch.randperm(len(x),device=x.device)
            for idx in order.tensor_split(c.critic_minibatches):
                self.opt_critic.zero_grad(set_to_none=True)
                # Exact mean gradient over each of the four upstream minibatches.
                total=0.
                for ids in idx.split(self.chunk*4):
                    loss=(self.critic(x[ids]).squeeze(-1)-y[ids]).square().sum()/len(idx)
                    loss.backward();total+=float(loss.detach())
                gn=clip_grads(self.critic.parameters(),c.grad_clip,float('inf'))
                if torch.isfinite(gn) and torch.isfinite(torch.tensor(total)):self.opt_critic.step()
                else:self.nonfinite_steps+=1
                losses.append(total)
        return dict(actor_loss=-float(torch.stack(objectives).sum())/c.imag_batch,
                    imagined_reward=float(torch.stack(reward_means).sum())/(c.imag_batch*c.horizon),
                    imagined_survival=float(torch.stack(survival).sum())/c.imag_batch,
                    actor_grad_norm=float(norm),critic_loss=sum(losses)/len(losses),
                    actor_updates=self.actor_updates,nonfinite_skipped=self.nonfinite_steps)

    def state_dict(self):
        return {**{k:getattr(self,k).state_dict() for k in ('actor','critic','wm','opt_actor','opt_critic','opt_wm')},
                'actor_updates':self.actor_updates,'lr_scale':self.lr_scale,'ramp_start':self.ramp_start}

    def rollback(self,checkpoint,it):
        ck=torch.load(checkpoint,map_location=self.b.env.device,weights_only=False)['learner']
        self.actor.load_state_dict(ck['actor']);self.critic.load_state_dict(ck['critic'])
        self.lr_scale*=.5;self.ramp_start=it
        self.opt_actor=torch.optim.Adam(self.actor.parameters(),lr=self.cfg.lr,betas=(.9,.95))
        self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=self.cfg.lr)
        # Deliberately KEEP learned dynamics/replay, matching Go2 collapse guard.
