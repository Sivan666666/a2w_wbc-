"""Concurrent state estimator + PPO, adapted from upstream ppo_cse (MIT).

Actor: history + predicted privileged state; critic: history + OUR critic state.
Estimator labels: base velocity(3), external EE force(3), EE position(3).
No force truth enters actor inference. PPO settings match our A configuration.
"""
from dataclasses import dataclass,asdict
import copy
import torch
from torch import nn
from torch.distributions import Normal


def mlp(n,dims,out):
    layers=[]
    for h in dims:layers.extend([nn.Linear(n,h),nn.ELU()]);n=h
    return nn.Sequential(*layers,nn.Linear(n,out))


class CSEPolicy(nn.Module):
    def __init__(self,history_dim,critic_dim,action_dim=22):
        super().__init__();self.history_dim=history_dim;self.critic_dim=critic_dim;self.action_dim=action_dim
        self.estimator=mlp(history_dim,[256,128],9)
        self.actor=mlp(history_dim+9,[512,256,128],action_dim)
        self.critic=mlp(history_dim+critic_dim,[512,256,128],1)
        self.log_std=nn.Parameter(torch.zeros(action_dim))

    def forward(self,history):
        estimated=self.estimator(history)
        return self.actor(torch.cat((history,estimated),-1))

    def distribution(self,history):
        mean=self(history)
        return Normal(mean,self.log_std.exp().expand_as(mean))

    def value(self,history,critic_state):return self.critic(torch.cat((history,critic_state),-1)).squeeze(-1)


class ExportPolicy(nn.Module):
    def __init__(self,model):
        super().__init__();self.estimator=copy.deepcopy(model.estimator).cpu();self.actor=copy.deepcopy(model.actor).cpu()
    def forward(self,history):return self.actor(torch.cat((history,self.estimator(history)),-1))
    @torch.jit.export
    def estimate(self,history):return self.estimator(history)


def estimator_loss(pred,target):
    # Upstream loss multiplies BOTH prediction and target by weight before MSE.
    return ((pred[:,:3]-target[:,:3])**2).mean()+((.05*(pred[:,3:6]-target[:,3:6]))**2).mean()+((10*(pred[:,6:9]-target[:,6:9]))**2).mean()

@dataclass
class PPOConfig:
    steps:int=24
    epochs:int=5
    minibatches:int=4
    clip:float=.2
    gamma:float=.99
    lam:float=.95
    value_coef:float=1.
    entropy_coef:float=.005
    learning_rate:float=.001
    estimator_learning_rate:float=.001
    desired_kl:float=.01
    max_grad_norm:float=1.


class CSEPPO:
    def __init__(self,model,cfg=None):
        self.model=model;self.cfg=cfg or PPOConfig();self.iteration=0
        self.optimizer=torch.optim.Adam(model.parameters(),lr=self.cfg.learning_rate)
        self.estimator_optimizer=torch.optim.Adam(model.estimator.parameters(),lr=self.cfg.estimator_learning_rate)
        self.learning_rate=self.cfg.learning_rate

    @torch.no_grad()
    def collect(self,env,obs):
        data={k:[] for k in ['h','c','target','a','logp','mu','sigma','value','r','done']}
        for _ in range(self.cfg.steps):
            h=obs['policy'];c=obs['critic'];target=obs['estimator'];dist=self.model.distribution(h);action=dist.sample();v=self.model.value(h,c)
            nxt,r,terminated,timeout,info=env.step(action)
            # Match A/RSL timeout convention: bootstrap stored pre-step value.
            r=r+self.cfg.gamma*v*timeout.float()
            values=[h,c,target,action,dist.log_prob(action).sum(-1),dist.mean,dist.stddev,v,r,terminated|timeout]
            for k,x in zip(data,values):data[k].append(x.detach().clone())
            obs=nxt
        data={k:torch.stack(v) for k,v in data.items()}
        last=self.model.value(obs['policy'],obs['critic']);gae=torch.zeros_like(last);returns=torch.empty_like(data['r'])
        for i in reversed(range(self.cfg.steps)):
            live=(~data['done'][i]).float();nextv=last if i==self.cfg.steps-1 else data['value'][i+1]
            delta=data['r'][i]+self.cfg.gamma*nextv*live-data['value'][i]
            gae=delta+self.cfg.gamma*self.cfg.lam*live*gae;returns[i]=gae+data['value'][i]
        data['returns']=returns;adv=returns-data['value'];data['adv']=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        return {k:v.flatten(0,1) for k,v in data.items()},obs

    def update(self,batch):
        c=self.cfg;n=len(batch['a']);logs=[]
        if n<c.minibatches:raise ValueError('Too few samples for minibatches')
        for _ in range(c.epochs):
            perm=torch.randperm(n,device=batch['a'].device)
            for idx in torch.tensor_split(perm,c.minibatches):
                b={k:v[idx] for k,v in batch.items()};dist=self.model.distribution(b['h']);value=self.model.value(b['h'],b['c'])
                with torch.no_grad():
                    kl=torch.sum(torch.log(dist.stddev/b['sigma']+1e-5)+(b['sigma'].square()+(b['mu']-dist.mean).square())/(2*dist.stddev.square())-.5,-1).mean()
                    if kl>2*c.desired_kl:self.learning_rate=max(1e-5,self.learning_rate/1.5)
                    elif 0<kl<c.desired_kl/2:self.learning_rate=min(1e-2,self.learning_rate*1.5)
                    for g in self.optimizer.param_groups:g['lr']=self.learning_rate
                ratio=(dist.log_prob(b['a']).sum(-1)-b['logp']).exp()
                surrogate=torch.maximum(-b['adv']*ratio,-b['adv']*ratio.clamp(1-c.clip,1+c.clip)).mean()
                clipped=b['value']+(value-b['value']).clamp(-c.clip,c.clip)
                vl=torch.maximum((value-b['returns']).square(),(clipped-b['returns']).square()).mean()
                loss=surrogate+c.value_coef*vl-c.entropy_coef*dist.entropy().sum(-1).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),c.max_grad_norm);self.optimizer.step()
                # As upstream: PPO may also update estimator through actor; separate supervised step follows.
                pred=self.model.estimator(b['h']);el=estimator_loss(pred,b['target'])
                if not torch.isfinite(el):raise FloatingPointError('Nonfinite estimator loss')
                self.estimator_optimizer.zero_grad();el.backward();self.estimator_optimizer.step()
                logs.append([float(surrogate.detach()),float(vl.detach()),float(el.detach()),float(kl)])
        self.iteration+=1
        x=torch.tensor(logs).mean(0)
        return dict(zip(['surrogate','value_loss','estimator_loss','kl'],x.tolist()))|{'learning_rate':self.learning_rate}

    def save(self,path,metadata=None):
        torch.save({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'estimator_optimizer':self.estimator_optimizer.state_dict(),'iteration':self.iteration,'learning_rate':self.learning_rate,'ppo_config':asdict(self.cfg),'dimensions':[self.model.history_dim,self.model.critic_dim,self.model.action_dim],'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'metadata':metadata},path)

    def load(self,path):
        s=torch.load(path,map_location=next(self.model.parameters()).device,weights_only=False)
        if s['ppo_config']!=asdict(self.cfg):raise ValueError('PPO configuration mismatch')
        if s['dimensions']!=[self.model.history_dim,self.model.critic_dim,self.model.action_dim]:raise ValueError('Observation/action ABI mismatch')
        self.model.load_state_dict(s['model']);self.optimizer.load_state_dict(s['optimizer']);self.estimator_optimizer.load_state_dict(s['estimator_optimizer'])
        self.iteration=s['iteration'];self.learning_rate=s['learning_rate'];torch.set_rng_state(s['torch_rng'].cpu())
        if s['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
        return s['metadata']

    def export(self,path):torch.jit.script(ExportPolicy(self.model).eval()).save(str(path))
