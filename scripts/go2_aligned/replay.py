"""Go2 uniform/prioritized/archive proportions; A2W transition storage.

Static context is indexed by environment, reset-varying gains/delays are stored
per transition. This avoids repeating 718 immutable floats 1000 times per env.
Both terminal and following transitions are excluded from WM fitting (Go2 mask).
"""
import torch
from . import state as st


class Replay:
    def __init__(self, cfg, static, dynamic_dim):
        self.cfg, self.static, self.device = cfg, static.clone(), static.device
        self.capacity, self.n = cfg.buffer_len, cfg.num_envs
        self.cursor = self.size = 0
        dims = dict(s=86,target=86,a=22,previous=22,history=234,commands=10,
                    dynamic=dynamic_dim,dpose=3,remaining=1,age=1,done=1)
        self.data = {k:torch.empty((self.capacity,self.n,d),device=self.device,
                     dtype=torch.float16 if k=='history' else (torch.bool if k=='done' else torch.float32)) for k,d in dims.items()}
        self.archive = {}
        self.archive_size = self.archive_cursor = 0

    def add(self, **data):
        for k,v in data.items(): self.data[k][self.cursor].copy_(v)
        self.cursor=(self.cursor+1)%self.capacity
        self.size=min(self.capacity,self.size+1)

    def windows(self, n, k):
        e=torch.randint(self.n,(n,),device=self.device)
        t=torch.randint(self.size-k+1,(n,),device=self.device)
        idx=(self.cursor-self.size+t[:,None]+torch.arange(k,device=self.device))%self.capacity
        b={key:v[idx,e[:,None]] for key,v in self.data.items() if key not in ('history','commands','remaining','age')}
        valid=(~b['done'].squeeze(-1)).float().cumprod(1)
        s=torch.cat((b['s'][:,:1],b['target']),1)
        return dict(s=s,a=b['a'],previous=b['previous'],
                    context=torch.cat((self.static[e],b['dynamic'][:,0]),-1),
                    dpose=b['dpose'],valid=valid)

    @staticmethod
    def feet_height(s):
        # Radius measured from collision STL max radial (x,z), in metres.
        return s[...,0:1]+s[...,st.FEET].reshape(*s.shape[:-1],4,3)[...,2]-.09486137

    def rarity(self,w):
        contact=(self.feet_height(w['s'])<.035).float()
        return (contact[:,1:]-contact[:,:-1]).abs().sum((1,2))+(contact.sum(-1)==0).float().sum(-1)+(w['valid'][:,-1]<.5).float()*4

    def archive_worthy(self,w):
        flight=(self.feet_height(w['s'])>=.035).all(-1).float().sum(-1)>=2
        hard=w['s'][:,:,6].abs().amax(-1)>.6
        return flight|hard|(w['valid'][:,-1]<.5)

    def archive_add(self,w):
        n=min(len(w['s']),self.cfg.archive_size)
        if not n:return
        if not self.archive:
            self.archive={k:torch.empty((self.cfg.archive_size,*v.shape[1:]),device=v.device,dtype=v.dtype) for k,v in w.items()}
        idx=(self.archive_cursor+torch.arange(n,device=self.device))%self.cfg.archive_size
        for k,v in w.items():self.archive[k][idx]=v[:n]
        self.archive_cursor=(self.archive_cursor+n)%self.cfg.archive_size
        self.archive_size=min(self.cfg.archive_size,self.archive_size+n)

    def wm_batch(self):
        c=self.cfg
        np=int(c.wm_batch*c.wm_prioritized_frac)
        na=int(c.wm_batch*c.wm_archive_frac) if self.archive_size else 0
        parts=[self.windows(c.wm_batch-np-na,c.wm_window)]
        cand=self.windows(4*max(np,1),c.wm_window)
        score=self.rarity(cand)
        keep=self.archive_worthy(cand)&(torch.rand_like(score)<c.archive_keep_prob)
        self.archive_add({k:v[keep] for k,v in cand.items()})
        top=(score+1e-3*torch.rand_like(score)).topk(np).indices
        parts.append({k:v[top] for k,v in cand.items()})
        if na:
            idx=torch.randint(self.archive_size,(na,),device=self.device)
            parts.append({k:v[idx] for k,v in self.archive.items()})
        return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}

    def seeds(self,n,seconds):
        pieces=[];total=0
        for _ in range(100):
            count=max(2*(n-total),32)
            e=torch.randint(self.n,(count,),device=self.device)
            t=(self.cursor-self.size+torch.randint(self.size,(count,),device=self.device))%self.capacity
            valid=(self.data['remaining'][t,e,0]>seconds)&(self.data['age'][t,e,0]>0)
            e,t=e[valid][:n-total],t[valid][:n-total]
            p={k:self.data[k][t,e].float() for k in ('s','a','previous','history','commands')}
            p['context']=torch.cat((self.static[e],self.data['dynamic'][t,e]),-1)
            pieces.append(p);total+=len(e)
            if total==n:break
        if total!=n:raise RuntimeError('Insufficient command-interior replay seeds; refuse silently smaller imag batch')
        return {k:torch.cat([p[k] for p in pieces]) for k in pieces[0]}
