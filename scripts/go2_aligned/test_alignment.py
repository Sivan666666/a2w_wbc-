"""CPU checks plus AST parity against the pinned read-only Go2 checkout."""
import ast
import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch
from .upstream import TrainCfg,Actor,lr_at
from .world_model import WorldModel
from .learner import Learner
from .replay import Replay
from .bridge import Bridge,heading_tilt_quat,rotate_inverse,yaw
from supermimic.core import qapply,qinv,qmul


class FakeBridge:
    env=NS(device='cpu')
    def critic_input(self,s,c,x,p):return torch.cat((s,c,x,p),-1)
    def imagined_step(self,s,p,h,c,x,actor,wm):
        critic=self.critic_input(s,c,x,p)
        a=actor(h);n,_=wm(s,a,p,x,torch.zeros(len(s),77))
        h=torch.cat((h[:,22:],a),-1)
        return n,a,h,-n[:,:10].square().sum(-1),torch.sigmoid(n[:,0]-10),critic


class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(10);torch.set_num_threads(2)

    def test_pinned_functions_ast(self):
        ref=Path('/tmp/a2w_supermimic_review_20261102/go2_code/go2_supermimic/supermimic')
        if not ref.exists():self.skipTest('Reference checkout absent; parity checked on development host')
        root=Path(__file__).parent
        for old,new,names in [('train.py','upstream.py',('TrainCfg','Actor','Critic','clip_grads','lr_at')),
                              ('world_model.py','world_model.py',('RunningNorm','mlp','WorldModel'))]:
            def nodes(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
            a,b=nodes(ref/old),nodes(root/new)
            for name in names:self.assertEqual(a[name],b[name],name)

    def test_defaults_and_schedule(self):
        c=TrainCfg()
        self.assertEqual((c.num_envs,c.iterations,c.collect_steps,c.wm_updates,c.wm_batch,c.imag_batch,c.horizon),
                         (4096,40000,1,2,2048,8192,32))
        self.assertEqual((c.critic_epochs,c.critic_minibatches,c.buffer_len,c.archive_size),(5,4,1000,100000))
        self.assertAlmostEqual(lr_at(c,30000),2e-4*.03)

    def test_yaw_velocity_roundtrip(self):
        def axis(angle,idx):
            q=torch.zeros(len(angle),4);q[:,0]=(angle/2).cos();q[:,idx]=(angle/2).sin();return q
        r,p,y=torch.randn(3,100)*.3
        root=qmul(axis(y,3),qmul(axis(p,2),axis(r,1)))
        down=torch.zeros(100,3);down[:,2]=-1
        g=qapply(qinv(root),down);v=torch.randn(100,3)
        vyaw=rotate_inverse(qapply(root,v),yaw(root))
        torch.testing.assert_close(qapply(qinv(heading_tilt_quat(g)),vyaw),v,atol=1e-6,rtol=1e-5)

    def test_analytic_torque_and_termination(self):
        b=Bridge.__new__(Bridge);b.env=NS(step_dt=.02,cfg=NS(decimation=4))
        b.default_active=torch.zeros(18);b.ki=list(range(22));b.di=list(range(22,44));b.li=[44]*22
        b.effort=torch.ones(22)*20;b.base_origin=torch.tensor([0.,0.,.01]);b.base_half=torch.tensor([.12,.14,.08]);b.fall_angle=1.
        s=torch.zeros(2,86);s[:,0]=.5;s[:,3]=-1
        ctx=torch.zeros(2,45);ctx[:,:18]=120;ctx[:,22:40]=3;ctx[:,40:44]=.6
        tau,acc=b.analytic_aux(s,s,torch.ones(2,22),torch.zeros(2,22),ctx)
        torch.testing.assert_close(tau[:,:18],torch.full((2,18),20.))
        torch.testing.assert_close(tau[:,18:],torch.full((2,4),3.))
        self.assertEqual(float(acc.abs().max()),0.)
        self.assertLess(float(b.fall_prob(s).max()),.001)
        s[:,3]=-.1;s[:,1]=.995
        self.assertGreater(float(b.fall_prob(s).min()),.99)

    def test_export_contains_normalizer(self):
        a=Actor(234,(16,8),22);x=torch.randn(8,234)*3
        a.norm.update(x)
        j=torch.jit.script(a)
        torch.testing.assert_close(j(x),a(x))
        self.assertIn('norm.mean',j.state_dict())

    def test_analytic_grad_accumulation_equivalence(self):
        c=dataclasses.replace(TrainCfg(),wm_hidden=(16,16),actor_hidden=(16,),critic_hidden=(16,),
                              imag_batch=16,horizon=3,compile=False,critic_epochs=1,critic_minibatches=4)
        a=Learner(FakeBridge(),c,16);b=Learner(FakeBridge(),c,4)
        for name in ('actor','critic','wm'):getattr(b,name).load_state_dict(getattr(a,name).state_dict())
        seed=dict(s=torch.randn(16,86)*.01,previous=torch.zeros(16,22),history=torch.randn(16,234)*.01,
                  commands=torch.randn(16,10)*.01,context=torch.randn(16,766)*.01)
        torch.manual_seed(3);ra=a.improve(seed)
        torch.manual_seed(3);rb=b.improve(seed)
        for x,y in zip(a.actor.parameters(),b.actor.parameters()):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        self.assertAlmostEqual(ra['imagined_reward'],rb['imagined_reward'],places=5)
        self.assertTrue(all(p.grad is None for p in a.wm.parameters()))
        self.assertEqual(a.actor_updates,1)
        self.assertFalse(hasattr(a,'target'))
        self.assertFalse(hasattr(a,'log_std'))

    def test_replay_context_and_terminal_masks(self):
        c=dataclasses.replace(TrainCfg(),num_envs=4,buffer_len=20,wm_window=3,wm_batch=8,archive_size=20)
        r=Replay(c,torch.zeros(4,718),48)
        for t in range(12):
            z=lambda n:torch.zeros(4,n)
            r.add(s=z(86)+t,target=z(86)+t+1,a=z(22),previous=z(22),history=z(234),commands=z(10),
                  dynamic=z(48)+t,dpose=z(3),remaining=z(1)+10,age=z(1)+1,done=torch.ones(4,1,dtype=torch.bool))
        w=r.windows(8,3)
        self.assertTrue((w['valid']==0).all())
        torch.testing.assert_close(w['context'][:,-1],w['s'][:,0,0])
        b=r.wm_batch();self.assertEqual(len(b['s']),8)
        self.assertEqual(r.seeds(16,.7)['s'].shape,(16,86))


if __name__=='__main__':unittest.main()
