"""Run: PYTHONPATH=scripts python -m unittest supermimic.test_core -v"""
import unittest
from pathlib import Path
import torch
from .core import *


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_history_term_order(self):
        old = torch.arange(234.).reshape(1,-1)
        latest = torch.arange(78.).reshape(1,-1)+1000
        result = push_history(old,latest)
        hi=fi=0
        for w in TERMS:
            torch.testing.assert_close(result[:,hi:hi+2*w], old[:,hi+w:hi+3*w])
            torch.testing.assert_close(result[:,hi+2*w:hi+3*w], latest[:,fi:fi+w])
            hi+=3*w; fi+=w

    def test_replay_wrap_and_holdout(self):
        replay=Replay(8,20,'cpu')
        for t in range(12):
            replay.add(time=torch.full((20,1),t),env=torch.arange(20)[:,None])
        for val in (False,True):
            b=replay.sample(100,4,val)
            self.assertTrue((b['time'][:,1:]-b['time'][:,:-1]==1).all())
            self.assertTrue((b['time']>=4).all())
            self.assertTrue((b['env']>=18).all() if val else (b['env']<18).all())

    def test_fk_gradient_and_gravity_rotation(self):
        asset=Path(__file__).resolve().parents[2]/'source/LeggedManip_Lab/LeggedManip_Lab/assets/a2w_z1/a2wz1_wbc.urdf'
        names=[f'{l}_{j}_joint' for l in ('FR','FL','RR','RL') for j in ('hip','thigh','calf')]+[f'joint{i}' for i in range(1,7)]
        fk=ArmFK(asset,names).double()
        q=(torch.randn(1,18,dtype=torch.double)*.1).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x:fk(x)['end_effector'][0],(q,),eps=1e-6,atol=1e-5))
        gravity=F.normalize(torch.randn(100,3),dim=-1)
        actual=qapply(tilt_quat(gravity),gravity)
        target=torch.zeros_like(actual);target[:,2]=-1
        torch.testing.assert_close(actual,target,atol=3e-5,rtol=3e-5)

    def test_bptt_freezes_parameters_not_inputs(self):
        actor=mlp(3,3,(8,)); dynamics=mlp(6,3,(8,))
        state=torch.zeros(4,3); opt=torch.optim.Adam(actor.parameters(),lr=.01)
        before=[p.clone() for p in actor.parameters()]
        with frozen(dynamics):
            for _ in range(8):
                action=actor(state)
                state=state+dynamics(torch.cat((state,action),-1))
            update(opt,(state-1).square().mean(),actor.parameters())
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(before,actor.parameters())))
        self.assertTrue(all(p.grad is None for p in dynamics.parameters()))
        self.assertTrue(all(p.requires_grad for p in dynamics.parameters()))

    def test_terminal_return(self):
        r=torch.ones(3,1); c=torch.tensor([[1.],[0.],[1.]])
        v=torch.ones(3,1)*100
        targets=lambda_returns(r,c,v,gamma=1,lam=1)
        torch.testing.assert_close(targets[:,0],torch.tensor([2.,1.,101.]))

    def test_action_target_clips_and_checkpoint(self):
        model=Dynamics(2,16)
        action=torch.full((1,22),1000.,requires_grad=True)
        x=model.inputs(torch.zeros(1,98),action,torch.zeros_like(action),torch.zeros(1,2),torch.ones(1,1))
        torch.testing.assert_close(x[:,50+18:50+22],torch.full((1,4),20.))
        x.sum().backward()
        torch.testing.assert_close(action.grad,torch.zeros_like(action))
        other=Dynamics(2,16);other.load_state_dict(model.state_dict(),strict=True)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)

    def test_window_never_crosses_reset_or_nan(self):
        model=Dynamics(2,16)
        state=torch.zeros(2,3,98);state[:,:,3]=-1
        batch=dict(state=state.clone(),target=state.clone(),action=torch.zeros(2,3,22),
                   previous=torch.zeros(2,3,22),context=torch.zeros(2,3,2),age=torch.ones(2,3,1),
                   terminated=torch.ones(2,3,dtype=torch.bool),done=torch.ones(2,3,dtype=torch.bool))
        loss=model.window_loss(batch)
        batch['target'][:,1:]=float('nan')
        torch.testing.assert_close(model.window_loss(batch),loss)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__': unittest.main()
