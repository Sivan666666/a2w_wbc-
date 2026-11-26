"""CPU regression and independent upstream-oracle tests; no simulator needed."""
import unittest,sys,ast,types,tempfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from force_control.core import *
from force_control.learning import *
from force_control.inference import PolicyHistory, ImpedanceCommand

class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(1)
    def test_source_reward_oracle(self):
        upstream=Path(__file__).resolve().parents[4]/'learning-compliance/b1_gym/rewards/b1_loco_z1_gaitfree_rewards.py'
        # Caller can pass an alternate checkout via environment; default sibling under work/.
        if not upstream.exists():upstream=Path(__file__).resolve().parents[3]/'learning-compliance/b1_gym/rewards/b1_loco_z1_gaitfree_rewards.py'
        self.assertTrue(upstream.exists())
        tree=ast.parse(upstream.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_reward_ee_force_x','_reward_ee_force_y','_reward_ee_force_z']]
        def euler(q):
            x,y,z,w=q.unbind(-1);return torch.zeros_like(w),torch.zeros_like(w),torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        def qyaw(r,p,y):return torch.stack((y*0,y*0,(y/2).sin(),(y/2).cos()),-1)
        def rotate(q,v):
            x,y,z,w=q.unbind(-1);c=1-2*z*z;s=2*w*z
            return torch.stack((c*v[:,0]+s*v[:,1],-s*v[:,0]+c*v[:,1],v[:,2]),-1)
        ns={'torch':torch,'get_euler_xyz':euler,'quat_from_euler_xyz':qyaw,'quat_rotate_inverse':rotate,'INDEX_EE_FORCE_X':0,'INDEX_EE_FORCE_Y':1,'INDEX_EE_FORCE_Z':2}
        exec(compile(ast.Module(body=funcs,type_ignores=[]),str(upstream),'exec'),ns)
        n=128;y=torch.rand(n)*6.28;q=torch.stack(((y/2).cos(),y*0,y*0,(y/2).sin()),-1);f=torch.randn(n,3)*50;cmd=torch.randn(n,3)*50;m=torch.arange(n)%2
        env=types.SimpleNamespace(num_envs=n,forces=f[:,None,:],gripper_stator_index=0,base_quat=q[:,[1,2,3,0]],commands=cmd,force_or_position_control=m,cfg=types.SimpleNamespace(rewards=types.SimpleNamespace(sigma_force_z=.02)))
        expected=torch.stack([ns['_reward_ee_force_'+a](types.SimpleNamespace(env=env)) for a in 'xyz'],-1)
        actual=force_reward(yaw_rotate(f,y,inverse=True),cmd,m)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    def test_spring_cap_sign_and_zero(self):
        x=torch.zeros(3,3);anchor=torch.tensor([[1.,-1,0],[1,0,1],[1,1,1]])
        f=spring_force(anchor,x,x,torch.full((3,1),100.),torch.full((3,1),10.),torch.tensor([True,False,True]))
        self.assertEqual(f[0].tolist(),[70.,-70.,0.]);self.assertEqual(f[1].tolist(),[0.,0.,0.])
        q=torch.tensor([[.70710678,0,0,.70710678]]);v=torch.tensor([[1.,0,0]])
        torch.testing.assert_close(yaw_rotate(v,yaw_angle(q),inverse=True),torch.tensor([[0.,-1,0.]]),atol=1e-6,rtol=0)
    def test_envelope(self):
        age=torch.tensor([0.,1.,2.,3.,4.,5.,6.]);r=torch.full_like(age,2.)
        torch.testing.assert_close(force_envelope(age,r,1.),torch.tensor([0.,.5,1.,1.,.5,0.,0.]))
    def test_partial_reset_and_position_mode(self):
        s=ForceState(8,'cpu',ForceConfig(force_probability=1,free_probability=0));ids=torch.arange(8);p=torch.rand(8,3);q=torch.tensor([[1.,0,0,0]]).expand(8,-1)
        s.reset(ids,p,p,q);s.advance(.5);s.apply(p+1,p,p*0,q);old=s.state_dict();s.reset(ids[:2],p,p,q)
        self.assertEqual(s.external_w[:2].abs().sum(),0)
        torch.testing.assert_close(s.external_w[2:],old['tensors']['external_w'][2:])
        s.mode[:]=False;s.advance(.02);self.assertEqual(s.command.abs().sum(),0)
        self.assertEqual(s.apply(p+1,p,p*0,q).abs().sum(),0)
    def test_estimator_learns_and_no_truth_input(self):
        m=CSEPolicy(20,7);x=torch.randn(32,20);y=torch.cat((x[:,:3],20*x[:,3:6],.1*x[:,6:9]),-1)
        opt=torch.optim.Adam(m.estimator.parameters(),lr=.003);start=estimator_loss(m.estimator(x),y).item()
        actor_before={k:v.clone() for k,v in m.actor.state_dict().items()}
        for _ in range(250):opt.zero_grad();loss=estimator_loss(m.estimator(x),y);loss.backward();opt.step()
        self.assertLess(estimator_loss(m.estimator(x),y).item(),start*.05)
        for k,v in m.actor.state_dict().items():self.assertTrue(torch.equal(v,actor_before[k]))
        self.assertEqual(m(x).shape,(32,22))
        exp=torch.jit.script(ExportPolicy(m));torch.testing.assert_close(exp(x),m(x));torch.testing.assert_close(exp.estimate(x),m.estimator(x))
    def test_ppo_checkpoint_resume(self):
        m=CSEPolicy(12,6);a=CSEPPO(m,PPOConfig(steps=4,epochs=1,minibatches=2));n=16
        h=torch.randn(n,12);c=torch.randn(n,6)
        with torch.no_grad():dist=m.distribution(h);actions=dist.sample();values=m.value(h,c)
        b={'h':h,'c':c,'target':torch.randn(n,9),'a':actions,'logp':dist.log_prob(actions).sum(-1).detach(),'mu':dist.mean.detach(),'sigma':dist.stddev.detach(),'value':values,'returns':values+torch.randn(n),'adv':torch.randn(n)}
        a.update(b)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'c.pt';a.save(p);a.update(b)
            restored=CSEPPO(CSEPolicy(12,6),PPOConfig(steps=4,epochs=1,minibatches=2));restored.load(p);restored.update(b)
            for k,v in a.model.state_dict().items():torch.testing.assert_close(v,restored.model.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(a.iteration,restored.iteration)
    def test_impedance_sign(self):
        x=torch.tensor([[0.,0.,0.]]);target=torch.tensor([[1.,0.,0.]])
        self.assertEqual(impedance_external_command(x,x,target,10,1).tolist(),[[-10.,0.,0.]])
        controller=ImpedanceCommand(dt=.02,kp=10,kd=1)
        torch.testing.assert_close(controller(x,target),torch.tensor([[-10.,0.,0.]]))
        torch.testing.assert_close(controller(x+.1,target),torch.tensor([[-4.,6.,6.]]))
    def test_history_reset_and_layout(self):
        h=PolicyHistory(2,length=3);x=torch.randn(2,82)
        first=h.push(x);torch.testing.assert_close(h.latest_frame(first,3),x)
        y=x+1;second=h.push(y,torch.tensor([True,False]))
        # First three angular-velocity frames occur together, oldest to newest.
        torch.testing.assert_close(second[0,:9].reshape(3,3),y[0,:3].expand(3,-1))
        torch.testing.assert_close(second[1,:9].reshape(3,3),torch.stack((x[1,:3],x[1,:3],y[1,:3])))
    def test_force_magnitude_penalties(self):
        f=torch.tensor([[-7.,3.,20.],[9.,-4.,-10.]])
        mode=torch.tensor([True,False])
        torch.testing.assert_close(force_magnitude_penalty(f,mode,0),torch.tensor([7.,0.]))
        torch.testing.assert_close(force_magnitude_penalty(f,mode,1),torch.tensor([3.,0.]))

if __name__=='__main__':unittest.main(verbosity=2)
