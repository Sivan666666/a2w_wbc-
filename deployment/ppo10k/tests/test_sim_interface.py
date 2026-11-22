"""Functional regression tests for the simulation adapter and shared inputs."""
import json
from pathlib import Path
import select
import socket
import unittest
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from a2w_deploy.controller import Controller
from a2w_deploy.mujoco_backend import MuJoCo
from a2w_deploy.teleop import VRReceiver
from a2w_deploy.kinematics import Kinematics

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'deploy.json').read_text())


class SimulationInterface(unittest.TestCase):
    def test_z_two_stage_preparation_then_c(self):
        b=MuJoCo()
        c=Controller(CONFIG,ROOT/CONFIG['actor'])
        c.key('c',b,0.)
        self.assertEqual(c.phase,'idle')
        c.key('z',b,0.)
        c.tick(b,0.)
        np.testing.assert_allclose(b.kp[:12],240.)
        np.testing.assert_allclose(b.kd[:12],6.)
        np.testing.assert_allclose(b.target_q[:12],c.start_q[:12])
        np.testing.assert_allclose(b.target_dq[:12],0.)
        c.tick(b,.25)
        np.testing.assert_allclose(b.target_q[:12],.75*c.start_q[:12]+.25*np.asarray(CONFIG['stand_pre_leg_q']))
        np.testing.assert_allclose(b.target_dq[:12],0.)
        c.tick(b,1.)
        np.testing.assert_allclose(b.kp[:12],240.)
        np.testing.assert_allclose(b.kd[:12],6.)
        np.testing.assert_allclose(b.target_q[:12],CONFIG['stand_pre_leg_q'])
        np.testing.assert_allclose(b.target_dq[:12],0.,atol=1e-12)
        c.tick(b,1.25)
        np.testing.assert_allclose(b.target_q[:12],.75*np.asarray(CONFIG['stand_pre_leg_q'])+.25*np.asarray(CONFIG['stand_final_leg_q']))
        c.tick(b,2.)
        np.testing.assert_allclose(b.target_q[:12],CONFIG['stand_final_leg_q'])
        np.testing.assert_allclose(b.target_dq[:12],0.)
        self.assertEqual(c.phase,'stand')  # Z1 retains its three-second preparation.
        c.tick(b,3.)
        self.assertEqual(c.phase,'ready')
        np.testing.assert_allclose(b.kp[:12],240.)
        np.testing.assert_allclose(b.kd[:12],6.)
        wheel_kp,wheel_kd=b.kp[18:].copy(),b.kd[18:].copy()
        np.testing.assert_array_equal(b.kp[12:18],CONFIG['prepare_arm_kp'])
        self.assertEqual(b.arm_ff_weight,1.)
        np.testing.assert_allclose(b.target_q[:12],CONFIG['stand_final_leg_q'])
        np.testing.assert_allclose(b.target_q[12:18],CONFIG['arm_ready_q'])
        np.testing.assert_allclose(b.target_dq,0.,atol=1e-12)
        c.key('c',b,3.1)
        self.assertEqual(c.phase,'policy')
        self.assertAlmostEqual(b.metrics(c.command)['ee_cm'],0.,places=10)
        self.assertAlmostEqual(b.metrics(c.command)['orientation_deg'],0.,places=10)
        c.tick(b,3.1)
        np.testing.assert_allclose(b.kp[:12],240.)
        np.testing.assert_allclose(b.kd[:12],6.)
        np.testing.assert_allclose(b.target_q[12:18],CONFIG['arm_ready_q'])
        c.tick(b,4.11)
        np.testing.assert_allclose(b.kp[:12],120.)
        np.testing.assert_allclose(b.kd[:12],3.)
        np.testing.assert_array_equal(b.kp[12:18],CONFIG['policy_arm_kp'])
        np.testing.assert_array_equal(b.kd[12:18],CONFIG['policy_arm_kd'])
        np.testing.assert_array_equal(b.kp[18:],wheel_kp)
        np.testing.assert_array_equal(b.kd[18:],wheel_kd)
        self.assertEqual(b.arm_ff_weight,0.)

    def test_arm_feedforward_is_preparation_only(self):
        b=MuJoCo()
        q=b.state()[0]
        q[12:18]=CONFIG['arm_ready_q']
        b.command(q,np.zeros(22),np.zeros(22),np.zeros(22),0.,arm_ff_weight=1.)
        b.advance()
        self.assertGreater(np.linalg.norm(b.last_arm_feedforward),1.)
        b.command(q,np.zeros(22),np.zeros(22),np.zeros(22),0.)
        b.advance()
        np.testing.assert_array_equal(b.last_arm_feedforward,np.zeros(6))

    def test_entry_captures_feedback_and_reanchors_keyboard_vr_home(self):
        b=MuJoCo()
        c=Controller(CONFIG,ROOT/CONFIG['actor'])
        c.key('z',b,0.)
        c.tick(b,3.)
        # Different feedback at c than at trajectory completion (PD sag/motion).
        b.data.qpos[b.qids[12:18]]=CONFIG['arm_ready_q']
        b.data.qpos[b.qids[13]]+=.04
        b.data.qpos[2]=.43
        b.data.qpos[3:7]=Rotation.from_euler('xyz',[.07,-.08,.3]).as_quat()[[3,0,1,2]]
        mujoco.mj_forward(b.model,b.data)
        c.vr({'right':{'pose':'0,0,0,0,0,0,1','grip':1.}})
        c.key('w',b,3.5)
        c.key('c',b,4.)
        expected=b.capture_ee()
        np.testing.assert_allclose(c.command.ee,expected,atol=1e-14)
        metric=b.metrics(c.command)
        np.testing.assert_allclose(metric['target_p'],metric['actual_p'],atol=1e-14)
        np.testing.assert_allclose(metric['target_r'],metric['actual_r'],atol=1e-14)
        self.assertIsNone(c.command.clutch)
        np.testing.assert_array_equal(c.command.velocity,np.zeros(3))
        c.tick(b,4.)
        np.testing.assert_allclose(c.observation.history[-1][-1],expected,atol=1e-7)
        c.key('i',b,4.)
        self.assertAlmostEqual(c.command.ee[0],expected[0]+CONFIG['position_step'])
        c.key('h',b,4.)
        np.testing.assert_allclose(c.command.ee,expected)
        c.vr({'right':{'pose':'1,2,3,0,0,0,1','grip':1.}})
        np.testing.assert_allclose(c.command.ee,expected)
        c.key('u',b,4.)
        c.vr({'right':{'secondaryButton':True}})
        np.testing.assert_allclose(c.command.ee,expected)

    def test_queued_c_captures_at_completion(self):
        b=MuJoCo()
        c=Controller(CONFIG,ROOT/CONFIG['actor'])
        c.key('z',b,0.)
        c.key('c',b,1.)
        b.data.qpos[b.qids[12:18]]=CONFIG['arm_ready_q']
        mujoco.mj_forward(b.model,b.data)
        c.tick(b,3.)
        self.assertEqual(c.phase,'policy')
        np.testing.assert_allclose(c.command.ee,b.capture_ee(),atol=1e-14)

    def test_hardware_fk_matches_mujoco_with_known_base_height(self):
        b=MuJoCo()
        fk=Kinematics()
        rng=np.random.default_rng(42)
        for _ in range(20):
            b.data.qpos[b.qids]=rng.uniform(-1.,1.,22)
            b.data.qpos[2]=rng.uniform(.3,.7)
            b.data.qpos[3:7]=Rotation.random(random_state=rng).as_quat()[[3,0,1,2]]
            mujoco.mj_forward(b.model,b.data)
            q,_,quat,_=b.state()
            np.testing.assert_allclose(fk.command(q,quat,b.data.qpos[2]),b.capture_ee(),atol=1e-12)

    def test_imu_uses_link_frame_not_principal_inertia_frame(self):
        b=MuJoCo()
        r=Rotation.from_euler('xyz',[.2,-.3,.7])
        b.data.qpos[3:7]=r.as_quat()[[3,0,1,2]]
        b.data.qvel[3:6]=[.3,-.4,.8]
        mujoco.mj_forward(b.model,b.data)
        np.testing.assert_allclose(b.state()[3],[.3,-.4,.8],atol=1e-12)
        gravity=r.inv().apply([0,0,-1])
        c=Controller(CONFIG,ROOT/CONFIG['actor'])
        c.key('z',b,-3.)
        c.tick(b,0.)
        c.key('c',b,0)
        c.tick(b,0)
        np.testing.assert_allclose(c.observation.history[1][-1],gravity,atol=1e-7)

    def test_keyboard_and_vr_produce_same_actor_action(self):
        b=MuJoCo()
        keyboard=Controller(CONFIG,ROOT/CONFIG['actor'])
        vr=Controller(CONFIG,ROOT/CONFIG['actor'])
        for c in (keyboard,vr):
            c.key('z',b,-3.)
            c.tick(b,0.)
        for key in 'c'+'i'*10+'j'*5+'u'*3+'1'*4+'3'*3+'5'*2:
            keyboard.key(key,b,0.)
        vr.key('c',b,0.)
        vr.vr({'right':{'pose':'0,0,0,0,0,0,1','grip':1.}})
        basis=np.asarray(CONFIG['vr_basis'])
        target_rotation=Rotation.from_euler('xyz',[.2,.15,.1]).as_matrix()
        pose=np.r_[basis.T@np.array([.1,.05,.03]),
                   Rotation.from_matrix(basis.T@target_rotation@basis).as_quat()]
        vr.vr({'right':{'pose':','.join(map(str,pose)),'grip':1.}})
        np.testing.assert_allclose(keyboard.command.ee,vr.command.ee,atol=1e-14)
        keyboard.tick(b,0.)
        vr.tick(b,0.)
        np.testing.assert_array_equal(keyboard.last_action,vr.last_action)

    def test_pxrea_udp_packet_reaches_shared_command(self):
        receiver=VRReceiver('127.0.0.1',0)
        sender=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            packet={'right':{'pose':'0,0,0,0,0,0,1','grip':0.},
                    'left':{'axisY':.5,'axisX':-.25}}
            sender.sendto(json.dumps(packet).encode(),receiver.socket.getsockname())
            select.select([receiver.socket],[],[],1.)
            received=list(receiver.drain())
            self.assertEqual(received,[packet])
            c=Controller(CONFIG,ROOT/CONFIG['actor'])
            c.vr(received[0])
            np.testing.assert_allclose(c.command.velocity,[.4,0.,.25])
        finally:
            sender.close()
            receiver.close()


if __name__=='__main__':unittest.main()
