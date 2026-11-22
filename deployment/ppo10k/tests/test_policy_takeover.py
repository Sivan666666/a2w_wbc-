"""Offline command continuity checks; no hardware transport or actor runtime."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from a2w_deploy.controller import Controller
from a2w_deploy.policy import action_targets, KD

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'deploy.json').read_text())


class FeedbackOnly:
    def state(self):
        return np.zeros(22), np.zeros(22), np.array([1., 0., 0., 0.]), np.zeros(3)

    def home_arm(self):
        self.homed = True
        self.home_calls = getattr(self, "home_calls", 0) + 1
        return np.zeros(6)

    def capture_ee(self):
        return np.array([.5, .1, .4, 1., 0., 0., 0.])

    def begin_arm_preparation(self, q, dq, target, duration):
        return np.asarray(target).copy()

    def command(self, q, dq, kp, kd, gripper, arm_ddq=None, arm_ff_weight=0., arm_preparation=False):
        self.sent = tuple(np.array(v).copy() for v in (q, dq, kp, kd))
        self.ff_weight = arm_ff_weight
        self.preparing = arm_preparation


class PolicyTakeover(unittest.TestCase):
    def setUp(self):
        self.action = np.full(22, .8, dtype=np.float32)
        with patch('a2w_deploy.controller.Actor', return_value=lambda obs: self.action.copy()):
            self.c = Controller(CONFIG, 'unused')
        self.b = FeedbackOnly()
        self.c.key('z', self.b, 0.)

    def test_first_midpoint_and_live_policy_endpoint(self):
        c, b = self.c, self.b
        c.tick(b, 3.)
        np.testing.assert_array_equal(b.sent[2][12:18], CONFIG['prepare_arm_kp'])
        np.testing.assert_array_equal(b.sent[3][12:18], CONFIG['prepare_arm_kd'])
        self.assertEqual(b.ff_weight, 1.)
        self.assertTrue(b.preparing)
        prepared = tuple(v.copy() for v in b.sent)
        c.key('c', b, 3.9)
        c.tick(b, 4.)  # Scheduler delay must not skip the initial anchor.
        self.assertEqual(b.ff_weight, 1.)
        self.assertFalse(b.preparing)
        for actual, expected in zip(b.sent, prepared):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(c.command.ee, b.capture_ee())
        # Use each new actor output, not a frozen first action.
        self.action[:] = -.4
        c.tick(b, 4.5)
        self.assertAlmostEqual(b.ff_weight, .5)
        q, dq = action_targets(self.action)
        for actual, initial, policy in zip(b.sent, prepared, (q, dq, c.policy_kp, c.policy_kd)):
            np.testing.assert_allclose(actual, .5 * (initial + policy))
        self.action[:] = .3
        c.tick(b, 5.)
        self.assertEqual(b.ff_weight, 0.)
        np.testing.assert_array_equal(b.sent[2][12:18], [50,50,80,30,30,20])
        np.testing.assert_array_equal(b.sent[3][12:18], [3,2,3,3,2.5,1])
        q, dq = action_targets(self.action)
        for actual, expected in zip(b.sent, (q, dq, c.policy_kp, c.policy_kd)):
            np.testing.assert_array_equal(actual, expected)
        self.action[:] = -.1
        c.tick(b, 5.02)
        np.testing.assert_array_equal(b.sent[0], action_targets(self.action)[0])
        np.testing.assert_array_equal(c.observation.previous_action, self.action)

    def test_queued_entry_and_damping_interrupt(self):
        c, b = self.c, self.b
        c.key('c', b, 1.)
        c.tick(b, 3.)
        prepared = tuple(v.copy() for v in b.sent)
        self.assertEqual(c.phase, 'policy')
        c.tick(b, 3.02)
        for actual, expected in zip(b.sent, prepared):
            np.testing.assert_array_equal(actual, expected)

        c.tick(b, 3.52)
        c.key('x', b, 3.6)
        c.tick(b, 3.6)
        self.assertEqual(b.ff_weight, 0.)
        self.assertFalse(b.preparing)
        np.testing.assert_array_equal(b.sent[2], np.zeros(22))
        np.testing.assert_array_equal(b.sent[3], KD)
        # A new preparation/entry must start a fresh transition.
        c.key('z', b, 4.)
        c.tick(b, 7.)
        prepared = tuple(v.copy() for v in b.sent)
        c.key('c', b, 7.1)
        c.tick(b, 7.1)
        for actual, expected in zip(b.sent, prepared):
            np.testing.assert_array_equal(actual, expected)

    def test_nx_blends_to_training_pd_with_feedforward(self):
        cfg = json.loads((ROOT / 'deploy.nx.json').read_text())
        action = np.full(22, .8, dtype=np.float32)
        with patch('a2w_deploy.controller.Actor', return_value=lambda obs: action.copy()):
            c = Controller(cfg, 'unused')
        b = FeedbackOnly()
        c.key('z', b, 0.)
        c.tick(b, 3.)
        prepared = tuple(v.copy() for v in b.sent)
        c.key('c', b, 4.)
        for t in [4., 4.5, 5., 6.]:
            c.tick(b, t)
            weight = {4.: 0., 4.5: .5, 5.: 1., 6.: 1.}[t]
            expected_kp = (1-weight)*np.array([512,768,768,512,384,256]) + weight*np.array([50,50,80,30,30,20])
            expected_kd = (1-weight)*np.array([25.6]*6) + weight*np.array([3,2,3,3,2.5,1])
            np.testing.assert_allclose(b.sent[2][12:18], expected_kp)
            np.testing.assert_allclose(b.sent[3][12:18], expected_kd)
            self.assertEqual(b.ff_weight, 1.)
            self.assertFalse(b.preparing)  # Do not freeze at the preparation endpoint.
            if t == 4.:
                for actual, expected in zip(b.sent, prepared):
                    np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(b.sent[0], action_targets(action)[0])
        np.testing.assert_array_equal(b.sent[2][:12], [120.]*12)
        np.testing.assert_array_equal(b.sent[3][:12], [3.]*12)
        np.testing.assert_array_equal(b.sent[3][18:], [.6]*4)
        action[:] = -.2
        c.tick(b, 6.02)
        np.testing.assert_array_equal(b.sent[0], action_targets(action)[0])
        c.key('x', b, 6.1)
        c.tick(b, 6.1)
        np.testing.assert_array_equal(b.sent[2], np.zeros(22))
        self.assertEqual(b.ff_weight, 0.)

    def test_r_stops_inference_returns_directly_and_requires_new_c(self):
        cfg = json.loads((ROOT / 'deploy.nx.json').read_text())
        calls = []
        def actor(obs):
            calls.append(obs.copy())
            return np.full(22, .8, dtype=np.float32)
        with patch('a2w_deploy.controller.Actor', return_value=actor):
            c = Controller(cfg, 'unused')
        b = FeedbackOnly()
        c.key('z', b, 0.)
        c.tick(b, 3.)
        c.key('c', b, 4.)
        c.tick(b, 4.)
        self.assertEqual(len(calls), 1)
        actual = np.r_[np.array(cfg['stand_final_leg_q']) + .12,
                       np.array(cfg['arm_ready_q']) + .03, np.zeros(4)]
        velocity = np.r_[np.full(18, .04), np.ones(4)]
        b.state = lambda: (actual.copy(), velocity.copy(), np.array([1.,0.,0.,0.]), np.zeros(3))
        c.policy_requested = True  # R also cancels any queued policy entry.
        c.key('w', b, 5.)
        c.key('r', b, 5.)
        self.assertEqual(c.phase, 'return_ready')
        self.assertFalse(c.policy_requested)
        np.testing.assert_array_equal(c.command.velocity, np.zeros(3))
        c.tick(b, 5.)
        np.testing.assert_allclose(b.sent[0][:18], actual[:18])
        np.testing.assert_allclose(b.sent[1][:18], velocity[:18])
        np.testing.assert_array_equal(b.sent[1][18:], np.zeros(4))
        c.key('c', b, 5.1)  # A key during recovery must not auto-restart the actor.
        self.assertFalse(c.policy_requested)
        c.tick(b, 6.)
        midpoint = .5 * (actual[:12] + np.array(cfg['stand_final_leg_q'])) + .3125 * velocity[:12]
        np.testing.assert_allclose(b.sent[0][:12], midpoint)
        self.assertGreater(np.min(b.sent[0][2:12:3]), -2.)  # No crouch waypoint.
        c.tick(b, 7.)
        np.testing.assert_allclose(b.sent[0][:12], cfg['stand_final_leg_q'])
        np.testing.assert_array_equal(b.sent[1][:12], np.zeros(12))
        c.tick(b, 8.1)
        self.assertEqual(c.phase, 'ready')
        np.testing.assert_allclose(b.sent[0][12:18], cfg['arm_ready_q'])
        np.testing.assert_array_equal(b.sent[2][:12], [240.]*12)
        np.testing.assert_array_equal(b.sent[3][:12], [6.]*12)
        self.assertEqual(b.ff_weight, 1.)
        c.tick(b, 9.)
        self.assertEqual(len(calls), 1)
        prepared = tuple(v.copy() for v in b.sent)
        c.key('c', b, 9.1)
        c.tick(b, 9.1)
        self.assertEqual(len(calls), 2)
        for sent, expected in zip(b.sent, prepared):
            np.testing.assert_array_equal(sent, expected)



if __name__ == '__main__':
    unittest.main()


class HomeCrouch(unittest.TestCase):
    def test_t_cancels_actor_and_preserves_support_through_descent(self):
        def forbidden_actor(obs):
            raise AssertionError('PPO must not run after T')
        with patch('a2w_deploy.controller.Actor', return_value=forbidden_actor):
            c = Controller(CONFIG, 'unused')
        b = FeedbackOnly()
        c.phase = 'policy'
        c.policy_requested = True
        c.command.velocity[:] = 1
        c.key('T', b, 10.)
        self.assertTrue(b.homed)
        self.assertFalse(c.policy_requested)
        np.testing.assert_array_equal(c.command.velocity, 0.)
        # Homing time must not consume the two-second leg trajectory.
        c.tick(b, 20.)
        np.testing.assert_array_equal(b.sent[0][:12], 0.)
        c.tick(b, 21.)
        np.testing.assert_allclose(b.sent[0][:12], CONFIG['crouch_pre_leg_q'])
        c.tick(b, 22.)
        self.assertEqual(c.phase, 'crouched')
        np.testing.assert_allclose(b.sent[0][:12], CONFIG['crouch_final_leg_q'])
        np.testing.assert_array_equal(b.sent[1], 0.)
        np.testing.assert_array_equal(b.sent[2][:12], 240.)
        np.testing.assert_array_equal(b.sent[3][:12], 6.)
        c.key('c', b, 23.)
        self.assertEqual(c.phase, 'crouched')
        c.key('T', b, 24.)
        self.assertEqual(b.home_calls, 1)
        self.assertEqual(c.phase, 'crouched')
