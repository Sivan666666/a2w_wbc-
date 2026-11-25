"""CPU reward checks; extract actual functions/config without launching Isaac Sim."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab'
FUNCTIONS = ('hip_deviation_l1', 'hip_comfort_l2', 'hip_mirror_l2')


def load_functions():
    tree = ast.parse((TASK / 'mdp/rewards.py').read_text())
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    body += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'actual_hip_rewards', 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in FUNCTIONS})


MDP = load_functions()
HIP_NAMES = [f'{leg}_hip_joint' for leg in ('FR', 'FL', 'RR', 'RL')]


class HipRegularizationTest(unittest.TestCase):
    def make_env(self, angles, gravity_z=-1.0):
        # Noncontiguous IDs and large non-hip values catch accidental wheel/arm inclusion.
        ids = [2, 7, 11, 19]
        q = torch.full((len(angles), 22), 100.0)
        q[:, ids] = torch.tensor(angles)
        data = SimpleNamespace(joint_pos=q, default_joint_pos=torch.zeros_like(q),
                               projected_gravity_b=torch.tensor([[0., 0., gravity_z]] * len(angles)))
        return SimpleNamespace(scene={'robot': SimpleNamespace(data=data)}), SimpleNamespace(name='robot', joint_ids=ids)

    def test_mirror_sign_and_front_rear_independence(self):
        env, cfg = self.make_env([[.1, -.1, .4, -.4], [.1, .1, .3, .3]])
        torch.testing.assert_close(MDP.hip_mirror_l2(env, cfg), torch.tensor([0., .4]))

    def test_deadband_and_combined_weighted_cost(self):
        env, cfg = self.make_env([[.25, -.25, .1, -.1], [0., 0., .5, 0.]])
        before = env.scene['robot'].data.joint_pos.clone()
        torch.testing.assert_close(MDP.hip_comfort_l2(env, cfg), torch.tensor([0., .0625]))
        cost = -.3 * MDP.hip_deviation_l1(env, cfg) - MDP.hip_comfort_l2(env, cfg) - .2 * MDP.hip_mirror_l2(env, cfg)
        torch.testing.assert_close(cost, torch.tensor([-.21, -.2625]))
        torch.testing.assert_close(env.scene['robot'].data.joint_pos, before)

    def test_home_offset_and_no_orientation_gate(self):
        for gravity_z in (-1., -.6, 0., 1.):
            env, cfg = self.make_env([[.1, -.1, .2, -.2]], gravity_z)
            torch.testing.assert_close(MDP.hip_deviation_l1(env, cfg), torch.tensor([.6]))
            env.scene['robot'].data.default_joint_pos[:, cfg.joint_ids] = torch.tensor([.1, -.1, .2, -.2])
            torch.testing.assert_close(MDP.hip_deviation_l1(env, cfg), torch.tensor([0.]))
            torch.testing.assert_close(MDP.hip_comfort_l2(env, cfg), torch.tensor([0.]))

    def test_precision_config_wires_only_four_hips(self):
        class Parent:
            def __init__(self):
                ranges = lambda: SimpleNamespace(pos_x=(.55, .85))
                self.commands = SimpleNamespace(ee_pose=SimpleNamespace(ranges=ranges(), limit_ranges=ranges()))
                self.rewards = SimpleNamespace(end_effector_position_tracking_exp=SimpleNamespace(weight=4.5, params={'std':.1}), joint_mirror=None, joint_deviation=SimpleNamespace(weight=-.02), arm_deviation=SimpleNamespace(weight=-.1))
            def __post_init__(self):
                pass
        tree = ast.parse((TASK / 'config/a2w_z1_wheel_v2_precision/wbc_env_cfg.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'A2WZ1WheelV2PrecisionEnvCfg')
        cls.decorator_list = []
        ns = {'A2WZ1WheelV2EnvCfg': Parent, 'deepcopy': deepcopy, 'mdp': MDP, 'RewTerm': SimpleNamespace,
              'SceneEntityCfg': lambda name, **kw: SimpleNamespace(name=name, **kw)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'actual_precision_config', 'exec'), ns)
        cfg = ns['A2WZ1WheelV2PrecisionEnvCfg'](); cfg.__post_init__()
        for term_name, weight, function in [('hip_deviation', -.3, MDP.hip_deviation_l1), ('hip_comfort', -1., MDP.hip_comfort_l2), ('hip_mirror', -.2, MDP.hip_mirror_l2)]:
            term = getattr(cfg.rewards, term_name)
            self.assertEqual(term.weight, weight); self.assertIs(term.func, function)
            self.assertEqual(term.params['asset_cfg'].joint_names, HIP_NAMES)
            self.assertTrue(term.params['asset_cfg'].preserve_order)
        self.assertEqual(cfg.rewards.hip_comfort.params['comfort_angle'], .25)
        self.assertIsNone(cfg.rewards.joint_mirror)
        self.assertEqual(cfg.rewards.joint_deviation.weight, -.02)
        self.assertEqual(cfg.rewards.arm_deviation.weight, -.1)


if __name__ == '__main__':
    unittest.main()
