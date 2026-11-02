"""Exercise actual deployment expressions without importing a GPU renderer."""
import ast
from pathlib import Path
import unittest
import numpy as np


class ActionProcessingTest(unittest.TestCase):
    def test_both_scripts_match_isaac_affine_then_clip(self):
        rng = np.random.default_rng(42)
        default = np.array([0., .8, -1.5]*4+[0.]*10)
        examples = [np.full(22, v, dtype=np.float32) for v in (-100., -3., -.3, 0., .3, 3., 100.)]
        examples += list(rng.uniform(-60, 60, (1000, 22)).astype(np.float32))
        for filename in ('repro_mujoco.py', 'repro_mujoco_draw_n.py', 'repro_mujoco_neowa.py'):
            tree = ast.parse(Path(__file__).with_name(filename).read_text())
            expressions = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    lhs = ast.unparse(node.targets[0])
                    if lhs in ('target[:18]', 'wheel_target') and isinstance(node.value, ast.Call):
                        if ast.unparse(node.value.func) == 'np.clip':
                            expressions[lhs] = compile(ast.Expression(node.value), filename, 'eval')
            self.assertEqual(set(expressions), {'target[:18]', 'wheel_target'}, filename)
            for action in examples:
                ns = dict(np=np, default=default, action=action)
                np.testing.assert_array_equal(eval(expressions['target[:18]'], ns), np.clip(default[:18]+.25*action[:18], -10., 10.))
                np.testing.assert_array_equal(eval(expressions['wheel_target'], ns), np.clip(action[18:]*10., -3., 3.))


if __name__ == '__main__':
    unittest.main()
