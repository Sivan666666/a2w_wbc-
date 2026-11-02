"""Config, legacy-isolation, reward routing and finite-physics tests. No training."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(a).app

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
import LeggedManip_Lab.tasks
from LeggedManip_Lab.assets.a2w_z1.a2w_z1_articulation_cfg import LEGS, ARM, WHEELS


def main():
    torch.set_num_threads(4)
    old = parse_env_cfg("A2W-Z1-WBC", device=a.device, num_envs=4)
    dump_yaml(str(a.output / "legacy_before.yaml"), old)
    cfg = parse_env_cfg("A2W-Z1-WBC-WheelV2", device=a.device, num_envs=4)
    play = parse_env_cfg("A2W-Z1-WBC-WheelV2-Play", device=a.device, num_envs=4)
    for c in (cfg, play):
        w = c.scene.robot.actuators["wheels"]
        assert (w.stiffness, w.damping, w.effort_limit, w.effort_limit_sim, w.velocity_limit_sim) == (0., .6, 20., 20., 50.)
        assert c.actions.wheel_vel.scale == 5.
        assert c.actions.wheel_vel.clip == {".*": (-100., 100.)}
        assert abs(c.rewards.dof_acc_l2.weight / c.rewards.wheel_acc_l2.weight - 100.) < 1e-9
        assert c.rewards.arm_acc_l2.weight == old.rewards.dof_acc_l2.weight
        for name in ("feet_slide", "feet_air_time", "feet_long_air", "air_time_variance", "joint_mirror"):
            assert getattr(c.rewards, name) is None
    assert cfg.commands.to_dict() == old.commands.to_dict()
    assert cfg.curriculum.to_dict() == old.curriculum.to_dict()
    assert cfg.observations.to_dict() == old.observations.to_dict()
    for name in ("end_effector_position_tracking_exp", "end_effector_orientation_tracking", "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "action_rate_l2"):
        assert getattr(cfg.rewards, name).to_dict() == getattr(old.rewards, name).to_dict()
    for name in ("arm", "hips_thighs", "calves"):
        assert cfg.scene.robot.actuators[name].to_dict() == old.scene.robot.actuators[name].to_dict()
    # Detect shared mutable config state, including contamination of existing instances.
    dump_yaml(str(a.output / "legacy_after_existing.yaml"), old)
    dump_yaml(str(a.output / "legacy_after_fresh.yaml"), parse_env_cfg("A2W-Z1-WBC", device=a.device, num_envs=4))
    before = (a.output / "legacy_before.yaml").read_bytes()
    assert before == (a.output / "legacy_after_existing.yaml").read_bytes()
    assert before == (a.output / "legacy_after_fresh.yaml").read_bytes()
    agent = load_cfg_from_registry("A2W-Z1-WBC-WheelV2", "rsl_rl_cfg_entry_point")
    assert agent.experiment_name == "a2w_z1_wbc_wheel_v2"
    dump_yaml(str(a.output / "agent.yaml"), agent)
    dump_yaml(str(a.output / "train_env.yaml"), cfg)
    dump_yaml(str(a.output / "play_env.yaml"), play)
    cfg.seed = 20260920
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.log_dir = str(a.output)
    env = gym.make("A2W-Z1-WBC-WheelV2", cfg=cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    obs, _ = env.reset()
    assert u.action_manager.total_action_dim == 22
    assert tuple(obs["policy"].shape) == (4, 234)
    wheel_action = u.action_manager.get_term("wheel_vel")
    sample = torch.tensor([[1., -1., 30., -30.]], device=u.device).repeat(4, 1)
    wheel_action.process_actions(sample)
    assert torch.equal(wheel_action.processed_actions, torch.tensor([[5., -5., 100., -100.]], device=u.device).repeat(4, 1))
    groups = {}
    expected = {"dof_acc_l2": LEGS, "arm_acc_l2": ARM, "wheel_acc_l2": WHEELS,
                "dof_torques_l2": LEGS, "arm_torques_l2": ARM, "joint_power": LEGS}
    for term_name, names in expected.items():
        term = u.reward_manager.get_term_cfg(term_name)
        ids = term.params["asset_cfg"].joint_ids
        resolved = [robot.joint_names[i] for i in ids]
        assert set(resolved) == set(names), (term_name, resolved)
        groups[term_name] = dict(joints=resolved, weight=term.weight)
    # Wheel-only acceleration/torque/power must not leak into leg or arm terms.
    n = robot.num_joints
    x = torch.zeros((4, n), device=u.device)
    x[:, robot.find_joints(WHEELS)[0]] = 2.
    fake = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
        joint_acc=x, applied_torque=x, joint_vel=x))})
    for name in expected:
        term = u.reward_manager.get_term_cfg(name)
        value = term.func(fake, **term.params)
        assert torch.all(value == (16. if name == "wheel_acc_l2" else 0.)), (name, value)
    actions = torch.zeros((4, 22), device=u.device)
    for i in range(128):
        obs, reward, terminated, truncated, info = env.step(actions)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        assert torch.isfinite(robot.data.joint_vel).all()
    result = dict(status="PASS", physics_steps=128, envs=4, actions=22, policy_obs=234,
                  legacy_yaml_identical=True, legacy_yaml_sha256=hashlib.sha256(before).hexdigest(),
                  action_test_raw=[1,-1,30,-30], action_test_processed=[5,-5,100,-100],
                  reward_groups=groups, note="Zero-action smoke, NOT a trained-policy performance evaluation.")
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("WHEEL_V2_TEST_PASS", json.dumps(result), flush=True)
    env.close()


try:
    main()
finally:
    app.close()
