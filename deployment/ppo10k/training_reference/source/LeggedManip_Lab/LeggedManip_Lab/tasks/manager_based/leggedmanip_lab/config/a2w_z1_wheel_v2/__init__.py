"""Explicit opt-in wheel configuration; legacy A2W tasks remain unchanged."""
import gymnasium as gym

for suffix, cfg in (("", "A2WZ1WheelV2EnvCfg"), ("-Play", "A2WZ1WheelV2EnvCfg_PLAY")):
    gym.register(
        id=f"A2W-Z1-WBC-WheelV2{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:{cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.wbc_env_cfg:A2WZ1WheelV2PPORunnerCfg",
        },
    )
