import gymnasium as gym

for suffix, cfg in (("", "A2WZ1WBCEnvCfg"), ("-Play", "A2WZ1WBCEnvCfg_PLAY")):
    gym.register(id=f"A2W-Z1-WBC{suffix}", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.wbc_env_cfg:{cfg}",
                "rsl_rl_cfg_entry_point": f"{__name__}.wbc_env_cfg:A2WZ1WBCPPORunnerCfg"})
