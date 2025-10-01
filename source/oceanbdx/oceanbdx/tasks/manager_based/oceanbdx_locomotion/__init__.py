# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OceanBDX locomotion environment."""

import gymnasium as gym

from .oceanbdx_locomotion_env import OceanBDXLocomotionEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ocean-BDX-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "oceanbdx.tasks.manager_based.oceanbdx_locomotion.config:OceanBDXLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": "oceanbdx.tasks.manager_based.oceanbdx_locomotion.config.agents.rsl_rl_ppo_cfg:OceanBDXPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ocean-BDX-Locomotion-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "oceanbdx.tasks.manager_based.oceanbdx_locomotion.config:OceanBDXLocomotionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "oceanbdx.tasks.manager_based.oceanbdx_locomotion.config.agents.rsl_rl_ppo_cfg:OceanBDXPPORunnerCfg",
    },
)