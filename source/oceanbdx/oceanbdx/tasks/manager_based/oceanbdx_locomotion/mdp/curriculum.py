# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for OceanBDX locomotion environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> float:
    """Curriculum based on the robot's linear velocity tracking performance."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get command and actual velocity
    cmd = env.command_manager.get_command(command_name)
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    
    # compute velocity tracking error
    vel_error = torch.norm(cmd[:, :2] - lin_vel, dim=1)
    
    # compute mean velocity tracking performance
    mean_vel_error = torch.mean(vel_error).item()
    
    # curriculum logic: progress when tracking error is small
    if mean_vel_error < 0.1:  # Good tracking
        return 1.0  # Advance curriculum
    elif mean_vel_error < 0.2:  # Moderate tracking
        return 0.0  # Maintain current level
    else:  # Poor tracking
        return -1.0  # Regress curriculum
