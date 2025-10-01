# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for OceanBDX locomotion environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_joints_by_scale(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float] = (0.8, 1.2),
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the joint positions and velocities by scaling default values."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get default joint positions and velocities
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    
    # apply random scaling to positions
    pos_scale = sample_uniform(position_range[0], position_range[1], joint_pos.shape, joint_pos.device)
    joint_pos *= pos_scale
    
    # apply random noise to velocities
    if velocity_range != (0.0, 0.0):
        vel_noise = sample_uniform(velocity_range[0], velocity_range[1], joint_vel.shape, joint_vel.device)
        joint_vel += vel_noise

    # clamp joint positions to limits
    joint_pos = joint_pos.clamp_(
        asset.data.soft_joint_pos_limits[env_ids, :, 0], asset.data.soft_joint_pos_limits[env_ids, :, 1]
    )

    # set joint positions and velocities
    asset.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def reset_root_state_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict = None,
    velocity_range: dict = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the robot root state with uniform sampling."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # set defaults if None
    if pose_range is None:
        pose_range = {}
    if velocity_range is None:
        velocity_range = {}

    # get default root state
    root_state = asset.data.default_root_state[env_ids].clone()

    # sample pose
    if "x" in pose_range:
        root_state[:, 0] += sample_uniform(
            pose_range["x"][0], pose_range["x"][1], (len(env_ids), 1), env.device
        ).squeeze(-1)
    if "y" in pose_range:
        root_state[:, 1] += sample_uniform(
            pose_range["y"][0], pose_range["y"][1], (len(env_ids), 1), env.device
        ).squeeze(-1)
    if "z" in pose_range:
        root_state[:, 2] += sample_uniform(
            pose_range["z"][0], pose_range["z"][1], (len(env_ids), 1), env.device
        ).squeeze(-1)

    # sample yaw orientation
    if "yaw" in pose_range:
        yaw = sample_uniform(
            pose_range["yaw"][0], pose_range["yaw"][1], (len(env_ids), 1), env.device
        ).squeeze(-1)
        # convert yaw to quaternion (w, x, y, z)
        root_state[:, 3] = torch.cos(yaw / 2.0)  # w
        root_state[:, 4] = 0.0  # x
        root_state[:, 5] = 0.0  # y
        root_state[:, 6] = torch.sin(yaw / 2.0)  # z

    # sample linear velocity
    for axis, idx in zip(["x", "y", "z"], [7, 8, 9]):
        if axis in velocity_range:
            root_state[:, idx] = sample_uniform(
                velocity_range[axis][0], velocity_range[axis][1], (len(env_ids), 1), env.device
            ).squeeze(-1)

    # sample angular velocity
    for axis, idx in zip(["roll", "pitch", "yaw"], [10, 11, 12]):
        if axis in velocity_range:
            root_state[:, idx] = sample_uniform(
                velocity_range[axis][0], velocity_range[axis][1], (len(env_ids), 1), env.device
            ).squeeze(-1)

    # set root state
    root_state[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_state_to_sim(root_state, env_ids)


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when episode length is reached."""
    return env.episode_length_buf >= env.max_episode_length - 1


def push_by_setting_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Push robot by setting random velocities to test robustness."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get current root state
    root_state = asset.data.root_state_w[env_ids].clone()
    
    # sample random velocities
    if "x" in velocity_range:
        root_state[:, 7] = sample_uniform(
            velocity_range["x"][0], velocity_range["x"][1], (len(env_ids), 1), env.device
        ).squeeze(-1)
    if "y" in velocity_range:
        root_state[:, 8] = sample_uniform(
            velocity_range["y"][0], velocity_range["y"][1], (len(env_ids), 1), env.device
        ).squeeze(-1)
    
    # set the root state
    asset.write_root_state_to_sim(root_state, env_ids)
