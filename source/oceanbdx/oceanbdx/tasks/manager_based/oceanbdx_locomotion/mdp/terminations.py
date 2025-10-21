# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for OceanBDX locomotion environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float = 1.57,  # 90 degrees in radians
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the robot orientation is too far from upright using robot base orientation."""
    # Use robot base orientation directly (more reliable than IMU)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get base quaternion in world frame (w, x, y, z)
    quat = asset.data.root_quat_w
    
    # Extract quaternion components
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Normalize quaternion to ensure unit length
    quat_norm = torch.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / quat_norm, x / quat_norm, y / quat_norm, z / quat_norm
    
    # Convert to rotation matrix and extract roll/pitch more robustly
    # Roll (rotation around x-axis) - side-to-side tilt
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    
    # Pitch (rotation around y-axis) - forward/backward tilt
    sin_pitch = torch.clamp(2 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    
    # Alternative method: use the z-component of the up vector
    # Up vector in world frame should be (0, 0, 1) for upright robot
    up_vector_z = 1 - 2 * (x * x + y * y)  # z-component of rotated up vector
    tilt_angle = torch.acos(torch.clamp(torch.abs(up_vector_z), 0.0, 1.0))
    
    # Check termination condition - use the more conservative approach
    tolerance = 0.05  # Small tolerance in radians
    
    # Method 1: Individual roll/pitch limits
    roll_exceeded = torch.abs(roll) > (limit_angle - tolerance)
    pitch_exceeded = torch.abs(pitch) > (limit_angle - tolerance)
    angle_terminate = roll_exceeded | pitch_exceeded
    
    # Method 2: Overall tilt angle (more robust)
    tilt_exceeded = tilt_angle > (limit_angle - tolerance)
    
    # Use both methods for maximum safety
    should_terminate = angle_terminate | tilt_exceeded
    
    # Debug: print angles and termination status every 500 steps
    if not hasattr(bad_orientation, '_step_count'):
        bad_orientation._step_count = 0
    bad_orientation._step_count += 1
    
    if bad_orientation._step_count % 24 == 0:
        max_roll_deg = torch.max(torch.abs(roll)).item() * 180.0 / 3.14159
        max_pitch_deg = torch.max(torch.abs(pitch)).item() * 180.0 / 3.14159
        max_tilt_deg = torch.max(tilt_angle).item() * 180.0 / 3.14159
        limit_deg = (limit_angle - tolerance) * 180.0 / 3.14159
        num_terminated = torch.sum(should_terminate).item()
        
        # 添加高度信息
        current_height = torch.mean(asset.data.root_pos_w[:, 2]).item()
        min_height = torch.min(asset.data.root_pos_w[:, 2]).item()
        max_height = torch.max(asset.data.root_pos_w[:, 2]).item()
        
        print(f"Angles - Roll: {max_roll_deg:.1f}°, Pitch: {max_pitch_deg:.1f}°, Tilt: {max_tilt_deg:.1f}°, Limit: {limit_deg:.1f}°, Height: {current_height:.2f}m({min_height:.2f}-{max_height:.2f}), Term: {num_terminated}/{env.num_envs}")
    
    # Terminate if either condition is met
    return should_terminate


def base_height(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.15,
    maximum_height: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the base height is below minimum or above maximum."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] < minimum_height) | (asset.data.root_pos_w[:, 2] > maximum_height)


def base_lin_vel(
    env: ManagerBasedRLEnv,
    max_velocity: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the base linear velocity exceeds maximum."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check linear velocity magnitude
    lin_vel_norm = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    return lin_vel_norm > max_velocity


def base_ang_vel(
    env: ManagerBasedRLEnv,
    max_velocity: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the base angular velocity exceeds maximum."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check angular velocity magnitude
    ang_vel_norm = torch.norm(asset.data.root_ang_vel_w, dim=1)
    return ang_vel_norm > max_velocity


def joint_pos_out_of_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if any joint position is out of its limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check joint limits (we use a small margin for safety)
    out_of_limits = torch.any(
        (asset.data.joint_pos < asset.data.soft_joint_pos_limits[..., 0])
        | (asset.data.joint_pos > asset.data.soft_joint_pos_limits[..., 1]),
        dim=1,
    )
    return out_of_limits


def joint_vel_out_of_limit(
    env: ManagerBasedRLEnv,
    max_velocity: float = 50.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if any joint velocity exceeds maximum."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check joint velocity limits
    out_of_limits = torch.any(torch.abs(asset.data.joint_vel) > max_velocity, dim=1)
    return out_of_limits


def undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float = 5.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Terminate if there are strong contacts at undesired body parts."""
    # extract the used quantities (to enable type-hinting)
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # get contact forces
    contact_forces = sensor.data.net_forces_w_history
    # compute contact force magnitude
    contact_force_norm = torch.norm(contact_forces.view(env.num_envs, -1, 3), dim=-1)
    # check for undesired contacts (skip feet contacts)
    undesired_contact_forces = contact_force_norm[:, 2:]  # skip first 2 (feet)
    # terminate if any undesired contact exceeds threshold
    terminate = torch.any(undesired_contact_forces > threshold, dim=1)
    return terminate


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when episode length is reached."""
    return env.episode_length_buf >= env.max_episode_length - 1


def knee_ground_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 5.0,
) -> torch.Tensor:
    """Terminate if any knee contacts the ground with force above threshold.
    
    This prevents the robot from crawling or falling with knees touching ground,
    encouraging proper bipedal locomotion.
    
    Args:
        env: The environment instance.
        threshold: Contact force threshold in Newtons to trigger termination.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    # Get knee contact sensors (hardcoded names, similar to foot sensors)
    left_knee_sensor: ContactSensor = env.scene["contact_forces_knee_L"]
    right_knee_sensor: ContactSensor = env.scene["contact_forces_knee_R"]
    
    # Get contact forces from both knees
    # Shape: [num_envs, history_length, 3] for single body contact sensors
    left_forces = left_knee_sensor.data.net_forces_w_history
    right_forces = right_knee_sensor.data.net_forces_w_history
    
    # Compute force magnitude (use view to handle potential multi-body format)
    # Reshape to [num_envs, -1, 3] and compute norm
    left_contact_norm = torch.norm(left_forces.view(env.num_envs, -1, 3), dim=-1)
    right_contact_norm = torch.norm(right_forces.view(env.num_envs, -1, 3), dim=-1)
    
    # Get maximum force across all timesteps/bodies per environment
    left_force_max = torch.max(left_contact_norm, dim=-1)[0]  # [num_envs]
    right_force_max = torch.max(right_contact_norm, dim=-1)[0]  # [num_envs]
    
    # Check if either knee exceeds threshold
    left_contact = left_force_max > threshold
    right_contact = right_force_max > threshold
    terminate = left_contact | right_contact
    
    # Debug: Print knee contact information periodically
    if not hasattr(knee_ground_contact, '_step_count'):
        knee_ground_contact._step_count = 0
    knee_ground_contact._step_count += 1

    if knee_ground_contact._step_count % 24 == 0:  # Every ~1 second at 24Hz
        num_terminated = torch.sum(terminate).item()
        if num_terminated > 0:
            max_left = torch.max(left_force_max).item()
            max_right = torch.max(right_force_max).item()
            print(f"🦵 Knee Contact - L: {max_left:.1f}N, R: {max_right:.1f}N, Threshold: {threshold:.1f}N, Terminated: {num_terminated}/{env.num_envs}")
    
    return terminate
