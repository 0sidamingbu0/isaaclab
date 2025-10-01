# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for OceanBDX locomotion environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for linear velocity along x-axis tracking."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get command
    cmd = env.command_manager.get_command("base_velocity")
    # compute the reward
    return torch.square(cmd[:, 0] - asset.data.root_lin_vel_b[:, 0])


def lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for linear velocity along y-axis tracking."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get command
    cmd = env.command_manager.get_command("base_velocity")
    # compute the reward
    return torch.square(cmd[:, 1] - asset.data.root_lin_vel_b[:, 1])


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for angular velocity around z-axis tracking."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get command
    cmd = env.command_manager.get_command("base_velocity")
    # compute the reward
    return torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for linear velocity along z-axis (vertical motion)."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for angular velocity around x and y-axis."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for non-flat base orientation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # penalize deviation from gravity direction
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def dof_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L2 penalty on joint torques."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def dof_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L2 penalty on joint accelerations."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on action differences."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0
) -> torch.Tensor:
    """Penalty for contacts at undesired body parts."""
    # extract the used quantities (to enable type-hinting)
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # undesired contacts: any contact except feet
    # assume first 2 contacts are feet, rest are undesired
    contact_forces = sensor.data.net_forces_w_history
    # compute contact force magnitude
    contact_force_norm = torch.norm(contact_forces.view(env.num_envs, -1, 3), dim=-1)
    # penalty for undesired contacts (exclude feet contacts)
    # threshold and penalty
    penalty = torch.sum((contact_force_norm > threshold).float(), dim=-1)
    return penalty


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    command_name: str = "base_velocity",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward long step time steps (feet in air)."""
    # extract the used quantities (to enable type-hinting)
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # get command
    command = env.command_manager.get_command(command_name)[:, :2]
    # compute the reward
    contact_forces = sensor.data.net_forces_w_history
    # compute contact force magnitude for feet
    feet_contact_forces = contact_forces.view(env.num_envs, -1, 3)[:, :2]  # first 2 are feet
    contact_force_norm = torch.norm(feet_contact_forces, dim=-1)
    # check if in contact
    in_contact = contact_force_norm > threshold
    # reward for longer air time when walking
    reward = (1.0 - in_contact.float()) * torch.norm(command, dim=1).unsqueeze(-1)
    return torch.sum(reward, dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.4
) -> torch.Tensor:
    """Reward for maintaining base height."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute height error
    base_height = asset.data.root_pos_w[:, 2]
    return torch.square(base_height - target_height)


def energy_expenditure(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for energy expenditure (torque * velocity)."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute power consumption
    power = torch.abs(asset.data.applied_torque * asset.data.joint_vel)
    return torch.sum(power, dim=1)


def joint_deviation_l1(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """L1 penalty on joint position deviation from default."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute joint deviation
    joint_deviation = torch.abs(asset.data.joint_pos - asset.data.default_joint_pos)
    return torch.sum(joint_deviation, dim=1)


def is_alive(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for staying alive (not terminating)."""
    return 1.0 - env.termination_manager.terminated.float()


def is_healthy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.2,
    max_height: float = 1.0,
) -> torch.Tensor:
    """Reward for staying healthy (proper height and orientation)."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check height bounds
    base_height = asset.data.root_pos_w[:, 2]
    height_ok = (base_height > min_height) & (base_height < max_height)
    # check orientation
    up_proj = asset.data.projected_gravity_b[:, 2]  # z component of gravity in base frame
    orientation_ok = up_proj < -0.5  # cos(120°) approximately
    # combine conditions
    healthy = height_ok & orientation_ok
    return healthy.float()


def track_lin_vel_xy_exp(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for tracking linear velocity commands (xy) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get command
    cmd = env.command_manager.get_command(command_name)
    # compute velocity error
    vel_error = torch.sum(torch.square(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-vel_error / std**2)


def track_ang_vel_z_exp(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for tracking angular velocity commands (z) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get command
    cmd = env.command_manager.get_command(command_name)
    # compute velocity error
    vel_error = torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-vel_error / std**2)


def base_pitch_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Specific penalty for pitch angle (forward/backward bending)."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get pitch component (y-axis rotation in base frame)
    pitch_component = asset.data.projected_gravity_b[:, 1]  # gravity y-component in base frame
    return torch.square(pitch_component)


def knee_position_reward(env: ManagerBasedRLEnv,
                        target_angle: float = 0.0,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for keeping knee joints (leg_l4, leg_r4) close to target angle."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get all joint names and find knee joint indices (only once globally)
    if not hasattr(knee_position_reward, '_knee_indices'):
        joint_names = asset.data.joint_names
        knee_indices = []
        
        # Debug: print joint names and indices (only once globally)
        print("Joint names and indices:")
        for i, name in enumerate(joint_names):
            print(f"  {i}: {name}")
        
        # Find knee joint indices
        for i, name in enumerate(joint_names):
            if name in ["leg_l4_joint", "leg_r4_joint"]:
                knee_indices.append(i)
                print(f"Found knee joint: {name} at index {i}")
        
        # Cache the indices globally
        knee_position_reward._knee_indices = knee_indices
        
        if len(knee_indices) == 0:
            print("Warning: No knee joints found!")
    
    # Use cached indices
    knee_indices = knee_position_reward._knee_indices
    
    # If no knee joints found, return zero reward
    if len(knee_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Get current knee positions
    knee_positions = asset.data.joint_pos[:, knee_indices]
    
    # Calculate deviation from target angle
    knee_error = torch.sum(torch.square(knee_positions - target_angle), dim=1)
    
    # Return reward (exponential kernel)
    return torch.exp(-knee_error / 0.25)  # σ² = 0.25


def upright_posture_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for maintaining upright posture (body vertical)."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the z-component of projected gravity (should be close to -1 for upright posture)
    upright_component = asset.data.projected_gravity_b[:, 2]
    target_upright = -1.0  # Perfect upright posture
    
    # Reward for being close to upright
    upright_error = torch.square(upright_component - target_upright)
    return torch.exp(-upright_error / 0.1)  # Tight tolerance for upright posture


# Aliases for common reward functions used in complete config
joint_torques_l2 = dof_torques_l2
joint_acc_l2 = dof_acc_l2


def contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 0.5,
) -> torch.Tensor:
    """Reward for contact forces at feet (encouraging ground contact)."""
    # extract the used quantities (to enable type-hinting)
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # get contact forces
    contact_forces = sensor.data.net_forces_w_history
    # compute contact force magnitude
    contact_force_norm = torch.norm(contact_forces.view(env.num_envs, -1, 3), dim=-1)
    # reward for contact above threshold
    contact_reward = (contact_force_norm > threshold).float()
    return torch.sum(contact_reward, dim=-1)


def gait_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for alternating gait pattern (encouraging walking)."""
    # Get contact sensor data for both feet
    try:
        left_sensor: ContactSensor = env.scene["contact_forces_LF"]
        right_sensor: ContactSensor = env.scene["contact_forces_RF"]
        
        # Get contact forces
        left_forces = left_sensor.data.net_forces_w_history
        right_forces = right_sensor.data.net_forces_w_history
        
        # Compute contact force magnitude for each foot
        left_contact_norm = torch.norm(left_forces.view(env.num_envs, -1, 3), dim=-1)
        right_contact_norm = torch.norm(right_forces.view(env.num_envs, -1, 3), dim=-1)
        
        # Determine if each foot is in contact (above threshold)
        # 🔧 降低阈值到0.3N，能识别"点地"状态，避免误判为"离地"
        left_in_contact = (torch.sum(left_contact_norm, dim=-1) > 0.3).float()
        right_in_contact = (torch.sum(right_contact_norm, dim=-1) > 0.3).float()
        
        # Reward for alternating pattern: one foot up, one foot down
        # XOR logic: reward when exactly one foot is in contact
        alternating_pattern = torch.abs(left_in_contact - right_in_contact)
        
        # Also reward for having at least one foot in contact (stability)
        at_least_one_contact = torch.clamp(left_in_contact + right_in_contact, 0.0, 1.0)
        
        # Combine both: encourage alternating but penalize double flight phase
        return alternating_pattern * at_least_one_contact
        
    except KeyError:
        # Fallback: if contact sensors not available, return zero reward
        return torch.zeros(env.num_envs, device=env.device)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_clearance: float = 0.02,  # 2cm minimum foot clearance
) -> torch.Tensor:
    """Reward for lifting feet during swing phase."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get positions of foot links (assume last links in each leg are feet)
    # This is a simplified version - you might need to adjust based on your robot structure
    
    # For now, use joint velocities as a proxy for foot movement
    # Higher joint velocities indicate more dynamic movement (walking vs standing)
    joint_vels = asset.data.joint_vel
    
    # Focus on leg joints (exclude neck joints)
    leg_joint_vels = joint_vels[:, :10]  # Assuming first 10 joints are legs
    
    # Reward for moderate joint velocities (not too fast, not too slow)
    vel_magnitude = torch.norm(leg_joint_vels, dim=1)
    
    # Optimal velocity range for walking
    optimal_vel = 2.0
    vel_reward = torch.exp(-torch.square(vel_magnitude - optimal_vel) / 2.0)
    
    return vel_reward


def air_time_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    threshold: float = 0.1,  # 最小速度阈值，低于此速度不需要抬脚
) -> torch.Tensor:
    """Reward for foot air time when robot should be walking."""
    # Get commanded velocity to determine if robot should be walking
    cmd = env.command_manager.get_command(command_name)
    speed_cmd = torch.norm(cmd[:, :2], dim=1)  # linear velocity magnitude
    
    # Only reward air time when there's a movement command
    should_walk = (speed_cmd > threshold).float()
    
    try:
        # Get contact sensor data for both feet
        left_sensor: ContactSensor = env.scene["contact_forces_LF"]
        right_sensor: ContactSensor = env.scene["contact_forces_RF"]
        
        # Get contact forces
        left_forces = left_sensor.data.net_forces_w_history
        right_forces = right_sensor.data.net_forces_w_history
        
        # Compute contact state (1 = in contact, 0 = in air)
        left_contact_norm = torch.norm(left_forces.view(env.num_envs, -1, 3), dim=-1)
        right_contact_norm = torch.norm(right_forces.view(env.num_envs, -1, 3), dim=-1)
        
        # Determine contact state (use lower threshold for air detection)
        left_in_contact = (torch.sum(left_contact_norm, dim=-1) > 1.0).float()
        right_in_contact = (torch.sum(right_contact_norm, dim=-1) > 1.0).float()
        
        # Calculate air time: reward when at least one foot is in air
        left_air_time = 1.0 - left_in_contact  # 1 when in air, 0 when in contact
        right_air_time = 1.0 - right_in_contact
        
        # Reward air time but ensure at least one foot maintains contact for stability
        # Ideal pattern: one foot in air, one foot in contact
        single_foot_air = (left_air_time * right_in_contact) + (right_air_time * left_in_contact)
        
        # Scale reward by movement command - more air time needed when moving faster
        air_time_reward = single_foot_air * should_walk * torch.clamp(speed_cmd, 0.0, 2.0)
        
        return air_time_reward
        
    except KeyError:
        # Fallback: if contact sensors not available, return zero reward
        return torch.zeros(env.num_envs, device=env.device)


def step_frequency_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    target_freq: float = 2.0,  # 目标步频 2Hz (每秒2步)
) -> torch.Tensor:
    """Reward for maintaining appropriate step frequency based on speed."""
    # Get commanded velocity
    cmd = env.command_manager.get_command(command_name)
    speed_cmd = torch.norm(cmd[:, :2], dim=1)
    
    try:
        # Get contact sensor data
        left_sensor: ContactSensor = env.scene["contact_forces_LF"]
        right_sensor: ContactSensor = env.scene["contact_forces_RF"]
        
        # Get contact state
        left_forces = left_sensor.data.net_forces_w_history
        right_forces = right_sensor.data.net_forces_w_history
        
        left_contact_norm = torch.norm(left_forces.view(env.num_envs, -1, 3), dim=-1)
        right_contact_norm = torch.norm(right_forces.view(env.num_envs, -1, 3), dim=-1)
        
        left_in_contact = (torch.sum(left_contact_norm, dim=-1) > 1.0).float()
        right_in_contact = (torch.sum(right_contact_norm, dim=-1) > 1.0).float()
        
        # Track contact state changes to estimate step frequency
        if not hasattr(step_frequency_reward, '_prev_left_contact'):
            step_frequency_reward._prev_left_contact = left_in_contact.clone()
            step_frequency_reward._prev_right_contact = right_in_contact.clone()
            step_frequency_reward._step_counter = torch.zeros_like(left_in_contact)
        
        # Detect foot strikes (transition from air to ground)
        left_strike = (left_in_contact > step_frequency_reward._prev_left_contact).float()
        right_strike = (right_in_contact > step_frequency_reward._prev_right_contact).float()
        
        # Count steps
        step_frequency_reward._step_counter += left_strike + right_strike
        
        # Update previous state
        step_frequency_reward._prev_left_contact = left_in_contact.clone()
        step_frequency_reward._prev_right_contact = right_in_contact.clone()
        
        # Calculate desired step frequency based on speed
        # Faster movement = higher step frequency
        desired_freq = target_freq * torch.clamp(speed_cmd / 1.0, 0.1, 2.0)
        
        # For now, return a simple reward based on movement
        # Full frequency analysis would require longer time windows
        movement_reward = torch.clamp(speed_cmd, 0.0, 1.0) * (left_strike + right_strike)
        
        return movement_reward
        
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)


def hip_abduction_reward(env: ManagerBasedRLEnv,
                         target_angle: float = 0.0,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for keeping hip abduction joints (leg_l1, leg_r1) close to target angle to prevent pigeon-toed gait."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get all joint names and find hip abduction joint indices (only once globally)
    if not hasattr(hip_abduction_reward, '_hip_indices'):
        joint_names = asset.data.joint_names
        hip_indices = []
        
        # Find hip abduction joint indices (leg_x1 joints)
        for i, name in enumerate(joint_names):
            if name in ["leg_l1_joint", "leg_r1_joint"]:
                hip_indices.append(i)
                print(f"Found hip abduction joint: {name} at index {i}")
        
        # Cache the indices globally
        hip_abduction_reward._hip_indices = hip_indices
        
        if len(hip_indices) == 0:
            print("Warning: No hip abduction joints found!")
    
    # Use cached indices
    hip_indices = hip_abduction_reward._hip_indices
    
    # If no hip joints found, return zero reward
    if len(hip_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Get current hip abduction positions
    hip_positions = asset.data.joint_pos[:, hip_indices]
    
    # Calculate deviation from target angle (0.0 for parallel legs)
    hip_error = torch.sum(torch.square(hip_positions - target_angle), dim=1)
    
    # Return exponential reward (higher reward for smaller error)
    return torch.exp(-10.0 * hip_error)


def track_base_height_exp(env: ManagerBasedRLEnv,
                          target_height: float = 0.4,
                          std: float = 0.1,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for tracking target base height using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get current height
    current_height = asset.data.root_pos_w[:, 2]
    
    # compute height error
    height_error = torch.square(current_height - target_height)
    
    # return exponential reward (higher reward for smaller error)
    return torch.exp(-height_error / std**2)
