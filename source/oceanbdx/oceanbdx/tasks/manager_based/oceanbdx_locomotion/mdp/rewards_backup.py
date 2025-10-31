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


def foot_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    min_force: float = 2.0,  # 最小有效接触力 (N)
    max_force: float = 200.0,  # 最大合理接触力 (N)，避免奖励撞击
    target_force: float = 50.0,  # 目标接触力 (N)，正常站立的力
    min_z_ratio: float = 0.2,  # Z向力占总力的最小比例，作为奖励的尺度因子
) -> torch.Tensor:
    """
    Reward for proper foot contact with ground - 简化版：只用Z向力判断接触质量.
    
    判断策略（简化，适用于复杂地形）:
    1. 接触力在合理范围内 (2N-200N)
    2. 接触力主要向上 (Z向法向力比例，起始值20%，与奖励成正比)
    3. 组合奖励：接触力质量 × Z向力比例系数
    
    优势：
    - 避免角度判断在不平地面的误判
    - 专注于力的大小和方向，更可靠
    - 适用于各种地形和坡度
    - 平滑奖励：比例越高奖励越大
    """
    try:
        # extract the used quantities (to enable type-hinting)
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        
        # get contact forces
        contact_forces = sensor.data.net_forces_w_history  # 世界坐标系下的接触力
        
        # 1. 接触力大小检查
        contact_force_norm = torch.norm(contact_forces.view(env.num_envs, -1, 3), dim=-1)
        total_contact_force = torch.sum(contact_force_norm, dim=-1)  # 每个环境的总接触力
        
        # 【关键】只处理合理范围内的接触力
        valid_contact_mask = (total_contact_force >= min_force) & (total_contact_force <= max_force)
        
        # 2. 【核心】Z向力（法向力）检查 - 判断是否主要为向上的支撑力
        if contact_forces.shape[-1] == 3:  # 确保有3D力向量
            contact_forces_3d = contact_forces.view(env.num_envs, -1, 3)
            
            # 计算Z方向(向上)的力分量
            total_force_z = torch.sum(contact_forces_3d[:, :, 2], dim=1)  # Z方向总力
            
            # 确保Z向力为正值（向上支撑）并计算比例
            upward_force_z = torch.clamp(total_force_z, 0.0, float('inf'))  # 只考虑向上的力
            force_z_ratio = upward_force_z / (total_contact_force + 1e-6)  # 向上Z向力比例
            
            # 【新策略】Z向力与奖励成正比例：比例越高奖励越大
            # 使用线性缩放，从 min_z_ratio 开始给奖励，到 1.0 时达到最大奖励
            z_force_quality = torch.clamp(
                (force_z_ratio - min_z_ratio) / (1.0 - min_z_ratio),  # 线性映射到 [0, 1]
                0.0, 1.0
            )  # 从 min_z_ratio 开始给奖励，比例越高奖励越大
        else:
            z_force_quality = torch.ones(env.num_envs, device=env.device) * 0.5
        
        # 3. 接触力大小质量奖励 (在目标力附近最高)
        force_error = torch.abs(total_contact_force - target_force) / (target_force * 0.5)
        contact_magnitude_quality = torch.exp(-0.5 * torch.square(force_error))
        
        # 4. 简化的综合奖励：接触力大小质量 × Z向力质量
        # 两个条件都满足才能获得高奖励
        combined_reward = contact_magnitude_quality * z_force_quality
        
        # 只对有效接触给奖励
        foot_reward = torch.where(valid_contact_mask, combined_reward, torch.zeros_like(combined_reward))
        
        return foot_reward
        
    except (KeyError, RuntimeError):
        # Fallback: if any error occurs, return zero reward
        return torch.zeros(env.num_envs, device=env.device)


def gait_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for true alternating gait pattern based on state transitions."""
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
        left_in_contact = (torch.sum(left_contact_norm, dim=-1) > 1.0).float()
        right_in_contact = (torch.sum(right_contact_norm, dim=-1) > 1.0).float()
        
        # Initialize gait history in environment if not exists
        if not hasattr(env, '_gait_history'):
            env._gait_history = {
                'prev_left': torch.zeros(env.num_envs, device=env.device),
                'prev_right': torch.zeros(env.num_envs, device=env.device),
                'last_transition_step': torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
                'current_step': torch.zeros(1, device=env.device, dtype=torch.long)
            }
        
        # Get previous contact states
        prev_left = env._gait_history['prev_left']
        prev_right = env._gait_history['prev_right']
        last_transition = env._gait_history['last_transition_step']
        current_step = env._gait_history['current_step']
        
        # Detect state transitions (contact state changes)
        left_changed = torch.abs(left_in_contact - prev_left) > 0.5
        right_changed = torch.abs(right_in_contact - prev_right) > 0.5
        
        # True alternating gait: when one foot changes state, the other should be stable
        # AND the change should create an alternating pattern (one up, one down)
        state_transition = left_changed | right_changed
        alternating_pattern = torch.abs(left_in_contact - right_in_contact) > 0.5
        
        # Reward for proper alternating transitions
        proper_transition = state_transition & alternating_pattern
        
        # Additional rewards for:
        # 1. Having at least one foot in contact (stability)
        stability_reward = torch.clamp(left_in_contact + right_in_contact, 0.0, 1.0)
        
        # Update transition tracking
        transition_mask = proper_transition
        last_transition[transition_mask] = current_step[0]
        
        # Penalize if no transition for too long (static gait)
        steps_since_transition = current_step[0] - last_transition
        static_penalty = (steps_since_transition > 50).float() * -0.5
        
        # Update history for next step
        env._gait_history['prev_left'] = left_in_contact.clone()
        env._gait_history['prev_right'] = right_in_contact.clone()
        env._gait_history['current_step'] += 1
        
        # Reset history for environments that just reset
        if hasattr(env, '_reset_env_ids') and len(env._reset_env_ids) > 0:
            env._gait_history['prev_left'][env._reset_env_ids] = 0.0
            env._gait_history['prev_right'][env._reset_env_ids] = 0.0
            env._gait_history['last_transition_step'][env._reset_env_ids] = current_step[0]
        
        # Combine rewards: transition reward + stability - static penalty
        total_reward = proper_transition.float() * 2.0 + stability_reward * 0.1 + static_penalty
        
        return total_reward
        
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


def step_length_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    min_step_length: float = 0.02,  # 降低最小步长到 2cm (更宽松)
    target_step_length: float = 0.08,  # 降低目标步长到 8cm (更现实)
    max_step_length: float = 0.2,  # 降低最大步长到 20cm
) -> torch.Tensor:
    """
    Reward for taking proper step lengths - 奖励与步长成指数正比，越长奖励越大.
    
    奖励策略（指数增长设计）:
    1. 微步惩罚: 步长 < min_step_length 时给予轻微惩罚
    2. 指数步长奖励: reward = exp(step_length / target_step_length) - 1 (与步长成指数正比)
    3. 超大步指数奖励: 步长超过目标时给予指数增长的额外奖励
    4. 前进方向指数奖励: 对x方向前进距离给予指数奖励
    
    指数设计的优势:
    - 小步长: 奖励较小 (例如: 0.5倍目标步长 → 奖励≈0.65)
    - 目标步长: 奖励适中 (例如: 1.0倍目标步长 → 奖励≈1.72)
    - 大步长: 奖励快速增长 (例如: 2.0倍目标步长 → 奖励≈6.39)
    
    这种设计强烈鼓励机器人迈更大的步而非小碎步。
    """
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
        
        # Get robot base position for step length calculation
        asset: Articulation = env.scene["robot"]
        current_pos = asset.data.root_pos_w[:, :2]  # x, y position only
        
        # Get commanded velocity to determine expected movement
        cmd = env.command_manager.get_command(command_name)
        speed_cmd = torch.norm(cmd[:, :2], dim=1)  # linear velocity magnitude
        
        # Initialize step length tracking
        if not hasattr(env, '_step_length_history'):
            env._step_length_history = {
                'prev_left': torch.zeros(env.num_envs, device=env.device),
                'prev_right': torch.zeros(env.num_envs, device=env.device),
                'step_start_pos': torch.zeros((env.num_envs, 2), device=env.device),
                'last_step_pos': torch.zeros((env.num_envs, 2), device=env.device),
                'step_in_progress': torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
                'debug_counter': torch.zeros(1, device=env.device, dtype=torch.long),  # 添加调试计数器
            }
        
        history = env._step_length_history
        
        # Detect step initiation (foot lifts off)
        left_lift_off = (history['prev_left'] > left_in_contact).float()
        right_lift_off = (history['prev_right'] > right_in_contact).float()
        
        # Detect step completion (foot touches down)
        left_touch_down = (left_in_contact > history['prev_left']).float()
        right_touch_down = (right_in_contact > history['prev_right']).float()
        
        # Track step initiation
        step_initiated = (left_lift_off + right_lift_off) > 0.5
        step_completed = (left_touch_down + right_touch_down) > 0.5
        
        # 调试信息：每100步打印一次状态
        history['debug_counter'] += 1
        if history['debug_counter'] % 100 == 0:
            num_active_steps = torch.sum(history['step_in_progress']).item()
            num_initiated = torch.sum(step_initiated).item()
            num_completed = torch.sum(step_completed).item()
            print(f"Step debug - Active: {num_active_steps}, Initiated: {num_initiated}, Completed: {num_completed}")
        
        # Update step start position when step begins
        start_new_step = step_initiated & (~history['step_in_progress'])
        history['step_start_pos'] = torch.where(
            start_new_step.unsqueeze(1),
            current_pos,
            history['step_start_pos']
        )
        history['step_in_progress'] = torch.where(start_new_step, True, history['step_in_progress'])
        
        # Calculate step length when step completes
        step_rewards = torch.zeros(env.num_envs, device=env.device)
        
        # Only calculate reward for completed steps
        completed_step_mask = step_completed & history['step_in_progress']
        
        if completed_step_mask.any():
            # Calculate actual step length (distance traveled)
            step_distance = torch.norm(current_pos - history['step_start_pos'], dim=1)
            
            # 调试：打印步长信息
            if history['debug_counter'] % 100 == 0:
                completed_distances = step_distance[completed_step_mask]
                if len(completed_distances) > 0:
                    avg_dist = torch.mean(completed_distances).item()
                    min_dist = torch.min(completed_distances).item()
                    max_dist = torch.max(completed_distances).item()
                    print(f"Step distances - Avg: {avg_dist:.3f}m, Min: {min_dist:.3f}m, Max: {max_dist:.3f}m")
            
            # Reward based on step length quality
            # 1. 更温和的微步惩罚
            micro_step_penalty = torch.where(
                step_distance < min_step_length,
                torch.full_like(step_distance, -0.2),  # 进一步减少惩罚，避免抑制学习
                torch.zeros_like(step_distance)
            )
            
            # 2. 【关键改进】指数步长奖励 - 与步长成指数正比，越长奖励越大
            # 使用指数函数：reward = exp(step_length / target_step_length) - 1
            # 这样可以确保步长越大，奖励增长越快
            exponential_step_reward = torch.exp(
                torch.clamp(step_distance / target_step_length, 0.0, 4.0)  # 限制最大指数避免数值溢出
            ) - 1.0  # 减去1使得0步长时奖励为0
            
            # 3. 超大步额外指数奖励：步长超过目标时给予指数增长的额外奖励
            super_step_bonus = torch.where(
                step_distance > target_step_length,
                torch.exp(torch.clamp((step_distance - target_step_length) / target_step_length, 0.0, 2.0)) - 1.0,  # 指数增长的超大步奖励
                torch.zeros_like(step_distance)
            )
            
            # 4. 前进方向指数奖励 (鼓励x方向的大步前进)
            forward_progress = torch.abs(current_pos[:, 0] - history['step_start_pos'][:, 0])  # x-direction progress
            forward_exponential_bonus = torch.exp(
                torch.clamp(forward_progress / target_step_length, 0.0, 3.0)
            ) - 1.0  # 前进距离的指数奖励
            forward_exponential_bonus *= 0.3  # 缩放系数
            
            # 5. Scale reward by movement command (don't reward big steps when standing still)
            movement_scale = torch.clamp(speed_cmd, 0.3, 1.0)  # 提高最小缩放值，确保有运动命令时才给予大奖励
            
            # Combine all components (强调指数步长奖励，步长越大奖励越大)
            total_step_reward = micro_step_penalty + (exponential_step_reward + super_step_bonus + forward_exponential_bonus) * movement_scale
            
            # Apply reward only to environments that completed a step
            step_rewards = torch.where(
                completed_step_mask,
                total_step_reward,
                torch.zeros_like(total_step_reward)
            )
        
        # Reset step tracking for completed steps
        history['step_in_progress'] = torch.where(completed_step_mask, False, history['step_in_progress'])
        history['last_step_pos'] = torch.where(
            completed_step_mask.unsqueeze(1),
            current_pos,
            history['last_step_pos']
        )
        
        # Update previous contact state
        history['prev_left'] = left_in_contact.clone()
        history['prev_right'] = right_in_contact.clone()
        
        # Reset history for environments that just reset
        if hasattr(env, '_reset_env_ids') and len(env._reset_env_ids) > 0:
            history['prev_left'][env._reset_env_ids] = 0.0
            history['prev_right'][env._reset_env_ids] = 0.0
            history['step_start_pos'][env._reset_env_ids] = current_pos[env._reset_env_ids]
            history['last_step_pos'][env._reset_env_ids] = current_pos[env._reset_env_ids]
            history['step_in_progress'][env._reset_env_ids] = False
        
        return step_rewards
        
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)


def air_time_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    threshold: float = 0.1,  # 最小速度阈值，低于此速度不需要抬脚
    min_air_time: float = 5.0,  # 最小空中时间（控制步数），约0.1秒@50Hz
    target_air_time: float = 15.0,  # 目标空中时间（控制步数），约0.3秒@50Hz
) -> torch.Tensor:
    """
    Reward for sustained foot air time with exponential scaling.
    
    奖励策略（指数持续时间设计）:
    1. 跟踪每只脚的连续空中时间
    2. 指数奖励公式: reward = exp(air_duration / target_air_time) - 1
    3. 鼓励更长的摆动相，抑制高频微抬脚
    4. 确保至少一只脚保持接触以维持稳定性
    
    指数设计的优势:
    - 短暂抬脚: 奖励很小 (5步空中时间 → 小奖励)
    - 适中摆动: 奖励适中 (15步空中时间 → 标准奖励)
    - 长摆动: 奖励快速增长 (30步空中时间 → 大奖励)
    """
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
        
        # Initialize air time tracking with exponential rewards
        if not hasattr(env, '_air_time_history'):
            env._air_time_history = {
                'left_air_duration': torch.zeros(env.num_envs, device=env.device),
                'right_air_duration': torch.zeros(env.num_envs, device=env.device),
                'prev_left_contact': torch.ones(env.num_envs, device=env.device),  # 假设开始时在地面
                'prev_right_contact': torch.ones(env.num_envs, device=env.device),
                'debug_counter': torch.zeros(1, device=env.device, dtype=torch.long),
            }
        
        history = env._air_time_history
        
        # 计算当前空中状态
        left_in_air = 1.0 - left_in_contact
        right_in_air = 1.0 - right_in_contact
        
        # 更新空中持续时间（向量化操作）
        # 如果脚在空中，增加持续时间；如果接触地面，重置为0
        history['left_air_duration'] = torch.where(
            left_in_air > 0.5,
            history['left_air_duration'] + 1.0,  # 空中时增加计数
            torch.zeros_like(history['left_air_duration'])  # 接触时重置
        )
        
        history['right_air_duration'] = torch.where(
            right_in_air > 0.5,
            history['right_air_duration'] + 1.0,  # 空中时增加计数
            torch.zeros_like(history['right_air_duration'])  # 接触时重置
        )
        
        # 调试信息：每100步打印一次空中时间统计
        history['debug_counter'] += 1
        if history['debug_counter'] % 100 == 0:
            left_air_active = torch.sum(left_in_air > 0.5).item()
            right_air_active = torch.sum(right_in_air > 0.5).item()
            avg_left_duration = torch.mean(history['left_air_duration'][left_in_air > 0.5]).item() if left_air_active > 0 else 0
            avg_right_duration = torch.mean(history['right_air_duration'][right_in_air > 0.5]).item() if right_air_active > 0 else 0
            max_left_duration = torch.max(history['left_air_duration']).item()
            max_right_duration = torch.max(history['right_air_duration']).item()
            
            # 换算成秒（50Hz控制频率，每步0.02秒）
            avg_left_seconds = avg_left_duration * 0.02
            avg_right_seconds = avg_right_duration * 0.02
            max_left_seconds = max_left_duration * 0.02
            max_right_seconds = max_right_duration * 0.02
            
            print(f"Air time debug - L active: {left_air_active}, R active: {right_air_active}")
            print(f"Air durations - L avg: {avg_left_seconds:.3f}s ({avg_left_duration:.1f} steps), R avg: {avg_right_seconds:.3f}s ({avg_right_duration:.1f} steps)")
            print(f"Air max times - L max: {max_left_seconds:.3f}s ({max_left_duration:.1f} steps), R max: {max_right_seconds:.3f}s ({max_right_duration:.1f} steps)")
        
        # 【关键改进】指数空中时间奖励计算 - 修复左右脚不平衡问题
        # 只有当空中时间超过最小阈值且在合理范围内时才给奖励
        max_reasonable_air_time = target_air_time * 2.0  # 最大合理空中时间：0.6秒
        
        left_qualified_air = torch.where(
            (history['left_air_duration'] >= min_air_time) & (history['left_air_duration'] <= max_reasonable_air_time),
            history['left_air_duration'],
            torch.zeros_like(history['left_air_duration'])
        )
        
        right_qualified_air = torch.where(
            (history['right_air_duration'] >= min_air_time) & (history['right_air_duration'] <= max_reasonable_air_time),
            history['right_air_duration'],
            torch.zeros_like(history['right_air_duration'])
        )
        
        # 【修复】使用更温和的奖励函数，避免指数爆炸
        # 使用限制性指数：reward = (1 - exp(-duration/target)) * max_reward
        # 这样在目标时间附近奖励最高，超出后不再增长
        max_air_reward = 2.0  # 最大空中时间奖励
        
        left_exponential_reward = torch.where(
            left_qualified_air > 0,
            (1.0 - torch.exp(-left_qualified_air / target_air_time)) * max_air_reward,
            torch.zeros_like(left_qualified_air)
        )
        
        right_exponential_reward = torch.where(
            right_qualified_air > 0,
            (1.0 - torch.exp(-right_qualified_air / target_air_time)) * max_air_reward,
            torch.zeros_like(right_qualified_air)
        )
        
        # 【新增】左右脚平衡奖励 - 鼓励双脚轮流摆动
        # 计算左右脚空中时间的差异，差异越小奖励越高
        left_air_ratio = history['left_air_duration'] / (target_air_time + 1e-6)
        right_air_ratio = history['right_air_duration'] / (target_air_time + 1e-6)
        air_time_balance = torch.exp(-torch.abs(left_air_ratio - right_air_ratio))  # 平衡奖励
        balance_bonus = air_time_balance * 0.5  # 平衡奖励权重
        
        # 稳定性约束：确保至少一只脚接触地面
        # 只有在单脚摆动时才给予奖励（另一只脚支撑）
        stable_left_swing = left_exponential_reward * right_in_contact  # 左脚摆动，右脚支撑
        stable_right_swing = right_exponential_reward * left_in_contact  # 右脚摆动，左脚支撑
        
        # 【新增】严格禁止双脚悬空 - 双脚都在空中时给予惩罚
        both_feet_airborne = (left_in_air > 0.5) & (right_in_air > 0.5)  # 检测双脚悬空
        airborne_penalty = torch.where(both_feet_airborne, torch.full_like(left_in_air, -2.0), torch.zeros_like(left_in_air))  # 双脚悬空惩罚
        
        # 【新增】鼓励至少一只脚接触地面的稳定性奖励
        at_least_one_contact = (left_in_contact > 0.5) | (right_in_contact > 0.5)  # 至少一只脚接触
        stability_bonus = torch.where(at_least_one_contact, torch.full_like(left_in_contact, 0.1), torch.zeros_like(left_in_contact))
        
        # 总的空中时间奖励：基础奖励 + 平衡奖励 + 稳定性奖励 - 双脚悬空惩罚
        basic_air_reward = stable_left_swing + stable_right_swing
        total_air_reward = basic_air_reward + balance_bonus + stability_bonus + airborne_penalty
        
        # 根据运动命令缩放奖励
        scaled_reward = total_air_reward * should_walk * torch.clamp(speed_cmd, 0.3, 1.5)
        
        # 调试：打印奖励统计
        if history['debug_counter'] % 100 == 0:
            num_rewarded = torch.sum(total_air_reward > 0.1).item()
            avg_reward = torch.mean(total_air_reward[total_air_reward > 0.1]).item() if num_rewarded > 0 else 0
            max_reward = torch.max(total_air_reward).item()
            
            # 新增：平衡性统计
            avg_balance = torch.mean(air_time_balance).item()
            left_right_ratio = avg_left_duration / (avg_right_duration + 1e-6) if avg_right_duration > 0 else float('inf')
            
            # 【新增】双脚悬空统计
            num_both_airborne = torch.sum(both_feet_airborne).item()
            num_single_support = torch.sum(at_least_one_contact).item()
            airborne_percentage = (num_both_airborne / env.num_envs) * 100.0
            
            print(f"Air time rewards - Rewarded envs: {num_rewarded}, Avg reward: {avg_reward:.3f}, Max reward: {max_reward:.3f}")
            print(f"Air time balance - L/R ratio: {left_right_ratio:.2f}, Balance score: {avg_balance:.3f} (1.0=perfect)")
            print(f"Stability check - Both airborne: {num_both_airborne} ({airborne_percentage:.1f}%), Single support: {num_single_support}")
            
            # 警告：检测异常的左右脚不平衡
            if left_right_ratio > 3.0 or left_right_ratio < 0.33:
                print(f"⚠️  WARNING: Severe foot imbalance detected! L/R ratio: {left_right_ratio:.2f}")
            if max_left_seconds > 1.0 or max_right_seconds > 1.0:
                print(f"⚠️  WARNING: Excessive air time detected! Max L: {max_left_seconds:.3f}s, Max R: {max_right_seconds:.3f}s")
            if airborne_percentage > 5.0:
                print(f"⚠️  WARNING: Too many robots with both feet airborne! {airborne_percentage:.1f}% > 5.0%")
        
        # Reset history for environments that just reset
        if hasattr(env, '_reset_env_ids') and len(env._reset_env_ids) > 0:
            history['left_air_duration'][env._reset_env_ids] = 0.0
            history['right_air_duration'][env._reset_env_ids] = 0.0
            history['prev_left_contact'][env._reset_env_ids] = 1.0
            history['prev_right_contact'][env._reset_env_ids] = 1.0
        
        return scaled_reward
        
    except KeyError:
        # Fallback: if contact sensors not available, return zero reward
        return torch.zeros(env.num_envs, device=env.device)


def step_frequency_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    target_freq: float = 1.0,  # 目标步频 1Hz (每秒1步，慢而稳)
    penalty_threshold: float = 2.0,  # 降低到2Hz开始惩罚，更严格控制
) -> torch.Tensor:
    """Penalty for high step frequency to encourage slow, stable gait (optimized version)."""
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
        
        # Initialize step frequency tracking (simplified)
        if not hasattr(env, '_step_freq_history'):
            env._step_freq_history = {
                'prev_left': torch.zeros(env.num_envs, device=env.device),
                'prev_right': torch.zeros(env.num_envs, device=env.device),
                'step_interval_counter': torch.zeros(env.num_envs, device=env.device),  # 简化：只跟踪步间隔
                'last_step_time': torch.zeros(env.num_envs, device=env.device),
                'current_time': torch.zeros(1, device=env.device),
                'debug_counter': torch.zeros(1, device=env.device, dtype=torch.long),  # 添加调试计数器
            }
        
        history = env._step_freq_history
        
        # Detect foot strikes (transition from air to ground) - 向量化操作
        left_strike = (left_in_contact > history['prev_left']).float()
        right_strike = (right_in_contact > history['prev_right']).float()
        
        # Any step occurred (向量化)
        step_occurred = (left_strike + right_strike) > 0.5
        
        # 调试信息：每100步打印一次步频状态
        history['debug_counter'] += 1
        if history['debug_counter'] % 100 == 0:
            num_steps = torch.sum(step_occurred).item()
            num_left_strikes = torch.sum(left_strike).item()
            num_right_strikes = torch.sum(right_strike).item()
            print(f"Step frequency debug - Steps: {num_steps}, L strikes: {num_left_strikes}, R strikes: {num_right_strikes}")
        
        # Update step timing (向量化操作，避免Python循环)
        current_time = history['current_time'][0]
        
        # 计算步间隔 (只对发生步态的环境)
        step_interval = current_time - history['last_step_time']
        
        # 更新上次步态时间
        history['last_step_time'] = torch.where(step_occurred, current_time, history['last_step_time'])
        
        # 简化的频率估计：基于步间隔的倒数 (向量化)
        # 步间隔以控制步为单位，转换为频率 (Hz)
        valid_interval_mask = (step_interval > 5.0) & (step_interval < 1000.0)  # 过滤异常值
        estimated_freq = torch.zeros_like(step_interval)
        
        # 频率 = 1 / (间隔秒数) = 50 / 间隔步数 (假设50Hz控制频率)
        estimated_freq[valid_interval_mask] = 50.0 / step_interval[valid_interval_mask]
        
        # 只对有效步态和合理间隔的环境计算惩罚
        active_mask = step_occurred & valid_interval_mask
        
        # 调试：打印步频信息
        if history['debug_counter'] % 100 == 0:
            active_frequencies = estimated_freq[active_mask]
            if len(active_frequencies) > 0:
                avg_freq = torch.mean(active_frequencies).item()
                min_freq = torch.min(active_frequencies).item()
                max_freq = torch.max(active_frequencies).item()
                num_high_freq = torch.sum(estimated_freq > penalty_threshold).item()
                print(f"Step frequencies - Avg: {avg_freq:.2f}Hz, Min: {min_freq:.2f}Hz, Max: {max_freq:.2f}Hz, High freq: {num_high_freq}")
        
        # Apply penalty for high frequency (向量化)
        freq_excess = torch.clamp(estimated_freq - penalty_threshold, 0.0, float('inf'))
        # 增强惩罚强度：使用平方惩罚而非线性惩罚
        frequency_penalty = torch.where(active_mask, -freq_excess**2, torch.zeros_like(freq_excess))
        
        # 额外奖励低频稳定步态 (向量化)
        stable_mask = active_mask & (estimated_freq > 0.5) & (estimated_freq <= target_freq * 1.2)
        stable_gait_bonus = torch.where(stable_mask, torch.full_like(estimated_freq, 0.5), torch.zeros_like(estimated_freq))
        
        # 调试：打印奖励分析
        if history['debug_counter'] % 100 == 0:
            num_penalized = torch.sum(frequency_penalty < 0).item()
            num_rewarded = torch.sum(stable_gait_bonus > 0).item()
            total_penalty = torch.sum(frequency_penalty).item()
            total_bonus = torch.sum(stable_gait_bonus).item()
            print(f"Step freq rewards - Penalized: {num_penalized}, Rewarded: {num_rewarded}, Total penalty: {total_penalty:.3f}, Total bonus: {total_bonus:.3f}")
        
        # Update history (向量化)
        history['prev_left'] = left_in_contact.clone()
        history['prev_right'] = right_in_contact.clone()
        history['current_time'] += 1
        
        # Reset history for environments that just reset (向量化)
        if hasattr(env, '_reset_env_ids') and len(env._reset_env_ids) > 0:
            history['prev_left'][env._reset_env_ids] = 0.0
            history['prev_right'][env._reset_env_ids] = 0.0
            history['last_step_time'][env._reset_env_ids] = current_time
        
        total_reward = frequency_penalty + stable_gait_bonus
        return total_reward
        
    except KeyError:
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
        
        # Simple reward based on movement and step occurrence
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


def gait_phase_reward(env: ManagerBasedRLEnv,
                      gait_period: float = 0.75,
                      std: float = 0.5,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for following reference gait trajectory.
    
    Args:
        env: The RL environment.
        gait_period: Period of the reference gait cycle in seconds (default: 0.75s).
        std: Standard deviation for exponential kernel (default: 0.5 rad).
        asset_cfg: Asset configuration.
    
    Returns:
        Exponential reward based on joint angle tracking error.
    """
    # Reference gait trajectory: 18 keyframes x 10 leg joints (degrees)
    # Joint order: L1, L2, L3, L4, L5, R1, R2, R3, R4, R5
    # Time interval: 0.0417s (approximately 24Hz sampling)
    reference_gait_deg = torch.tensor([
        [0, -15, -45, 0, 60, 0, 15, 45, 0, -60],
        [0, -10.6, -37.5, -7.5, 52.5, 0, 17.3, 48.8, 3.8, -60],
        [0, -6.2, -30, -15, 45, 0, 19.6, 52.5, 7.5, -60],
        [0, -1.9, -22.5, -22.5, 37.5, 0, 21.9, 56.2, 11.2, -60],
        [0, 2.5, -15, -30, 30, 0, 24.2, 60, 15, -60],
        [0, 6.9, -7.5, -37.5, 22.5, 0, 17.3, 48.8, 3.8, -45],
        [0, 11.2, 0, -45, 15, 0, 10.4, 37.5, -7.5, -30],
        [0, 15.6, 7.5, -52.5, 7.5, 0, 3.5, 26.2, -18.8, -15],
        [0, 20, 15, -60, 0, 0, -3.5, 15, -30, 0],
        [0, 17.3, 18.8, -52.5, -3.8, 0, -6.2, 7.5, -22.5, 15],
        [0, 14.6, 22.5, -45, -7.5, 0, -8.8, 0, -15, 30],
        [0, 11.9, 26.2, -37.5, -11.2, 0, -11.5, -7.5, -7.5, 45],
        [0, 9.2, 30, -30, -15, 0, -14.2, -15, 0, 60],
        [0, 12.7, 33.8, -22.5, -18.8, 0, -10.4, -7.5, 7.5, 52.5],
        [0, 16.2, 37.5, -15, -22.5, 0, -6.5, 0, 15, 45],
        [0, 19.6, 41.2, -7.5, -26.2, 0, -2.7, 7.5, 22.5, 37.5],
        [0, 23.1, 45, 0, -30, 0, 1.2, 15, 30, 30],
        [0, 11.5, 15, 0, 7.5, 0, 8.1, 30, 15, -7.5]
    ], device=env.device, dtype=torch.float32)
    
    # Convert reference from degrees to radians
    reference_gait_rad = reference_gait_deg * (torch.pi / 180.0)
    
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get leg joint indices (cache them on first call)
    if not hasattr(gait_phase_reward, '_leg_joint_indices'):
        joint_names = asset.data.joint_names
        leg_joints = ["leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
                      "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint"]
        
        leg_indices = []
        for target_joint in leg_joints:
            for i, name in enumerate(joint_names):
                if name == target_joint:
                    leg_indices.append(i)
                    break
        
        if len(leg_indices) != 10:
            print(f"⚠️  WARNING: Found only {len(leg_indices)} leg joints, expected 10!")
            print(f"Available joints: {joint_names}")
            print(f"Found indices: {leg_indices}")
        
        gait_phase_reward._leg_joint_indices = torch.tensor(leg_indices, device=env.device, dtype=torch.long)
        gait_phase_reward._initialized = True
        print(f"🎯 Gait phase reward initialized with leg joint indices: {leg_indices}")
        print(f"   Joint names: {[joint_names[i] for i in leg_indices]}")
    
    leg_indices = gait_phase_reward._leg_joint_indices
    
    # Get current leg joint positions [num_envs, 10]
    current_joint_pos = asset.data.joint_pos[:, leg_indices]
    
    # Calculate phase in gait cycle for each environment
    # Use episode time to determine phase
    time_in_cycle = torch.fmod(env.episode_length_buf.float() * env.step_dt, gait_period)
    phase_ratio = time_in_cycle / gait_period  # [num_envs], range [0, 1)
    
    # Convert phase to keyframe index (0 to 17)
    keyframe_idx = (phase_ratio * 18.0).long()  # [num_envs]
    keyframe_idx = torch.clamp(keyframe_idx, 0, 17)  # Ensure valid range
    
    # Get reference joint angles for current phase [num_envs, 10]
    reference_pos = reference_gait_rad[keyframe_idx]
    
    # Calculate tracking error
    # Use mean squared error per joint (averaged across joints) for better scaling
    joint_diff = current_joint_pos - reference_pos  # [num_envs, 10]
    mse_per_joint = torch.mean(torch.square(joint_diff), dim=1)  # [num_envs], averaged across 10 joints
    
    # Calculate reward using exponential kernel
    # With std=3.0, mse=1.0 gives reward ≈ 0.90, mse=5.0 gives reward ≈ 0.57
    reward = torch.exp(-mse_per_joint / (std ** 2))
    
    # Debug print for first few calls or every 500 steps
    if not hasattr(gait_phase_reward, '_debug_count'):
        gait_phase_reward._debug_count = 0
    
    gait_phase_reward._debug_count += 1
    
    if gait_phase_reward._debug_count <= 5 or gait_phase_reward._debug_count % 500 == 0:
        avg_error = mse_per_joint.mean().item()
        avg_reward = reward.mean().item()
        max_reward = reward.max().item()
        min_reward = reward.min().item()
        avg_phase = phase_ratio.mean().item()
        
        print(f"🎯 Gait Reward [{gait_phase_reward._debug_count}] - "
              f"MSE/joint: {avg_error:.4f}, Reward: {avg_reward:.4f} (min:{min_reward:.4f}, max:{max_reward:.4f}), "
              f"Phase: {avg_phase:.2f}, Keyframe: {keyframe_idx[0].item()}")
        
        if gait_phase_reward._debug_count <= 2:
            print(f"   Current joints (env 0, deg): {(current_joint_pos[0][:5] * 180/torch.pi).cpu().numpy()}")
            print(f"   Reference joints (env 0, deg): {(reference_pos[0][:5] * 180/torch.pi).cpu().numpy()}")
            print(f"   Joint diff (env 0, deg): {(joint_diff[0][:5] * 180/torch.pi).cpu().numpy()}")
    
    # Return exponential reward (higher reward for smaller error)
    return reward
