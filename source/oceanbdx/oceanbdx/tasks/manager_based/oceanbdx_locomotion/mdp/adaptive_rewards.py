# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
自适应步态奖励函数 (Adaptive Gait Reward Functions)
基于Disney BDX训练指南的14个核心奖励
参考: legged_gym, walk-these-ways, isaac-orbit
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================
# 3.2 任务奖励 (Task Rewards)
# ============================================================

def reward_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    速度跟踪奖励 (指数核)
    来源: legged_gym/envs/base/legged_robot.py
    
    奖励机器人跟踪xy方向的线速度指令
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    target_vel = command[:, :2]  # (vx, vy)
    actual_vel = asset.data.root_lin_vel_b[:, :2]
    
    error = torch.sum((actual_vel - target_vel) ** 2, dim=1)
    reward = torch.exp(-error / (std ** 2))
    
    return reward


def reward_angular_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    角速度跟踪奖励
    来源: legged_gym 标准实现
    
    奖励机器人跟踪z轴的角速度指令(转向)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    target_ang_vel = command[:, 2]  # yaw rate
    actual_ang_vel = asset.data.root_ang_vel_b[:, 2]
    
    error = (actual_ang_vel - target_ang_vel) ** 2
    reward = torch.exp(-error / (std ** 2))
    
    return reward


# ============================================================
# 3.3 稳定性约束 (Stability Constraints)
# ============================================================

def reward_orientation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    姿态惩罚 (roll & pitch 应接近0)
    来源: 所有双足/四足项目的标准实现
    
    惩罚机器人躯干的roll和pitch偏离
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 从四元数提取roll和pitch
    # quat: [w, x, y, z]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Roll (rotation around x-axis) - side-to-side tilt
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    
    # Pitch (rotation around y-axis) - forward/backward tilt
    sin_pitch = torch.clamp(2 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    
    penalty = roll ** 2 + pitch ** 2
    
    return -penalty  # 负值 = 惩罚


def reward_base_height_tracking(
    env: ManagerBasedRLEnv,
    target_height: float = 0.35,  # ⬇️ 从0.39降低到0.35 (Disney BDX参考值,减少膝盖过直和前倾)
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    躯干高度跟踪
    来源: legged_gym
    
    奖励机器人保持目标躯干高度
    """
    asset: Articulation = env.scene[asset_cfg.name]
    actual_height = asset.data.root_pos_w[:, 2]  # z坐标
    
    error = (actual_height - target_height) ** 2
    reward = torch.exp(-error / (std ** 2))
    
    return reward


# ============================================================
# 3.4 步态质量约束 (Gait Quality) - 核心防作弊
# ============================================================

def reward_feet_alternating_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    改进版自适应交替接触奖励 - 加入过渡相
    来源: 改进自 legged_gym 的 feet_air_time
    这是防止振动作弊的最关键奖励!
    
    相位划分 (更精细):
    - phase 0.0-0.1: 过渡相(双脚支撑,准备右腿摆动)
    - phase 0.1-0.4: 右腿摆动,左腿支撑
    - phase 0.4-0.6: 过渡相(双脚支撑,准备左腿摆动)
    - phase 0.6-0.9: 左腿摆动,右腿支撑
    - phase 0.9-1.0: 过渡相(双脚支撑,准备下一周期)
    """
    # 获取相位管理器
    if not hasattr(env, 'phase_manager'):
        return torch.zeros(env.num_envs, device=env.device)
    
    phase = env.phase_manager.current_phase
    
    # 获取左右脚接触力
    left_sensor: ContactSensor = env.scene["contact_forces_LF"]
    right_sensor: ContactSensor = env.scene["contact_forces_RF"]
    
    left_forces = left_sensor.data.net_forces_w_history
    right_forces = right_sensor.data.net_forces_w_history
    
    # 计算接触力大小
    left_contact_norm = torch.norm(left_forces.view(env.num_envs, -1, 3), dim=-1)
    right_contact_norm = torch.norm(right_forces.view(env.num_envs, -1, 3), dim=-1)
    
    # 二值化接触状态
    left_in_contact = (torch.sum(left_contact_norm, dim=-1) > threshold).float()
    right_in_contact = (torch.sum(right_contact_norm, dim=-1) > threshold).float()
    
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # 🆕 过渡相 1 (0.0-0.1): 允许双脚着地,准备右腿摆动
    transition1_mask = phase < 0.1
    both_contact_ok_1 = left_in_contact * right_in_contact
    reward[transition1_mask] = both_contact_ok_1[transition1_mask] * 0.3  # 轻微奖励
    
    # 摆动相 1 (0.1-0.4): 期望右腿摆动(离地), 左腿支撑(着地)
    swing1_mask = (phase >= 0.1) & (phase < 0.4)
    ideal_state_1 = (1 - right_in_contact) * left_in_contact  # 左着右离
    both_contact_1 = right_in_contact * left_in_contact        # 双脚着地(惩罚)
    reward[swing1_mask] = ideal_state_1[swing1_mask] * 1.0 - both_contact_1[swing1_mask] * 0.8
    
    # 🆕 过渡相 2 (0.4-0.6): 允许双脚着地,准备左腿摆动
    transition2_mask = (phase >= 0.4) & (phase < 0.6)
    both_contact_ok_2 = left_in_contact * right_in_contact
    reward[transition2_mask] = both_contact_ok_2[transition2_mask] * 0.3
    
    # 摆动相 2 (0.6-0.9): 期望左腿摆动, 右腿支撑
    swing2_mask = (phase >= 0.6) & (phase < 0.9)
    ideal_state_2 = (1 - left_in_contact) * right_in_contact  # 右着左离
    both_contact_2 = left_in_contact * right_in_contact
    reward[swing2_mask] = ideal_state_2[swing2_mask] * 1.0 - both_contact_2[swing2_mask] * 0.8
    
    # 🆕 过渡相 3 (0.9-1.0): 允许双脚着地,准备下一周期
    transition3_mask = phase >= 0.9
    both_contact_ok_3 = left_in_contact * right_in_contact
    reward[transition3_mask] = both_contact_ok_3[transition3_mask] * 0.3
    
    return reward


def reward_stride_length_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    自适应步长跟踪
    来源: 自定义，概念类似 cassie-mujoco-sim
    
    奖励机器人的实际步幅接近期望步幅
    """
    # 获取期望步幅
    if not hasattr(env, 'phase_manager'):
        return torch.zeros(env.num_envs, device=env.device)
    
    _, desired_stride, _ = env.phase_manager.get_current_targets()
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取左右脚位置（需要从body_pos获取）
    # 简化：使用关节位置估算脚部位置
    # 这里假设有body_pos字典或者可以从关节角度计算
    # 为了简化实现，使用base前进方向的关节位置差异
    joint_pos = asset.data.joint_pos
    
    # 简化版本：使用左右腿的髋关节pitch角度差异估算步幅
    # 实际应该用FK计算脚尖位置，这里用简化方法
    # TODO: 使用正运动学计算精确脚部位置
    
    # 暂时返回固定小奖励，等待完整FK实现
    error = torch.abs(0.2 - desired_stride)  # 假设当前步幅0.2m
    reward = torch.exp(-error / 0.1)
    
    return reward * 0.5  # 降低权重因为是简化实现


def reward_foot_clearance(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    自适应脚抬起高度
    来源: legged_gym 的 foot_clearance_cmd_linear
    
    奖励摆动腿抬起到期望高度
    """
    # 获取期望抬脚高度
    if not hasattr(env, 'phase_manager'):
        return torch.zeros(env.num_envs, device=env.device)
    
    phase = env.phase_manager.current_phase
    _, _, desired_clearance = env.phase_manager.get_current_targets()
    
    # 获取接触状态
    left_sensor: ContactSensor = env.scene["contact_forces_LF"]
    right_sensor: ContactSensor = env.scene["contact_forces_RF"]
    
    left_contact = torch.norm(left_sensor.data.net_forces_w_history.view(env.num_envs, -1, 3), dim=-1).sum(dim=-1)
    right_contact = torch.norm(right_sensor.data.net_forces_w_history.view(env.num_envs, -1, 3), dim=-1).sum(dim=-1)
    
    left_swing = left_contact < threshold
    right_swing = right_contact < threshold
    
    # 获取脚高度（简化版本，实际需要FK）
    # TODO: 使用正运动学计算精确脚部高度
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    
    # 简化：假设脚高度 = 基座高度 - 腿长 + 抬起量
    # 真实实现需要FK
    estimated_clearance = torch.ones_like(base_height) * 0.03  # 假设3cm抬起
    
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # 右腿摆动相 (phase < 0.5)
    right_swing_mask = (phase < 0.5) & right_swing
    if right_swing_mask.any():
        clearance_achieved = estimated_clearance - desired_clearance
        reward[right_swing_mask] = torch.clamp(clearance_achieved[right_swing_mask], 0, 0.1)
    
    # 左腿摆动相
    left_swing_mask = (phase >= 0.5) & left_swing
    if left_swing_mask.any():
        clearance_achieved = estimated_clearance - desired_clearance
        reward[left_swing_mask] = torch.clamp(clearance_achieved[left_swing_mask], 0, 0.1)
    
    return reward * 0.5  # 降低权重因为是简化实现


# ============================================================
# 3.5 安全约束 (Safety Constraints)
# ============================================================

def reward_undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    惩罚非足部接触
    来源: legged_gym 标准实现
    
    惩罚膝盖、大腿、躯干接触地面
    """
    # 检查膝盖接触
    try:
        left_knee: ContactSensor = env.scene["contact_forces_knee_L"]
        right_knee: ContactSensor = env.scene["contact_forces_knee_R"]
        
        left_knee_contact = torch.norm(left_knee.data.net_forces_w_history.view(env.num_envs, -1, 3), dim=-1).sum(dim=-1)
        right_knee_contact = torch.norm(right_knee.data.net_forces_w_history.view(env.num_envs, -1, 3), dim=-1).sum(dim=-1)
        
        penalty = ((left_knee_contact > threshold).float() + 
                  (right_knee_contact > threshold).float())
        
        return -penalty
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)


def reward_joint_limits_penalty(
    env: ManagerBasedRLEnv,
    soft_limit_ratio: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    关节限位惩罚
    来源: legged_gym/envs/base/legged_robot.py
    
    惩罚接近关节限位的动作
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    joint_pos = asset.data.joint_pos
    # soft_joint_pos_limits shape: [num_joints, 2] or [num_envs, num_joints, 2]
    # 需要广播到 [num_envs, num_joints]
    joint_limits = asset.data.soft_joint_pos_limits
    if joint_limits.dim() == 2:
        # [num_joints, 2] -> [1, num_joints] for broadcasting
        joint_limits_low = joint_limits[:, 0].unsqueeze(0)
        joint_limits_high = joint_limits[:, 1].unsqueeze(0)
    else:
        # [num_envs, num_joints, 2]
        joint_limits_low = joint_limits[:, :, 0]
        joint_limits_high = joint_limits[:, :, 1]
    
    # 归一化到 [-1, 1]
    normalized_pos = 2 * (joint_pos - joint_limits_low) / \
                     (joint_limits_high - joint_limits_low + 1e-8) - 1
    
    # 惩罚超出软限位的关节
    out_of_soft_limits = (torch.abs(normalized_pos) > soft_limit_ratio).float()
    penalty = torch.sum(out_of_soft_limits * normalized_pos ** 2, dim=1)
    
    return -penalty


def reward_feet_slip_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    支撑腿滑动惩罚
    来源: legged_gym 的 feet_slip
    
    惩罚支撑腿的水平速度（应该接近0）
    """
    # 获取接触状态
    left_sensor: ContactSensor = env.scene["contact_forces_LF"]
    right_sensor: ContactSensor = env.scene["contact_forces_RF"]
    
    left_contact = torch.norm(left_sensor.data.net_forces_w_history.view(env.num_envs, -1, 3), dim=-1).sum(dim=-1)
    right_contact = torch.norm(right_sensor.data.net_forces_w_history.view(env.num_envs, -1, 3), dim=-1).sum(dim=-1)
    
    left_in_stance = left_contact > threshold
    right_in_stance = right_contact > threshold
    
    # 获取脚部速度（简化：使用base速度）
    asset: Articulation = env.scene[asset_cfg.name]
    base_vel = asset.data.root_lin_vel_b[:, :2]  # (vx, vy)
    
    # 支撑腿的滑动量（简化实现）
    slip = torch.sum(base_vel ** 2, dim=1)
    
    left_slip = slip * left_in_stance.float()
    right_slip = slip * right_in_stance.float()
    
    return -(left_slip + right_slip)


# ============================================================
# 3.6 能耗与平滑性 (Energy & Smoothness)
# ============================================================

def reward_action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    动作平滑性奖励
    来源: legged_gym 的 action_rate
    防止高频振动作弊的关键!
    """
    action_diff = env.action_manager.action - env.action_manager.prev_action
    penalty = torch.sum(action_diff ** 2, dim=1)
    
    return -penalty


def reward_joint_torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    关节力矩惩罚 (能耗)
    来源: legged_gym 的 torques
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque
    penalty = torch.sum(torques ** 2, dim=1)
    
    return -penalty


def reward_joint_acceleration_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    关节加速度惩罚
    来源: legged_gym 的 dof_acc
    防止剧烈运动和振动
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc
    penalty = torch.sum(joint_acc ** 2, dim=1)
    
    return -penalty


def reward_joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    关节速度惩罚
    来源: legged_gym 的 dof_vel
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel
    penalty = torch.sum(joint_vel ** 2, dim=1)
    
    return -penalty


def reward_termination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    终止惩罚
    严厉惩罚导致episode终止的行为（摔倒等）
    """
    return -env.termination_manager.terminated.float() * 100.0


# ============================================================
# 🆕 重心转移与抬腿奖励 (Weight Transfer & Lift)
# ============================================================

def reward_weight_transfer(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    🆕 重心转移奖励 - 相位感知,鼓励单腿支撑和抬腿
    
    核心思想: 根据步态相位奖励正确的重心转移
    - 支撑相: 奖励支撑腿有力接触 + 轻微惩罚摆动腿接触
    - 摆动相: 奖励摆动腿离地
    - 过渡相: 允许双脚着地(准备重心转移)
    
    相位划分:
    - 0.0-0.1: 过渡(双脚支撑)
    - 0.1-0.4: 左腿支撑 + 右腿摆动
    - 0.4-0.6: 过渡(双脚支撑)
    - 0.6-0.9: 右腿支撑 + 左腿摆动
    - 0.9-1.0: 过渡(双脚支撑)
    """
    # 获取相位管理器
    if not hasattr(env, 'phase_manager'):
        return torch.zeros(env.num_envs, device=env.device)
    
    phase = env.phase_manager.current_phase
    
    # 获取左右脚接触力
    left_sensor: ContactSensor = env.scene["contact_forces_LF"]
    right_sensor: ContactSensor = env.scene["contact_forces_RF"]
    
    left_forces = left_sensor.data.net_forces_w_history
    right_forces = right_sensor.data.net_forces_w_history
    
    # 计算接触力大小 (归一化到 [0, 1])
    left_contact_norm = torch.norm(left_forces.view(env.num_envs, -1, 3), dim=-1)
    right_contact_norm = torch.norm(right_forces.view(env.num_envs, -1, 3), dim=-1)
    
    # 接触强度 (0=离地, 1=着地)
    left_contact_strength = torch.clamp(torch.sum(left_contact_norm, dim=-1) / (threshold * 10), 0, 1)
    right_contact_strength = torch.clamp(torch.sum(right_contact_norm, dim=-1) / (threshold * 10), 0, 1)
    
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # 支撑相 1 + 摆动相 1 (0.1-0.4): 左腿支撑,右腿摆动
    phase1_mask = (phase >= 0.1) & (phase < 0.4)
    if phase1_mask.any():
        # 奖励左腿有力接触
        reward[phase1_mask] += left_contact_strength[phase1_mask] * 0.5
        # 奖励右腿离地
        reward[phase1_mask] += (1.0 - right_contact_strength[phase1_mask]) * 0.8
        # 轻微惩罚右腿着地(鼓励抬起)
        reward[phase1_mask] -= right_contact_strength[phase1_mask] * 0.3
    
    # 支撑相 2 + 摆动相 2 (0.6-0.9): 右腿支撑,左腿摆动
    phase2_mask = (phase >= 0.6) & (phase < 0.9)
    if phase2_mask.any():
        # 奖励右腿有力接触
        reward[phase2_mask] += right_contact_strength[phase2_mask] * 0.5
        # 奖励左腿离地
        reward[phase2_mask] += (1.0 - left_contact_strength[phase2_mask]) * 0.8
        # 轻微惩罚左腿着地
        reward[phase2_mask] -= left_contact_strength[phase2_mask] * 0.3
    
    # 过渡相: 允许双脚着地,不给奖励也不惩罚
    # (phase < 0.1 or 0.4 <= phase < 0.6 or phase >= 0.9)
    
    return reward
