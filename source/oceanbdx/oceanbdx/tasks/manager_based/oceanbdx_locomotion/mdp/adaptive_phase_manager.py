# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
自适应相位管理器 (Adaptive Phase Manager)
根据速度指令动态调整期望步态周期、步幅、抬脚高度
参考: Disney BDX步态视频分析 + legged_gym实践
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class VideoGaitReference:
    """从Disney BDX参考视频中提取的步态参数"""
    
    # 参考行走速度 (m/s) - 视频中测量
    reference_velocity: float = 0.35
    
    # 步态周期 (秒) - 从一只脚着地到下次该脚着地
    reference_period: float = 0.75
    
    # 典型步幅 (米) - 一步跨出的距离
    reference_stride: float = 0.131
    
    # 正常行走时的躯干高度 (米)
    nominal_base_height: float = 0.35
    
    # 摆动腿抬起高度 (米)
    foot_clearance: float = 0.037
    
    # 双支撑相占比 (0-1)
    double_support_ratio: float = 0.3
    
    # 机器人腿长 (米) - 从URDF测量
    leg_length: float = 0.35


class AdaptiveGaitTable:
    """
    速度-步态参数映射表
    基于Disney BDX视频分析 + 生物力学原理
    """
    
    # 速度(m/s): (周期(s), 步幅(m)两步距离, 抬脚高度(m))
    # 验证: v = stride/period
    GAIT_PARAMS = {
        0.0:  (0.8,  0.0,   0.0),      # 静止
        0.1:  (0.8,  0.08,  0.025),    # 极慢走
        0.25: (0.8,  0.2,   0.03),     # 慢走
        0.35: (0.75, 0.262, 0.037),    # 正常走 (0.131*2, 参考视频)
        0.5:  (0.65, 0.325, 0.045),    # 快走
        0.6:  (0.6,  0.36,  0.055),    # 非常快
        0.74: (0.5,  0.37,  0.07),     # 最快 (0.185*2)
    }
    
    @staticmethod
    def interpolate(speed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据速度插值获取期望步态参数
        
        Args:
            speed: [N] 速度大小 (m/s)
            
        Returns:
            period: [N] 期望周期 (s)
            stride: [N] 期望步幅 (m)
            clearance: [N] 期望抬脚高度 (m)
        """
        device = speed.device
        num_envs = speed.shape[0]
        
        # 转换为Python列表用于插值
        velocities = sorted(AdaptiveGaitTable.GAIT_PARAMS.keys())
        periods = [AdaptiveGaitTable.GAIT_PARAMS[v][0] for v in velocities]
        strides = [AdaptiveGaitTable.GAIT_PARAMS[v][1] for v in velocities]
        clearances = [AdaptiveGaitTable.GAIT_PARAMS[v][2] for v in velocities]
        
        # 转为tensor
        vel_tensor = torch.tensor(velocities, device=device, dtype=torch.float32)
        period_tensor = torch.tensor(periods, device=device, dtype=torch.float32)
        stride_tensor = torch.tensor(strides, device=device, dtype=torch.float32)
        clearance_tensor = torch.tensor(clearances, device=device, dtype=torch.float32)
        
        # 对每个环境进行线性插值
        period = torch.zeros(num_envs, device=device, dtype=torch.float32)
        stride = torch.zeros(num_envs, device=device, dtype=torch.float32)
        clearance = torch.zeros(num_envs, device=device, dtype=torch.float32)
        
        for i in range(num_envs):
            s = speed[i].item()
            
            # Clamp速度到表格范围
            s = max(min(s, velocities[-1]), velocities[0])
            
            # 查找插值区间
            for j in range(len(velocities) - 1):
                if velocities[j] <= s <= velocities[j + 1]:
                    # 线性插值
                    alpha = (s - velocities[j]) / (velocities[j + 1] - velocities[j])
                    period[i] = period_tensor[j] * (1 - alpha) + period_tensor[j + 1] * alpha
                    stride[i] = stride_tensor[j] * (1 - alpha) + stride_tensor[j + 1] * alpha
                    clearance[i] = clearance_tensor[j] * (1 - alpha) + clearance_tensor[j + 1] * alpha
                    break
        
        return period, stride, clearance


class AdaptivePhaseManager:
    """
    自适应步态相位管理
    - 根据速度指令动态调整期望步态周期
    - 生成多频率相位观测（与真机一致）
    - 提供步态参数供奖励函数使用
    """
    
    def __init__(self, num_envs: int, device: str, video_config: VideoGaitReference):
        """
        Args:
            num_envs: 环境数量
            device: 计算设备
            video_config: 视频参考步态配置
        """
        self.num_envs = num_envs
        self.device = device
        self.config = video_config
        
        # 相位状态 [0, 1] 循环
        self.current_phase = torch.zeros(num_envs, device=device)
        
        # 累计运动时间（用于相位编码）
        self.motion_time = torch.zeros(num_envs, device=device)
        
        # 当前期望步态参数（供奖励函数使用）
        self.desired_period = torch.ones(num_envs, device=device) * video_config.reference_period
        self.desired_stride = torch.ones(num_envs, device=device) * video_config.reference_stride * 2.0  # 双倍（两步）
        self.desired_clearance = torch.ones(num_envs, device=device) * video_config.foot_clearance
        
        # 相位速率 (1/period)
        self.phase_rate = torch.ones(num_envs, device=device) / video_config.reference_period
        
    def update(self, velocity_command: torch.Tensor, dt: float) -> torch.Tensor:
        """
        更新相位并返回当前相位
        
        Args:
            velocity_command: [N, 3] 速度指令 (vx, vy, vyaw)
            dt: 时间步长
            
        Returns:
            phase: [N] 当前相位 [0, 1]
        """
        # 计算速度大小
        speed = torch.norm(velocity_command[:, :2], dim=1)
        
        # 从查找表插值获取期望参数
        period, stride, clearance = AdaptiveGaitTable.interpolate(speed)
        
        # 更新内部状态
        self.desired_period = period
        self.desired_stride = stride
        self.desired_clearance = clearance
        self.phase_rate = 1.0 / (period + 1e-8)
        
        # 更新相位
        phase_increment = dt / (period + 1e-8)
        self.current_phase = (self.current_phase + phase_increment) % 1.0
        
        # 更新运动时间
        self.motion_time = self.current_phase * period
        
        return self.current_phase
    
    def get_phase_observation(self) -> torch.Tensor:
        """
        生成多频率相位观测（与真机部署一致）
        
        Returns:
            phase_obs: [N, 9] 相位观测
                - 6维: sin/cos 多频率编码 (1x, 0.5x, 0.25x)
                - 1维: phase_rate (归一化)
                - 1维: desired_stride (归一化)
                - 1维: desired_clearance (归一化)
        """
        # 计算theta (与真机一致)
        theta = torch.pi * self.motion_time / 2.0
        
        # 多频率sin/cos编码
        phase_feat = torch.stack([
            torch.sin(theta),
            torch.cos(theta),
            torch.sin(theta / 2.0),
            torch.cos(theta / 2.0),
            torch.sin(theta / 4.0),
            torch.cos(theta / 4.0),
        ], dim=-1)  # [N, 6]
        
        # 归一化期望参数
        max_stride = 0.5   # 经验最大值
        max_clearance = 0.1
        max_phase_rate = 2.0  # 最快步频 (1/0.5s)
        
        phase_rate_norm = torch.clamp(self.phase_rate / max_phase_rate, 0.0, 1.0).unsqueeze(-1)
        stride_norm = torch.clamp(self.desired_stride / max_stride, 0.0, 1.0).unsqueeze(-1)
        clearance_norm = torch.clamp(self.desired_clearance / max_clearance, 0.0, 1.0).unsqueeze(-1)
        
        # 拼接所有特征
        phase_obs = torch.cat([
            phase_feat,          # 6 dim
            phase_rate_norm,     # 1 dim
            stride_norm,         # 1 dim
            clearance_norm,      # 1 dim
        ], dim=-1)  # [N, 9]
        
        return phase_obs
    
    def get_current_targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回当前期望步态参数（供奖励函数使用）
        
        Returns:
            period: [N] 期望周期 (s)
            stride: [N] 期望步幅 (m, 两步距离)
            clearance: [N] 期望抬脚高度 (m)
        """
        return self.desired_period, self.desired_stride, self.desired_clearance
    
    def reset_idx(self, env_ids: torch.Tensor):
        """重置指定环境的相位"""
        self.current_phase[env_ids] = 0.0
        self.motion_time[env_ids] = 0.0
