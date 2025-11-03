# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
训练课程调度器 (Training Curriculum Scheduler)
三阶段权重调度：早期稳定性 → 中期步态 → 后期优化
参考: legged_gym curriculum learning
"""

from __future__ import annotations

from typing import Dict


class TrainingCurriculum:
    """
    四阶段训练课程：从站立到行走，逐步引入约束
    
    阶段0 (0-5%): 🆕 站立稳定期 - 双足支撑,建立平衡,重心控制
    阶段1 (5-30%): 学会站立、不摔倒、粗略前进
    阶段2 (30-70%): 形成正常步态、提高速度、对称行走
    阶段3 (70-100%): 优化能效、适应全速度范围、精细调节
    """
    
    # 🆕 阶段0: 站立稳定期 (0-20% 训练迭代) - ⬆️ 延长4倍,强化鲁棒站立
    STAGE0_STANDING = {
        'progress_range': (0.0, 0.20),  # 🔧 从5%延长到20% (0-2000 iter)
        
        # 任务奖励 - 🔧 只关注原地站立,不要求前进!
        'velocity_tracking': 0.0,           # 🔧 关闭速度跟踪!
        'angular_velocity_tracking': 0.0,   # 🔧 关闭转向跟踪!
        
        # 稳定性约束 - 🔧 修复: 降低高度权重,避免"躺平"策略
        'orientation_penalty': 50.0,        # ⬇️ 从100降到50 (避免过度主导)
        'base_height_tracking': 3.0,        # ⬇️ 从15降到3.0 (关键修复!防止躺平获得高奖励)
        'termination_penalty': 20.0,        # ⬇️ 从30降到20 (平衡惩罚)
        
        # 步态约束 - 完全不启用,保持双足支撑!
                # 步态约束 - 🔧 只鼓励保持双足支撑,完全不启用单腿摆动!
        'feet_alternating_contact': 0.0,
        'stride_length_tracking': 0.0,
        'foot_clearance': 0.0,
        'weight_transfer': 0.0,  # 🆕 Stage 0 不要求重心转移
        
        # 安全约束 - 🚨 修正符号!函数返回负值,权重用正数! 🔧 降低权重避免干扰主目标
        'undesired_contacts': 2.0,          # ✅ 降低 5.0→2.0
        'joint_limits_penalty': 0.05,       # ✅ 降低 0.1→0.05
        'feet_slip_penalty': 1.0,           # ✅ 降低 2.0→1.0
        
        # 能耗优化 - 🔧 鼓励保持静止 - 🚨 修正符号!
        'action_smoothness': 0.01,          # ✅ 惩罚大动作
        'joint_torque_penalty': 1e-5,       # ✅
        'joint_acceleration': 1e-7,         # ✅
        'joint_velocity_penalty': 1e-4,     # ✅
        
        # 命令范围 - 🔧 零速度!
        'velocity_command_range': (0.0, 0.0),  # 🔧 不给速度命令! (站立阶段)
    }
    
    # 阶段1: 早期训练 (20-45% 训练迭代) - 开始学习行走
    STAGE1_EARLY = {
        'progress_range': (0.20, 0.45),  # 🔧 调整到20-45%
        
        # 任务奖励 - 🔧 修复：提高权重，成为主导
        'velocity_tracking': 2.0,           # 从0.5提升到2.0
        'angular_velocity_tracking': 1.0,   # 从0.25提升到1.0
        
        # 稳定性约束 - 🔧 修复: 降低高度权重,保持平衡
        'orientation_penalty': 2.0,         # ⬆️ 从1.0提升到2.0 (强化姿态)
        'base_height_tracking': 1.5,        # ⬇️ 从2.5降到1.5 (避免过度关注高度)
        'termination_penalty': 2.0,         # ✅ 保持不变
        
        # 步态约束 - 🆕 提前激活!低权重引导,防止小碎步固化
        'feet_alternating_contact': 0.3,    # 🆕 从0提升到0.3 (早期引导交替步态)
        'stride_length_tracking': 0.2,      # 🆕 从0提升到0.2 (引导合理步长)
        'foot_clearance': 0.2,              # 🆕 从0提升到0.2 (引导抬脚)
        'weight_transfer': 0.5,             # 🆕 从0提升到0.5 (引导重心转移和抬腿)
        
        # 安全约束 - 🔧 修复：大幅降低joint_limits权重 - 🚨 修正符号!
        'undesired_contacts': 5.0,          # ✅
        'joint_limits_penalty': 0.1,        # ✅ 从-2.0降至-0.1 → 改正数0.1
        'feet_slip_penalty': 1.0,           # ✅
        
        # 能耗优化 - 暂不考虑 - 🚨 修正符号!
        'action_smoothness': 0.001,         # ✅
        'joint_torque_penalty': 1e-6,       # ✅
        'joint_acceleration': 1e-8,         # ✅
        'joint_velocity_penalty': 1e-5,     # ✅
        
        # 命令范围 - 负值向前 (适配反向IMU: X+ points backward in hardware)
        'velocity_command_range': (-0.35, 0.0),  # 负值 = 向前移动
    }
    
    # 阶段2: 中期训练 (45-75% 训练迭代)
    STAGE2_MID = {
        'progress_range': (0.45, 0.75),  # 🔧 调整到45-75%
        
        # 任务奖励 - 🔧 修复：保持高权重
        'velocity_tracking': 2.5,           # 从1.2提升到2.5
        'angular_velocity_tracking': 1.2,   # 从0.6提升到1.2
        
        # 稳定性约束 - ⬆️ 提高高度权重
        'orientation_penalty': 0.8,         # ✅ 姿态控制已经很好
        'base_height_tracking': 2.0,        # ⬆️ 从0.6提升到2.0 (继续强调高度)
        'termination_penalty': 1.5,         # ✅ 摔倒率很低
        
        # 步态约束 - 核心阶段，⬆️ 提高权重强化步态质量!
        'feet_alternating_contact': 1.2,    # ⬆️ 从1.0提升到1.2 (强化交替)
        'stride_length_tracking': 1.5,      # ⬆️ 从1.2提升到1.5 (强化步长)
        'foot_clearance': 1.0,              # ⬆️ 从0.9提升到1.0 (强化抬腿)
        'weight_transfer': 1.0,             # ⬆️ 从0.5提升到1.0 (强化重心转移)
        
        # 安全约束 - 🔧 修复：保持低权重 - 🚨 修正符号!
        'undesired_contacts': 2.0,          # ✅
        'joint_limits_penalty': 0.05,       # ✅ 从-1.0降至-0.05 → 改正数0.05
        'feet_slip_penalty': 1.5,           # ✅
        
        # 能耗优化 - 开始引入 - 🚨 修正符号!
        'action_smoothness': 0.01,          # ✅ 要求平滑
        'joint_torque_penalty': 5e-5,       # ✅
        'joint_acceleration': 2.5e-7,       # ✅ 防作弊关键
        'joint_velocity_penalty': 5e-4,     # ✅
        
        # 命令范围 - 扩大到中速 (负值向前)
        'velocity_command_range': (-0.5, 0.0),  # 负值 = 向前移动
    }
    
    # 阶段3: 后期训练 (75-100% 训练迭代)
    STAGE3_LATE = {
        'progress_range': (0.75, 1.0),  # 🔧 调整到75-100%
        
        # 任务奖励 - 🔧 修复：最高权重
        'velocity_tracking': 3.0,           # 从1.5提升到3.0
        'angular_velocity_tracking': 1.5,   # 从0.75提升到1.5
        
        # 稳定性约束 - ⬆️ 后期仍需保持高度
        'orientation_penalty': 0.6,         # ✅ 姿态控制已经很好
        'base_height_tracking': 1.5,        # ⬆️ 从0.4提升到1.5 (后期也要保持高度!)
        'termination_penalty': 1.0,         # ✅ 摔倒率很低
        
        # 步态约束 - 保持中等权重,让速度跟踪主导
        'feet_alternating_contact': 1.0,    # 保持交替
        'stride_length_tracking': 1.2,      # 保持步长
        'foot_clearance': 0.8,              # 保持抬腿
        'weight_transfer': 1.0,             # 保持重心转移
        
        # 安全约束 - 🔧 修复：保持最低 - 🚨 修正符号!
        'undesired_contacts': 2.0,          # ✅
        'joint_limits_penalty': 0.02,       # ✅ 从-1.0降至-0.02 → 改正数0.02
        'feet_slip_penalty': 2.0,           # ✅ 最高，完全不允许滑动
        
        # 能耗优化 - 最高权重阶段 - 🚨 修正符号!
        'action_smoothness': 0.05,          # ✅ 5倍提升
        'joint_torque_penalty': 1e-4,       # ✅ 2倍提升
        'joint_acceleration': 1e-6,         # ✅ 4倍提升
        'joint_velocity_penalty': 1e-3,     # ✅ 2倍提升
        
        # 命令范围 - 全速度 (负值向前)
        'velocity_command_range': (-0.74, 0.0),  # 负值 = 向前移动
    }
    
    @staticmethod
    def get_current_weights(training_progress: float) -> Dict[str, float]:
        """
        根据训练进度 (0.0-1.0) 返回当前权重
        使用线性插值在不同阶段间平滑过渡
        
        Args:
            training_progress: 训练进度 [0.0, 1.0]
            
        Returns:
            current_weights: 当前阶段的权重字典
        """
        if training_progress < 0.20:
            # 🆕 阶段0: 站立稳定期 (0-20%)
            return TrainingCurriculum.STAGE0_STANDING.copy()
        elif training_progress < 0.45:
            # 阶段0到1的过渡 (20-45%)
            alpha = (training_progress - 0.20) / 0.25
            return TrainingCurriculum._interpolate_weights(
                TrainingCurriculum.STAGE0_STANDING,
                TrainingCurriculum.STAGE1_EARLY,
                alpha
            )
        elif training_progress < 0.75:
            # 在阶段1和2之间插值 (45-75%)
            alpha = (training_progress - 0.45) / 0.30
            return TrainingCurriculum._interpolate_weights(
                TrainingCurriculum.STAGE1_EARLY,
                TrainingCurriculum.STAGE2_MID,
                alpha
            )
        else:
            # 在阶段2和3之间插值 (75-100%)
            alpha = (training_progress - 0.75) / 0.25
            return TrainingCurriculum._interpolate_weights(
                TrainingCurriculum.STAGE2_MID,
                TrainingCurriculum.STAGE3_LATE,
                alpha
            )
    
    @staticmethod
    def _interpolate_weights(weights1: Dict, weights2: Dict, alpha: float) -> Dict:
        """在两个权重字典间线性插值"""
        result = {}
        for key in weights1.keys():
            if key not in ['progress_range', 'velocity_command_range']:
                result[key] = weights1[key] * (1 - alpha) + weights2[key] * alpha
            else:
                result[key] = weights2[key]  # 非数值项使用新阶段的值
        return result
    
    @staticmethod
    def get_velocity_command_range(training_progress: float) -> tuple[float, float]:
        """
        获取当前阶段的速度命令范围
        
        Args:
            training_progress: 训练进度 [0.0, 1.0]
            
        Returns:
            (min_vel, max_vel): 速度命令范围 (m/s)
        """
        weights = TrainingCurriculum.get_current_weights(training_progress)
        return weights['velocity_command_range']


def get_current_stage(training_progress: float) -> int:
    """
    根据训练进度返回当前阶段编号
    
    Args:
        training_progress: 训练进度 [0.0, 1.0]
        
    Returns:
        stage: 当前阶段编号 (0, 1, 2, 或 3)
    """
    if training_progress < 0.20:
        return 0  # 🆕 站立稳定期 (0-20%)
    elif training_progress < 0.45:
        return 1  # 早期行走 (20-45%)
    elif training_progress < 0.75:
        return 2  # 中期优化 (45-75%)
    else:
        return 3  # 后期精炼 (75-100%)

