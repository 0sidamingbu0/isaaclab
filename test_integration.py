#!/usr/bin/env python3
"""
快速测试脚本 - 验证自适应奖励系统集成
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source/oceanbdx"))

print("=" * 80)
print("🧪 测试Disney BDX自适应奖励系统集成")
print("=" * 80)

# 测试1: 导入核心模块
print("\n📦 测试1: 导入核心模块...")
try:
    from oceanbdx.tasks.manager_based.oceanbdx_locomotion.mdp import (
        AdaptivePhaseManager,
        TrainingCurriculum,
        VideoGaitReference,
        get_current_stage,
    )
    print("✅ 核心模块导入成功")
    print(f"   - AdaptivePhaseManager: {AdaptivePhaseManager}")
    print(f"   - TrainingCurriculum: {TrainingCurriculum}")
    print(f"   - VideoGaitReference: {VideoGaitReference}")
except Exception as e:
    print(f"❌ 核心模块导入失败: {e}")
    sys.exit(1)

# 测试2: 导入奖励函数
print("\n🎁 测试2: 导入奖励函数...")
try:
    from oceanbdx.tasks.manager_based.oceanbdx_locomotion.mdp import (
        reward_velocity_tracking_exp,
        reward_feet_alternating_contact,
        reward_action_smoothness,
        reward_joint_acceleration_penalty,
    )
    print("✅ 奖励函数导入成功")
    print(f"   - reward_velocity_tracking_exp: {reward_velocity_tracking_exp}")
    print(f"   - reward_feet_alternating_contact: {reward_feet_alternating_contact}")
    print(f"   - reward_action_smoothness: {reward_action_smoothness}")
    print(f"   - reward_joint_acceleration_penalty: {reward_joint_acceleration_penalty}")
except Exception as e:
    print(f"❌ 奖励函数导入失败: {e}")
    sys.exit(1)

# 测试3: 导入观测函数
print("\n👀 测试3: 导入观测函数...")
try:
    from oceanbdx.tasks.manager_based.oceanbdx_locomotion.mdp import (
        adaptive_gait_phase_observation,
    )
    print("✅ 观测函数导入成功")
    print(f"   - adaptive_gait_phase_observation: {adaptive_gait_phase_observation}")
except Exception as e:
    print(f"❌ 观测函数导入失败: {e}")
    sys.exit(1)

# 测试4: 导入环境类
print("\n🏃 测试4: 导入环境类...")
try:
    from oceanbdx.tasks.manager_based.oceanbdx_locomotion import OceanBDXLocomotionEnv
    print("✅ 环境类导入成功")
    print(f"   - OceanBDXLocomotionEnv: {OceanBDXLocomotionEnv}")
except Exception as e:
    print(f"❌ 环境类导入失败: {e}")
    sys.exit(1)

# 测试5: 创建VideoGaitReference实例
print("\n📹 测试5: 创建VideoGaitReference实例...")
try:
    video_ref = VideoGaitReference()
    print("✅ VideoGaitReference创建成功")
    print(f"   - 参考速度: {video_ref.reference_velocity} m/s")
    print(f"   - 步态周期: {video_ref.reference_period} s")
    print(f"   - 步幅: {video_ref.reference_stride} m")
    print(f"   - 躯干高度: {video_ref.nominal_base_height} m")
    print(f"   - 抬脚高度: {video_ref.foot_clearance} m")
except Exception as e:
    print(f"❌ VideoGaitReference创建失败: {e}")
    sys.exit(1)

# 测试6: 测试TrainingCurriculum
print("\n📚 测试6: 测试TrainingCurriculum...")
try:
    curriculum = TrainingCurriculum()
    
    # 测试不同阶段的权重
    stage1_weights = curriculum.get_current_weights(0.15)  # Stage1中期
    stage2_weights = curriculum.get_current_weights(0.50)  # Stage2中期
    stage3_weights = curriculum.get_current_weights(0.85)  # Stage3中期
    
    print("✅ TrainingCurriculum工作正常")
    print(f"   Stage1 (15%): velocity_tracking={stage1_weights['velocity_tracking']:.2f}")
    print(f"   Stage2 (50%): velocity_tracking={stage2_weights['velocity_tracking']:.2f}")
    print(f"   Stage3 (85%): velocity_tracking={stage3_weights['velocity_tracking']:.2f}")
    
    print(f"   Stage1 (15%): feet_alternating_contact={stage1_weights['feet_alternating_contact']:.2f}")
    print(f"   Stage2 (50%): feet_alternating_contact={stage2_weights['feet_alternating_contact']:.2f}")
    print(f"   Stage3 (85%): feet_alternating_contact={stage3_weights['feet_alternating_contact']:.2f}")
except Exception as e:
    print(f"❌ TrainingCurriculum测试失败: {e}")
    sys.exit(1)

# 测试7: 测试AdaptivePhaseManager（需要torch）
print("\n⏱️  测试7: 测试AdaptivePhaseManager...")
try:
    import torch
    
    num_envs = 4
    dt = 0.02
    device = "cpu"
    
    phase_mgr = AdaptivePhaseManager(num_envs=num_envs, dt=dt, device=device)
    print("✅ AdaptivePhaseManager创建成功")
    print(f"   - 环境数: {num_envs}")
    print(f"   - 时间步: {dt}s")
    print(f"   - 设备: {device}")
    
    # 模拟一步更新
    velocities = torch.tensor([[0.35, 0.0], [0.5, 0.0], [0.1, 0.0], [0.0, 0.0]], device=device)
    phase_mgr.update(velocities)
    phase_obs = phase_mgr.get_phase_observation()
    
    print(f"   - 相位观测维度: {phase_obs.shape}")
    print(f"   - 当前相位: {phase_mgr.current_phase[:2]}")
    print(f"   - 期望周期: {phase_mgr.desired_period[:2]}")
    print(f"   - 期望步长: {phase_mgr.desired_stride[:2]}")
    
except Exception as e:
    print(f"❌ AdaptivePhaseManager测试失败: {e}")
    sys.exit(1)

# 最终总结
print("\n" + "=" * 80)
print("🎉 所有集成测试通过！")
print("=" * 80)
print("\n✅ 自适应奖励系统已完全集成，可以开始训练")
print("\n下一步:")
print("  1. 运行快速测试训练（512环境，100迭代）")
print("  2. 检查TensorBoard日志")
print("  3. 开始全规模训练（4096环境，5000迭代）")
print("\n训练命令:")
print("  python scripts/rsl_rl/train_with_curriculum.py \\")
print("      --task=Isaac-OceanBDX-Locomotion-Main-v0 \\")
print("      --num_envs=512 \\")
print("      --headless \\")
print("      --max_iterations=100")
print("=" * 80 + "\n")
