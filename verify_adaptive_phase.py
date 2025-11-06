#!/usr/bin/env python3
"""
验证训练环境 Step 0 时 adaptive_phase 的实际值
"""

import torch
import sys
sys.path.append("/home/ocean/oceanbdx/oceanbdx/source/oceanbdx")

from oceanbdx.tasks.manager_based.oceanbdx_locomotion.mdp.adaptive_phase_manager import (
    AdaptivePhaseManager,
    VideoGaitReference
)


def test_phase_manager():
    """测试 AdaptivePhaseManager 在不同场景下的输出"""
    
    print("=" * 80)
    print("🧪 AdaptivePhaseManager 测试")
    print("=" * 80)
    
    # 创建 phase manager
    num_envs = 1
    device = "cpu"
    video_config = VideoGaitReference()
    
    phase_manager = AdaptivePhaseManager(
        num_envs=num_envs,
        device=device,
        video_config=video_config
    )
    
    print("\n📋 VideoGaitReference 默认配置:")
    print(f"  reference_period: {video_config.reference_period}")
    print(f"  reference_stride: {video_config.reference_stride}")
    print(f"  foot_clearance: {video_config.foot_clearance}")
    
    print("\n📋 初始化后的内部状态:")
    print(f"  desired_period: {phase_manager.desired_period[0]:.4f}")
    print(f"  desired_stride: {phase_manager.desired_stride[0]:.4f}")
    print(f"  desired_clearance: {phase_manager.desired_clearance[0]:.4f}")
    print(f"  phase_rate: {phase_manager.phase_rate[0]:.4f}")
    
    # 场景1: 初始化后立即获取观测 (未调用 update)
    print("\n" + "=" * 80)
    print("📊 场景1: 初始化后立即获取观测 (未调用 update)")
    print("=" * 80)
    
    phase_obs_init = phase_manager.get_phase_observation()
    print(f"\nadaptive_phase 输出 (9维):")
    print(f"  完整: {phase_obs_init[0].tolist()}")
    print(f"  前6维 (sin/cos): {phase_obs_init[0, :6].tolist()}")
    print(f"  后3维 (归一化参数): {phase_obs_init[0, 6:].tolist()}")
    
    # 手动验证归一化计算
    print(f"\n🔍 手动验证归一化:")
    max_phase_rate = 2.0
    max_stride = 0.5
    max_clearance = 0.1
    
    phase_rate_norm = phase_manager.phase_rate[0] / max_phase_rate
    stride_norm = phase_manager.desired_stride[0] / max_stride
    clearance_norm = phase_manager.desired_clearance[0] / max_clearance
    
    print(f"  phase_rate_norm = {phase_manager.phase_rate[0]:.4f} / {max_phase_rate} = {phase_rate_norm:.4f}")
    print(f"  stride_norm = {phase_manager.desired_stride[0]:.4f} / {max_stride} = {stride_norm:.4f}")
    print(f"  clearance_norm = {phase_manager.desired_clearance[0]:.4f} / {max_clearance} = {clearance_norm:.4f}")
    
    # 场景2: 调用 update([0, 0], dt) 后 (静止速度)
    print("\n" + "=" * 80)
    print("📊 场景2: 调用 update([0, 0], dt=0.02) 后 (静止速度)")
    print("=" * 80)
    
    velocity = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device)
    dt = 0.02
    
    phase_manager.update(velocity, dt)
    
    print(f"\n更新后的内部状态:")
    print(f"  desired_period: {phase_manager.desired_period[0]:.4f}")
    print(f"  desired_stride: {phase_manager.desired_stride[0]:.4f}")
    print(f"  desired_clearance: {phase_manager.desired_clearance[0]:.4f}")
    print(f"  phase_rate: {phase_manager.phase_rate[0]:.4f}")
    
    phase_obs_static = phase_manager.get_phase_observation()
    print(f"\nadaptive_phase 输出 (9维):")
    print(f"  完整: {phase_obs_static[0].tolist()}")
    print(f"  前6维 (sin/cos): {phase_obs_static[0, :6].tolist()}")
    print(f"  后3维 (归一化参数): {phase_obs_static[0, 6:].tolist()}")
    
    # 手动验证归一化计算
    print(f"\n🔍 手动验证归一化:")
    phase_rate_norm = phase_manager.phase_rate[0] / max_phase_rate
    stride_norm = phase_manager.desired_stride[0] / max_stride
    clearance_norm = phase_manager.desired_clearance[0] / max_clearance
    
    print(f"  phase_rate_norm = {phase_manager.phase_rate[0]:.4f} / {max_phase_rate} = {phase_rate_norm:.4f}")
    print(f"  stride_norm = {phase_manager.desired_stride[0]:.4f} / {max_stride} = {stride_norm:.4f}")
    print(f"  clearance_norm = {phase_manager.desired_clearance[0]:.4f} / {max_clearance} = {clearance_norm:.4f}")
    
    # 场景3: 0.35 m/s 前进速度
    print("\n" + "=" * 80)
    print("📊 场景3: 调用 update([0.35, 0], dt=0.02) 后 (正常行走)")
    print("=" * 80)
    
    # 重置 phase manager
    phase_manager = AdaptivePhaseManager(num_envs, device, video_config)
    velocity = torch.tensor([[0.35, 0.0]], dtype=torch.float32, device=device)
    phase_manager.update(velocity, dt)
    
    print(f"\n更新后的内部状态:")
    print(f"  desired_period: {phase_manager.desired_period[0]:.4f}")
    print(f"  desired_stride: {phase_manager.desired_stride[0]:.4f}")
    print(f"  desired_clearance: {phase_manager.desired_clearance[0]:.4f}")
    print(f"  phase_rate: {phase_manager.phase_rate[0]:.4f}")
    
    phase_obs_walk = phase_manager.get_phase_observation()
    print(f"\nadaptive_phase 输出 (9维):")
    print(f"  完整: {phase_obs_walk[0].tolist()}")
    print(f"  后3维 (归一化参数): {phase_obs_walk[0, 6:].tolist()}")
    
    # 手动验证归一化计算
    print(f"\n🔍 手动验证归一化:")
    phase_rate_norm = phase_manager.phase_rate[0] / max_phase_rate
    stride_norm = phase_manager.desired_stride[0] / max_stride
    clearance_norm = phase_manager.desired_clearance[0] / max_clearance
    
    print(f"  phase_rate_norm = {phase_manager.phase_rate[0]:.4f} / {max_phase_rate} = {phase_rate_norm:.4f}")
    print(f"  stride_norm = {phase_manager.desired_stride[0]:.4f} / {max_stride} = {stride_norm:.4f}")
    print(f"  clearance_norm = {phase_manager.desired_clearance[0]:.4f} / {max_clearance} = {clearance_norm:.4f}")
    
    # 总结
    print("\n" + "=" * 80)
    print("📋 总结: 训练环境 Step 0 应该使用的值")
    print("=" * 80)
    
    print(f"\n如果 Step 0 未调用 update (使用初始化默认值):")
    print(f"  adaptive_phase[-3:] = {phase_obs_init[0, 6:].tolist()}")
    
    print(f"\n如果 Step 0 调用了 update([0,0]) (静止速度):")
    print(f"  adaptive_phase[-3:] = {phase_obs_static[0, 6:].tolist()}")
    
    print(f"\n如果 Step 0 调用了 update([0.35,0]) (参考速度):")
    print(f"  adaptive_phase[-3:] = {phase_obs_walk[0, 6:].tolist()}")
    
    print(f"\ntest_model_output.py 中硬编码的值:")
    print(f"  adaptive_phase[-3:] = [0.6667, 0.0, 0.37]")
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    test_phase_manager()
