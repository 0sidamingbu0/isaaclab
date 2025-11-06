#!/usr/bin/env python3
"""
简化版本: 直接计算 adaptive_phase 的值,不依赖完整 Isaac Lab 环境
"""

import torch


class VideoGaitReference:
    """从Disney BDX参考视频中提取的步态参数"""
    reference_period = 0.75
    reference_stride = 0.131
    foot_clearance = 0.037


class AdaptiveGaitTable:
    """速度-步态参数映射表"""
    GAIT_PARAMS = {
        0.0:  (0.8,  0.0,   0.0),
        0.1:  (0.8,  0.08,  0.025),
        0.25: (0.8,  0.2,   0.03),
        0.35: (0.75, 0.262, 0.037),
        0.5:  (0.65, 0.325, 0.045),
        0.6:  (0.6,  0.36,  0.055),
        0.74: (0.5,  0.37,  0.07),
    }
    
    @staticmethod
    def interpolate(speed_value):
        """根据速度插值获取期望步态参数"""
        velocities = sorted(AdaptiveGaitTable.GAIT_PARAMS.keys())
        
        # Clamp速度到表格范围
        s = max(min(speed_value, velocities[-1]), velocities[0])
        
        # 查找插值区间
        for j in range(len(velocities) - 1):
            if velocities[j] <= s <= velocities[j + 1]:
                # 线性插值
                alpha = (s - velocities[j]) / (velocities[j + 1] - velocities[j])
                
                params_j = AdaptiveGaitTable.GAIT_PARAMS[velocities[j]]
                params_j1 = AdaptiveGaitTable.GAIT_PARAMS[velocities[j + 1]]
                
                period = params_j[0] * (1 - alpha) + params_j1[0] * alpha
                stride = params_j[1] * (1 - alpha) + params_j1[1] * alpha
                clearance = params_j[2] * (1 - alpha) + params_j1[2] * alpha
                
                return period, stride, clearance
        
        # 边界情况
        return AdaptiveGaitTable.GAIT_PARAMS[velocities[0]]


def calculate_adaptive_phase(period, stride, clearance, motion_time=0.0):
    """
    计算 adaptive_phase 的9维观测
    
    Args:
        period: 期望周期 (s)
        stride: 期望步幅 (m, 两步距离)
        clearance: 期望抬脚高度 (m)
        motion_time: 累计运动时间
    
    Returns:
        9维观测向量
    """
    # 计算theta
    theta = torch.tensor(motion_time) * torch.pi / 2.0
    
    # 多频率sin/cos编码
    phase_feat = [
        torch.sin(theta).item(),
        torch.cos(theta).item(),
        torch.sin(theta / 2.0).item(),
        torch.cos(theta / 2.0).item(),
        torch.sin(theta / 4.0).item(),
        torch.cos(theta / 4.0).item(),
    ]
    
    # 归一化参数
    max_stride = 0.5
    max_clearance = 0.1
    max_phase_rate = 2.0
    
    phase_rate = 1.0 / (period + 1e-8)
    phase_rate_norm = min(max(phase_rate / max_phase_rate, 0.0), 1.0)
    stride_norm = min(max(stride / max_stride, 0.0), 1.0)
    clearance_norm = min(max(clearance / max_clearance, 0.0), 1.0)
    
    return phase_feat + [phase_rate_norm, stride_norm, clearance_norm]


def main():
    print("=" * 80)
    print("🧪 adaptive_phase 计算验证")
    print("=" * 80)
    
    video_config = VideoGaitReference()
    
    print("\n📋 VideoGaitReference 默认配置:")
    print(f"  reference_period: {video_config.reference_period}")
    print(f"  reference_stride: {video_config.reference_stride}")
    print(f"  foot_clearance: {video_config.foot_clearance}")
    
    # 场景1: 初始化默认值 (未调用 update)
    print("\n" + "=" * 80)
    print("📊 场景1: 初始化默认值 (未调用 update)")
    print("=" * 80)
    
    period_init = video_config.reference_period
    stride_init = video_config.reference_stride * 2.0  # 双倍
    clearance_init = video_config.foot_clearance
    
    print(f"\n内部状态:")
    print(f"  desired_period: {period_init}")
    print(f"  desired_stride: {stride_init}")
    print(f"  desired_clearance: {clearance_init}")
    print(f"  phase_rate: {1.0/period_init:.4f}")
    
    phase_obs_init = calculate_adaptive_phase(period_init, stride_init, clearance_init, motion_time=0.0)
    
    print(f"\nadaptive_phase 输出 (9维):")
    print(f"  完整: {phase_obs_init}")
    print(f"  前6维 (sin/cos): {phase_obs_init[:6]}")
    print(f"  后3维 (归一化): {phase_obs_init[6:]}")
    
    # 手动验证
    print(f"\n手动验证:")
    phase_rate_norm = (1.0 / period_init) / 2.0
    stride_norm = stride_init / 0.5
    clearance_norm = clearance_init / 0.1
    print(f"  phase_rate_norm = (1/{period_init}) / 2.0 = {phase_rate_norm:.4f}")
    print(f"  stride_norm = {stride_init} / 0.5 = {stride_norm:.4f}")
    print(f"  clearance_norm = {clearance_init} / 0.1 = {clearance_norm:.4f}")
    
    # 场景2: 静止速度 (speed=0.0)
    print("\n" + "=" * 80)
    print("📊 场景2: update(speed=0.0) 静止速度")
    print("=" * 80)
    
    period_static, stride_static, clearance_static = AdaptiveGaitTable.interpolate(0.0)
    
    print(f"\n从 GAIT_PARAMS[0.0] 获取:")
    print(f"  desired_period: {period_static}")
    print(f"  desired_stride: {stride_static}")
    print(f"  desired_clearance: {clearance_static}")
    print(f"  phase_rate: {1.0/period_static:.4f}")
    
    phase_obs_static = calculate_adaptive_phase(period_static, stride_static, clearance_static, motion_time=0.0)
    
    print(f"\nadaptive_phase 输出 (9维):")
    print(f"  完整: {phase_obs_static}")
    print(f"  后3维 (归一化): {phase_obs_static[6:]}")
    
    # 手动验证
    print(f"\n手动验证:")
    phase_rate_norm = (1.0 / period_static) / 2.0
    stride_norm = stride_static / 0.5
    clearance_norm = clearance_static / 0.1
    print(f"  phase_rate_norm = (1/{period_static}) / 2.0 = {phase_rate_norm:.4f}")
    print(f"  stride_norm = {stride_static} / 0.5 = {stride_norm:.4f}")
    print(f"  clearance_norm = {clearance_static} / 0.1 = {clearance_norm:.4f}")
    
    # 场景3: 参考速度 (speed=0.35)
    print("\n" + "=" * 80)
    print("📊 场景3: update(speed=0.35) 参考行走速度")
    print("=" * 80)
    
    period_walk, stride_walk, clearance_walk = AdaptiveGaitTable.interpolate(0.35)
    
    print(f"\n从 GAIT_PARAMS[0.35] 获取:")
    print(f"  desired_period: {period_walk}")
    print(f"  desired_stride: {stride_walk}")
    print(f"  desired_clearance: {clearance_walk}")
    print(f"  phase_rate: {1.0/period_walk:.4f}")
    
    phase_obs_walk = calculate_adaptive_phase(period_walk, stride_walk, clearance_walk, motion_time=0.0)
    
    print(f"\nadaptive_phase 输出 (9维):")
    print(f"  完整: {phase_obs_walk}")
    print(f"  后3维 (归一化): {phase_obs_walk[6:]}")
    
    # 总结
    print("\n" + "=" * 80)
    print("📋 总结对比")
    print("=" * 80)
    
    print(f"\n场景1 - 初始化默认值:")
    print(f"  adaptive_phase[-3:] = [{phase_obs_init[6]:.4f}, {phase_obs_init[7]:.4f}, {phase_obs_init[8]:.4f}]")
    
    print(f"\n场景2 - 静止速度 (speed=0.0):")
    print(f"  adaptive_phase[-3:] = [{phase_obs_static[6]:.4f}, {phase_obs_static[7]:.4f}, {phase_obs_static[8]:.4f}]")
    
    print(f"\n场景3 - 参考速度 (speed=0.35):")
    print(f"  adaptive_phase[-3:] = [{phase_obs_walk[6]:.4f}, {phase_obs_walk[7]:.4f}, {phase_obs_walk[8]:.4f}]")
    
    print(f"\ntest_model_output.py 硬编码值:")
    print(f"  adaptive_phase[-3:] = [0.6667, 0.0000, 0.3700]")
    
    print("\n" + "=" * 80)
    print("🎯 推断: 训练环境 Step 0 的实际行为")
    print("=" * 80)
    
    print("\n分析 test_model_output.py 的值 [0.6667, 0.0, 0.37]:")
    print("  - 第7维 0.6667 = 场景1的 phase_rate_norm")
    print("  - 第8维 0.0000 = 场景2的 stride_norm")
    print("  - 第9维 0.3700 = 场景1的 clearance_norm")
    
    print("\n这是一个 '混合值',可能的原因:")
    print("  1. 训练环境在 reset 时使用特殊逻辑")
    print("  2. 静止时保持默认 phase_rate 和 clearance,但 stride=0")
    print("  3. 或者 test_model_output.py 的值是手动调试的,不完全准确")
    
    print("\n" + "=" * 80)
    print("🔧 给部署 AI 的建议")
    print("=" * 80)
    
    print("\n方案1 (最安全): 直接使用 test_model_output.py 的值")
    print("  adaptive_phase[-3:] = [0.6667, 0.0, 0.37]")
    
    print("\n方案2 (逻辑推导): 静止时的混合逻辑")
    print("  if (speed < 0.001):")
    print("    phase_rate_norm = (1.0/0.75) / 2.0 = 0.6667  // 保持默认节奏")
    print("    stride_norm = 0.0                           // 不移动")
    print("    clearance_norm = 0.037 / 0.1 = 0.37        // 保持默认抬脚高度")
    
    print("\n✅ 建议先使用方案1快速修复,验证效果!")


if __name__ == "__main__":
    main()
