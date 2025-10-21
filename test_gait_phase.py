#!/usr/bin/env python3
"""测试参考步态相位跟踪奖励函数"""

import torch
import math

# 测试参考步态数据转换和相位计算
def test_gait_phase():
    # 参考步态数据（度）
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
    ], dtype=torch.float32)
    
    # 转换为弧度
    reference_gait_rad = reference_gait_deg * (math.pi / 180.0)
    
    print("✅ 参考步态数据加载成功")
    print(f"   - 关键帧数量: {reference_gait_rad.shape[0]}")
    print(f"   - 关节数量: {reference_gait_rad.shape[1]}")
    print(f"   - 数据范围 (弧度): [{reference_gait_rad.min():.3f}, {reference_gait_rad.max():.3f}]")
    print(f"   - 数据范围 (度): [{reference_gait_deg.min():.1f}, {reference_gait_deg.max():.1f}]")
    
    # 测试相位计算
    gait_period = 0.75  # 步态周期
    dt = 0.02  # 50Hz 控制频率
    
    print("\n📊 相位计算测试 (步态周期: 0.75秒, 频率: 50Hz)")
    print("步数 | 时间(s) | 相位比例 | 关键帧索引 | 对应时间(s)")
    print("-" * 70)
    
    for step in [0, 10, 18, 37, 50, 75, 100]:
        time_in_cycle = (step * dt) % gait_period
        phase_ratio = time_in_cycle / gait_period
        keyframe_idx = int(phase_ratio * 18)
        keyframe_idx = min(keyframe_idx, 17)  # 确保范围
        keyframe_time = keyframe_idx * 0.0417
        
        print(f"{step:4d} | {step*dt:6.3f} | {phase_ratio:9.3f} | {keyframe_idx:13d} | {keyframe_time:13.4f}")
    
    # 测试奖励计算
    print("\n🎯 奖励计算示例")
    # 模拟当前关节角度（假设完美跟踪第0帧）
    current_pos = reference_gait_rad[0].unsqueeze(0)  # [1, 10]
    reference_pos = reference_gait_rad[0].unsqueeze(0)  # [1, 10]
    
    # 计算误差
    joint_error = torch.sum(torch.square(current_pos - reference_pos), dim=1)
    std = 0.5
    reward = torch.exp(-joint_error / (std ** 2))
    
    print(f"   - 完美跟踪 (误差=0): 奖励 = {reward.item():.6f}")
    
    # 添加一些误差
    noisy_pos = current_pos + torch.randn_like(current_pos) * 0.1
    joint_error_noisy = torch.sum(torch.square(noisy_pos - reference_pos), dim=1)
    reward_noisy = torch.exp(-joint_error_noisy / (std ** 2))
    
    print(f"   - 带噪声跟踪 (误差={joint_error_noisy.item():.4f}): 奖励 = {reward_noisy.item():.6f}")
    
    # 完全错误的姿态
    wrong_pos = torch.zeros_like(current_pos)
    joint_error_wrong = torch.sum(torch.square(wrong_pos - reference_pos), dim=1)
    reward_wrong = torch.exp(-joint_error_wrong / (std ** 2))
    
    print(f"   - 全零姿态 (误差={joint_error_wrong.item():.4f}): 奖励 = {reward_wrong.item():.6f}")
    
    print("\n✅ 测试完成！")
    print("\n💡 建议:")
    print("   1. 步态周期0.75秒对应18个关键帧，每帧间隔0.0417秒")
    print("   2. 50Hz控制频率下，每37-38步完成一个完整步态周期")
    print("   3. 标准差std=0.5可以调整，越小对跟踪精度要求越高")
    print("   4. 奖励权重weight=1.0可以根据训练效果调整")

if __name__ == "__main__":
    test_gait_phase()
