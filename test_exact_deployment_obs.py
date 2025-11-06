#!/usr/bin/env python3
"""
测试 Call #0 的观测值是否导致模型输出极端动作
"""

import torch
import numpy as np

# Load model
model_path = "logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt"
model = torch.jit.load(model_path)
model.eval()

# Call #0 的观测值（从日志中复制）
# 🔍 BEFORE contiguous() - First 20 values:
#    [0.0036, -1.6202, 0.1770, 0.2074, 0.1335, -0.9691, -0.0248, 0.0006, -0.0307, -0.0028, 
#     0.0887, 0.0244, -0.0005, 0.0360, -0.0057, -0.0248, -0.0183, 0.0079, 0.0011, -0.0001]
# 📊 Last 9 observation values (adaptive_phase):
#    [0.4818, 0.8763, 0.6845, 0.7290, 0.8443, 0.5358, 0.6667, 0.0000, 0.3700]

obs_call0 = torch.tensor([[
    0.0036, -1.6202, 0.1770,  # ang_vel
    0.2074, 0.1335, -0.9691,  # gravity_vec
    -0.0248, 0.0006, -0.0307, -0.0028, 0.0887,  # dof_pos前5个（Left leg）
    0.0244, -0.0005, 0.0360, -0.0057, -0.0248,  # dof_pos 6-10（Right leg）
    -0.0183, 0.0079, 0.0011, -0.0001,  # dof_pos 11-14 (Neck)
    # 接下来应该是 dof_vel (14), joint_torques (14), commands (3), last_actions (14)
    # 从日志中没有完整的74维，我们需要补全
    # 假设后面都是合理的小值
    ]], dtype=torch.float32)

print("=" * 80)
print("测试 Call #0 观测值")
print("=" * 80)

# 先构造完整的74维观测
# 根据日志，我们知道前20维和后9维，中间的需要合理估计
# 74 = 3 (ang_vel) + 3 (gravity) + 14 (dof_pos) + 14 (dof_vel) + 14 (torques) + 3 (commands) + 14 (actions) + 9 (phase)

# 从日志 "First 20 observation values" 和 "Last 9 observation values" 重建
first_20 = [0.0036, -1.6202, 0.1770, 0.2074, 0.1335, -0.9691, -0.0248, 0.0006, -0.0307, -0.0028, 
            0.0887, 0.0244, -0.0005, 0.0360, -0.0057, -0.0248, -0.0183, 0.0079, 0.0011, -0.0001]
last_9 = [0.4818, 0.8763, 0.6845, 0.7290, 0.8443, 0.5358, 0.6667, 0.0000, 0.3700]

# 中间的 74 - 20 - 9 = 45 维
# 这些应该是：dof_vel(14) + torques(14) + commands(3) + last_actions(14) = 45
# 从日志看commands是[0,0,0]，last_actions初始应该也是0

middle_45 = [0.0] * 45  # 简化处理，用零填充

obs_full = torch.tensor([first_20 + middle_45 + last_9], dtype=torch.float32)

print(f"\n观测维度: {obs_full.shape}")
print(f"前20维: {obs_full[0, :20].tolist()}")
print(f"后9维: {obs_full[0, -9:].tolist()}")

# 推理
with torch.no_grad():
    actions = model(obs_full)

print(f"\n模型输出:")
print(f"  Shape: {actions.shape}")
print(f"  Range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
print(f"  Mean: {actions.mean().item():.4f}")
print(f"  Std: {actions.std().item():.4f}")

print(f"\n所有14个关节的动作:")
for i, act in enumerate(actions[0]):
    marker = " ⚠️ " if abs(act.item()) > 1.5 else "    "
    print(f"  Joint {i:2d}: {act.item():7.4f}{marker}")

extreme_count = (torch.abs(actions) > 1.5).sum().item()
print(f"\n极端动作 (|a|>1.5) 数量: {extreme_count}/14")

if extreme_count > 7:
    print("\n❌ 模型认为机器人处于BAD状态！")
else:
    print("\n✅ 模型输出正常范围的动作")

# 再测试一个完全静止的状态
print("\n" + "=" * 80)
print("对比测试：完全静止的初始状态")
print("=" * 80)

obs_zero = torch.zeros((1, 74), dtype=torch.float32)
obs_zero[0, 2] = 0.0  # ang_vel[2]
obs_zero[0, 5] = -1.0  # gravity_vec[2]
obs_zero[0, -9:] = torch.tensor(last_9)  # adaptive_phase

print(f"观测: ang_vel=[0,0,0], gravity=[0,0,-1], dof_pos=all_0, ...")

with torch.no_grad():
    actions_zero = model(obs_zero)

print(f"\n模型输出:")
print(f"  Range: [{actions_zero.min().item():.4f}, {actions_zero.max().item():.4f}]")
print(f"  所有动作:")
for i, act in enumerate(actions_zero[0]):
    print(f"    Joint {i:2d}: {act.item():7.4f}")

extreme_count_zero = (torch.abs(actions_zero) > 1.5).sum().item()
print(f"\n极端动作数量: {extreme_count_zero}/14")
