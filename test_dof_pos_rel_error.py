#!/usr/bin/env python3
"""
测试: 如果 dof_pos_rel 计算错误,是否会导致极端输出
"""

import torch

# 加载模型
model_path = "logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt"
model = torch.jit.load(model_path)
model.eval()

print("=" * 80)
print("🧪 测试: dof_pos_rel 错误是否导致极端输出")
print("=" * 80)

# 场景1: 完全正确的观测
obs_correct = torch.tensor([[
    0.0, 0.0, 0.0,  # ang_vel
    0.0, 0.0, -1.0,  # gravity_vec
    # dof_pos_rel - 全0 (在default_dof_pos)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # dof_vel_scaled
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # joint_torques
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # commands
    0.0, 0.0, 0.0,
    # last_actions
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # adaptive_phase
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37
]], dtype=torch.float32)

with torch.no_grad():
    action_correct = model(obs_correct)

print(f"\n1️⃣  正确观测 (dof_pos_rel 全0):")
print(f"   动作范围: [{action_correct.min():.4f}, {action_correct.max():.4f}]")
print(f"   极端值 (|a|>1.5): {(torch.abs(action_correct) > 1.5).sum().item()} / 14")
print(f"   Left leg:  {action_correct[0, :5].tolist()}")
print(f"   Right leg: {action_correct[0, 5:10].tolist()}")
print(f"   Neck:      {action_correct[0, 10:14].tolist()}")

# 场景2: dof_pos_rel 第7维错误 (R3 关节)
# 部署日志显示: raw_dof_pos[7]=-0.2, default[7]=-0.0 → dof_pos_rel[7]=-0.2
obs_wrong_r3 = obs_correct.clone()
obs_wrong_r3[0, 6+7] = -0.2  # dof_pos_rel[7] = -0.2 instead of 0

with torch.no_grad():
    action_wrong_r3 = model(obs_wrong_r3)

print(f"\n2️⃣  dof_pos_rel[7]=-0.2 (R3 关节误差):")
print(f"   动作范围: [{action_wrong_r3.min():.4f}, {action_wrong_r3.max():.4f}]")
print(f"   极端值 (|a|>1.5): {(torch.abs(action_wrong_r3) > 1.5).sum().item()} / 14")
print(f"   Left leg:  {action_wrong_r3[0, :5].tolist()}")
print(f"   Right leg: {action_wrong_r3[0, 5:10].tolist()}")
print(f"   Neck:      {action_wrong_r3[0, 10:14].tolist()}")

# 场景3: 多个 dof_pos_rel 错误
obs_wrong_multi = obs_correct.clone()
obs_wrong_multi[0, 6:20] = torch.tensor([
    0.1, 0.05, 0.1, 0.05, -0.05,  # L1-L5 轻微误差
    -0.1, -0.05, -0.2, -0.05, 0.05,  # R1-R5 R3有较大误差
    0.0, 0.0, 0.0, 0.0  # N1-N4 正确
])

with torch.no_grad():
    action_wrong_multi = model(obs_wrong_multi)

print(f"\n3️⃣  多个 dof_pos_rel 错误:")
print(f"   动作范围: [{action_wrong_multi.min():.4f}, {action_wrong_multi.max():.4f}]")
print(f"   极端值 (|a|>1.5): {(torch.abs(action_wrong_multi) > 1.5).sum().item()} / 14")
print(f"   Left leg:  {action_wrong_multi[0, :5].tolist()}")
print(f"   Right leg: {action_wrong_multi[0, 5:10].tolist()}")
print(f"   Neck:      {action_wrong_multi[0, 10:14].tolist()}")

# 场景4: dof_pos_rel 有大范围错误 (模拟严重的计算错误)
obs_wrong_large = obs_correct.clone()
obs_wrong_large[0, 6:20] = torch.rand(14) * 0.4 - 0.2  # 随机 [-0.2, 0.2]

with torch.no_grad():
    action_wrong_large = model(obs_wrong_large)

print(f"\n4️⃣  dof_pos_rel 大范围随机错误:")
print(f"   动作范围: [{action_wrong_large.min():.4f}, {action_wrong_large.max():.4f}]")
print(f"   极端值 (|a|>1.5): {(torch.abs(action_wrong_large) > 1.5).sum().item()} / 14")
print(f"   Left leg:  {action_wrong_large[0, :5].tolist()}")
print(f"   Right leg: {action_wrong_large[0, 5:10].tolist()}")
print(f"   Neck:      {action_wrong_large[0, 10:14].tolist()}")

print("\n" + "=" * 80)
print("🎯 分析")
print("=" * 80)

print(f"\n部署环境 Step 0 的实际输出:")
print(f"   Left leg:  [-1.96, -0.75, -1.10, 0.77, 1.23]")
print(f"   Right leg: [-0.79, 1.41, 2.00, 1.64, -0.91]")
print(f"   Neck:      [0.23, -1.55, 1.36, 0.71]")
print(f"   极端值: 10/14  ❌")

print(f"\n如果以上测试都没有产生类似的极端输出,说明:")
print(f"   1. 问题不是 dof_pos_rel 计算错误")
print(f"   2. 可能是其他观测项 (dof_vel, joint_torques, etc.)")
print(f"   3. 或者是部署环境的模型推理方式有问题")

print("\n✅ 测试完成!")
