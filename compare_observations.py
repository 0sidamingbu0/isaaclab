#!/usr/bin/env python3
"""
逐项对比两次测试的观测,找出差异
"""

import torch

print("=" * 80)
print("🔍 逐项对比观测向量")
print("=" * 80)

# 测试1: test_model_output.py - 输出正常
obs1 = torch.tensor([[
    0.0, 0.0, 0.0,  # ang_vel
    0.0, 0.0, -1.0,  # gravity_vec
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dof_pos_rel
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dof_vel
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # torques
    0.0, 0.0, 0.0,  # commands
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # last_actions
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37  # adaptive_phase (正确)
]], dtype=torch.float32)

# 测试2: test_deployment_observation.py - 也输出正常
obs2 = torch.tensor([[
    0.0, 0.0, 0.0,  # ang_vel
    0.0, -0.0, -1.0,  # gravity_vec ← 注意这里有 -0.0
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dof_pos_rel
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dof_vel
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # torques
    0.0, 0.0, 0.0,  # commands
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # last_actions
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0  # adaptive_phase (错误)
]], dtype=torch.float32)

print("\n📋 观测向量对比:")
print(f"obs1 (正确 adaptive_phase): {obs1.shape}")
print(f"obs2 (错误 adaptive_phase): {obs2.shape}")

# 逐项对比
diff = torch.abs(obs1 - obs2)
max_diff = diff.max().item()
diff_indices = torch.where(diff > 1e-6)[1].tolist()

print(f"\n最大差异: {max_diff:.6f}")
print(f"差异维度数量: {len(diff_indices)}")

if len(diff_indices) > 0:
    print(f"\n差异维度列表:")
    
    obs_names = [
        (0, 3, "ang_vel_body"),
        (3, 6, "gravity_vec"),
        (6, 20, "dof_pos_rel"),
        (20, 34, "dof_vel_scaled"),
        (34, 48, "joint_torques"),
        (48, 51, "commands"),
        (51, 65, "last_actions"),
        (65, 74, "adaptive_phase"),
    ]
    
    for idx in diff_indices:
        # 找到对应的观测名称
        obs_name = "unknown"
        local_idx = idx
        for start, end, name in obs_names:
            if start <= idx < end:
                obs_name = name
                local_idx = idx - start
                break
        
        print(f"  [{idx:2d}] {obs_name:16s} [{local_idx:2d}]: {obs1[0,idx].item():8.4f} vs {obs2[0,idx].item():8.4f} (diff={diff[0,idx].item():.4f})")

# 加载模型测试
print("\n" + "=" * 80)
print("🤖 模型输出对比")
print("=" * 80)

model_path = "logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt"
model = torch.jit.load(model_path)
model.eval()

with torch.no_grad():
    action1 = model(obs1)
    action2 = model(obs2)

print(f"\n模型输出对比:")
print(f"action1 (正确 adaptive_phase): 范围 [{action1.min().item():.4f}, {action1.max().item():.4f}]")
print(f"action2 (错误 adaptive_phase): 范围 [{action2.min().item():.4f}, {action2.max().item():.4f}]")

action_diff = torch.abs(action1 - action2)
print(f"\n动作差异: 最大 {action_diff.max().item():.4f}, 平均 {action_diff.mean().item():.4f}")

print("\n动作详细对比:")
for i in range(14):
    print(f"  Joint {i:2d}: {action1[0,i].item():7.4f} vs {action2[0,i].item():7.4f} (diff={action_diff[0,i].item():.4f})")

print("\n" + "=" * 80)
print("🎯 结论")
print("=" * 80)

if action_diff.max().item() < 0.5:
    print("\n✅ adaptive_phase 最后3维的差异对模型输出影响较小 (<0.5)")
    print("   这说明模型对这3维不太敏感,或者已经学会了忽略异常值")
else:
    print("\n⚠️ adaptive_phase 最后3维的差异对模型输出有明显影响 (>0.5)")

print("\n❓ 既然训练环境中两种观测都输出正常,为什么部署环境输出极端值?")
print("\n可能的原因:")
print("1. 部署环境的其他观测项(不是adaptive_phase)有问题")
print("2. 部署环境的模型加载/推理方式有差异")
print("3. 部署环境的数值精度或数据类型不一致")
print("4. 部署环境可能使用了不同的观测顺序或映射")

print("\n🔍 下一步调试:")
print("1. 让部署AI打印完整的74维观测向量")
print("2. 逐项对比部署观测 vs 训练观测")
print("3. 检查部署环境的数据类型 (float32 vs float64)")
print("4. 检查部署环境的模型推理代码")
