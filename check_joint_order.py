#!/usr/bin/env python3
"""检查训练时的实际关节顺序和 default_joint_pos"""

import torch
from oceanbdx.assets.oceanusd import OCEAN_ROBOT_CFG

# 从配置中获取关节名称
joint_pos_dict = OCEAN_ROBOT_CFG.init_state.joint_pos

# 按照Isaac Lab的排序规则（字母顺序）
sorted_joints = sorted(joint_pos_dict.keys())

print("=" * 80)
print("训练时的实际关节顺序 (Isaac Lab按字母排序):")
print("=" * 80)

default_joint_pos = []
for i, joint_name in enumerate(sorted_joints):
    value = joint_pos_dict[joint_name]
    default_joint_pos.append(value)
    print(f"{i:2d}. {joint_name:20s} = {value:+7.4f}")

print("\n" + "=" * 80)
print("训练时的 default_dof_pos 数组:")
print("=" * 80)
print(default_joint_pos)
print()

# 转换为你配置文件中的格式
print("配置文件格式 (适用于 config.yaml):")
print("-" * 80)
print("default_dof_pos: [", end="")
for i, val in enumerate(default_joint_pos):
    if i > 0:
        print(", ", end="")
    if i == 5:
        print("\n                  ", end="")
    elif i == 10:
        print("\n                  ", end="")
    print(f"{val:+.2f}", end="")
print("]")
print()

# 检查与部署配置的差异
print("\n" + "=" * 80)
print("与你的 config.yaml 对比:")
print("=" * 80)
deployment_default = [0.13, 0.07, 0.2, 0.052, -0.05,    # Left leg (L1-L5)
                      -0.13, -0.07, -0.2, -0.052, 0.05,  # Right leg (R1-R5)
                      0.0, 0.0, 0.0, 0.0]                # Neck (N1-N4)

print(f"{'Index':<8} {'Joint Name':<20} {'Training':<12} {'Deployment':<12} {'Diff':<10}")
print("-" * 80)
for i, joint_name in enumerate(sorted_joints):
    train_val = default_joint_pos[i]
    deploy_val = deployment_default[i]
    diff = train_val - deploy_val
    status = "✅" if abs(diff) < 0.001 else "❌ MISMATCH"
    print(f"{i:<8} {joint_name:<20} {train_val:+8.4f}     {deploy_val:+8.4f}     {diff:+8.4f}  {status}")

print("\n" + "=" * 80)
