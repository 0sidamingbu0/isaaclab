#!/usr/bin/env python3
"""
直接读取配置文件检查关节顺序和默认位置
"""

import yaml

# 读取训练配置中的初始关节位置
print("=" * 80)
print("📋 检查训练配置中的关节默认位置")
print("=" * 80)

# 从 __init__.py 中手动提取的初始关节位置
training_default_joint_pos = {
    "leg_r1_joint": -0.13,
    "leg_r2_joint": -0.07,
    "leg_r3_joint": -0.20,
    "leg_r4_joint": -0.052,
    "leg_r5_joint": 0.05,
    "leg_l1_joint": 0.13,
    "leg_l2_joint": 0.07,
    "leg_l3_joint": 0.20,
    "leg_l4_joint": 0.052,
    "leg_l5_joint": -0.05,
    "neck_n1_joint": 0.0,
    "neck_n2_joint": 0.0,
    "neck_n3_joint": 0.0,
    "neck_n4_joint": 0.0,
}

print("\n🔍 训练配置中的初始关节位置 (source/oceanbdx/oceanbdx/assets/oceanusd/__init__.py):")
print("=" * 80)
for joint_name, value in training_default_joint_pos.items():
    print(f"  {joint_name:15s}: {value:7.3f}")

# Isaac Lab 会按照字母顺序排序关节
sorted_joints = sorted(training_default_joint_pos.keys())
print("\n🔤 Isaac Lab 字母序排序后的关节顺序:")
print("=" * 80)
for i, joint_name in enumerate(sorted_joints):
    value = training_default_joint_pos[joint_name]
    print(f"  [{i:2d}] {joint_name:15s}: {value:7.3f}")

# 提取训练时的 default_dof_pos 数组
training_array = [training_default_joint_pos[joint_name] for joint_name in sorted_joints]
print("\n📊 训练时的 default_dof_pos 数组 (Isaac Lab 内部顺序):")
print("=" * 80)
print(f"  {training_array}")

# 读取部署配置
print("\n" + "=" * 80)
print("📋 检查部署配置中的关节默认位置")
print("=" * 80)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

deployment_default_pos = config['ocean/robot_lab']['default_dof_pos']
print("\n🚀 部署配置中的 default_dof_pos (config.yaml):")
print("=" * 80)
print(f"  {deployment_default_pos}")

# 读取部署配置的注释说明
print("\n📝 部署配置注释说明:")
print("  Left leg: l1,l2,l3,l4,l5 (TRAINING VALUES!)")
print("  Right leg: r1,r2,r3,r4,r5 (TRAINING VALUES!)")
print("  Neck: n1,n2,n3,n4 (neutral)")

# 比较两者
print("\n" + "=" * 80)
print("🔍 对比分析")
print("=" * 80)

print("\n部署配置假设的顺序 (根据注释):")
deployment_assumed_order = [
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",  # Left leg
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint",  # Right leg
    "neck_n1_joint", "neck_n2_joint", "neck_n3_joint", "neck_n4_joint",  # Neck
]

for i, joint_name in enumerate(deployment_assumed_order):
    expected_value = training_default_joint_pos[joint_name]
    actual_value = deployment_default_pos[i]
    match = "✅" if abs(expected_value - actual_value) < 0.001 else "❌"
    print(f"  [{i:2d}] {joint_name:15s}: 训练={expected_value:7.3f}, 部署={actual_value:7.3f} {match}")

# 检查是否需要映射
print("\n" + "=" * 80)
print("🔄 关节映射检查")
print("=" * 80)

joint_mapping = config['ocean/robot_lab']['joint_mapping']
print(f"\n当前 joint_mapping: {joint_mapping}")

print("\n映射后的顺序检查:")
for i, mapped_idx in enumerate(joint_mapping):
    training_joint = sorted_joints[i]
    deployment_joint = deployment_assumed_order[mapped_idx]
    training_val = training_array[i]
    deployment_val = deployment_default_pos[mapped_idx]
    match = "✅" if abs(training_val - deployment_val) < 0.001 else "❌"
    print(f"  训练[{i:2d}] {training_joint:15s} ({training_val:7.3f}) -> 部署[{mapped_idx:2d}] {deployment_joint:15s} ({deployment_val:7.3f}) {match}")

print("\n" + "=" * 80)
print("💡 结论")
print("=" * 80)
print("1. 检查训练时 Isaac Lab 的实际关节顺序是否与字母序一致")
print("2. 检查部署时的 default_dof_pos 是否与训练时匹配")
print("3. 检查 joint_mapping 是否正确映射了关节顺序")
print("=" * 80)
