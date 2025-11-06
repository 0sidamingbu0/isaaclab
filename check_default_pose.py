#!/usr/bin/env python3
"""检查训练时的 default_dof_pos 实际顺序"""

import sys
sys.path.append("/home/ocean/oceanbdx/oceanbdx/source/oceanbdx")

# 直接读取配置文件
from oceanbdx.assets.oceanusd import OCEAN_ROBOT_CFG

print("=" * 80)
print("训练时的 default_joint_pos 配置")
print("=" * 80)

# 获取初始关节位置
init_state = OCEAN_ROBOT_CFG.init_state
joint_pos = init_state.joint_pos

print(f"\n字典内容 (关节名 -> 角度):")
print("-" * 80)
for joint_name, angle in joint_pos.items():
    print(f"  {joint_name:20s}: {angle:7.3f}")

print("\n\n关节名列表 (Python字典的键顺序):")
print("-" * 80)
joint_names = list(joint_pos.keys())
for i, name in enumerate(joint_names):
    print(f"  [{i:2d}] {name:20s}: {joint_pos[name]:7.3f}")

print("\n\n数组形式的 default_dof_pos (Isaac Lab 实际使用的顺序):")
print("-" * 80)
default_values = [joint_pos[name] for name in joint_names]
print(f"default_dof_pos = {default_values}")

# 分组显示
print("\n\n按组显示:")
print("-" * 80)
leg_joints = [name for name in joint_names if 'leg' in name]
neck_joints = [name for name in joint_names if 'neck' in name]

print("\n腿部关节:")
for name in leg_joints:
    print(f"  {name:20s}: {joint_pos[name]:7.3f}")

print("\n颈部关节:")
for name in neck_joints:
    print(f"  {name:20s}: {joint_pos[name]:7.3f}")

# 检查训练顺序与部署顺序的映射
print("\n\n" + "=" * 80)
print("训练顺序 vs 部署顺序对比")
print("=" * 80)

training_order = joint_names
deployment_order_description = [
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint",
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
    "neck_n1_joint", "neck_n2_joint", "neck_n3_joint", "neck_n4_joint"
]

print("\n训练时关节顺序 (Isaac Lab 内部):")
for i, name in enumerate(training_order):
    print(f"  [{i:2d}] {name:20s}: {joint_pos[name]:7.3f}")

print("\n\n部署时期望的关节顺序 (Gazebo/ROS2):")
for i, name in enumerate(deployment_order_description):
    if name in joint_pos:
        print(f"  [{i:2d}] {name:20s}: {joint_pos[name]:7.3f}")
    else:
        print(f"  [{i:2d}] {name:20s}: NOT FOUND")

print("\n\n生成 joint_mapping:")
print("-" * 80)
if set(training_order) == set(deployment_order_description):
    mapping = []
    for deployment_name in deployment_order_description:
        training_idx = training_order.index(deployment_name)
        mapping.append(training_idx)
    print(f"joint_mapping = {mapping}")
    print("\n解释: deployment[i] = training[mapping[i]]")
    print("即: 部署代码中的第 i 个关节对应训练时的第 mapping[i] 个关节")
else:
    print("ERROR: 关节名不匹配！")
    print(f"训练有但部署没有: {set(training_order) - set(deployment_order_description)}")
    print(f"部署有但训练没有: {set(deployment_order_description) - set(training_order)}")
