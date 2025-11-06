#!/usr/bin/env python3
"""
生成标准测试用例供部署验证使用
不依赖 Isaac Lab，可直接运行
"""

import json
import math


def generate_standard_test_cases():
    """生成标准测试用例"""
    
    print("=" * 80)
    print("🔧 生成 OceanBDX 标准测试用例")
    print("=" * 80)
    
    # 训练时的关节顺序（Isaac Lab 字母序）
    joint_order_training = [
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint",
        "neck_n1_joint", "neck_n2_joint", "neck_n3_joint", "neck_n4_joint",
    ]
    
    # 默认关节位置（训练值）
    default_dof_pos = [
        0.13, 0.07, 0.2, 0.052, -0.05,    # Left leg
        -0.13, -0.07, -0.2, -0.052, 0.05,  # Right leg
        0.0, 0.0, 0.0, 0.0                 # Neck
    ]
    
    test_cases = {}
    
    # ============================================================
    # Test Case 1: 完美初始状态 (default_dof_pos, 静止命令)
    # ============================================================
    print("\n📋 生成 Test Case 1: 完美初始状态")
    
    # 构建观测向量 (74 维)
    observation_1 = (
        [0.0, 0.0, 0.0] +                          # 1. ang_vel_body (3)
        [0.0, 0.0, +9.81] +                        # 2. gravity_vec (3) - 注意是+9.81!
        [0.0] * 14 +                               # 3. dof_pos_rel (14) - 全零因为在default
        [0.0] * 14 +                               # 4. dof_vel_scaled (14)
        [0.0] * 14 +                               # 5. joint_torques (14)
        [0.0, 0.0, 0.0] +                          # 6. commands (3) - 静止
        [0.0] * 14 +                               # 7. last_actions (14)
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37]  # 8. adaptive_phase (9)
    )
    
    test_cases["test_case_1"] = {
        "name": "完美初始状态",
        "description": "机器人在 default_dof_pos，静止命令，直立无旋转",
        "robot_state": {
            "base_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "base_angular_velocity": [0.0, 0.0, 0.0],
            "joint_positions": default_dof_pos,
            "joint_velocities": [0.0] * 14,
            "joint_torques": [0.0] * 14,
            "velocity_commands": [0.0, 0.0, 0.0],
        },
        "observation_vector": observation_1,
        "observation_breakdown": {
            "ang_vel_body": [0.0, 0.0, 0.0],
            "gravity_vec": [0.0, 0.0, +9.81],
            "dof_pos_rel": [0.0] * 14,
            "dof_vel_scaled": [0.0] * 14,
            "joint_torques": [0.0] * 14,
            "commands": [0.0, 0.0, 0.0],
            "last_actions": [0.0] * 14,
            "adaptive_phase": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37],
        },
        "expected_behavior": {
            "action_range": "接近零的小值，例如 [-0.3, 0.3]",
            "action_pattern": "不应该全是极端值 ±2",
            "robot_should": "保持站立，不倒下",
        },
        "joint_order_training": joint_order_training,
        "default_dof_pos": default_dof_pos,
    }
    
    # ============================================================
    # Test Case 2: 前倾 15 度
    # ============================================================
    print("📋 生成 Test Case 2: 前倾 15 度")
    
    angle = math.radians(15)
    quat_pitch = [
        math.cos(angle/2),  # qw
        0.0,                 # qx
        math.sin(angle/2),  # qy (绕 Y 轴)
        0.0                  # qz
    ]
    
    # 前倾 15° 时重力投影: gx ≈ -9.81*sin(15°), gz ≈ 9.81*cos(15°)
    gx = -9.81 * math.sin(angle)  # ≈ -2.54
    gz = 9.81 * math.cos(angle)   # ≈ 9.47
    
    observation_2 = (
        [0.0, 0.0, 0.0] +                          # ang_vel
        [gx, 0.0, gz] +                            # gravity_vec (前倾)
        [0.0] * 14 +                               # dof_pos_rel
        [0.0] * 14 +                               # dof_vel
        [0.0] * 14 +                               # torques
        [0.0, 0.0, 0.0] +                          # commands
        [0.0] * 14 +                               # actions
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37]
    )
    
    test_cases["test_case_2"] = {
        "name": "前倾 15 度",
        "description": "base 绕 Y 轴前倾 15 度，测试 gravity_vec 方向",
        "robot_state": {
            "base_quaternion_wxyz": quat_pitch,
            "base_angular_velocity": [0.0, 0.0, 0.0],
            "joint_positions": default_dof_pos,
            "joint_velocities": [0.0] * 14,
            "joint_torques": [0.0] * 14,
            "velocity_commands": [0.0, 0.0, 0.0],
        },
        "observation_vector": observation_2,
        "observation_breakdown": {
            "gravity_vec": [round(gx, 2), 0.0, round(gz, 2)],
        },
        "expected_behavior": {
            "action_pattern": "应该输出向后倾的纠正动作",
            "key_joints": "L3/R3 (hip pitch) 应该有明显动作",
        },
    }
    
    # ============================================================
    # Test Case 3: 左腿 L3 抬高 0.3 rad
    # ============================================================
    print("📋 生成 Test Case 3: 左腿 L3 抬高")
    
    dof_pos_3 = default_dof_pos.copy()
    dof_pos_3[2] += 0.3  # leg_l3_joint 索引 2
    
    dof_pos_rel_3 = [dof_pos_3[i] - default_dof_pos[i] for i in range(14)]
    
    observation_3 = (
        [0.0, 0.0, 0.0] +
        [0.0, 0.0, +9.81] +
        dof_pos_rel_3 +  # L3 = 0.3, 其他 = 0
        [0.0] * 14 +
        [0.0] * 14 +
        [0.0, 0.0, 0.0] +
        [0.0] * 14 +
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37]
    )
    
    test_cases["test_case_3"] = {
        "name": "左腿 L3 抬高",
        "description": "leg_l3_joint 偏离 default +0.3 rad，测试关节映射",
        "robot_state": {
            "base_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "base_angular_velocity": [0.0, 0.0, 0.0],
            "joint_positions": dof_pos_3,
            "joint_velocities": [0.0] * 14,
            "joint_torques": [0.0] * 14,
            "velocity_commands": [0.0, 0.0, 0.0],
        },
        "observation_vector": observation_3,
        "observation_breakdown": {
            "dof_pos_rel": dof_pos_rel_3,
        },
        "expected_behavior": {
            "action_pattern": "主要调整 L3 关节（索引 2），应该输出负值",
            "key_joint_index": 2,
            "key_joint_name": "leg_l3_joint",
            "other_joints": "其他关节动作应该很小",
        },
    }
    
    # ============================================================
    # Test Case 4: 右腿 R3 抬高 0.3 rad
    # ============================================================
    print("📋 生成 Test Case 4: 右腿 R3 抬高")
    
    dof_pos_4 = default_dof_pos.copy()
    dof_pos_4[7] += 0.3  # leg_r3_joint 索引 7
    
    dof_pos_rel_4 = [dof_pos_4[i] - default_dof_pos[i] for i in range(14)]
    
    observation_4 = (
        [0.0, 0.0, 0.0] +
        [0.0, 0.0, +9.81] +
        dof_pos_rel_4 +  # R3 = 0.3, 其他 = 0
        [0.0] * 14 +
        [0.0] * 14 +
        [0.0, 0.0, 0.0] +
        [0.0] * 14 +
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37]
    )
    
    test_cases["test_case_4"] = {
        "name": "右腿 R3 抬高",
        "description": "leg_r3_joint 偏离 default +0.3 rad，验证左右腿不对称",
        "robot_state": {
            "base_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "base_angular_velocity": [0.0, 0.0, 0.0],
            "joint_positions": dof_pos_4,
            "joint_velocities": [0.0] * 14,
            "joint_torques": [0.0] * 14,
            "velocity_commands": [0.0, 0.0, 0.0],
        },
        "observation_vector": observation_4,
        "observation_breakdown": {
            "dof_pos_rel": dof_pos_rel_4,
        },
        "expected_behavior": {
            "action_pattern": "主要调整 R3 关节（索引 7），应该输出负值",
            "key_joint_index": 7,
            "key_joint_name": "leg_r3_joint",
            "other_joints": "其他关节动作应该很小",
            "comparison": "与 Test Case 3 对比，验证左右腿独立控制",
        },
    }
    
    # ============================================================
    # 保存测试用例
    # ============================================================
    output_file = "standard_test_cases.json"
    with open(output_file, 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ 标准测试用例已保存到: {output_file}")
    print("=" * 80)
    print("\n📤 使用方法:")
    print("1. 部署方加载模型: model_7500.pt")
    print("2. 读取测试用例: standard_test_cases.json")
    print("3. 对每个测试用例:")
    print("   - 构建观测向量（observation_vector）")
    print("   - 模型推理得到 actions")
    print("   - 对比 expected_behavior")
    print("4. 提交验证报告")
    
    return test_cases


if __name__ == "__main__":
    test_cases = generate_standard_test_cases()
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("📊 测试用例摘要")
    print("=" * 80)
    for key, case in test_cases.items():
        print(f"\n{key}:")
        print(f"  名称: {case['name']}")
        print(f"  描述: {case['description']}")
        print(f"  观测维度: {len(case['observation_vector'])}")
        if 'key_joint_index' in case['expected_behavior']:
            print(f"  关键关节: {case['expected_behavior']['key_joint_name']} (索引 {case['expected_behavior']['key_joint_index']})")
    
    print("\n" + "=" * 80)
    print("⚠️  关键提醒:")
    print("=" * 80)
    print("1. gravity_vec 必须是 [0, 0, +9.81]，不是 [0, 0, -1] 或 [0, 0, -9.81]")
    print("2. Test Case 1 的模型输出不应该全是极端值 ±2")
    print("3. 关节顺序必须与训练时一致（字母序）")
    print("4. 所有观测值的单位和缩放必须匹配训练时的定义")
    print("=" * 80)
