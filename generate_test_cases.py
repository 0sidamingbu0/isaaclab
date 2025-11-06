#!/usr/bin/env python3
"""
生成标准测试用例供部署验证使用
不依赖 Isaac Lab，可直接运行
"""

import json
import math


def generate_test_cases():
    """生成标准测试用例"""
    
    print("=" * 80)
    print("🔧 生成 OceanBDX 部署验证测试用例")
    print("=" * 80)
    
    # 训练时的关节顺序（Isaac Lab 字母序）
    joint_order_training = [
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint",
        "neck_n1_joint", "neck_n2_joint", "neck_n3_joint", "neck_n4_joint",
    ]
    
    # 默认关节位置（训练值）
    default_dof_pos = torch.tensor([
        0.13, 0.07, 0.2, 0.052, -0.05,    # Left leg
        -0.13, -0.07, -0.2, -0.052, 0.05,  # Right leg
        0.0, 0.0, 0.0, 0.0                 # Neck
    ])
    
    test_cases = {}
    
    # ============================================================
    # Test Case 1: 默认站立姿态
    # ============================================================
    print("\n📋 Test Case 1: 默认站立姿态")
    
    # 机器人状态
    base_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 无旋转
    base_ang_vel = torch.zeros(3)
    dof_pos = default_dof_pos.clone()
    dof_vel = torch.zeros(14)
    
    # 计算观测向量（手动构建，不依赖环境）
    # 1. ang_vel_body (3)
    obs_ang_vel = base_ang_vel.clone()
    
    # 2. gravity_vec (3) - 直立时 Z 向上
    obs_gravity = torch.tensor([0.0, 0.0, 9.81])
    
    # 3. dof_pos_rel (14)
    obs_dof_pos = dof_pos - default_dof_pos
    
    # 4. dof_vel (14)
    obs_dof_vel = dof_vel * 0.05  # scaled
    
    # 5. joint_torques (14) - 假设为零
    obs_torques = torch.zeros(14)
    
    # 6. commands (3)
    obs_commands = torch.zeros(3)
    
    # 7. last_actions (14)
    obs_actions = torch.zeros(14)
    
    # 8. adaptive_phase (9) - 初始状态
    obs_phase = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37])
    
    # 拼接观测向量
    observation_1 = torch.cat([
        obs_ang_vel,      # 3
        obs_gravity,      # 3
        obs_dof_pos,      # 14
        obs_dof_vel,      # 14
        obs_torques,      # 14
        obs_commands,     # 3
        obs_actions,      # 14
        obs_phase         # 9
    ])  # Total: 74
    
    test_cases["test_case_1"] = {
        "name": "默认站立姿态",
        "description": "机器人在 default_dof_pos，直立无旋转",
        "robot_state": {
            "base_quaternion": base_quat_w.tolist(),
            "base_angular_velocity": base_ang_vel.tolist(),
            "joint_positions": dof_pos.tolist(),
            "joint_velocities": dof_vel.tolist(),
            "joint_torques": obs_torques.tolist(),
            "velocity_commands": obs_commands.tolist(),
        },
        "observation_vector": observation_1.tolist(),
        "observation_breakdown": {
            "ang_vel_body": obs_ang_vel.tolist(),
            "gravity_vec": obs_gravity.tolist(),
            "dof_pos_rel": obs_dof_pos.tolist(),
            "dof_vel_scaled": obs_dof_vel.tolist(),
            "joint_torques": obs_torques.tolist(),
            "commands": obs_commands.tolist(),
            "last_actions": obs_actions.tolist(),
            "adaptive_phase": obs_phase.tolist(),
        },
        "expected_model_output": "接近零的动作（因为已在目标姿态）",
        "joint_order_training": joint_order_training,
    }
    
    # ============================================================
    # Test Case 2: 前倾 15 度
    # ============================================================
    print("📋 Test Case 2: 前倾 15 度")
    
    import math
    angle = math.radians(15)  # 绕 Y 轴旋转（pitch）
    base_quat_w = torch.tensor([
        math.cos(angle/2), 
        0.0, 
        math.sin(angle/2), 
        0.0
    ])
    
    # 计算旋转后的重力向量
    # 前倾时，重力会有 X 分量（向前）
    gravity_world = torch.tensor([0.0, 0.0, 9.81])
    # 使用四元数旋转公式（简化）
    # 前倾 15 度: gx ≈ -9.81*sin(15°) ≈ -2.54
    #            gz ≈ 9.81*cos(15°) ≈ 9.47
    obs_gravity = torch.tensor([-2.54, 0.0, 9.47])
    
    observation_2 = torch.cat([
        torch.zeros(3),       # ang_vel
        obs_gravity,          # gravity_vec (前倾)
        torch.zeros(14),      # dof_pos_rel
        torch.zeros(14),      # dof_vel
        torch.zeros(14),      # torques
        torch.zeros(3),       # commands
        torch.zeros(14),      # actions
        obs_phase             # phase
    ])
    
    test_cases["test_case_2"] = {
        "name": "前倾 15 度",
        "description": "base 绕 Y 轴前倾 15 度",
        "robot_state": {
            "base_quaternion": base_quat_w.tolist(),
            "base_angular_velocity": [0.0, 0.0, 0.0],
            "joint_positions": default_dof_pos.tolist(),
            "joint_velocities": [0.0] * 14,
            "joint_torques": [0.0] * 14,
            "velocity_commands": [0.0, 0.0, 0.0],
        },
        "observation_vector": observation_2.tolist(),
        "observation_breakdown": {
            "gravity_vec": obs_gravity.tolist(),
        },
        "expected_model_output": "模型应输出向后倾的动作来纠正前倾",
    }
    
    # ============================================================
    # Test Case 3: 左腿 L3 抬高
    # ============================================================
    print("📋 Test Case 3: 左腿 L3 抬高")
    
    dof_pos_3 = default_dof_pos.clone()
    dof_pos_3[2] += 0.3  # leg_l3_joint 索引 2，增加 0.3 弧度
    
    obs_dof_pos_3 = dof_pos_3 - default_dof_pos
    
    observation_3 = torch.cat([
        torch.zeros(3),       # ang_vel
        torch.tensor([0.0, 0.0, 9.81]),  # gravity_vec (直立)
        obs_dof_pos_3,        # dof_pos_rel (L3 有偏差)
        torch.zeros(14),      # dof_vel
        torch.zeros(14),      # torques
        torch.zeros(3),       # commands
        torch.zeros(14),      # actions
        obs_phase             # phase
    ])
    
    test_cases["test_case_3"] = {
        "name": "左腿 L3 抬高",
        "description": "leg_l3_joint 偏离 default +0.3 rad",
        "robot_state": {
            "base_quaternion": [1.0, 0.0, 0.0, 0.0],
            "base_angular_velocity": [0.0, 0.0, 0.0],
            "joint_positions": dof_pos_3.tolist(),
            "joint_velocities": [0.0] * 14,
            "joint_torques": [0.0] * 14,
            "velocity_commands": [0.0, 0.0, 0.0],
        },
        "observation_vector": observation_3.tolist(),
        "observation_breakdown": {
            "dof_pos_rel": obs_dof_pos_3.tolist(),
        },
        "expected_model_output": "主要调整 L3 关节（索引 2），动作应为负值",
        "key_joint_index": 2,
        "key_joint_name": "leg_l3_joint",
    }
    
    # ============================================================
    # Test Case 4: 右腿 R3 抬高
    # ============================================================
    print("📋 Test Case 4: 右腿 R3 抬高")
    
    dof_pos_4 = default_dof_pos.clone()
    dof_pos_4[7] += 0.3  # leg_r3_joint 索引 7，增加 0.3 弧度
    
    obs_dof_pos_4 = dof_pos_4 - default_dof_pos
    
    observation_4 = torch.cat([
        torch.zeros(3),       # ang_vel
        torch.tensor([0.0, 0.0, 9.81]),  # gravity_vec (直立)
        obs_dof_pos_4,        # dof_pos_rel (R3 有偏差)
        torch.zeros(14),      # dof_vel
        torch.zeros(14),      # torques
        torch.zeros(3),       # commands
        torch.zeros(14),      # actions
        obs_phase             # phase
    ])
    
    test_cases["test_case_4"] = {
        "name": "右腿 R3 抬高",
        "description": "leg_r3_joint 偏离 default +0.3 rad",
        "robot_state": {
            "base_quaternion": [1.0, 0.0, 0.0, 0.0],
            "base_angular_velocity": [0.0, 0.0, 0.0],
            "joint_positions": dof_pos_4.tolist(),
            "joint_velocities": [0.0] * 14,
            "joint_torques": [0.0] * 14,
            "velocity_commands": [0.0, 0.0, 0.0],
        },
        "observation_vector": observation_4.tolist(),
        "observation_breakdown": {
            "dof_pos_rel": obs_dof_pos_4.tolist(),
        },
        "expected_model_output": "主要调整 R3 关节（索引 7），动作应为负值",
        "key_joint_index": 7,
        "key_joint_name": "leg_r3_joint",
    }
    
    # ============================================================
    # 保存测试用例
    # ============================================================
    output_file = "test_cases.json"
    with open(output_file, 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ 测试用例已保存到: {output_file}")
    print("=" * 80)
    print("\n📤 请将此文件发送给部署 AI 进行验证")
    print("📋 验证协议: DEPLOYMENT_VERIFICATION_PROTOCOL.md")
    
    return test_cases

if __name__ == "__main__":
    test_cases = generate_test_cases()
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("📊 测试用例摘要")
    print("=" * 80)
    for key, case in test_cases.items():
        print(f"\n{key}:")
        print(f"  名称: {case['name']}")
        print(f"  描述: {case['description']}")
        print(f"  观测维度: {len(case['observation_vector'])}")
        if 'key_joint_index' in case:
            print(f"  关键关节: {case['key_joint_name']} (索引 {case['key_joint_index']})")
