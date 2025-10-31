# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ocean robot configuration with IMU support."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

OCEAN_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ocean/oceanbdx/oceanbdx/source/oceanbdx/oceanbdx/assets/oceanusd/ocean.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,  # 大幅降低去穿透速度
            max_linear_velocity=5.0,  # 大幅提高线性速度限制
            max_angular_velocity=2,  # 大幅提高角速度限制
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=False,  # 确保根链接不被固定
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),  # 调整到合理的初始高度
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # 🎯 Disney BDX标准站立姿态 (从视频分析提取)
            # Right leg joints
            "leg_r1_joint": -0.13,  # 大腿yaw = -7.5° (外八字)
            "leg_r2_joint": -0.07,  # 大腿roll = -4° (髋关节侧向)
            "leg_r3_joint": -0.20,  # 大腿pitch = -11.5° (髋关节前后，负值=后倾)
            "leg_r4_joint": -0.052, # 膝关节pitch = -3° (膝盖弯曲)
            "leg_r5_joint": 0.05,   # 脚pitch = +5.7° (脚尖上翘，重心后移)
            # Left leg joints
            "leg_l1_joint": 0.13,   # 大腿yaw = 7.5° (外八字)
            "leg_l2_joint": 0.07,   # 大腿roll = 4° (髋关节侧向)
            "leg_l3_joint": 0.20,   # 大腿pitch = 11.5° (髋关节前后，正值=后倾)
            "leg_l4_joint": 0.052,  # 膝关节pitch = 3° (膝盖弯曲)
            "leg_l5_joint": -0.05,  # 脚pitch = -5.7° (脚尖上翘，重心后移)
            # Neck joints - 保持中性
            "neck_n1_joint": 0.0,   # neck yaw
            "neck_n2_joint": 0.0,   # neck pitch
            "neck_n3_joint": 0.0,   # neck roll
            "neck_n4_joint": 0.0,   # head tilt
            # IMU joint is fixed, no initial position needed
        },
        joint_vel={
            # Left leg joints
            "leg_l1_joint": 0.0,
            "leg_l2_joint": 0.0,
            "leg_l3_joint": 0.0,
            "leg_l4_joint": 0.0,
            "leg_l5_joint": 0.0,
            # Right leg joints
            "leg_r1_joint": 0.0,
            "leg_r2_joint": 0.0,
            "leg_r3_joint": 0.0,
            "leg_r4_joint": 0.0,
            "leg_r5_joint": 0.0,
            # Neck joints
            "neck_n1_joint": 0.0,
            "neck_n2_joint": 0.0,
            "neck_n3_joint": 0.0,
            "neck_n4_joint": 0.0,
            # IMU joint is fixed, no velocity
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                "leg_l1_joint", "leg_l2_joint", "leg_l3_joint",
                "leg_l4_joint", "leg_l5_joint",
                "leg_r1_joint", "leg_r2_joint", "leg_r3_joint",
                "leg_r4_joint", "leg_r5_joint"
            ],
            effort_limit=50.0,  # 大幅增加到100Nm确保有足够力量站立
            saturation_effort=90.0,
            velocity_limit=15.0,  # 大幅提高速度限制
            stiffness=50.0,  # 🔧 适中刚性避免振荡 (30→50, 提供足够支撑力)
            damping=4.0,     # 🔧 接近临界阻尼 (2→4, Kp/Kd=12.5:1)
            friction=0.8,
        ),
        "neck": DCMotorCfg(
            joint_names_expr=[
                "neck_n1_joint", "neck_n2_joint",
                "neck_n3_joint", "neck_n4_joint"
            ],
            effort_limit=10.0,  # 增加颈部力量
            saturation_effort=8.0,
            velocity_limit=10.0,  # 提高颈部速度限制
            stiffness=15.0,  # 🔧 适中刚性稳定上身 (8→15)
            damping=1.5,    # 🔧 匹配阻尼比 (Kp/Kd=10:1)
            friction=0.3,
        ),
        # IMU joint is fixed, no actuator needed
    },
)