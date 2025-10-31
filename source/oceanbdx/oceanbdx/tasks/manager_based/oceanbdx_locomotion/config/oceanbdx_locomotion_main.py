# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg  # 移除了RayCasterCfg和patterns（高度扫描相关）
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import OceanBDX robot configuration
from oceanbdx.assets.oceanusd import OCEAN_ROBOT_CFG

# Import MDP functions
from . import mdp

##
# Scene definition
##


@configclass
class OceanBDXLocomotionSceneCfg(InteractiveSceneCfg):
    """Configuration for the locomotion scene with OceanBDX robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAAC_NUCLEUS_DIR}/Materials/TilesAiMaterial/TilesAiMaterial.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    
    # robot
    robot: ArticulationCfg = OCEAN_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # IMU sensor attached to the robot's imu_link
    imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/imu_link",
        update_period=0.005,  # 200 Hz
        debug_vis=True,  # 启用IMU可视化
    )

    # 【足部接触传感器】- 只在脚部链接(leg_l5/leg_r5)上检测接触力
    contact_forces_LF = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_l5_link",  # 【修正】只监测左脚接触，不是所有左腿链接
        update_period=0.005,  # 200 Hz
        history_length=3,  # 【可调】保存3帧历史数据，用于步态分析
        debug_vis=True,  # 【可调】显示接触力可视化(绿色球)
    )

    contact_forces_RF = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_r5_link",  # 【修正】只监测右脚接触，不是所有右腿链接
        update_period=0.005,  # 200 Hz
        history_length=3,  # 【可调】保存3帧历史数据，用于步态分析
        debug_vis=True,  # 【可调】显示接触力可视化(绿色球)
    )

    # 【膝盖接触传感器】- 检测膝盖是否触地，用于终止条件
    contact_forces_knee_L = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_l3_link",  # 左膝盖链接
        update_period=0.005,  # 200 Hz
        history_length=1,  # 只需要当前帧数据
        debug_vis=True,  # 显示膝盖接触可视化
        force_threshold=2.0,  # 膝盖接触力阈值
    )

    contact_forces_knee_R = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_r3_link",  # 右膝盖链接
        update_period=0.005,  # 200 Hz
        history_length=1,  # 只需要当前帧数据
        debug_vis=True,  # 显示膝盖接触可视化
        force_threshold=2.0,  # 膝盖接触力阈值
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """
    速度命令配置 - 支持三阶段课程学习
    
    三阶段速度范围演化：
    - Stage1 (0-30%): 0-0.35 m/s - 学习稳定站立和基础步态
    - Stage2 (30-70%): 0-0.5 m/s - 形成完整步态模式
    - Stage3 (70-100%): 0-0.74 m/s - 优化步态质量和速度
    
    注：训练时需要动态调整lin_vel_x范围，参考training_curriculum.py
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(15.0, 20.0),  # 命令重新采样间隔，Stage1可用较长值保持稳定
        rel_standing_envs=0.2,  # 🎯 Stage1提升到20%静止环境，学习稳定站立
        rel_heading_envs=0.0,   # 关闭朝向命令，专注直线行走
        heading_command=False,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # 🎯 Stage1初始范围：只训练低速前进 (0-0.35 m/s)
            # 训练脚本需要动态调整此范围：
            # - Stage2: (0.0, 0.5)
            # - Stage3: (0.0, 0.74)
            lin_vel_x=(0.0, 0.35),   # � 关键：从Disney BDX参考速度0.35m/s开始
            lin_vel_y=(-0.1, 0.1),   # 🎯 Stage1限制侧移，Stage3再放宽到(-0.3, 0.3)
            ang_vel_z=(-0.3, 0.3),   # 🎯 Stage1限制旋转，Stage3再放宽到(-0.5, 0.5)
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """【动作空间配置】- 控制机器人关节运动范围"""

    # 【关节位置控制配置】
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["leg_.*", "neck_.*"],  # 控制的关节：腿部和颈部关节
        scale=0.5,  # 【重要可调】动作幅度缩放(0-1)，0.25=25%关节范围，减小幅度使动作更平稳
        use_default_offset=True,  # 【可调】是否使用默认姿态作为偏移基准
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 【IMU传感器观测数据】- 完全基于真实IMU输出，确保训练部署一致性
        
        # ✅ IMU角速度（陀螺仪）- 与实际硬件匹配
        base_ang_vel = ObsTerm(
            func=mdp.imu_angular_velocity,  # IMU陀螺仪3轴角速度
            params={"sensor_cfg": SceneEntityCfg("imu_sensor")},
            noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        # ✅ 重力投影 - 纯重力方向，不含运动加速度冲击
        # 训练：使用robot姿态计算重力投影（准确且稳定）
        # 部署：使用IMU姿态+低通滤波计算重力投影
        # OceanBDX坐标系：直立[0,0,+9.81], 前倾[+9.81,0,+xxx], 左倾[0,+9.81,+xxx]
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,  # 🔧 改用纯重力投影，避免运动冲击噪声
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05)  # 🔧 减小噪声，因为重力更稳定
        )
        
        # ❌ 移除四元数观测 - 部署时不需要，重力投影已经包含姿态信息
        # base_quat_w = ObsTerm(
        #     func=mdp.imu_quaternion,
        #     params={"sensor_cfg": SceneEntityCfg("imu_sensor")}
        # )

        # Joint states
        joint_pos_rel = ObsTerm(
            func=isaaclab_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_rel = ObsTerm(
            func=isaaclab_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        # 【新增】关节转矩反馈 - 提供实际电机输出转矩信息，增强接触感知和力控能力
        joint_torques = ObsTerm(
            func=mdp.joint_torques_obs,  # 需要创建这个观测函数
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )

        # Commands
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # Actions
        last_actions = ObsTerm(func=mdp.last_action)

        # 🆕 【自适应步态相位观测】- 根据速度动态调整期望步态参数
        # 9维观测：6个sin/cos多频率编码 + phase_rate + desired_stride + desired_clearance
        # 与真机部署完全一致，提供显式的时间/相位信息
        adaptive_phase = ObsTerm(
            func=mdp.adaptive_gait_phase_observation
            # 无需参数，自动从env.phase_manager获取
        )

        # Height scan for terrain awareness - 注释：部署时如果没有高度扫描传感器则移除此观测
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # Reset robot joints with some randomization
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Reset robot base with randomization
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # Add random forces to test robustness
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
    )


@configclass
class RewardsCfg:
    """
    🎯 Disney BDX自适应奖励系统 - 17个核心奖励函数
    配合4阶段课程学习 (training_curriculum.py)
    所有奖励函数来自 adaptive_rewards.py
    参考: Disney BDX训练指南, legged_gym, walk-these-ways
    """

    # ============================================================================
    # 【任务奖励】(2个) - 让机器人跟踪速度指令
    # ============================================================================
    
    velocity_tracking = RewTerm(
        func=mdp.reward_velocity_tracking_exp,
        weight=2.0,  # 主导奖励,课程会动态调整
        params={"command_name": "base_velocity", "std": 0.5}
    )
    
    angular_velocity_tracking = RewTerm(
        func=mdp.reward_angular_velocity_tracking,
        weight=1.0,  # 次要任务奖励
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # ============================================================================
    # 【稳定性约束】(2个) - 保持直立不摔倒
    # ============================================================================
    
    orientation_penalty = RewTerm(
        func=mdp.reward_orientation_penalty,
        weight=-1.0  # Stage 0会提升到-100
    )
    
    base_height_tracking = RewTerm(
        func=mdp.reward_base_height_tracking,
        weight=0.8,  # Stage 0会提升到15.0
        params={"target_height": 0.35, "std": 0.1}  # 降低目标高度减少膝盖过直
    )

    # ============================================================================
    # 【步态质量】(4个) - 🔑 核心防作弊机制
    # ============================================================================
    
    # ⭐ 自适应交替接触 - 防止振动作弊 (改进版:支持过渡相)
    feet_alternating_contact = RewTerm(
        func=mdp.reward_feet_alternating_contact,
        weight=0.0,  # Stage 1: 0.3, Stage 2: 1.0, Stage 3: 0.8
        params={"threshold": 1.0}
    )
    
    # 🆕 重心转移奖励 - 鼓励单腿支撑和抬腿迈步
    weight_transfer = RewTerm(
        func=mdp.reward_weight_transfer,
        weight=0.0,  # Stage 1: 0.5, Stage 2: 1.0, Stage 3: 1.0
        params={"threshold": 1.0}
    )
    
    # 自适应步长跟踪
    stride_length_tracking = RewTerm(
        func=mdp.reward_stride_length_tracking,
        weight=0.0  # Stage 1: 0.2, Stage 2: 0.8, Stage 3: 0.6
    )
    
    # 自适应抬脚高度
    foot_clearance = RewTerm(
        func=mdp.reward_foot_clearance,
        weight=0.0,  # Stage 1: 0.2, Stage 2: 0.6, Stage 3: 0.4
        params={"threshold": 1.0}
    )

    # ============================================================================
    # 【安全约束】(3个) - 防止危险动作
    # ============================================================================
    
    # 惩罚膝盖、躯干接触地面
    undesired_contacts = RewTerm(
        func=mdp.reward_undesired_contacts,
        weight=-5.0,  # Stage 0: -5.0, Stage 3: -2.0
        params={"threshold": 1.0}
    )
    
    # 关节限位惩罚
    joint_limits_penalty = RewTerm(
        func=mdp.reward_joint_limits_penalty,
        weight=-0.1,  # 避免过度主导总奖励
        params={"soft_limit_ratio": 0.9}
    )
    
    # 支撑腿滑动惩罚
    feet_slip_penalty = RewTerm(
        func=mdp.reward_feet_slip_penalty,
        weight=-1.0,  # Stage 1: -1.0, Stage 2: -1.5, Stage 3: -2.0
        params={"threshold": 1.0}
    )

    # ============================================================================
    # 【能耗与平滑性】(4个) - 🔑 防高频振动作弊
    # ============================================================================
    
    # ⭐ 动作平滑性 - 防止高频振动的关键
    action_smoothness = RewTerm(
        func=mdp.reward_action_smoothness,
        weight=-0.001  # Stage 1: -0.001, Stage 2: -0.01, Stage 3: -0.05
    )
    
    # 关节力矩惩罚（能耗）
    joint_torque_penalty = RewTerm(
        func=mdp.reward_joint_torque_penalty,
        weight=-1e-6  # Stage 1: -1e-6, Stage 2: -5e-5, Stage 3: -1e-4
    )
    
    # ⭐ 关节加速度惩罚 - 防止剧烈运动
    joint_acceleration = RewTerm(
        func=mdp.reward_joint_acceleration_penalty,
        weight=-1e-8  # Stage 1: -1e-8, Stage 2: -2.5e-7, Stage 3: -1e-6
    )
    
    # 关节速度惩罚
    joint_velocity_penalty = RewTerm(
        func=mdp.reward_joint_velocity_penalty,
        weight=-1e-5  # Stage 1: -1e-5, Stage 2: -5e-4, Stage 3: -1e-3
    )

    # ============================================================================
    # 【终止惩罚】(1个) - 严厉惩罚摔倒
    # ============================================================================
    
    termination_penalty = RewTerm(
        func=mdp.reward_termination_penalty,
        weight=1.0  # 函数内部已乘以-100，总权重-100 to -200
    )


@configclass
class TerminationsCfg:
    """【终止条件配置】- 决定何时结束训练episode"""

    # 【自动终止】时间到达episode_length_s后自动结束
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 【可调】机器人摔倒高度检测
    base_height = DoneTerm(
        func=mdp.base_height,
        params={"minimum_height": 0.15, "asset_cfg": SceneEntityCfg("robot")},  # 🔧 Stage 0: 放宽到0.15m，避免过早终止探索
    )
    # 【重要可调】机器人倾倒角度检测
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi / 3, "asset_cfg": SceneEntityCfg("robot")},  # 🔧 修复：从30度放宽到60度，给机器人更多学习空间
    )
    # 【新增】膝盖触地终止 - 防止机器人跪倒或摔倒时膝盖撞击地面
    # knee_contact = DoneTerm(
    #     func=mdp.knee_ground_contact,
    #     params={
    #         "threshold": 5.0,  # 膝盖接触力阈值(N)
    #     },
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    # 暂时禁用课程学习以简化配置
    pass


##
# Environment configuration
##


@configclass
class OceanBDXLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the OceanBDX locomotion environment."""

    # 【场景设置】
    scene: OceanBDXLocomotionSceneCfg = OceanBDXLocomotionSceneCfg(
        num_envs=4096,  # 【重要可调】并行环境数量，4096个机器人同时训练，越多越快但显存要求高
        env_spacing=2.5  # 【可调】环境间距2.5米，防止机器人碰撞
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """【环境初始化后配置】- 关键训练参数"""
        # 【重要可调】基础设置
        self.decimation = 4  # 控制频率分频，4=50Hz控制频率(200Hz/4)
        self.episode_length_s = 35.0  # 【关键可调】单次训练时长35秒，影响学习复杂行为的时间
        # 【可调】仿真设置
        self.sim.dt = 0.005  # 仿真步长=200Hz，越小越精确但越慢
        self.sim.physics_material = self.scene.terrain.physics_material

        # ============================================================================
        # 🔑 自适应步态系统初始化标记
        # ============================================================================
        # 注：AdaptivePhaseManager需要在环境构造时手动创建并附加到env实例
        # 示例代码（需添加到环境类的__init__方法）：
        #
        #   from oceanbdx.tasks.manager_based.oceanbdx_locomotion.mdp import AdaptivePhaseManager
        #
        #   self.phase_manager = AdaptivePhaseManager(
        #       num_envs=self.num_envs,
        #       dt=self.step_dt,  # 控制时间步 = decimation * sim.dt
        #       device=self.device
        #   )
        #
        # 每个仿真步需要更新：
        #   self.phase_manager.update(self.robot.data.root_lin_vel_w[:, :2])
        #
        # 观测和奖励函数会自动从env.phase_manager获取参数
        # ============================================================================
        
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics dt)
        # if self.scene.height_scanner is not None:  # 注释：高度扫描传感器已移除
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces_LF is not None:
            self.scene.contact_forces_LF.update_period = self.sim.dt
        if self.scene.contact_forces_RF is not None:
            self.scene.contact_forces_RF.update_period = self.sim.dt
        if self.scene.imu_sensor is not None:
            self.scene.imu_sensor.update_period = self.sim.dt
        # 【新增】膝盖接触传感器更新配置
        if self.scene.contact_forces_knee_L is not None:
            self.scene.contact_forces_knee_L.update_period = self.sim.dt
        if self.scene.contact_forces_knee_R is not None:
            self.scene.contact_forces_knee_R.update_period = self.sim.dt


@configclass
class OceanBDXLocomotionEnvCfg_PLAY(OceanBDXLocomotionEnvCfg):
    """Configuration for the OceanBDX locomotion environment for play mode."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
