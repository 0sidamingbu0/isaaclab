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
    """Command specifications for the MDP."""

    # 【速度命令生成配置】- 控制机器人接收到的运动指令范围
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(15.0, 20.0),  # 【可调】命令重新采样时间间隔(秒)，越长命令越稳定
        rel_standing_envs=0.1,  # 【可调】静止环境比例(0-1)，0.1=10%环境要求机器人站立不动
        rel_heading_envs=0.0,   # 【可调】朝向命令环境比例(0-1)，0=不使用朝向命令
        heading_command=False,  # 【可调】是否启用朝向控制，True=启用转向到特定方向
        heading_control_stiffness=1.0,  # 【可调】朝向控制刚度，影响转向速度
        debug_vis=True,  # 【可调】是否显示命令箭头可视化
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0, 0.5),  # 【重要可调】前进速度范围(m/s)，(0,1.0)=只前进不后退
            lin_vel_y=(-0.5, 0.5),  # 【重要可调】侧向速度范围(m/s)，(-1,1)=左右移动各1m/s
            ang_vel_z=(-0.5, 0.5),  # 【重要可调】旋转速度范围(rad/s)，控制原地转圈速度
            heading=(-math.pi, math.pi),  # 【可调】目标朝向范围，全方向
        ),
    )


@configclass
class ActionsCfg:
    """【动作空间配置】- 控制机器人关节运动范围"""

    # 【关节位置控制配置】
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["leg_.*", "neck_.*"],  # 控制的关节：腿部和颈部关节
        scale=0.9,  # 【重要可调】动作幅度缩放(0-1)，0.25=25%关节范围，减小幅度使动作更平稳
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
        # ✅ 重力投影 - 使用IMU加速度数据，匹配OceanBDX坐标系
        # OceanBDX坐标系：直立[0,0,+9.81], 前倾[+9.81,0,+xxx], 左倾[0,+9.81,+xxx]
        # 部署时实现：直接使用IMU加速度计输出，无需额外计算
        projected_gravity = ObsTerm(
            func=mdp.imu_acceleration,  # 使用IMU加速度，包含重力投影+运动加速度
            params={"sensor_cfg": SceneEntityCfg("imu_sensor")},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        # ✅ IMU姿态四元数 - 与实际硬件匹配
        base_quat_w = ObsTerm(
            func=mdp.imu_quaternion,  # IMU输出的姿态四元数
            params={"sensor_cfg": SceneEntityCfg("imu_sensor")}
        )

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
    """Reward terms for the MDP."""

    # 【任务目标奖励】- 鼓励机器人完成运动任务
    # 【重要可调】跟踪线性速度命令奖励，鼓励按指令前进/侧移
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0,  # 权重4.0，越高越重视速度跟踪
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}  # std控制容忍度，越小要求越精确
    )
    # 【重要可调】跟踪角速度命令奖励，鼓励按指令转向
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0,  # 权重2.0，控制转向重要性
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # 【稳定性奖励】- 保持机器人稳定直立
    # 【重要可调】身体高度跟踪奖励，鼓励保持目标站立高度
    track_base_height = RewTerm(
        func=mdp.track_base_height_exp, weight=1.0,  # 正权重=2.0，接近目标高度越近奖励越高
        params={"target_height": 0.33, "std": 0.1}  # 目标高度0.4m，标准差0.05m控制容忍度
    )
    # 【重要可调】姿态平衡奖励，鼓励保持直立（合并了原有的三个重复项）
    # 统一使用upright_posture，它既包含了姿态稳定性又提供正向激励
    upright_posture = RewTerm(func=mdp.upright_posture_reward, weight=3.0)  # 权重1.0，平衡后的直立奖励
    # 【关键可调】膝关节位置奖励，控制腿部弯曲度
    # knee_position = RewTerm(
    #     func=mdp.knee_position_reward,
    #     weight=0.8,  # 权重1.0，对膝关节位置要求很高
    #     params={"target_angle": 0}  # 目标角度0弧度(约0度)，保持直立
    # )
    # 【新增】髋关节外展角度奖励，防止外八字步态
    # hip_abduction_position = RewTerm(
    #     func=mdp.hip_abduction_reward,
    #     weight=0.2,  # 权重1.5，强调腿部平行重要性
    #     params={"target_angle": 0.0}  # 目标角度0度，保持腿部平行
    # )
    # 【可调】线性加速度平滑性惩罚
    lin_acc_penalty = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)  # 🔧 大幅增加到-2.0，强烈惩罚Z轴跳跃

    # 【可调】角加速度平滑性惩罚
    ang_acc_penalty = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # 权重-0.05，防止翻滚

    # 【正则化奖励】- 控制能耗和动作平滑性
    # 【可调】关节力矩惩罚，降低能耗
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5e-6)  # 权重-5e-6，很小的力矩惩罚

    # 【可调】关节加速度惩罚，保证动作平滑
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5e-8)  # 权重-5e-8，防止关节剧烈变化

    # 【可调】动作变化率惩罚，避免动作突变
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.003)  # 权重-0.003，动作连续性

    # 【接触力奖励】- 控制足部与地面接触
    # 🔧 左脚接触力奖励
    feet_contact_forces_LF = RewTerm(
        func=mdp.contact_forces,
        weight=0.5,  # 降低权重，避免过度鼓励单脚支撑
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_LF"),
            "threshold": 0.5,
        },
    )
    # 🔧 右脚接触力奖励（新增）
    feet_contact_forces_RF = RewTerm(
        func=mdp.contact_forces,
        weight=0.5,  # 相同权重，鼓励双脚平衡支撑
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_RF"),
            "threshold": 0.5,
        },
    )
    
    # 【步态奖励】- 鼓励类人行走步态
    # 【重要可调】交替步态模式奖励，强烈鼓励左右脚轮流着地
    gait_pattern = RewTerm(
        func=mdp.gait_reward,
        weight=5.0,  # 🔧 提高到5.0，强化交替步态控制
    )
    
    # 【禁用】足部抬起奖励 - 实现有问题，依赖gait_pattern和air_time来控制步态
    # foot_movement = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=0.0,  # 禁用
    #     params={"min_clearance": 0.01}
    # )
    
    # 【关键可调】腾空时间奖励，轻微鼓励抬脚离地
    # air_time = RewTerm(
    #     func=mdp.air_time_reward,
    #     weight=0.3,  # 🔧 降低到0.3，避免过度鼓励腾空导致单脚支撑
    #     params={"command_name": "base_velocity", "threshold": 0.1}  # 提高速度阈值，只在实际运动时触发
    # )
    
    # 【重要可调】步频奖励，鼓励慢而稳的行走节奏
    step_frequency = RewTerm(
        func=mdp.step_frequency_reward,
        weight=3.0,  # 🔧 提高到2.0，强化步频控制
        params={"command_name": "base_velocity", "target_freq": 2.0}  # 🔧 降低到1.0Hz（每秒1步），鼓励慢步态
    )


@configclass
class TerminationsCfg:
    """【终止条件配置】- 决定何时结束训练episode"""

    # 【自动终止】时间到达episode_length_s后自动结束
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 【可调】机器人摔倒高度检测
    base_height = DoneTerm(
        func=mdp.base_height,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("robot")},  # 身体低于0.3m时终止
    )
    # 【重要可调】机器人倾倒角度检测
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi / 4, "asset_cfg": SceneEntityCfg("robot")},  # 倾斜超过45度终止
    )
        # 【新增】膝盖触地终止 - 防止机器人跪倒或摔倒时膝盖撞击地面
    knee_contact = DoneTerm(
        func=mdp.knee_ground_contact,
        params={
            "threshold": 5.0,  # 膝盖接触力阈值(N)
        },
    )


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
