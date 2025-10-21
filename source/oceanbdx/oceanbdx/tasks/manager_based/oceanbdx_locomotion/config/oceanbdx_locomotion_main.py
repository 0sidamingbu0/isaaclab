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
            lin_vel_x=(-1.5, 0.0),   # 🔧 只允许前进：0-1.5m/s，匹配参考步态方向
            lin_vel_y=(-0.3, 0.3),  # 🔧 减小侧移范围，专注前进训练
            ang_vel_z=(-0.5, 0.5),  # 保持转向能力
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

        # 🆕 【步态相位观测】- 提供显式的时间/相位信息，帮助策略理解步态周期
        # 多频率编码：[sin(φ), cos(φ), sin(φ/2), cos(φ/2), sin(φ/4), cos(φ/4)]
        # 提供粗到细的时间尺度信息，与参考步态奖励配合使用
        gait_phase = ObsTerm(
            func=mdp.gait_phase_observation,
            params={"gait_period": 0.75}  # 与参考步态周期一致
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
    """Reward terms for the MDP - Simplified version focusing on gait tracking."""

    # ============================================================================
    # 【核心奖励】- 基于参考步态的模仿学习
    # ============================================================================
    
    # 🌟【最重要】参考步态相位跟踪奖励 - 教机器人如何正确地动
    gait_phase_tracking = RewTerm(
        func=mdp.gait_phase_reward,
        weight=5.0,  # 🔧 大幅增加权重到5.0，让步态跟踪成为主导奖励
        params={
            "gait_period": 0.75,  # 步态周期0.75秒（与参考轨迹一致）
            "std": 3.0            # 标准差3.0弧度，容忍初始误差
        }
    )
    
    # ============================================================================
    # 【任务目标奖励】- 让机器人知道要往哪里走
    # ============================================================================

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,  # 🔧 降低到3.0，让步态优先
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,  # 🔧 降低到1.5，让步态优先
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # ============================================================================
    # 【稳定性奖励】- 保持直立，防止摔倒
    # ============================================================================

    upright_posture = RewTerm(
        func=mdp.upright_posture_reward,
        weight=2.0  # 🔧 降低到2.0，步态应该自然包含姿态稳定性
    )
    
    # ============================================================================
    # 【安全约束】- 防止危险动作
    # ============================================================================
    
    # 防止Z轴跳跃（必须保留）
    lin_acc_penalty = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)  # 🔧 降低到-1.0
    
    # 防止翻滚（必须保留）
    ang_acc_penalty = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    
    # ============================================================================
    # 【平滑性约束】- 保持动作连贯（权重降低）
    # ============================================================================
    
    # 🔧 大幅降低权重，因为参考步态本身就是平滑的
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-6)  # 从-5e-6降低到-1e-6
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-8)         # 从-5e-8降低到-1e-8
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)    # 从-0.003降低到-0.001
    
    # ============================================================================
    # 【已取消的奖励】- 由步态跟踪隐式提供
    # ============================================================================
    # ❌ track_base_height - 步态隐含了正确高度
    # ❌ feet_contact_forces - 步态隐含了正确的脚部接触
    # ❌ gait_pattern - 已经有更精确的phase tracking
    # ❌ air_time, step_frequency, step_length - 步态定义了运动节奏
    
    # 【禁用】足部抬起奖励 - 实现有问题，依赖gait_pattern和air_time来控制步态
    # foot_movement = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=0.0,  # 禁用
    #     params={"min_clearance": 0.01}
    # )
    
    # 【关键可调】腾空时间奖励，使用修复后的平衡摆动奖励
    # air_time = RewTerm(
    #     func=mdp.air_time_reward,
    #     weight=0.3,  # 🔧 降低权重，修复后的算法更安全，避免主导训练
    #     params={"command_name": "base_velocity", "threshold": 0.1, "min_air_time": 5.0, "target_air_time": 15.0}  # 指数持续时间参数
    # )
    
    # # 【重要可调】步频奖励，鼓励慢而稳的行走节奏，惩罚高频换步
    # step_frequency = RewTerm(
    #     func=mdp.step_frequency_penalty,  # 🔧 惩罚函数，内部返回负值
    #     weight=2.0,  # 🔧 增加权重到2.0，加强对高频步态的抑制
    #     params={"command_name": "base_velocity", "target_freq": 1.0, "penalty_threshold": 2.0}  # 目标1Hz，超过2Hz开始惩罚（更严格）
    # )
    
    # 【新增】步长奖励，鼓励迈大步而不是小碎步
    # step_length = RewTerm(
    #     func=mdp.step_length_reward,  # 🔧 步长奖励函数，惩罚小碎步，奖励合理步长
    #     weight=2.0,  # 🔧 进一步降低权重到1.0，让其作为辅助奖励
    #     params={
    #         "command_name": "base_velocity",
    #         "min_step_length": 0.02,     # 降低最小有效步长到 2cm，更宽松
    #         "target_step_length": 0.1,  # 降低目标步长到 10cm，更现实
    #         "max_step_length": 0.3       # 降低最大合理步长到 30cm
    #     }
    # )


@configclass
class TerminationsCfg:
    """【终止条件配置】- 决定何时结束训练episode"""

    # 【自动终止】时间到达episode_length_s后自动结束
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 【可调】机器人摔倒高度检测
    base_height = DoneTerm(
        func=mdp.base_height,
        params={"minimum_height": 0.25, "asset_cfg": SceneEntityCfg("robot")},  # 身体低于0.3m时终止
    )
    # 【重要可调】机器人倾倒角度检测
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi / 6, "asset_cfg": SceneEntityCfg("robot")},  # 倾斜超过30度终止（更严格）
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
