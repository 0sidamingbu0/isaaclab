# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment for OceanBDX locomotion task."""

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .config.oceanbdx_locomotion_main import OceanBDXLocomotionEnvCfg
from .mdp.adaptive_phase_manager import AdaptivePhaseManager


class OceanBDXLocomotionEnv(ManagerBasedRLEnv):
    """Environment for OceanBDX locomotion task.

    This environment implements a bipedal locomotion task for the OceanBDX robot.
    The robot needs to track velocity commands (x, y, yaw) while maintaining balance.
    
    自适应步态系统：
    - 使用AdaptivePhaseManager根据速度动态调整步态参数
    - 相位、步长、抬脚高度自适应于当前速度指令
    """

    cfg: OceanBDXLocomotionEnvCfg

    def __init__(self, cfg: OceanBDXLocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment. Defaults to None, which
                is similar to "human".
        """
        # ============================================================
        # 🔑 在父类初始化之前创建phase_manager占位符
        # 因为ObservationManager在父类__init__中会立即调用observation函数
        # ============================================================
        # 先创建一个临时的phase_manager（使用默认参数）
        # 注意：此时self.num_envs, self.device等还不存在，需要从cfg获取
        import torch as torch_temp
        from .mdp.adaptive_phase_manager import VideoGaitReference
        device_temp = kwargs.get('device', 'cuda:0')
        video_config = VideoGaitReference()  # 使用默认视频参考配置
        
        self.phase_manager = AdaptivePhaseManager(
            num_envs=cfg.scene.num_envs,
            device=device_temp,
            video_config=video_config
        )
        
        # 调用父类初始化
        super().__init__(cfg, render_mode, **kwargs)
        
        print(f"✅ AdaptivePhaseManager initialized: {self.num_envs} envs, dt={self.step_dt:.4f}s")
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-process actions before stepping through the physics.
        
        在物理步之前更新相位管理器，确保奖励函数和观测能获取到最新的相位信息。
        """
        # 获取机器人当前速度（世界坐标系下的xy速度）
        robot = self.scene["robot"]
        robot_velocity_xy = robot.data.root_lin_vel_w[:, :2]
        
        # 🔑 更新相位管理器（每个控制步）
        self.phase_manager.update(robot_velocity_xy)
        
        # 调用父类的预处理
        super()._pre_physics_step(actions)
