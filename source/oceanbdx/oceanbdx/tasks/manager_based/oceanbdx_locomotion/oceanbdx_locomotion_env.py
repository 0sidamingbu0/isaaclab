# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment for OceanBDX locomotion task."""

from isaaclab.envs import ManagerBasedRLEnv

from .config.oceanbdx_locomotion_main import OceanBDXLocomotionEnvCfg


class OceanBDXLocomotionEnv(ManagerBasedRLEnv):
    """Environment for OceanBDX locomotion task.

    This environment implements a bipedal locomotion task for the OceanBDX robot.
    The robot needs to track velocity commands (x, y, yaw) while maintaining balance.
    """

    cfg: OceanBDXLocomotionEnvCfg

    def __init__(self, cfg: OceanBDXLocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment. Defaults to None, which
                is similar to "human".
        """
        super().__init__(cfg, render_mode, **kwargs)