# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action functions for OceanBDX locomotion environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class JointPositionAction(ActionTerm):
    """Joint position action term.

    This action term applies joint position commands to the specified joints of an articulation.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        # call parent
        super().__init__(cfg, env)

        # resolve the joint names
        self._asset: Articulation = self._env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=self.cfg.preserve_order)
        
        # Log information about the action
        print(f"[INFO] Joint position action term: {self._joint_names}")

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self._env.num_envs, len(self._joint_ids), device=self._env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # parse offset - if not provided, use default joint positions
        if cfg.offset is None:
            if cfg.use_default_offset:
                self._offset = self._asset.data.default_joint_pos[:, self._joint_ids]
            else:
                self._offset = torch.zeros_like(self._raw_actions)
        else:
            self._offset = torch.full_like(self._raw_actions, cfg.offset)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store raw actions
        self._raw_actions[:] = actions
        # process actions
        self._processed_actions = self._raw_actions * self.cfg.scale + self._offset

    def apply_actions(self):
        # set joint position targets
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # reset action tensors
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class JointPositionActionCfg(ActionTermCfg):
    """Configuration for joint position action."""

    class_type: type = JointPositionAction

    asset_name: str = "robot"
    """Name of the asset in the scene."""
    joint_names: list[str] | str = []
    """List of joint names or regex patterns to match joint names."""
    scale: float = 1.0
    """Scaling factor for the action."""
    offset: float | None = None
    """Offset to add to the action. Defaults to None, in which case the offset is the asset's default joint positions."""
    preserve_order: bool = False
    """Whether to preserve the order of joint names in the action space."""
    use_default_offset: bool = True
    """Whether to use default joint positions as offset."""