# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generators for velocity commands with arrow visualization."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, quat_from_angle_axis
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformVelocityCommand(CommandTerm):
    """Command generator for velocity commands with arrow visualization."""
    
    cfg: "UniformVelocityCommandCfg"

    def __init__(self, cfg: "UniformVelocityCommandCfg", env: ManagerBasedRLEnv):
        """Initialize the command term."""
        super().__init__(cfg, env)
        
        # create buffers for the command
        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_b = torch.zeros_like(self._command)
        self.heading_target = torch.zeros((self.num_envs,), device=self.device)
        self.heading_error = torch.zeros_like(self.heading_target)
        
        # create timer for resampling
        self.time_left = torch.zeros((self.num_envs,), device=self.device)
        
        # Set initial resampling time for each environment
        self.time_left[:] = sample_uniform(
            self.cfg.resampling_time_range[0],
            self.cfg.resampling_time_range[1],
            (self.num_envs,),
            device=self.device
        )

        # Visualization markers for command arrows
        self._command_visualizer = None

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for given environments."""
        # number of environments to resample
        n = len(env_ids)
        
        # sample linear velocity x
        self._command[env_ids, 0] = sample_uniform(
            self.cfg.ranges.lin_vel_x[0],
            self.cfg.ranges.lin_vel_x[1],
            (n,),
            device=self.device,
        )
        
        # sample linear velocity y
        self._command[env_ids, 1] = sample_uniform(
            self.cfg.ranges.lin_vel_y[0],
            self.cfg.ranges.lin_vel_y[1],
            (n,),
            device=self.device,
        )
        
        # sample angular velocity z
        self._command[env_ids, 2] = sample_uniform(
            self.cfg.ranges.ang_vel_z[0],
            self.cfg.ranges.ang_vel_z[1],
            (n,),
            device=self.device,
        )
        
        # update timer
        self.time_left[env_ids] = sample_uniform(
            self.cfg.resampling_time_range[0],
            self.cfg.resampling_time_range[1],
            (n,),
            device=self.device,
        )

    def _update_command(self):
        """Update the command."""
        # decrease timer
        self.time_left -= self._env.step_dt
        
        # resample commands for environments where timer has expired
        resample_envs = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(resample_envs) > 0:
            self._resample_command(resample_envs)
            
        # copy command to buffer (in base frame)
        self.command_b[:] = self._command

        # update debug visualization if enabled
        if self._command_visualizer is not None:
            self._update_command_visualization()

    def _update_command_visualization(self):
        """Update command visualization with velocity arrows."""
        if self._command_visualizer is None:
            return
            
        # Get robot positions (base of the robots)
        robot_positions = self._env.scene["robot"].data.root_pos_w[:, :3]
        
        # Create arrow positions (above each robot)
        arrow_positions = robot_positions.clone()
        arrow_positions[:, 2] += 1.0  # Raise arrows 1m above robot base
        
        # Calculate velocity command magnitudes
        vel_magnitudes = torch.norm(self._command[:, :2], dim=1)
        
        # Create arrow orientations based on velocity commands
        yaw_angles = torch.zeros(self.num_envs, device=self.device)
        
        # Only calculate yaw for environments with significant linear velocity
        # This prevents jittery rotation when velocity is near zero
        non_zero_mask = vel_magnitudes > 0.1  # Increased threshold for stability
        if non_zero_mask.any():
            yaw_angles[non_zero_mask] = torch.atan2(
                self._command[non_zero_mask, 1],  # vel_y 
                self._command[non_zero_mask, 0]   # vel_x
            )
        
        # Add some smoothing to prevent rapid angle changes
        # Store previous angles and smooth the transition
        if not hasattr(self, '_prev_yaw_angles'):
            self._prev_yaw_angles = yaw_angles.clone()
        
        # Simple smoothing: blend with previous angles
        alpha = 0.8  # Smoothing factor (0 = no change, 1 = full change)
        yaw_angles = alpha * yaw_angles + (1 - alpha) * self._prev_yaw_angles
        self._prev_yaw_angles = yaw_angles.clone()
        
        # Create quaternions for arrow orientations
        arrow_orientations = quat_from_angle_axis(
            yaw_angles,
            torch.tensor([0.0, 0.0, 1.0], device=self.device)
        )
        
        # Scale arrow positions based on command magnitude for better visualization
        # Larger commands = arrows further from robot center
        scale_factor = torch.clamp(vel_magnitudes / 2.0, 0.5, 2.0).unsqueeze(1)
        offset_direction = torch.stack([
            torch.cos(yaw_angles),
            torch.sin(yaw_angles), 
            torch.zeros_like(yaw_angles)
        ], dim=1)
        
        # Adjust arrow positions slightly in command direction
        arrow_positions += 0.3 * scale_factor * offset_direction
        
        # Use marker index 0 for all arrows (single arrow type)
        marker_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Visualize all arrows
        self._command_visualizer.visualize(arrow_positions, arrow_orientations, marker_indices=marker_indices)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization using Isaac Lab markers."""
        if debug_vis:
            # Create visualization markers for command arrows
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/VelocityCommandArrows", 
                markers={
                    "velocity_arrow": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(2.0, 0.5, 0.5),  # Long thin arrow for direction clarity
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.2, 0.2),  # Bright red arrows for visibility
                            metallic=0.0,
                            roughness=0.4,
                            emissive_color=(0.5, 0.0, 0.0),  # Slight glow
                        ),
                    ),
                },
            )
            self._command_visualizer = VisualizationMarkers(marker_cfg)
            print("[INFO] Command velocity arrows enabled - red arrows show command direction above each robot")
        else:
            self._command_visualizer = None

    def _debug_vis_callback(self, event):
        """Debug visualization callback - required by Isaac Lab framework."""
        # This method is required by the framework but we don't need to implement it
        # since our visualization is handled in _update_debug_vis
        pass

    def compute_metrics(self) -> dict[str, float]:
        """Compute metrics for the command term."""
        return {}

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command."""
        if env_ids is None:
            env_ids = slice(None)
        
        # resample commands for reset environments
        if isinstance(env_ids, slice):
            if env_ids == slice(None):
                self._resample_command(list(range(self.num_envs)))
            else:
                self._resample_command(list(range(*env_ids.indices(self.num_envs))))
        else:
            self._resample_command(env_ids)
            
        return self.compute_metrics()


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for uniform velocity command."""

    @configclass
    class Ranges:
        """Ranges for the velocity commands."""
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
        heading: tuple[float, float] = (-3.14159, 3.14159)

    class_type: type = UniformVelocityCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    rel_standing_envs: float = 0.02
    rel_heading_envs: float = 1.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: Ranges = Ranges()
