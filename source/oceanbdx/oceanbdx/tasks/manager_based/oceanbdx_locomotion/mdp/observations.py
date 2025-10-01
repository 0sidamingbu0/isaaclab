# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.

All observation functions have the signature:

.. code-block:: python

    def obs_func(env: ManagerBasedEnv, **kwargs) -> torch.Tensor:
        pass

Where ``env`` is the environment instance and ``kwargs`` are the arguments that can be passed to the observation
function. The function should return a tensor of shape (num_envs, ...) with the observation values.

"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster, ContactSensor, Imu

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

"""
Root state.
"""


def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in base frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def base_quat_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root quaternion (w, x, y, z) in world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def base_quat_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root quaternion (w, x, y, z) in base frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_b


"""
Joint state.
"""


def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset relative to default joint positions."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def joint_pos_norm(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset normalized with limits."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos,
        asset.data.soft_joint_pos_limits[:, :, 0],
        asset.data.soft_joint_pos_limits[:, :, 1],
    )


def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint velocities of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.joint_vel


"""
Actions.
"""


def last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """The last actions that were applied to the environment."""
    return env.action_manager.action


"""
Commands.
"""


def generated_commands(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)


"""
Contact forces.
"""


def contact_forces(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """Contact forces as reported by the contact sensor."""
    # extract the used quantities (to enable type-hinting)
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    return sensor.data.net_forces_w_history[:, 0, :].view(env.num_envs, -1)


"""
Height measurements.
"""


def height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset is subtracted from the height measurement. This is useful when the height scanner
    is mounted at a certain height above the ground and we want to get the height of the ground w.r.t.
    a reference point (e.g., the base of the robot).
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene[sensor_cfg.name]
    # height data: (num_envs, num_rays, 1) -> (num_envs, num_rays)
    return sensor.data.ray_hits_w[..., -1] - offset


"""
IMU measurements from Isaac Lab native IMU sensor.
"""


def imu_acceleration(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("imu_sensor")) -> torch.Tensor:
    """Linear acceleration from IMU sensor in the sensor frame."""
    # extract the used quantities (to enable type-hinting)
    sensor: Imu = env.scene[sensor_cfg.name]
    return sensor.data.lin_acc_b


def imu_angular_velocity(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("imu_sensor")) -> torch.Tensor:
    """Angular velocity from IMU sensor in the sensor frame."""
    # extract the used quantities (to enable type-hinting)
    sensor: Imu = env.scene[sensor_cfg.name]
    return sensor.data.ang_vel_b


def imu_quaternion(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("imu_sensor")) -> torch.Tensor:
    """Orientation quaternion from IMU sensor in world frame."""
    # extract the used quantities (to enable type-hinting)
    sensor: Imu = env.scene[sensor_cfg.name]
    return sensor.data.quat_w


def joint_torques_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint torques applied by the actuators."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.applied_torque
