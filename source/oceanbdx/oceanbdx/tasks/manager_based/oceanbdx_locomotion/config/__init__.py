# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OceanBDX locomotion environment."""

# Main configurations (complete environment with all sensors and features)
from .oceanbdx_locomotion_main import (
    OceanBDXLocomotionEnvCfg,
    OceanBDXLocomotionEnvCfg_PLAY,
)

# Backup configurations available for reference
# from .oceanbdx_locomotion_simple import (
#     OceanBDXLocomotionEnvCfg as OceanBDXLocomotionEnvCfg_Simple,
#     OceanBDXLocomotionEnvCfg_PLAY as OceanBDXLocomotionEnvCfg_Simple_PLAY,
# )