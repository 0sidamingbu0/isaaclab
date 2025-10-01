#!/usr/bin/env python

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script to verify OceanBDX locomotion environment setup."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test OceanBDX locomotion environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import traceback

print("[INFO] Testing OceanBDX locomotion environment...")

try:
    # Import gym first 
    import gymnasium as gym
    print("[INFO] ✓ gymnasium imported successfully")
    
    # Import Isaac Lab
    from isaaclab.utils.dict import print_dict
    print("[INFO] ✓ isaaclab imported successfully")
    
    # Import oceanbdx tasks
    import oceanbdx.tasks  # noqa: F401
    print("[INFO] ✓ oceanbdx.tasks imported successfully")
    
    # Try to check if the environment is registered
    env_ids = [spec.id for spec in gym.envs.registry.values() if "Ocean" in spec.id]
    print(f"[INFO] Found registered Ocean environments: {env_ids}")
    
    if "Isaac-Ocean-BDX-Locomotion-v0" in env_ids:
        print("[INFO] ✓ Isaac-Ocean-BDX-Locomotion-v0 is registered")
        
        # Try to create the environment
        print("[INFO] Attempting to create environment...")
        env = gym.make(
            "Isaac-Ocean-BDX-Locomotion-v0",
            num_envs=args_cli.num_envs,
            render_mode=None
        )
        print("[INFO] ✓ Environment created successfully")
        
        # Print environment info
        print(f"[INFO] Action space: {env.action_space}")
        print(f"[INFO] Observation space: {env.observation_space}")
        
        # Test reset
        print("[INFO] Testing environment reset...")
        obs, info = env.reset()
        print(f"[INFO] ✓ Reset successful, observation shape: {obs['policy'].shape}")
        
        # Test a few steps
        print("[INFO] Testing environment steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"[INFO] Step {i+1}: reward={reward.mean():.4f}, terminated={terminated.sum()}, truncated={truncated.sum()}")
        
        print("[INFO] ✓ Environment test completed successfully!")
        env.close()
        
    else:
        print(f"[ERROR] Isaac-Ocean-BDX-Locomotion-v0 not found in registered environments")
        print(f"Available environments: {env_ids}")
        
except Exception as e:
    print(f"[ERROR] Test failed with exception: {e}")
    print(f"[ERROR] Traceback: {traceback.format_exc()}")

finally:
    # close sim app
    simulation_app.close()
    print("[INFO] Test completed.")