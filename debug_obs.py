#!/usr/bin/env python3

"""Debug script to check observation dimensions."""

import gymnasium as gym
import torch

# Import the environment
import oceanbdx  # noqa: F401

def debug_observations():
    """Debug observation dimensions."""
    try:
        # Create environment
        env = gym.make("Isaac-Ocean-BDX-Locomotion-v0", num_envs=1, render_mode=None)
        
        # Reset environment
        obs, info = env.reset()
        
        print("=== Observation Debug ===")
        print(f"Total observation shape: {obs['policy'].shape}")
        
        # Check individual observation terms
        obs_manager = env.observation_manager
        policy_group = obs_manager._group_obs_term_cfgs["policy"]
        
        print("\n=== Individual Observation Terms ===")
        for term_name, term_cfg in policy_group.items():
            if term_name.startswith("_"):
                continue
            try:
                # Get the observation function
                func = term_cfg.func
                params = term_cfg.params if term_cfg.params else {}
                
                # Call the function
                obs_data = func(env, **params)
                print(f"{term_name}: {obs_data.shape}")
                
            except Exception as e:
                print(f"{term_name}: ERROR - {e}")
        
        env.close()
        
    except Exception as e:
        print(f"Environment creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_observations()