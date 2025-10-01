#!/usr/bin/env python3

"""
Script to debug observation dimensions and validate the exported model.
"""

import os
import sys
import torch

# Add the source directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "source"))

from omni.isaac.lab.app import AppLauncher

# Create minimal app launcher for headless debugging
app_launcher = AppLauncher(args_cli=type('Args', (), {'headless': True, 'device': 'cpu'})())
simulation_app = app_launcher.app

# Import after app launch
import omni.isaac.lab.envs  # noqa: F401
import omni.isaac.lab_tasks  # noqa: F401
import oceanbdx.tasks  # noqa: F401
import gymnasium as gym

def main():
    """Debug observation dimensions."""
    print("🔍 Debugging observation dimensions...")
    
    # Create environment
    env_name = "Isaac-Ocean-BDX-Locomotion-v0"
    print(f"Creating environment: {env_name}")
    
    try:
        env = gym.make(env_name, num_envs=1, headless=True)
        
        # Reset environment to get observations
        obs, _ = env.reset()
        
        print(f"\n📊 Environment successfully created!")
        print(f"🎯 Observation space shape: {obs.shape}")
        print(f"📏 Total observation dimensions: {obs.shape[-1]}")
        
        # Break down observations by term if possible
        if hasattr(env.unwrapped, 'observation_manager'):
            obs_manager = env.unwrapped.observation_manager
            print(f"\n🔧 Observation Manager Details:")
            
            if hasattr(obs_manager, '_group_obs_term_names'):
                for group_name, term_names in obs_manager._group_obs_term_names.items():
                    print(f"  📋 Group '{group_name}':")
                    total_dims = 0
                    for term_name in term_names:
                        if hasattr(obs_manager, '_group_obs_term_dim') and group_name in obs_manager._group_obs_term_dim:
                            term_dims = obs_manager._group_obs_term_dim[group_name].get(term_name, 'Unknown')
                            total_dims += term_dims if isinstance(term_dims, int) else 0
                            print(f"    - {term_name}: {term_dims} dims")
                        else:
                            print(f"    - {term_name}: Unknown dims")
                    print(f"    Total: {total_dims} dims")
        
        # Check model input requirements
        print(f"\n🤖 Checking exported model...")
        model_path = "logs/rsl_rl/oceanbdx_locomotion/2025-09-30_15-29-30/exported/policy.pt"
        
        if os.path.exists(model_path):
            try:
                model = torch.jit.load(model_path, map_location='cpu')
                print(f"✅ Model loaded successfully from: {model_path}")
                
                # Try to get model input shape from the first layer
                # This is a bit hacky but works for most models
                state_dict = dict(model.named_parameters())
                first_weight_key = list(state_dict.keys())[0]
                if 'weight' in first_weight_key:
                    input_shape = state_dict[first_weight_key].shape[1]
                    print(f"🎯 Model expects input dimensions: {input_shape}")
                    
                    if input_shape != obs.shape[-1]:
                        print(f"❌ MISMATCH DETECTED!")
                        print(f"   - Environment provides: {obs.shape[-1]} dims")
                        print(f"   - Model expects: {input_shape} dims")
                        print(f"   - Difference: {input_shape - obs.shape[-1]} dims")
                    else:
                        print(f"✅ Dimensions match perfectly!")
                
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
        else:
            print(f"❌ Model not found at: {model_path}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()