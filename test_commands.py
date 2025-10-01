#!/usr/bin/env python3

"""
简单测试脚本，验证OceanBDX机器人是否正确接收和执行速度命令
"""

import torch
print("测试环境设置...")

# 测试基础Isaac Lab导入
try:
    import isaaclab
    print("✅ Isaac Lab 导入成功")
except ImportError as e:
    print(f"❌ Isaac Lab 导入失败: {e}")

# 测试环境创建
try:
    import gymnasium as gym
    from oceanbdx.tasks.manager_based.oceanbdx_locomotion import OceanBDXLocomotionSimpleEnvCfg
    
    # 创建一个小规模环境进行测试
    env_cfg = OceanBDXLocomotionSimpleEnvCfg()
    env_cfg.scene.num_envs = 4  # 只测试4个环境
    
    env = gym.make("Isaac-Ocean-BDX-Locomotion-v0", cfg=env_cfg)
    print("✅ 环境创建成功")
    
    # 重置环境
    obs, info = env.reset()
    print(f"✅ 环境重置成功, 观测维度: {obs['policy'].shape}")
    
    # 检查命令管理器
    if hasattr(env.unwrapped, 'command_manager'):
        cmd_mgr = env.unwrapped.command_manager
        print(f"✅ 命令管理器存在: {type(cmd_mgr)}")
        
        # 获取当前命令
        if 'base_velocity' in cmd_mgr._group_command_term_names:
            base_vel_cmd = cmd_mgr.get_command('base_velocity')
            print(f"✅ 速度命令获取成功:")
            print(f"   命令形状: {base_vel_cmd.shape}")
            print(f"   命令值示例: {base_vel_cmd[:4]}")  # 显示前4个环境的命令
            
            # 检查命令是否为零（这是问题所在）
            non_zero_cmds = torch.norm(base_vel_cmd, dim=1) > 0.1
            print(f"   非零命令环境数: {non_zero_cmds.sum().item()}/{base_vel_cmd.shape[0]}")
            
            if non_zero_cmds.sum().item() == 0:
                print("⚠️  所有命令都接近零！这可能是问题所在。")
            else:
                print("✅ 发现非零速度命令")
        else:
            print("❌ 未找到base_velocity命令")
    else:
        print("❌ 命令管理器不存在")
        
    # 运行几步检查
    print("运行5步测试...")
    for i in range(5):
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # 重新获取命令检查是否变化
        if hasattr(env.unwrapped, 'command_manager'):
            base_vel_cmd = env.unwrapped.command_manager.get_command('base_velocity')
            non_zero = torch.norm(base_vel_cmd, dim=1) > 0.1
            print(f"步骤 {i+1}: 非零命令环境数: {non_zero.sum().item()}")
    
    env.close()
    print("✅ 测试完成")
    
except Exception as e:
    print(f"❌ 环境测试失败: {e}")
    import traceback
    traceback.print_exc()