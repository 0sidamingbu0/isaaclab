#!/usr/bin/env python3
"""
测试机器人初始姿态
让机器人保持出生姿态不动，检查是否会摔倒
"""

import argparse
import torch
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="测试机器人初始姿态")
parser.add_argument("--num_envs", type=int, default=16, help="环境数量")
parser.add_argument("--headless", action="store_true", default=False, help="无头模式")
args_cli = parser.parse_args()

# 启动仿真器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import oceanbdx.tasks  # noqa: F401
from oceanbdx.tasks.manager_based.oceanbdx_locomotion.config import OceanBDXLocomotionEnvCfg

def main():
    """主函数:创建环境并让机器人保持静止"""
    
    # 创建环境配置
    env_cfg = OceanBDXLocomotionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 创建环境
    env = gym.make("Isaac-Ocean-BDX-Locomotion-v0", cfg=env_cfg)
    
    print("\n" + "="*80)
    print("🤖 机器人初始姿态测试")
    print("="*80)
    print("\n📋 配置信息：")
    print(f"   环境数量: {args_cli.num_envs}")
    print(f"   初始高度: 0.4m (配置值)")
    print(f"   控制频率: 50Hz (decimation=4)")
    print(f"   仿真步长: 0.005s")
    
    # 获取机器人信息
    robot = env.unwrapped.scene["robot"]
    print(f"\n🔧 关节配置：")
    print(f"   总关节数: {robot.num_joints}")
    print(f"   控制关节: {robot.data.joint_names}")
    
    # 打印初始关节位置
    print(f"\n📐 初始关节位置 (弧度)：")
    initial_joint_pos = robot.data.default_joint_pos[0]
    for i, (name, pos) in enumerate(zip(robot.data.joint_names, initial_joint_pos)):
        print(f"   {name:20s}: {pos:7.4f} rad ({pos*57.3:6.1f}°)")
    
    # 重置环境
    obs, _ = env.reset()
    print(f"\n✅ 环境已重置")
    
    # 获取初始状态
    base_pos = robot.data.root_pos_w[0]
    base_quat = robot.data.root_quat_w[0]
    
    print(f"\n🎯 初始状态：")
    print(f"   基座位置: x={base_pos[0]:.3f}, y={base_pos[1]:.3f}, z={base_pos[2]:.3f} m")
    print(f"   基座四元数: w={base_quat[0]:.3f}, x={base_quat[1]:.3f}, y={base_quat[2]:.3f}, z={base_quat[3]:.3f}")
    
    # 计算欧拉角
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat([base_quat[1].item(), base_quat[2].item(), 
                               base_quat[3].item(), base_quat[0].item()])  # xyzw格式
    euler = rot.as_euler('xyz', degrees=True)
    print(f"   欧拉角: Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
    
    print(f"\n⏱️  开始测试 - 机器人将保持静止30秒...")
    print(f"   👀 请观察机器人是否：")
    print(f"      1. 保持直立")
    print(f"      2. 不会摔倒")
    print(f"      3. 脚是否接触地面")
    print(f"      4. 身体高度是否合适")
    print(f"\n   按 Ctrl+C 可提前结束\n")
    
    # 保持静止30秒 (每步0.02秒，1500步 = 30秒)
    try:
        # 获取实际的动作维度(14个关节)
        action_dim = env.unwrapped.action_manager.total_action_dim
        zero_action = torch.zeros((args_cli.num_envs, action_dim), device=env.unwrapped.device)
        
        for step in range(1500):
            # 发送零动作（保持初始姿态）
            obs, reward, terminated, truncated, info = env.step(zero_action)
            
            # 每5秒打印一次状态
            if step % 250 == 0 and step > 0:
                current_base_pos = robot.data.root_pos_w[0]
                current_base_quat = robot.data.root_quat_w[0]
                
                # 计算当前欧拉角
                rot = Rotation.from_quat([current_base_quat[1].item(), current_base_quat[2].item(), 
                                         current_base_quat[3].item(), current_base_quat[0].item()])
                euler = rot.as_euler('xyz', degrees=True)
                
                # 计算倾斜角度
                tilt = float(torch.tensor(euler[0]**2 + euler[1]**2).sqrt())
                
                print(f"⏱️  {step*0.02:.1f}s - "
                      f"高度: {current_base_pos[2]:.3f}m, "
                      f"Roll: {euler[0]:5.1f}°, "
                      f"Pitch: {euler[1]:5.1f}°, "
                      f"Tilt: {tilt:.1f}° "
                      f"{'✅' if tilt < 30 else '⚠️ 倾斜过大'}")
                
                # 统计终止情况
                num_terminated = terminated.sum().item()
                if num_terminated > 0:
                    print(f"   ⚠️  {num_terminated}/{args_cli.num_envs} 个机器人已终止（摔倒）")
            
            # 如果所有机器人都摔倒了，提前结束
            if terminated.all():
                print(f"\n❌ 所有机器人都摔倒了！测试在 {step*0.02:.1f} 秒后结束")
                break
        
        else:
            print(f"\n✅ 测试完成！机器人在30秒内保持稳定")
        
        # 最终统计
        final_base_pos = robot.data.root_pos_w
        final_heights = final_base_pos[:, 2]
        
        print(f"\n📊 最终统计（{args_cli.num_envs}个环境）：")
        print(f"   平均高度: {final_heights.mean():.3f}m (标准差: {final_heights.std():.3f}m)")
        print(f"   最高: {final_heights.max():.3f}m")
        print(f"   最低: {final_heights.min():.3f}m")
        
        num_survived = (~terminated).sum().item()
        print(f"   存活: {num_survived}/{args_cli.num_envs} ({num_survived/args_cli.num_envs*100:.1f}%)")
        
        if num_survived == args_cli.num_envs:
            print(f"\n✅ 结论：初始姿态稳定，机器人不会自然摔倒")
        elif num_survived > args_cli.num_envs * 0.8:
            print(f"\n⚠️  结论：大部分机器人稳定，少数可能由于数值误差摔倒")
        else:
            print(f"\n❌ 结论：初始姿态不稳定，需要调整！")
            print(f"\n💡 建议：")
            print(f"   1. 检查初始关节角度是否合理")
            print(f"   2. 检查初始高度是否过高/过低")
            print(f"   3. 检查重心是否在支撑面内")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  用户中断测试")
    
    # 关闭环境
    env.close()
    print(f"\n" + "="*80)

if __name__ == "__main__":
    main()
    simulation_app.close()
