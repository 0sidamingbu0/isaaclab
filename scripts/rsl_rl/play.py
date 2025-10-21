# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import oceanbdx.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # 获取机器人资产用于关节角度输出
    robot = env.unwrapped.scene["robot"]
    print("=" * 80)
    print("🤖 OceanBDX 关节角度监测模式")
    print("=" * 80)
    print(f"关节总数: {len(robot.data.joint_names)}")
    print("每100步输出一次关节角度信息...")
    print("按Ctrl+C停止并输出最终配置")
    
    # === 输出电机/动作顺序信息 ===
    print("\n" + "=" * 80)
    print("🔧 电机控制顺序信息 (用于部署对应)")
    print("=" * 80)
    print("动作索引 | 关节名称          | 关节类型    | 描述")
    print("-" * 60)
    
    for i, joint_name in enumerate(robot.data.joint_names):
        # 解析关节类型
        if "leg_l1" in joint_name:
            joint_type = "左髋外展"
        elif "leg_l2" in joint_name:
            joint_type = "左髋前后"
        elif "leg_l3" in joint_name:
            joint_type = "左膝关节"
        elif "leg_l4" in joint_name:
            joint_type = "左踝前后"
        elif "leg_l5" in joint_name:
            joint_type = "左踝侧摆"
        elif "leg_r1" in joint_name:
            joint_type = "右髋外展"
        elif "leg_r2" in joint_name:
            joint_type = "右髋前后"
        elif "leg_r3" in joint_name:
            joint_type = "右膝关节"
        elif "leg_r4" in joint_name:
            joint_type = "右踝前后"
        elif "leg_r5" in joint_name:
            joint_type = "右踝侧摆"
        elif "neck_n1" in joint_name:
            joint_type = "颈部转向"
        elif "neck_n2" in joint_name:
            joint_type = "颈部俯仰"
        elif "neck_n3" in joint_name:
            joint_type = "颈部侧倾"
        elif "neck_n4" in joint_name:
            joint_type = "头部倾斜"
        else:
            joint_type = "未知"
        
        print(f"   {i:2d}    | {joint_name:<17} | {joint_type:<8} | 电机{i+1}")
    
    print("\n📋 部署时电机映射参考:")
    print("# 模型动作输出 actions[i] 对应的电机:")
    for i, joint_name in enumerate(robot.data.joint_names):
        print(f"# actions[{i}] -> {joint_name}")
    
    print(f"\n🎯 动作向量维度: {env.num_actions}")
    print("=" * 80)
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        
        timestep += 1
        
        # === 添加关节角度输出功能 ===
        if timestep % 100 == 0:  # 每100步输出一次
            print(f"\n--- 时间步: {timestep} ---")
            
            # 获取第一个环境的关节状态
            joint_positions = robot.data.joint_pos[0]
            joint_velocities = robot.data.joint_vel[0]
            joint_torques = robot.data.applied_torque[0]  # 获取当前关节力矩
            
            # === 新增：显示当前动作输出 ===
            print("\n🎮 当前模型输出动作值:")
            print("动作索引 | 关节名称          | 动作值     | 当前角度(°) | 力矩(N⋅m)")
            print("-" * 80)
            for i, (name, action_val, pos_val, torque_val) in enumerate(zip(robot.data.joint_names, actions[0], joint_positions, joint_torques)):
                pos_deg = pos_val.item() * 180.0 / 3.14159
                action_val_item = action_val.item()
                torque_item = torque_val.item()
                print(f"   {i:2d}    | {name:<17} | {action_val_item:8.4f} | {pos_deg:8.2f}° | {torque_item:8.3f}")
            
            print("\n📊 关节状态详情:")
            print("关节名称".ljust(20), "弧度".ljust(10), "度数".ljust(10), "速度".ljust(10), "力矩(N⋅m)")
            print("-" * 70)
            
            for name, pos, vel, torque in zip(robot.data.joint_names, joint_positions, joint_velocities, joint_torques):
                pos_val = pos.item()
                vel_val = vel.item()
                torque_val = torque.item()
                deg_val = pos_val * 180.0 / 3.14159
                print(f"{name:<20} {pos_val:7.4f} {deg_val:7.2f}° {vel_val:7.4f} {torque_val:8.3f}")
            
            # 检查是否稳定
            max_vel = torch.max(torch.abs(joint_velocities)).item()
            avg_vel = torch.mean(torch.abs(joint_velocities)).item()
            max_torque = torch.max(torch.abs(joint_torques)).item()
            avg_torque = torch.mean(torch.abs(joint_torques)).item()
            
            # 降低稳定性要求：平均速度 < 1.0 或最大速度 < 2.0
            is_stable = (avg_vel < 1.0) or (max_vel < 2.0)
            
            print(f"关节速度统计 - 最大: {max_vel:.4f} rad/s, 平均: {avg_vel:.4f} rad/s")
            print(f"关节力矩统计 - 最大: {max_torque:.3f} N⋅m, 平均: {avg_torque:.3f} N⋅m")
            
            if is_stable:  # 更宽松的稳定阈值
                print(f"\n🎯 机器人相对稳定! (平均速度: {avg_vel:.4f} rad/s, 最大速度: {max_vel:.4f} rad/s)")
                print("=" * 80)
                print("📋 用于 OCEAN_ROBOT_CFG 初始姿态的配置:")
                print("=" * 80)
                print("joint_pos={")
                for name, pos in zip(robot.data.joint_names, joint_positions):
                    pos_val = pos.item()
                    deg_val = pos_val * 180.0 / 3.14159
                    print(f'    "{name}": {pos_val:.6f},  # {deg_val:.2f}°')
                print("},")
                
                # 保存到文件
                import json
                config_dict = {}
                for name, pos in zip(robot.data.joint_names, joint_positions):
                    config_dict[name] = float(pos.item())
                
                timestamp = int(time.time())
                filename = f"/tmp/ocean_joint_angles_{timestamp}.json"
                with open(filename, "w") as f:
                    json.dump(config_dict, f, indent=2)
                
                print(f"\n💾 关节角度已保存到: {filename}")
                print("=" * 80)
                
                # 同时输出当前姿态分析
                print("\n📊 当前姿态分析:")
                leg_joints = [name for name in robot.data.joint_names if "leg_" in name]
                neck_joints = [name for name in robot.data.joint_names if "neck_" in name]
                
                print("腿部关节角度:")
                for name in leg_joints:
                    idx = robot.data.joint_names.index(name)
                    pos_val = joint_positions[idx].item()
                    torque_val = joint_torques[idx].item()
                    deg_val = pos_val * 180.0 / 3.14159
                    print(f"  {name}: {deg_val:6.2f}° (力矩: {torque_val:6.3f} N⋅m)")
                
                print("颈部关节角度:")
                for name in neck_joints:
                    idx = robot.data.joint_names.index(name)
                    pos_val = joint_positions[idx].item()
                    torque_val = joint_torques[idx].item()
                    deg_val = pos_val * 180.0 / 3.14159
                    print(f"  {name}: {deg_val:6.2f}° (力矩: {torque_val:6.3f} N⋅m)")
                print("=" * 80)
        # === 关节角度输出功能结束 ===
        
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
