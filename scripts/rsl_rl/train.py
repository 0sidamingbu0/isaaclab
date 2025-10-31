# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import oceanbdx.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # ============================================================================
    # 🎓 三阶段课程学习训练（如果使用OceanBDX任务）
    # ============================================================================
    task_lower = args_cli.task.lower()
    if "oceanbdx" in task_lower or "ocean-bdx" in task_lower or "ocean_bdx" in task_lower:
        print("\n" + "=" * 80)
        print(f"🎓 启用Disney BDX三阶段课程学习 (任务: {args_cli.task})")
        print("=" * 80)
        
        # 导入课程学习模块
        try:
            from oceanbdx.tasks.manager_based.oceanbdx_locomotion.mdp import TrainingCurriculum
            
            curriculum = TrainingCurriculum()
            
            # 获取奖励管理器
            reward_manager = env.unwrapped.reward_manager
            
            # 🔧 设置初始Stage 0权重（站立稳定期）
            initial_weights = curriculum.get_current_weights(0.0)
            
            # 直接修改reward_manager的活动项权重
            for term_name, weight in initial_weights.items():
                if term_name not in ['velocity_command_range', 'progress_range']:
                    if hasattr(reward_manager, '_term_names') and term_name in reward_manager._term_names:
                        idx = reward_manager._term_names.index(term_name)
                        # _term_cfgs是列表，使用索引访问
                        reward_manager._term_cfgs[idx].weight = weight
                        # 强制刷新内部缓存（如果存在）
                        if hasattr(reward_manager, '_term_weights'):
                            reward_manager._term_weights[idx] = weight
            
            # 验证权重是否生效
            vel_idx = reward_manager._term_names.index('velocity_tracking')
            feet_idx = reward_manager._term_names.index('feet_alternating_contact')
            
            print(f"\n✅ 已设置Stage 0初始权重（站立稳定期，0-20%）")  # 🔧 修正打印信息
            print(f"   velocity_tracking 配置: {reward_manager._term_cfgs[vel_idx].weight}")
            print(f"   feet_alternating_contact 配置: {reward_manager._term_cfgs[feet_idx].weight}")
            
            # 调试：打印reward_manager的属性
            print(f"\n🔍 调试信息：")
            print(f"   RewardManager类型: {type(reward_manager)}")
            print(f"   _term_cfgs类型: {type(reward_manager._term_cfgs)}")
            print(f"   奖励项数量: {len(reward_manager._term_names)}")
            if hasattr(reward_manager, '_term_weights'):
                print(f"   velocity_tracking 缓存权重: {reward_manager._term_weights[vel_idx]}")
            else:
                print(f"   ⚠️ 没有_term_weights属性")
            
            print(f"\n   velocity_command_range: {initial_weights.get('velocity_command_range', 'N/A')}")
            print(f"   orientation_penalty: {initial_weights['orientation_penalty']}")
            print(f"   base_height_tracking: {initial_weights['base_height_tracking']}")
            
            # 🔧 包装learn方法，动态更新课程权重
            original_learn = runner.learn
            
            def learn_with_curriculum(*args, **kwargs):
                """包装learn方法以支持课程学习"""
                # 保存原始参数
                num_iterations = kwargs.get('num_learning_iterations', args[0] if args else 1000)
                
                # 重写runner的log方法以插入权重更新逻辑
                original_log = runner.log
                last_stage = -1
                
                def log_with_curriculum(locs: dict):
                    nonlocal last_stage
                    # 计算训练进度
                    current_iter = locs.get('it', 0)
                    progress = current_iter / num_iterations
                    
                    # 更新权重（每个iteration都检查）
                    new_weights = curriculum.get_current_weights(progress)
                    for term_name, weight in new_weights.items():
                        if term_name not in ['velocity_command_range', 'progress_range']:
                            if hasattr(reward_manager, '_term_names') and term_name in reward_manager._term_names:
                                idx = reward_manager._term_names.index(term_name)
                                # _term_cfgs是列表，使用索引
                                reward_manager._term_cfgs[idx].weight = weight
                                if hasattr(reward_manager, '_term_weights'):
                                    reward_manager._term_weights[idx] = weight
                    
                    # 🔧 关键修复：动态更新velocity command范围！
                    if 'velocity_command_range' in new_weights:
                        vel_range = new_weights['velocity_command_range']
                        command_manager = env.unwrapped.command_manager
                        if hasattr(command_manager, '_terms') and 'base_velocity' in command_manager._terms:
                            base_vel_term = command_manager._terms['base_velocity']
                            if hasattr(base_vel_term, 'cfg'):
                                # 更新lin_vel_x范围
                                base_vel_term.cfg.ranges.lin_vel_x = vel_range
                                # 如果是Stage 0，也需要将其他速度设为0
                                if vel_range == (0.0, 0.0):
                                    base_vel_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
                                    base_vel_term.cfg.ranges.ang_vel_z = (0.0, 0.0)
                    
                    # 检测阶段切换
                    from oceanbdx.tasks.manager_based.oceanbdx_locomotion.mdp import get_current_stage
                    current_stage = get_current_stage(progress)
                    if current_stage != last_stage:
                        last_stage = current_stage
                        print(f"\n{'='*80}")
                        print(f"🎓 课程阶段切换: Stage {current_stage} (进度: {progress*100:.1f}%)")
                        print(f"   velocity_tracking: {new_weights['velocity_tracking']}")
                        print(f"   feet_alternating_contact: {new_weights['feet_alternating_contact']}")
                        print(f"   velocity_command_range: {new_weights.get('velocity_command_range', 'N/A')}")
                        print(f"{'='*80}\n")
                    
                    # 调用原始log方法
                    return original_log(locs)
                
                runner.log = log_with_curriculum
                
                # 调用原始learn
                result = original_learn(*args, **kwargs)
                
                # 恢复原始log方法
                runner.log = original_log
                return result
            
            runner.learn = learn_with_curriculum
            
            print("\n💡 课程学习已启用：权重将在每次iteration自动更新")
            print("   Stage 0 (0-5%): 站立稳定 - 不要求移动和抬腿")
            print("   Stage 1 (5-30%): 学习行走 - 开始速度跟踪")
            print("   Stage 2 (30-70%): 优化步态 - 引入交替接触")
            print("   Stage 3 (70-100%): 精细调节 - 能效优化")
            print("=" * 80 + "\n")
            
        except ImportError as e:
            print(f"⚠️  课程学习模块导入失败，使用标准训练: {e}")
    
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
