#!/usr/bin/env python3
# Copyright (c) 2025, OceanBDX Project.
# All rights reserved.

"""
OceanBDX 键盘控制脚本 - 用于在 Isaac Lab 中进行 sim2sim 推理测试

使用方法:
    方法1 (推荐 - 在 Isaac Lab 目录):
        cd /path/to/IsaacLab
        ./isaaclab.sh -p /home/ocean/oceanbdx/oceanbdx/scripts/play_keyboard_control.py \\
            --checkpoint /home/ocean/oceanbdx/oceanbdx/logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt

    方法2 (本地运行 - 需要配置环境):
        cd /home/ocean/oceanbdx/oceanbdx
        python scripts/play_keyboard_control.py --checkpoint logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt
            --checkpoint logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt

键盘控制:
    W: 向前移动
    S: 向后移动
    A: 向左移动
    D: 向右移动
    Q: 逆时针旋转
    E: 顺时针旋转
    R: 重置环境
    SPACE: 停止
    ESC: 退出
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="OceanBDX keyboard control for sim2sim testing.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt",
    help="Path to trained policy checkpoint (.pt file)",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1 for easier control)")
# 添加 AppLauncher 参数 (会自动添加 --device 等参数)
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import gymnasium as gym
import carb
import omni.appwindow

# 导入 OceanBDX 环境配置 - 这会自动注册 gym 环境
import oceanbdx.tasks  # noqa: F401


class OceanBDXKeyboardControl:
    """OceanBDX 键盘控制类 - 用于测试训练好的策略"""

    def __init__(self):
        """初始化环境和键盘控制"""

        print("\n" + "=" * 80)
        print("🤖 OceanBDX 键盘控制 - Sim2Sim 推理测试")
        print("=" * 80)

        # 检查 checkpoint 是否存在
        if not os.path.exists(args_cli.checkpoint):
            raise FileNotFoundError(f"❌ Checkpoint not found: {args_cli.checkpoint}")

        print(f"✅ Loading checkpoint: {args_cli.checkpoint}")

        # 导入 Play 模式的环境配置
        from oceanbdx.tasks.manager_based.oceanbdx_locomotion.config import OceanBDXLocomotionEnvCfg_PLAY

        # 创建环境配置 (Play 模式)
        self.env_cfg = OceanBDXLocomotionEnvCfg_PLAY()
        self.env_cfg.scene.num_envs = args_cli.num_envs
        self.env_cfg.episode_length_s = 1000000  # 无限长episode

        # 设置速度命令范围 (更适合键盘控制)
        self.env_cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.env_cfg.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        print(f"✅ Creating environment with {args_cli.num_envs} env(s)...")

        # 使用 gym.make 创建环境
        self.env = gym.make("Isaac-Ocean-BDX-Locomotion-Play-v0", cfg=self.env_cfg)
        self.device = self.env.unwrapped.device

        # 加载训练好的模型
        print("✅ Loading policy from checkpoint...")
        self.policy = torch.jit.load(args_cli.checkpoint).to(self.device)
        self.policy.eval()

        # 初始化命令 (使用环境数量)
        self.commands = torch.zeros(args_cli.num_envs, 3, device=self.device)

        # 打开日志文件记录观测和动作
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"inference_log_{timestamp}.txt"
        self.log_file = open(self.log_filename, "w")
        self.log_file.write("# Isaac Lab Inference Log\n")
        self.log_file.write(f"# Timestamp: {timestamp}\n")
        self.log_file.write(f"# Checkpoint: {args_cli.checkpoint}\n")
        self.log_file.write("# Observation dim: 74, Action dim: 14\n")
        self.log_file.write("#\n")
        self.log_file.write("# Format per step:\n")
        self.log_file.write("#   STEP <step_num>\n")
        self.log_file.write("#   OBS <74 float values> (space separated)\n")
        self.log_file.write("#   ACT <14 float values> (space separated)\n")
        self.log_file.write("#   INFO gravity=[x,y,z] tilt=deg roll=deg pitch=deg height=m ang_vel=[x,y,z] dof_pos=[...] dof_vel=[...]\n")
        self.log_file.write("#\n")
        self.log_file.write("# Observation structure (74-dim):\n")
        self.log_file.write("#   [0-2]   ang_vel (3)         - Angular velocity\n")
        self.log_file.write("#   [3-5]   gravity (3)         - Gravity vector (used to compute tilt/roll/pitch)\n")
        self.log_file.write("#   [6-19]  dof_pos_rel (14)    - Joint positions\n")
        self.log_file.write("#   [20-33] dof_vel (14)        - Joint velocities\n")
        self.log_file.write("#   [34-47] torques (14)        - Joint torques\n")
        self.log_file.write("#   [48-50] commands (3)        - Velocity commands [lin_x, lin_y, ang_z]\n")
        self.log_file.write("#   [51-64] last_actions (14)   - Previous actions\n")
        self.log_file.write("#   [65-73] adaptive_phase (9)  - Adaptive phase variables\n")
        self.log_file.write("#\n")
        self.log_file.write("=" * 100 + "\n\n")
        self.step_num = 0
        
        print(f"📝 Logging to: {self.log_filename}")

        # 设置键盘控制
        self.set_up_keyboard()

        print("\n" + "=" * 80)
        print("📋 键盘控制说明:")
        print("=" * 80)
        print("  W       - 向前移动 (lin_vel_x = +1.0)")
        print("  S       - 向后移动 (lin_vel_x = -1.0)")
        print("  A       - 向左移动 (lin_vel_y = +0.5)")
        print("  D       - 向右移动 (lin_vel_y = -0.5)")
        print("  Q       - 逆时针旋转 (ang_vel_z = +1.0)")
        print("  E       - 顺时针旋转 (ang_vel_z = -1.0)")
        print("  SPACE   - 停止 (所有速度归零)")
        print("  R       - 重置环境")
        print("  ESC     - 退出程序")
        print("=" * 80 + "\n")

    def set_up_keyboard(self):
        """设置键盘监听"""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        # 定义按键到命令的映射
        # 格式: [lin_vel_x, lin_vel_y, ang_vel_z]
        # 提高速度命令,让机器人移动更明显
        self._key_to_control = {
            "W": torch.tensor([1.0, 0.0, 0.0], device=self.device),  # 前进 (加大到1.0)
            "S": torch.tensor([-1.0, 0.0, 0.0], device=self.device),  # 后退
            "A": torch.tensor([0.0, 0.5, 0.0], device=self.device),  # 左移
            "D": torch.tensor([0.0, -0.5, 0.0], device=self.device),  # 右移
            "Q": torch.tensor([0.0, 0.0, 1.0], device=self.device),  # 左转 (加大到1.0)
            "E": torch.tensor([0.0, 0.0, -1.0], device=self.device),  # 右转
            "SPACE": torch.tensor([0.0, 0.0, 0.0], device=self.device),  # 停止
        }

        self._reset_requested = False
        self._quit_requested = False

    def _on_keyboard_event(self, event):
        """键盘事件回调"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # 方向控制键
            if event.input.name in self._key_to_control:
                self.commands[:] = self._key_to_control[event.input.name]
                print(f"🎮 Command: {event.input.name} -> {self.commands[0].cpu().numpy()}")

            # 重置
            elif event.input.name == "R":
                self._reset_requested = True
                print("🔄 Reset requested...")

            # 退出
            elif event.input.name == "ESCAPE":
                self._quit_requested = True
                print("👋 Quit requested...")

        # 按键释放时保持命令不变（不归零）
        # 这样可以持续移动，直到按下其他键

    def run(self):
        """运行主循环"""
        print("🚀 Starting simulation loop...")
        print("✅ Environment ready! Use keyboard to control the robot.\n")

        # 重置环境
        obs_dict, _ = self.env.reset()

        step_count = 0

        try:
            while simulation_app.is_running() and not self._quit_requested:
                # 检查是否需要重置
                if self._reset_requested:
                    obs_dict, _ = self.env.reset()
                    self.commands.zero_()
                    self._reset_requested = False
                    step_count = 0
                    print("✅ Environment reset!\n")

                # 提取 policy 观测 (字典格式 -> tensor)
                obs = obs_dict["policy"]

                # 覆盖观测中的命令（使用键盘输入）
                # 注意: OceanBDX 的观测维度是 74, commands 在第 57:60 位置
                # 观测结构: [ang_vel(3), gravity(3), dof_pos_rel(14), dof_vel(14),
                #            torques(14), commands(3), last_actions(14), adaptive_phase(9)]
                obs[:, 57:60] = self.commands

                # 使用策略计算动作
                with torch.inference_mode():
                    action = self.policy(obs)

                # 记录观测和动作到日志文件
                self.step_num += 1
                self.log_file.write(f"STEP {self.step_num}\n")
                
                # 写入观测 (74维)
                obs_str = " ".join([f"{v:.6f}" for v in obs[0].cpu().numpy()])
                self.log_file.write(f"OBS {obs_str}\n")
                
                # 写入动作 (14维)
                act_str = " ".join([f"{v:.6f}" for v in action[0].cpu().numpy()])
                self.log_file.write(f"ACT {act_str}\n")
                
                # 提取并计算额外的调试信息
                import math
                obs_cpu = obs[0].cpu().numpy()
                
                # 从观测中提取关键信息
                ang_vel = obs_cpu[0:3]  # 角速度
                gravity = obs_cpu[3:6]  # 重力向量
                dof_pos = obs_cpu[6:20]  # 关节位置 (14个)
                dof_vel = obs_cpu[20:34]  # 关节速度 (14个)
                
                # 计算倾斜角度 (从重力向量)
                gx, gy, gz = gravity[0], gravity[1], gravity[2]
                gravity_norm = math.sqrt(gx**2 + gy**2 + gz**2)
                
                # Roll (侧倾) 和 Pitch (俯仰)
                roll_rad = math.atan2(gy, -gz)
                pitch_rad = math.atan2(-gx, -gz)
                roll_deg = math.degrees(roll_rad)
                pitch_deg = math.degrees(pitch_rad)
                
                # 总倾斜角度
                tilt_rad = math.acos(min(abs(gz) / max(gravity_norm, 1e-6), 1.0))
                tilt_deg = math.degrees(tilt_rad)
                
                # 获取机器人位置和高度
                base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0, :3].cpu().numpy()
                
                # 写入额外信息
                dof_pos_str = ",".join([f"{v:.3f}" for v in dof_pos[:6]])
                dof_vel_str = ",".join([f"{v:.1f}" for v in dof_vel[:6]])
                self.log_file.write(f"INFO gravity=[{gx:.6f},{gy:.6f},{gz:.6f}] ")
                self.log_file.write(f"tilt={tilt_deg:.2f} roll={roll_deg:.2f} pitch={pitch_deg:.2f} ")
                self.log_file.write(f"height={base_pos[2]:.3f} ")
                self.log_file.write(f"ang_vel=[{ang_vel[0]:.3f},{ang_vel[1]:.3f},{ang_vel[2]:.3f}] ")
                self.log_file.write(f"dof_pos=[{dof_pos_str}...] ")
                self.log_file.write(f"dof_vel=[{dof_vel_str}...]\n")
                
                self.log_file.write("\n")
                self.log_file.flush()  # 立即写入磁盘

                # 执行动作
                obs_dict, rewards, terminated, truncated, info = self.env.step(action)

                step_count += 1

                # 每 50 步打印一次详细状态
                if step_count % 50 == 0:
                    cmd = self.commands[0]
                    # 打印详细信息 (使用之前计算的值)
                    print(
                        f"📊 Step {step_count:6d} | "
                        f"Pos: [{base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.2f}] | "
                        f"Tilt: {tilt_deg:5.2f}° | Roll: {roll_deg:6.2f}° | Pitch: {pitch_deg:6.2f}° | "
                        f"Cmd: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]"
                    )
                    print(
                        f"         Gravity: [{gx:.4f}, {gy:.4f}, {gz:.4f}] | "
                        f"AngVel: [{ang_vel[0]:.2f}, {ang_vel[1]:.2f}, {ang_vel[2]:.2f}] | "
                        f"Action[0-2]: [{action[0,0]:.3f}, {action[0,1]:.3f}, {action[0,2]:.3f}]"
                    )

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user (Ctrl+C)")

        # 关闭日志文件
        self.log_file.close()
        print(f"\n📝 Log saved to: {self.log_filename}")
        print(f"   Total steps logged: {self.step_num}")

        print("\n" + "=" * 80)
        print("✅ Simulation finished!")
        print("=" * 80)


def main():
    """主函数"""
    # 创建控制器
    controller = OceanBDXKeyboardControl()
    
    # 运行
    controller.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
