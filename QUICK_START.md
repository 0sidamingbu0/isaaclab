# 🚀 快速启动指南

## 问题
当前 `oceanbdx` 工程**没有** `isaaclab.sh` 脚本,需要到 Isaac Lab 主目录运行。

## 解决方案

### ✅ 方案1: 使用快速启动脚本 (最简单!)

```bash
cd /home/ocean/oceanbdx/oceanbdx

# 第一次使用,指定你的 Isaac Lab 路径
./run_keyboard_control.sh /path/to/IsaacLab

# 例如:
./run_keyboard_control.sh ~/IsaacLab
# 或
./run_keyboard_control.sh /home/ocean/IsaacLab
```

脚本会自动:
- ✅ 检查 Isaac Lab 路径是否正确
- ✅ 使用绝对路径运行键盘控制脚本
- ✅ 自动加载默认 checkpoint
- ✅ 设置 num_envs=1

### ✅ 方案2: 手动运行

```bash
# 1. 进入 Isaac Lab 目录
cd /path/to/IsaacLab

# 2. 运行脚本 (使用绝对路径)
./isaaclab.sh -p /home/ocean/oceanbdx/oceanbdx/scripts/play_keyboard_control.py \
    --checkpoint /home/ocean/oceanbdx/oceanbdx/logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt \
    --num_envs 1
```

### ✅ 方案3: 创建软链接

```bash
# 在 oceanbdx 目录创建链接
cd /home/ocean/oceanbdx/oceanbdx
ln -s /path/to/IsaacLab/isaaclab.sh ./isaaclab.sh

# 之后就可以直接用
./isaaclab.sh -p scripts/play_keyboard_control.py
```

## 键盘控制

| 按键 | 功能 |
|------|------|
| W | 前进 |
| S | 后退 |
| A | 左移 |
| D | 右移 |
| Q | 左转 |
| E | 右转 |
| SPACE | 停止 |
| R | 重置 |
| ESC | 退出 |

## 示例输出

```
====================================================================================
🤖 OceanBDX 键盘控制 - Sim2Sim 推理测试
====================================================================================
✅ Loading checkpoint: logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt
✅ Creating environment with 1 env(s)...
✅ Loading policy from checkpoint...

====================================================================================
📋 键盘控制说明:
====================================================================================
  W       - 向前移动 (lin_vel_x = +0.5)
  S       - 向后移动 (lin_vel_x = -0.5)
  A       - 向左移动 (lin_vel_y = +0.3)
  D       - 向右移动 (lin_vel_y = -0.3)
  Q       - 逆时针旋转 (ang_vel_z = +0.5)
  E       - 顺时针旋转 (ang_vel_z = -0.5)
  SPACE   - 停止 (所有速度归零)
  R       - 重置环境
  ESC     - 退出程序
====================================================================================

🚀 Starting simulation loop...
✅ Environment ready! Use keyboard to control the robot.

🎮 Command: W -> [0.5 0.  0. ]
📊 Step    100 | Pos: [0.50, 0.00, 0.85] | Cmd: [0.50, 0.00, 0.00]
📊 Step    200 | Pos: [1.20, 0.00, 0.85] | Cmd: [0.50, 0.00, 0.00]
```

## 调试目的

通过这个脚本验证:
1. ✅ 模型在 Isaac Lab (PyTorch JIT) 中是否正常
2. ✅ 对比与 Gazebo (LibTorch C++) 的差异
3. ✅ 排查 LibTorch 推理问题

如果 Isaac Lab 正常,Gazebo 异常 → **问题在 LibTorch C++ 实现!**

---

**更多详情**: 查看 `KEYBOARD_CONTROL_README.md`
