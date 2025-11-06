# OceanBDX 键盘控制测试 - 使用说明

## 📋 功能说明

这个脚本用于在 Isaac Lab 环境中测试训练好的 OceanBDX 策略,通过键盘控制机器人移动,方便进行 sim2sim 推理验证。

## ⚠️ 重要提示

**当前工程 (`oceanbdx`) 没有 `isaaclab.sh` 脚本!** 需要到 **Isaac Lab 主目录**运行。

## 🚀 使用方法

### 方法1: 在 Isaac Lab 目录运行 (推荐)

```bash
# 1. 进入 Isaac Lab 安装目录
cd /path/to/IsaacLab  # 例如: cd ~/IsaacLab

# 2. 使用绝对路径运行脚本
./isaaclab.sh -p /home/ocean/oceanbdx/oceanbdx/scripts/play_keyboard_control.py \
    --checkpoint /home/ocean/oceanbdx/oceanbdx/logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt \
    --num_envs 1
```

### 方法2: 创建软链接 (可选)

```bash
# 在 oceanbdx 目录创建软链接
cd /home/ocean/oceanbdx/oceanbdx
ln -s /path/to/IsaacLab/isaaclab.sh ./isaaclab.sh

# 然后就可以直接运行
./isaaclab.sh -p scripts/play_keyboard_control.py \
    --checkpoint logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt
```

### 方法3: 使用相对路径

如果你的 `oceanbdx` 工程在 Isaac Lab 扩展目录中:

```bash
# 例如: IsaacLab/source/extensions/oceanbdx/
cd /path/to/IsaacLab
./isaaclab.sh -p source/extensions/oceanbdx/scripts/play_keyboard_control.py \
    --checkpoint source/extensions/oceanbdx/logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt
```

## ⌨️ 键盘控制

| 按键 | 功能 | 命令值 [x, y, yaw] |
|------|------|-------------------|
| **W** | 向前移动 | [+0.5, 0, 0] |
| **S** | 向后移动 | [-0.5, 0, 0] |
| **A** | 向左平移 | [0, +0.3, 0] |
| **D** | 向右平移 | [0, -0.3, 0] |
| **Q** | 逆时针旋转 | [0, 0, +0.5] |
| **E** | 顺时针旋转 | [0, 0, -0.5] |
| **SPACE** | 停止 | [0, 0, 0] |
| **R** | 重置环境 | - |
| **ESC** | 退出程序 | - |

## 🔍 与 Gazebo 部署的对比

### Isaac Lab (训练环境)
- ✅ 使用 **PyTorch JIT** (`torch.jit.load`) 进行推理
- ✅ Python 环境,tensor 操作简单
- ✅ 观测构造在 Python 中,清晰可调试
- ✅ 数据类型: `torch.float32` 自动处理

### Gazebo (部署环境)
- ⚠️ 使用 **LibTorch C++** (`torch::jit::load`) 进行推理
- ⚠️ C++ 环境,tensor 构造需要注意
- ⚠️ 观测构造在 C++ 中,可能有类型/维度错误
- ⚠️ 数据类型: 需要显式指定 `torch::kFloat32`

## 🎯 调试目的

通过这个脚本,你可以:

1. **验证模型本身没问题** - 在 Isaac Lab 中模型应该输出正常
2. **对比 sim2sim 输出差异** - 如果 Isaac Lab 正常但 Gazebo 异常,问题在部署
3. **测试不同命令** - 通过键盘输入各种命令组合,观察机器人响应
4. **检查观测是否合理** - 每 100 步打印机器人位置和命令

## 📊 输出信息

脚本会每 100 步打印一次状态:
```
📊 Step    100 | Pos: [0.50, 0.00, 0.85] | Cmd: [0.50, 0.00, 0.00]
```
- **Pos**: 机器人基座位置 [x, y, z]
- **Cmd**: 当前速度命令 [lin_vel_x, lin_vel_y, ang_vel_z]

## ⚠️ 已知问题

### adaptive_phase 错误
- **训练环境**: [0.6667, 0.0, 0.37] (正确)
- **Gazebo部署**: [0.0, 0.0, 0.0] (错误)
- **影响**: 测试显示影响很小 (~0.2 动作差异)
- **结论**: **不是极端输出的主因**

### 真正的问题
根据之前的调试,极端输出 (±2) 的原因可能在于:
1. **LibTorch 模型加载** - `torch::jit::load()` 可能有问题
2. **Tensor 数据类型** - C++ 中 float32 vs float64 混用
3. **观测向量构造** - `at::Tensor` 创建方式不对
4. **模型推理调用** - `model.forward()` 参数传递错误

## 🔧 下一步调试

### 步骤 1: 在 Isaac Lab 中测试
```bash
./isaaclab.sh -p scripts/play_keyboard_control.py
```
按 **W** 键让机器人前进,观察:
- ✅ 机器人是否能正常行走?
- ✅ 动作是否在合理范围 (不是 ±2 极值)?
- ✅ 机器人是否会摔倒?

### 步骤 2: 对比 Gazebo 日志
如果 Isaac Lab 正常,Gazebo 异常,则:
1. 检查 **LibTorch C++** 的 `rl_sdk.cpp` 代码
2. 打印中间值:观测 tensor、输出 tensor
3. 对比数据类型、shape、数值范围
4. 验证 `torch::from_blob()` 的使用是否正确

### 步骤 3: 使用相同观测测试
在 C++ 中硬编码一个 Isaac Lab 中的观测向量,看 LibTorch 输出是否一致。

## 📝 注意事项

1. **环境数量**: 建议使用 `--num_envs 1` 方便观察单个机器人
2. **Checkpoint 路径**: 确保路径指向 `exported/policy.pt` (JIT 模型)
3. **观测维度**: OceanBDX 是 **74 维**, commands 在索引 **57:60**
4. **按键响应**: 按下按键后机器人会持续执行该命令,按其他键改变命令

## 💡 提示

- 如果机器人一直摔倒 → 模型训练有问题,或者环境配置不对
- 如果机器人不动 → 检查命令是否正确传递到观测中
- 如果动作跳变 → 可能是观测某些项错误 (但不是 adaptive_phase)
- 如果动作正常但 Gazebo 异常 → **问题在 LibTorch C++ 推理!**

## 🎯 预期结果

**正常情况** (Isaac Lab):
- 机器人能站立
- 按 W 能向前走
- 动作值在 [-1, 1] 范围内
- 不会突然摔倒

**异常情况** (当前 Gazebo):
- 首帧输出 10/14 个 ±2 极值
- 机器人站起后立即摔倒
- 动作完全不合理

**如果 Isaac Lab 正常** → 证明:
1. ✅ 模型本身没问题
2. ✅ adaptive_phase 错误影响很小
3. ❌ **问题出在 Gazebo 的 LibTorch 推理实现!**

---

**作者**: GitHub Copilot  
**日期**: 2025-11-05  
**用途**: sim2sim 推理调试,验证 LibTorch vs PyTorch 差异
