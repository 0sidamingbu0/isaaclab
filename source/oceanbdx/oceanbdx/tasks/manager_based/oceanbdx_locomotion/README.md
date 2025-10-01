# OceanBDX 双足机器人 Manager-Based 训练环境

这个目录包含了为 OceanBDX 双足机器人创建的 Isaac Lab manager-based 训练环境。该环境允许机器人学习基于输入指令 (x, y, yaw) 的运动控制。

## 环境结构

```
oceanbdx_locomotion/
├── config/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── rsl_rl_ppo_cfg.py          # PPO训练配置
│   ├── __init__.py
│   └── oceanbdx_locomotion_simple.py  # 环境配置
├── mdp/
│   ├── __init__.py
│   ├── actions.py                     # 动作管理
│   ├── commands.py                    # 指令生成
│   ├── events.py                      # 事件处理
│   ├── observations.py               # 观测定义
│   ├── rewards.py                    # 奖励函数
│   └── terminations.py               # 终止条件
└── __init__.py
```

## 环境特性

### 观测空间
- 基座线速度 (base_lin_vel)
- 基座角速度 (base_ang_vel)  
- 重力投影 (projected_gravity)
- 速度指令 (velocity_commands)
- 关节位置相对值 (joint_pos_rel)
- 关节速度 (joint_vel_rel)
- 上一动作 (last_actions)

### 动作空间
- 14个关节的位置控制 (10个腿部关节 + 4个颈部关节)
- 动作范围: [-1, 1] 经过缩放

### 指令空间  
- x 方向线速度: [-1.0, 1.5] m/s
- y 方向线速度: [-0.5, 0.5] m/s
- z 方向角速度: [-1.0, 1.0] rad/s

### 奖励函数
1. **跟踪奖励**: 鼓励跟随速度指令
   - x,y 线速度跟踪 (权重: 1.5)
   - z 角速度跟踪 (权重: 0.75)

2. **稳定奖励**: 
   - 基座高度保持 (目标: 0.4m, 权重: -0.5)
   - 平坦姿态保持 (权重: -1.0)
   - 存活奖励 (权重: 1.0)

3. **惩罚项**:
   - z 方向线速度 (权重: -2.0)
   - x,y 方向角速度 (权重: -0.05)
   - 关节扭矩 (权重: -5e-5)
   - 动作变化率 (权重: -0.01)
   - 关节位置偏差 (权重: -0.01)

### 终止条件
- 基座高度超出范围 [0.15, 1.0]m
- 基座姿态超过90度
- 基座线速度超过10m/s
- 关节位置或速度超限
- 回合时长达到上限 (20秒)

## 训练配置

### PPO 参数
- 隐藏层: [512, 256, 128]
- 学习率: 1e-3
- 回合长度: 24 steps
- 最大迭代次数: 2000
- 环境数量: 4096
- 激活函数: ELU

### 仿真参数
- 物理时间步: 0.005s (200 Hz)
- 控制频率: 0.01s (100 Hz, decimation=2)  
- 回合时长: 20 秒

## 使用方法

### 训练
```bash
# 基本训练
python scripts/train_oceanbdx_locomotion.py

# 指定环境数量
python scripts/train_oceanbdx_locomotion.py --num_envs 2048

# 录制训练视频
python scripts/train_oceanbdx_locomotion.py --video --video_interval 1000

# 自定义训练迭代次数
python scripts/train_oceanbdx_locomotion.py --max_iterations 5000
```

### 测试训练好的策略
```bash
# 基本测试
python scripts/play_oceanbdx_locomotion.py --checkpoint logs/rsl_rl/oceanbdx_locomotion/YYYY-MM-DD_HH-MM-SS/model_XXXX.pt

# 指定环境数量
python scripts/play_oceanbdx_locomotion.py --checkpoint /path/to/checkpoint --num_envs 16
```

### 使用 Isaac Lab 标准脚本
```bash
# 训练
python scripts/rsl_rl/train.py --task Isaac-Ocean-BDX-Locomotion-v0 --num_envs 4096

# 测试  
python scripts/rsl_rl/play.py --task Isaac-Ocean-BDX-Locomotion-Play-v0 --checkpoint /path/to/checkpoint
```

## 环境注册

环境已注册为:
- `Isaac-Ocean-BDX-Locomotion-v0`: 训练环境 (4096个环境)
- `Isaac-Ocean-BDX-Locomotion-Play-v0`: 测试环境 (50个环境)

## 机器人配置

机器人使用 `oceanbdx.assets.oceanusd.OCEAN_ROBOT_CFG` 配置，包含:
- 10个腿部关节 (每条腿5个关节)
- 4个颈部关节
- IMU传感器
- 接触力传感器
- 合适的物理属性和初始姿态

## 开发说明

这个环境基于 Isaac Lab 的 manager-based 架构，具有:
- 模块化的 MDP 组件设计
- 灵活的奖励函数配置
- 可扩展的观测和动作空间
- 完整的事件处理系统

如需修改环境行为，可以:
1. 调整 `oceanbdx_locomotion_simple.py` 中的配置参数
2. 修改 `mdp/` 目录下的各个模块
3. 更新 `rsl_rl_ppo_cfg.py` 中的训练超参数