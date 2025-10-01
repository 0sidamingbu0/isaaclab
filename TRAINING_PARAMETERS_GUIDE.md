# 🤖 OceanBDX 双足机器人训练参数调整指南

## 📋 目录
1. [命令行参数](#命令行参数)
2. [环境配置参数](#环境配置参数)
3. [机器人物理参数](#机器人物理参数)
4. [PPO算法参数](#ppo算法参数)
5. [神经网络参数](#神经网络参数)
6. [奖励函数参数](#奖励函数参数)
7. [终止条件参数](#终止条件参数)
8. [命令参数](#命令参数)

---

## 🖥️ 命令行参数

```bash
python scripts/rsl_rl/train.py --task Isaac-Ocean-BDX-Locomotion-v0 [参数]
```

| 参数 | 默认值 | 作用 | 建议调整 |
|------|--------|------|----------|
| `--num_envs` | 4096 | 并行环境数量 | 根据GPU内存调整：RTX 4090→4096, RTX 3080→2048 |
| `--max_iterations` | 2000 | 最大训练迭代数 | 增加到5000-10000获得更好效果 |
| `--seed` | None | 随机种子 | 设置固定值（如42）确保结果可重现 |
| `--video` | False | 录制训练视频 | 开启后会降低训练速度但便于分析 |
| `--video_interval` | 2000 | 视频录制间隔(步) | **重要**: 500-2000, 太小会严重影响性能 |
| `--video_length` | 200 | 每段视频长度(步) | 100-500步，根据需要调整 |
| `--headless` | False | 无界面模式 | 录制视频时建议开启，提高性能 |

---

## 🌍 环境配置参数
**文件位置**: `config/oceanbdx_locomotion_simple.py`

### 🎬 场景配置
```python
class OceanBDXLocomotionSceneCfg:
    # 地面配置
    terrain.physics_material.static_friction = 1.0    # 静摩擦系数：0.5-2.0
    terrain.physics_material.dynamic_friction = 1.0   # 动摩擦系数：0.5-2.0
    terrain.physics_material.restitution = 0.0        # 弹性系数：0.0-1.0
```

### 🔧 基础环境设置
```python
class OceanBDXLocomotionEnvCfg:
    # 环境数量和间距
    scene.num_envs = 4096              # 并行环境数：512-8192
    scene.env_spacing = 2.5            # 环境间距(m)：2.0-5.0
    
    # 时间设置
    episode_length_s = 20.0            # 回合长度(秒)：10-30
    decimation = 2                     # 控制频率分频：1-4
    
    # 物理设置
    sim.dt = 0.005                     # 物理步长(秒)：0.001-0.01
    sim.render_interval = 2            # 渲染间隔：1-4
```

---

## 🤖 机器人物理参数
**文件位置**: `assets/oceanusd/__init__.py`

### ⚙️ 关节驱动器参数
```python
actuators = {
    "legs": DCMotorCfg(
        effort_limit=100.0,        # 最大力矩(Nm)：50-200
        saturation_effort=90.0,    # 饱和力矩(Nm)：45-180
        velocity_limit=50.0,       # 最大速度(rad/s)：20-100
        stiffness=60.0,           # 刚度系数：20-100
        damping=1.5,              # 阻尼系数：0.5-5.0
        friction=0.8,             # 摩擦系数：0.1-2.0
    ),
    "neck": DCMotorCfg(
        effort_limit=10.0,        # 颈部力矩限制：5-20
        stiffness=8.0,            # 颈部刚度：3-15
        damping=2.0,              # 颈部阻尼：1.0-5.0
    ),
}
```

### 🔩 刚体属性
```python
rigid_props = sim_utils.RigidBodyPropertiesCfg(
    max_depenetration_velocity=1.0,   # 去穿透速度：0.5-3.0
    max_linear_velocity=50.0,         # 最大线速度：20-100
    max_angular_velocity=50.0,        # 最大角速度：20-100
)
```

### 🦴 关节属性
```python
articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    solver_position_iteration_count=8,  # 位置求解器迭代：4-16
    solver_velocity_iteration_count=0,  # 速度求解器迭代：0-8
    enabled_self_collisions=True,       # 自碰撞检测：True/False
)
```

---

## 🧠 PPO算法参数
**文件位置**: `config/agents/rsl_rl_ppo_cfg.py`

### 🎯 训练参数
```python
class OceanBDXPPORunnerCfg:
    num_steps_per_env = 24           # 每环境步数：16-48
    max_iterations = 2000            # 最大迭代：2000-10000
    save_interval = 50               # 保存间隔：25-100
```

### 🔄 PPO算法核心参数
```python
algorithm = RslRlPpoAlgorithmCfg(
    # 学习率设置
    learning_rate = 1.0e-3           # 学习率：1e-4到1e-2
    schedule = "adaptive"            # 学习率调度："adaptive"/"linear"
    
    # PPO特有参数
    clip_param = 0.2                 # 裁剪参数：0.1-0.3
    entropy_coef = 0.01              # 熵系数：0.001-0.1
    value_loss_coef = 1.0            # 价值损失系数：0.5-2.0
    
    # 训练批次
    num_learning_epochs = 5          # 学习轮数：3-10
    num_mini_batches = 4             # 小批次数：2-8
    
    # 其他重要参数
    gamma = 0.99                     # 折扣因子：0.95-0.999
    lam = 0.95                       # GAE参数：0.9-0.98
    desired_kl = 0.01                # 目标KL散度：0.005-0.02
    max_grad_norm = 1.0              # 梯度裁剪：0.5-2.0
)
```

---

## 🧮 神经网络参数
**文件位置**: `config/agents/rsl_rl_ppo_cfg.py`

```python
policy = RslRlPpoActorCriticCfg(
    # 网络结构
    actor_hidden_dims = [512, 256, 128]     # 演员网络层：[256,128] 到 [1024,512,256]
    critic_hidden_dims = [512, 256, 128]    # 评论家网络层：[256,128] 到 [1024,512,256]
    activation = "elu"                       # 激活函数："elu"/"relu"/"tanh"
    
    # 初始化
    init_noise_std = 1.0                     # 初始噪声标准差：0.1-2.0
    
    # 归一化
    actor_obs_normalization = False          # 演员观测归一化：True/False
    critic_obs_normalization = False         # 评论家观测归一化：True/False
)
```

---

## 🎯 奖励函数参数
**文件位置**: `config/oceanbdx_locomotion_simple.py`

```python
@configclass
class RewardsCfg:
    # 🎯 速度跟踪奖励
    track_lin_vel_xy_exp = RewTerm(
        weight=1.5,                    # 权重：0.5-3.0
        params={"std": math.sqrt(0.25)}  # 标准差：0.3-1.0
    )
    track_ang_vel_z_exp = RewTerm(
        weight=0.75,                   # 权重：0.25-1.5
    )
    
    # 📐 姿态惩罚
    flat_orientation_l2 = RewTerm(weight=-1.0)    # 姿态惩罚：-0.5到-2.0
    base_height_l2 = RewTerm(
        weight=-0.5,                   # 高度惩罚：-0.1到-1.0
        params={"target_height": 0.4}  # 目标高度：0.3-0.5m
    )
    
    # ⚡ 能耗惩罚
    dof_torques_l2 = RewTerm(weight=-5.0e-5)      # 力矩惩罚：-1e-5到-1e-4
    action_rate_l2 = RewTerm(weight=-0.01)        # 动作变化率：-0.005到-0.02
    
    # 🚫 不良行为惩罚
    lin_vel_z_l2 = RewTerm(weight=-2.0)           # Z轴速度惩罚：-1.0到-5.0
    ang_vel_xy_l2 = RewTerm(weight=-0.05)         # XY轴角速度：-0.01到-0.1
    
    # 🎁 生存奖励
    is_alive = RewTerm(weight=1.0)                # 生存奖励：0.5-2.0
```

---

## ⛔ 终止条件参数
**文件位置**: `config/oceanbdx_locomotion_simple.py`

```python
@configclass
class TerminationsCfg:
    # 高度限制
    base_height = DoneTerm(
        params={
            "minimum_height": 0.15,        # 最低高度：0.1-0.25m
            "maximum_height": 1.0          # 最高高度：0.8-1.5m
        }
    )
    
    # 姿态限制
    bad_orientation = DoneTerm(
        params={"limit_angle": math.pi/2}  # 最大倾斜角：π/3到π/2
    )
    
    # 速度限制
    base_lin_vel = DoneTerm(
        params={"max_velocity": 10.0}      # 最大线速度：5.0-20.0m/s
    )
```

---

## 🎮 命令参数
**文件位置**: `config/oceanbdx_locomotion_simple.py`

```python
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(10.0, 10.0),    # 命令更新间隔：(5.0,15.0)
        rel_standing_envs=0.02,                 # 静止环境比例：0.0-0.1
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),             # X方向速度：(-2.0, 3.0)
            lin_vel_y=(-0.5, 0.5),             # Y方向速度：(-1.0, 1.0)
            ang_vel_z=(-1.0, 1.0),             # 偏航角速度：(-2.0, 2.0)
        ),
    )
```

---

## 🚀 动作控制参数
**文件位置**: `config/oceanbdx_locomotion_simple.py`

```python
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        scale=0.25,                    # 动作缩放：0.1-0.5
        use_default_offset=True        # 使用默认偏移：True/False
    )
```

---

## 📈 参数调优建议

### 🎯 提高训练稳定性
- 降低 `learning_rate` 到 `5e-4`
- 增加 `num_mini_batches` 到 `6-8`
- 调整 `clip_param` 到 `0.15`

### 🚀 提高训练速度
- 增加 `num_envs` 到GPU内存允许的最大值
- 降低 `num_learning_epochs` 到 `3-4`
- 增加 `learning_rate` 到 `3e-3`

### 🎪 改善locomotion性能
- 增加 `track_lin_vel_xy_exp.weight` 到 `2.0-3.0`
- 降低能耗惩罚权重
- 调整机器人驱动器 `stiffness` 和 `damping`

### 🔧 解决训练问题
- **机器人摔倒**：增加姿态奖励权重，降低速度命令范围
- **动作不平滑**：增加 `action_rate_l2` 权重
- **收敛慢**：检查奖励函数设计，调整学习率和网络结构

---

## 📝 使用示例

```bash
# 基础训练
python scripts/rsl_rl/train.py --task Isaac-Ocean-BDX-Locomotion-v0 --num_envs 2048

# 长时间训练with视频录制 
python scripts/rsl_rl/train.py --task Isaac-Ocean-BDX-Locomotion-v0 --num_envs 4096 --max_iterations 5000 --headless --video --video_interval 500

# 高质量视频录制（较低频率）
python scripts/rsl_rl/train.py --task Isaac-Ocean-BDX-Locomotion-v0 --num_envs 2048 --video --video_interval 1000 --video_length 300

# 调试训练（少环境，固定种子）
python scripts/rsl_rl/train.py --task Isaac-Ocean-BDX-Locomotion-v0 --num_envs 128 --seed 42

# 无头模式高性能训练
python scripts/rsl_rl/train.py --task Isaac-Ocean-BDX-Locomotion-v0 --num_envs 4096 --headless --max_iterations 8000
```

## 📹 视频录制注意事项

⚠️ **重要提醒**：
- `--video` 是布尔标志，不要加 `true/false`
- `--video_interval` 不要设置太小（<100），会严重影响训练性能
- 录制视频时建议使用 `--headless` 模式提高性能
- 视频文件保存在 `logs/rsl_rl/oceanbdx_locomotion/` 目录下

🎬 **最佳实践**：
```bash
# 推荐的视频录制命令
python scripts/rsl_rl/train.py --task Isaac-Ocean-BDX-Locomotion-v0 --num_envs 4096 --headless --video --video_interval 500 --video_length 200 --max_iterations 3000
```

记住：调整参数时建议每次只改变1-2个参数，观察效果后再进一步调整！🎯