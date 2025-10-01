# OceanBDX 机器人训练参数详细说明

## 奖励函数参数 (Reward Function Parameters)

### 1. 速度指令跟踪奖励
| 参数名称 | 权重 | 功能说明 | 参数设置 |
|---------|------|----------|----------|
| `track_lin_vel_xy_exp` | 1.5 | 跟踪xy方向的线速度指令，这是主要目标 | `std=0.5` |
| `track_ang_vel_z_exp` | 0.75 | 跟踪z轴角速度指令（转向），略低于线速度 | `std=0.5` |

**说明**: 这些是正奖励，鼓励机器人按照给定的x,y,yaw指令移动。

### 2. 非期望方向速度惩罚
| 参数名称 | 权重 | 功能说明 |
|---------|------|----------|
| `lin_vel_z_l2` | -2.0 | 惩罚垂直方向(z轴)线速度，防止机器人跳跃或下沉 |
| `ang_vel_xy_l2` | -0.05 | 惩罚roll和pitch角速度，保持机器人稳定不摇摆 |

**说明**: 这些惩罚确保机器人只在需要的方向移动，避免不必要的运动。

### 3. 机体姿态惩罚
| 参数名称 | 权重 | 功能说明 |
|---------|------|----------|
| `flat_orientation_l2` | -5.0 | 强烈惩罚机体姿态偏离直立状态，防止roll和pitch偏角 |
| `base_pitch_penalty` | -3.0 | 专门的俯仰角惩罚，防止前后弯腰 |

**说明**: 总计-8.0的姿态惩罚，确保机器人保持挺直姿态，这是解决"弯腰"问题的关键。

### 4. 能量消耗惩罚
| 参数名称 | 权重 | 功能说明 |
|---------|------|----------|
| `dof_torques_l2` | -5.0e-5 | 惩罚关节扭矩，鼓励机器人使用较小的力矩完成任务 |
| `action_rate_l2` | -0.01 | 惩罚动作变化率，鼓励平滑的运动，避免剧烈抖动 |

**说明**: 这些惩罚促进能效和平滑性，让机器人动作更自然。

### 5. 高度和存活奖励
| 参数名称 | 权重 | 功能说明 | 参数设置 |
|---------|------|----------|----------|
| `base_height_l2` | -0.5 | 惩罚偏离目标高度0.4m，鼓励机器人保持合适的站立高度 | `target_height=0.4` |
| `is_alive` | 1.0 | 只要机器人保持运行状态就给予奖励，鼓励稳定性 | - |

## 终止条件参数 (Termination Conditions)

### 1. 机体高度终止条件
```python
base_height = DoneTerm(
    func=mdp.base_height,
    params={"minimum_height": 0.15, "maximum_height": 1.0}
)
```
**说明**: 当机器人高度低于0.15m或高于1.0m时终止训练

### 2. 姿态终止条件
```python
bad_orientation = DoneTerm(
    func=mdp.bad_orientation,
    params={"limit_angle": math.pi / 4}  # 45度
)
```
**说明**: 当机器人倾斜角度超过45度时终止，防止跌倒

### 3. 高速度终止条件
```python
base_lin_vel = DoneTerm(
    func=mdp.base_lin_vel,
    params={"max_velocity": 10.0}
)
```
**说明**: 当线速度超过10.0 m/s时终止，避免过度失控

### 4. 超时终止条件
```python
time_out = DoneTerm(func=mdp.time_out, time_out=True)
```
**说明**: 当episode达到最大长度时正常终止

## 训练监控指标解读

### 关键指标说明
- **Mean reward**: 平均总奖励，正值且上升表示学习进展良好
- **Mean episode length**: 平均episode长度，应该接近最大值(8000步)
- **Episode_Termination/bad_orientation**: 因倾斜终止的比例，应该逐渐降低
- **Episode_Termination/time_out**: 正常完成的比例，应该逐渐增加

### 收敛判断标准
- ✅ **好的收敛迹象**: Mean reward > 50, bad_orientation < 0.1, time_out > 0.8
- ⚠️ **需要调整**: 如果reward下降或bad_orientation持续高于0.3

## 参数调整建议

### 如果机器人不稳定
- 增加 `flat_orientation_l2` 权重 (如-7.0)
- 增加 `base_pitch_penalty` 权重 (如-5.0)
- 降低 `track_lin_vel_xy_exp` 权重 (如1.0)

### 如果机器人过于保守
- 降低姿态惩罚权重
- 增加速度跟踪奖励权重
- 调整终止条件的角度限制

### 如果动作过于剧烈
- 增加 `action_rate_l2` 权重
- 增加 `dof_torques_l2` 权重
- 在PPO配置中降低学习率

## 视频录制参数说明

- `--video_interval 1000`: 每1000个环境交互步录制一次视频
- `--video_length 40000`: 每个视频长度40000步 = 200秒 = 3分20秒
- 建议调整: `--video_interval 100 --video_length 2000` (更频繁，更短的视频)