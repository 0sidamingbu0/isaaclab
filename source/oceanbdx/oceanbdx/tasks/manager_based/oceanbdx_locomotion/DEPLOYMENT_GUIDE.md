# OceanBDX 部署指南 - 观测数据处理

## 📋 概述

本文档说明如何在真实机器人上实现与训练环境一致的观测数据处理。

---

## 🎯 关键观测项处理

### 观测列表对照

| 部署名称 | 训练名称 | 维度 | 说明 |
|---------|---------|------|------|
| `ang_vel` | `base_ang_vel` | [3] | IMU角速度（陀螺仪） |
| `gravity_vec` | `projected_gravity` | [3] | 重力投影（无运动加速度） |
| `commands` | `velocity_commands` | [3] | 速度命令 [vx, vy, wz] |
| `dof_pos` | `joint_pos_rel` | [10] | 关节位置（相对默认姿态） |
| `dof_vel` | `joint_vel_rel` | [10] | 关节速度 |
| `actions` | `last_actions` | [10] | 上一次的动作 |
| `phase` | `gait_phase` | [6] | 🆕 步态相位编码 |

**总计：** 3+3+3+10+10+10+6 = **45维**

---

### 1. 重力投影 (gravity_vec / projected_gravity)

**训练环境：**
```python
# 使用robot姿态计算纯重力投影（无运动加速度）
projected_gravity = quat_rotate_inverse(robot.orientation, [0, 0, -9.81])
```

**部署实现（推荐方案）：**

#### 方法A：低通滤波（简单鲁棒）
```python
class GravityFilter:
    """低通滤波器提取重力分量"""
    def __init__(self, alpha=0.98):
        self.alpha = alpha  # 滤波系数 (0.95-0.99)
        self.gravity = np.array([0.0, 0.0, 9.81])
    
    def update(self, imu_acc):
        """
        输入: IMU原始加速度 (包含重力+运动加速度)
        输出: 滤波后的重力投影
        """
        self.gravity = self.alpha * self.gravity + (1 - self.alpha) * imu_acc
        return self.gravity

# 使用示例
gravity_filter = GravityFilter(alpha=0.98)

while True:
    imu_data = imu.read()
    observation['projected_gravity'] = gravity_filter.update(imu_data.linear_acceleration)
```

**调参建议：**
- `alpha=0.98`：适合50Hz IMU，响应时间~1秒
- `alpha=0.95`：更快响应（~0.5秒），但噪声稍大
- `alpha=0.99`：更平滑（~2秒），但响应慢

#### 方法B：使用IMU姿态估计（最准确）
```python
# 如果IMU芯片提供姿态估计（MPU6050/BNO055等）
imu_quat = imu.get_orientation()  # 读取IMU内部融合的姿态
gravity_world = np.array([0.0, 0.0, 9.81])
gravity_body = quat_rotate_inverse(imu_quat, gravity_world)
observation['projected_gravity'] = gravity_body
```

**四元数旋转实现：**
```python
def quat_rotate_inverse(quat, vec):
    """
    将向量从世界坐标系转到body坐标系
    quat: [w, x, y, z] 或 [x, y, z, w]（注意你的IMU格式）
    vec: [x, y, z]
    """
    # 假设quat格式为[w, x, y, z]
    w, x, y, z = quat
    
    # 共轭四元数（逆旋转）
    quat_conj = np.array([w, -x, -y, -z])
    
    # 四元数乘法: quat_conj * [0, vec] * quat
    vec_quat = np.array([0, vec[0], vec[1], vec[2]])
    result = quat_multiply(quat_multiply(quat_conj, vec_quat), quat)
    
    return result[1:]  # 返回[x, y, z]

def quat_multiply(q1, q2):
    """四元数乘法"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
```

---

### 2. 角速度 (base_ang_vel)

**训练环境：**
```python
base_ang_vel = imu_sensor.angular_velocity  # IMU陀螺仪输出
```

**部署实现：**
```python
# 直接使用IMU陀螺仪输出
observation['base_ang_vel'] = imu.angular_velocity  # [wx, wy, wz] (rad/s)
```

⚠️ **注意坐标系一致性**：确认你的IMU坐标系与训练环境一致。

---

### 3. 步态相位 (phase / gait_phase) 🆕

**训练环境：**
```python
# 多频率相位编码
motion_time = episode_length * dt
phase = 2π * motion_time / gait_period  # 0.75秒周期
phase_encoding = [sin(φ), cos(φ), sin(φ/2), cos(φ/2), sin(φ/4), cos(φ/4)]
```

**部署实现：**
```cpp
// C++ 实现（与你的代码一致）
float motion_time = episode_length_buf * dt * decimation;
float phase = M_PI * motion_time / 2.0;  // 4秒周期，如需0.75秒则改为: 2*M_PI*motion_time/0.75

std::vector<float> phase_obs = {
    std::sin(phase),
    std::cos(phase),
    std::sin(phase / 2.0),
    std::cos(phase / 2.0),
    std::sin(phase / 4.0),
    std::cos(phase / 4.0)
};
```

**Python 实现：**
```python
class PhaseEncoder:
    def __init__(self, gait_period=0.75):
        self.gait_period = gait_period
        self.episode_time = 0.0
    
    def update(self, dt):
        """每个控制步调用一次"""
        self.episode_time += dt
        phase = 2 * np.pi * self.episode_time / self.gait_period
        
        return np.array([
            np.sin(phase),
            np.cos(phase),
            np.sin(phase / 2),
            np.cos(phase / 2),
            np.sin(phase / 4),
            np.cos(phase / 4),
        ])
    
    def reset(self):
        """每个episode开始时调用"""
        self.episode_time = 0.0

# 使用
phase_encoder = PhaseEncoder(gait_period=0.75)
observation['phase'] = phase_encoder.update(dt=0.02)  # 50Hz控制
```

**重要说明：**
- ⚠️ **周期匹配**：确保部署的 `gait_period` 与训练一致（0.75秒）
- ⚠️ **时间重置**：每个episode开始时重置 `episode_time = 0`
- ✅ **多尺度信息**：
  - `phase`: 完整周期（0.75s）
  - `phase/2`: 半周期（0.375s）
  - `phase/4`: 四分之一周期（0.1875s）

---

### 4. 姿态四元数 ❌ 已移除

### 4. 姿态四元数 ❌ 已移除

**训练环境：**
```python
# 已从观测中移除，重力投影已包含姿态信息
```

**部署实现：**
```python
# 不需要四元数观测
# 重力投影 + 角速度已经提供足够的姿态信息
```

---

### 5. 关节状态

**训练环境：**
```python
joint_pos = robot.joint_positions
joint_vel = robot.joint_velocities
joint_torques = robot.applied_torques
```

**部署实现：**
```python
observation['joint_pos_rel'] = motor.get_positions()  # 相对默认位置
observation['joint_vel_rel'] = motor.get_velocities()
observation['joint_torques'] = motor.get_torques()
```

---

## 🔧 完整观测构建示例

```python
class ObservationBuilder:
    def __init__(self, gait_period=0.75):
        self.gravity_filter = GravityFilter(alpha=0.98)
        self.phase_encoder = PhaseEncoder(gait_period=gait_period)
        self.last_action = np.zeros(10)  # 10个腿部关节
    
    def get_observation(self, imu, motors, command, dt):
        """构建与训练环境一致的观测"""
        
        # 1. IMU数据
        imu_data = imu.read()
        projected_gravity = self.gravity_filter.update(imu_data.linear_acceleration)
        
        # 2. 相位编码
        phase_obs = self.phase_encoder.update(dt)
        
        obs = {
            # IMU观测
            'ang_vel': imu_data.angular_velocity,        # [3]
            'gravity_vec': projected_gravity,             # [3]
            
            # 关节状态
            'dof_pos': motors.get_positions(),            # [10]
            'dof_vel': motors.get_velocities(),           # [10]
            
            # 命令
            'commands': command,                          # [3]: [vx, vy, wz]
            
            # 历史动作
            'actions': self.last_action,                  # [10]
            
            # 🆕 相位信息
            'phase': phase_obs,                           # [6]
        }
        
        # 展平成一维数组（按顺序拼接）
        obs_flat = np.concatenate([
            obs['ang_vel'],        # 3
            obs['gravity_vec'],    # 3
            obs['commands'],       # 3
            obs['dof_pos'],        # 10
            obs['dof_vel'],        # 10
            obs['actions'],        # 10
            obs['phase'],          # 6
        ])  # 总计: 45维
        
        return obs_flat
    
    def update_last_action(self, action):
        self.last_action = action
    
    def reset(self):
        """Episode重置时调用"""
        self.phase_encoder.reset()
        self.last_action = np.zeros(10)
```

---

## ⚠️ 常见陷阱

### 1. 坐标系不一致
```python
# 训练环境：Z轴向上
# 真实IMU：可能Z轴向下

# 解决：检查并转换
if imu_z_down:
    imu_acc = imu_acc * np.array([1, 1, -1])
```

### 2. 单位不一致
```python
# 训练：弧度/秒
# 部署：度/秒

# 解决：统一转换
ang_vel_rad = np.deg2rad(ang_vel_deg)
```

### 3. 四元数格式
```python
# 训练：[w, x, y, z]
# IMU库：[x, y, z, w]

# 解决：重新排列
quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
```

### 4. 忘记归一化/去中心化
```python
# 如果训练时有归一化，部署也要做
joint_pos_normalized = (joint_pos - default_pos) / position_range
```

---

## 🧪 测试验证

### 1. 静态测试
```python
# 机器人静止直立时，检查观测值
assert np.allclose(projected_gravity, [0, 0, 9.81], atol=0.5)
assert np.allclose(base_ang_vel, [0, 0, 0], atol=0.1)
```

### 2. 动态测试
```python
# 轻微倾斜机器人，检查重力投影变化
# 前倾30度时，预期 projected_gravity ≈ [4.9, 0, 8.5]
```

### 3. 频率测试
```python
# 确保观测频率与训练一致（通常50Hz）
import time
start = time.time()
for _ in range(100):
    obs = obs_builder.get_observation(imu, motors, cmd)
duration = time.time() - start
freq = 100 / duration
print(f"Observation frequency: {freq:.1f} Hz")
```

---

## 📚 参考资料

- Isaac Lab IMU文档: https://isaac-sim.github.io/IsaacLab/
- 四元数旋转: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
- 低通滤波器设计: https://en.wikipedia.org/wiki/Low-pass_filter

---

## 🆘 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 机器人持续倾斜 | 重力投影计算错误 | 检查坐标系和四元数格式 |
| 动作抖动 | IMU噪声太大 | 增加滤波器alpha值 |
| 响应迟钝 | 滤波器太慢 | 降低alpha值到0.95 |
| 策略输出异常 | 观测值范围不对 | 检查单位和归一化 |

---

**最后更新：** 2025-10-21  
**维护者：** OceanBDX Team
