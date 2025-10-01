# OceanBDX 机器人部署输入数据说明

## 修改后的模型输入数据 (使用IMU原始数据)

**推荐配置**: 直接使用IMU原始数据，避免复杂的速度估计

### 新的输入向量维度分析
```
总观测维度: 56
```

### 具体输入数据组成

| 数据类型 | 维度 | 说明 | 直接对应IMU输出 |
|---------|------|------|----------------|
| `last_actions` | (14,) | 上一步的关节动作指令 | 电机控制器反馈 |
| `base_acceleration` | (3,) | **机体加速度 [ax, ay, az]** | ✅ **IMU加速度计直接输出** |
| `base_ang_vel` | (3,) | **机体角速度 [ωx, ωy, ωz]** | ✅ **IMU陀螺仪直接输出** |
| `projected_gravity` | (3,) | 重力在机体坐标系投影 | 从IMU四元数计算 |
| `velocity_commands` | (3,) | 速度指令 [vx_cmd, vy_cmd, ωz_cmd] | 上位机指令 |
| `joint_pos_rel` | (14,) | 关节相对位置 | 编码器读数 |
| `joint_vel_rel` | (14,) | 关节速度(缩放) | 编码器微分 |

**新配置总计: 56维** ✅

## 实际硬件部署优势

### ✅ 优势:
1. **无需速度估计**: 直接使用加速度计数据，避免积分误差
2. **实时性更好**: 不需要复杂的滤波和状态估计
3. **硬件对应性强**: 完全对应实际IMU输出
4. **部署简单**: 减少了数据处理的复杂度

### 🔄 数据映射关系:
```python
# 你的IMU输出 → 模型输入 (1:1对应)
imu_output = {
    "acceleration": [ax, ay, az],      # → base_acceleration
    "gyroscope": [wx, wy, wz],         # → base_ang_vel  
    "quaternion": [qw, qx, qy, qz]     # → projected_gravity (需计算)
}
```

## 完整版模型输入数据 (oceanbdx_locomotion_env_cfg.py)

如果使用完整配置，模型输入包括：

### IMU 数据部分
| 数据类型 | 维度 | 说明 | 对应IMU传感器 |
|---------|------|------|---------------|
| `base_lin_vel` | (3,) | 机体线速度 [vx, vy, vz] | IMU加速度计积分 |
| `base_ang_vel` | (3,) | 机体角速度 [ωx, ωy, ωz] | IMU陀螺仪 |
| `projected_gravity` | (3,) | 重力在机体坐标系投影 | IMU姿态估计 |

### 命令和关节数据
| 数据类型 | 维度 | 说明 | 获取方式 |
|---------|------|------|----------|
| `velocity_commands` | (3,) | 速度指令 [vx_cmd, vy_cmd, ωz_cmd] | 上位机指令 |
| `last_actions` | (14,) | 上一步关节动作 | 电机控制器反馈 |
| `joint_pos_rel` | (14,) | 关节相对位置 | 编码器读数 |
| `joint_vel_rel` | (14,) | 关节速度(缩放) | 编码器微分 |

**完整版总计: 54维 (3+3+3+3+14+14+14)**

## 实际硬件部署需求

### 1. 必需的IMU传感器数据
```python
# IMU应提供的原始数据
imu_data = {
    "linear_acceleration": [ax, ay, az],    # m/s² (加速度计)
    "angular_velocity": [wx, wy, wz],       # rad/s (陀螺仪)
    "orientation": [qw, qx, qy, qz]         # 四元数 (姿态估计)
}
```

### 2. 关节编码器数据
```python
# 14个关节的编码器反馈
joint_data = {
    # 左腿 (5个关节)
    "leg_l1_joint": {"position": 0.0, "velocity": 0.0},  # hip yaw
    "leg_l2_joint": {"position": 0.0, "velocity": 0.0},  # hip pitch  
    "leg_l3_joint": {"position": 0.0, "velocity": 0.0},  # knee pitch
    "leg_l4_joint": {"position": 0.0, "velocity": 0.0},  # ankle pitch
    "leg_l5_joint": {"position": 0.0, "velocity": 0.0},  # ankle roll
    
    # 右腿 (5个关节)
    "leg_r1_joint": {"position": 0.0, "velocity": 0.0},  # hip yaw
    "leg_r2_joint": {"position": 0.0, "velocity": 0.0},  # hip pitch
    "leg_r3_joint": {"position": 0.0, "velocity": 0.0},  # knee pitch  
    "leg_r4_joint": {"position": 0.0, "velocity": 0.0},  # ankle pitch
    "leg_r5_joint": {"position": 0.0, "velocity": 0.0},  # ankle roll
    
    # 颈部 (4个关节)
    "neck_n1_joint": {"position": 0.0, "velocity": 0.0},  # neck yaw
    "neck_n2_joint": {"position": 0.0, "velocity": 0.0},  # neck pitch
    "neck_n3_joint": {"position": 0.0, "velocity": 0.0},  # neck roll
    "neck_n4_joint": {"position": 0.0, "velocity": 0.0},  # head tilt
}
```

## 数据预处理函数

### IMU数据处理
```python
import numpy as np

def quaternion_to_projected_gravity(quat_wxyz):
    """从四元数计算重力在机体坐标系中的投影"""
    qw, qx, qy, qz = quat_wxyz
    
    # 世界坐标系中的重力向量 [0, 0, -9.81]
    gravity_world = np.array([0.0, 0.0, -9.81])
    
    # 四元数到旋转矩阵 (世界到机体)
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    # 重力向量转换到机体坐标系
    projected_gravity = R @ gravity_world
    return projected_gravity

def process_imu_data(imu_raw):
    """处理IMU原始数据为模型输入格式"""
    
    # 你的IMU输出格式:
    # imu_raw = {
    #     "acceleration": [ax, ay, az],    # 加速度计 m/s²
    #     "gyroscope": [wx, wy, wz],       # 陀螺仪 rad/s
    #     "quaternion": [qw, qx, qy, qz]   # 姿态四元数
    # }
    
    # 1. 角速度直接使用陀螺仪数据
    base_ang_vel = np.array(imu_raw["gyroscope"])
    
    # 2. 从四元数计算重力投影 (不是四元数本身!)
    projected_gravity = quaternion_to_projected_gravity(imu_raw["quaternion"])
    
    # 3. 线速度估计 (需要积分加速度或使用其他方法)
    base_lin_vel = estimate_velocity_from_imu(imu_raw)
    
    return base_lin_vel, base_ang_vel, projected_gravity
```

### 关节数据处理
```python
def process_joint_data(joint_positions, joint_velocities, default_positions):
    """处理关节数据为模型输入格式"""
    # 相对位置
    joint_pos_rel = np.array(joint_positions) - np.array(default_positions)
    
    # 速度缩放
    joint_vel_scaled = np.array(joint_velocities) * 0.05
    
    return joint_pos_rel, joint_vel_scaled
```

## 部署示例代码

```python
def create_model_input(imu_data, joint_data, velocity_cmd, last_action):
    """创建完整的模型输入向量"""
    
    # 处理IMU数据
    base_lin_vel, base_ang_vel, projected_gravity = process_imu_data(imu_data)
    
    # 处理关节数据
    joint_pos_rel, joint_vel_scaled = process_joint_data(
        joint_data["positions"], 
        joint_data["velocities"],
        DEFAULT_JOINT_POSITIONS
    )
    
    # 拼接所有观测数据
    observation = np.concatenate([
        last_action,        # (14,)
        base_lin_vel,       # (3,) 
        base_ang_vel,       # (3,)
        projected_gravity,  # (3,)
        velocity_cmd,       # (3,)
        joint_pos_rel,      # (14,)
        joint_vel_scaled    # (14,)
    ])  # 总计: 54维
    
    return observation
```

## 建议升级到完整输入

当前的简化输入(42维)缺少关键的IMU数据，建议升级观测配置以包含：
1. **IMU数据** - 提供机体状态感知
2. **速度指令** - 提供任务目标信息  
3. **更好的状态估计** - 提高控制精度

这将使模型在实际部署中表现更好！