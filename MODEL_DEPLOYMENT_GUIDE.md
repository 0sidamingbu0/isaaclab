# 🤖 OceanBDX模型部署指南

## 🚨 重要警告: 观测维度必须严格匹配!

**训练模型观测维度: 74维**

如果你的部署代码观测不是74维,模型将**无法正常工作**!

### 🔍 快速诊断: 你的部署观测是69维?

如果你当前的观测是**69维**,说明存在以下问题:

| 问题 | 你的配置(69维) | 应该是(74维) | 修改方法 |
|------|--------------|-------------|----------|
| ❌ 多了4维 | 包含 `quaternion` | 应删除 | 只保留 `gravity_vec` |
| ❌ 少了9维 | 缺少 `adaptive_phase` | 必须添加 | 见下方AdaptivePhaseManager代码 |

### 常见错误清单:

- ❌ **错误1**: 包含了 `quaternion` (4维) → 应该删除,训练时已用 `projected_gravity` 替代
- ❌ **错误2**: 缺少 `adaptive_phase` (9维) → 必须添加 AdaptivePhaseManager 生成步态相位
- ❌ **错误3**: 观测顺序错误 → 必须严格按照训练时的顺序拼接

### ✅ 正确的观测顺序 (74维):

```
1. ang_vel_body        (3维)  ← IMU陀螺仪
2. gravity_vec         (3维)  ← 重力投影 (不要四元数!)
3. dof_pos             (14维) ← 关节位置
4. dof_vel             (14维) ← 关节速度
5. joint_torques       (14维) ← 关节转矩
6. commands            (3维)  ← 速度命令
7. actions             (14维) ← 上一步动作
8. adaptive_phase      (9维)  ← 步态相位 (新增!)
────────────────────────────
   总计: 74维 ✅
```

---

## �📋 新旧模型观测数据对比

### ✅ 当前训练模型 (Iter 7405 最佳)

训练日期: 2025-10-31
最佳checkpoint: `model_7500.pt` (推荐使用 iter 7000-7500)

#### 观测空间维度计算

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `base_ang_vel` | **3** | IMU陀螺仪角速度 (roll_rate, pitch_rate, yaw_rate) |
| `projected_gravity` | **3** | 重力投影 (gx, gy, gz) |
| `joint_pos_rel` | **14** | 关节相对位置 (10腿+4颈) |
| `joint_vel_rel` | **14** | 关节速度 (10腿+4颈) |
| `joint_torques` | **14** | 关节转矩反馈 (10腿+4颈) |
| `velocity_commands` | **3** | 速度命令 (vx, vy, wz) |
| `last_actions` | **14** | 上一步动作 (10腿+4颈) |
| `adaptive_phase` | **9** | 自适应步态相位 (详见下文) |
| **总计** | **74维** | |

#### adaptive_phase 详细结构 (9维)

```python
[
    sin(theta),                # 1.0x频率正弦  (维度1)
    cos(theta),                # 1.0x频率余弦  (维度2)
    sin(theta/2),              # 0.5x频率正弦  (维度3)
    cos(theta/2),              # 0.5x频率余弦  (维度4)
    sin(theta/4),              # 0.25x频率正弦 (维度5)
    cos(theta/4),              # 0.25x频率余弦 (维度6)
    phase_rate,                # 归一化步频 (1/period, max=2.0Hz) (维度7)
    desired_stride,            # 归一化期望步幅 (max=0.5m) (维度8)
    desired_clearance          # 归一化抬脚高度 (max=0.1m) (维度9)
]
# 其中 theta = π * motion_time / 2.0
```

**关键特征**:
- ✅ 速度自适应: 速度越快,phase_rate越高,步幅越大
- ✅ 多频率编码: 提供更丰富的周期信息
- ✅ 显式步态参数: 防止振动作弊

---

## 🆕 与旧模型的主要差异

### 1. **观测空间变化**

#### 新增观测项:
- ✅ `joint_torques` (14维) - 关节转矩反馈
- ✅ `adaptive_phase` (9维) - 自适应步态相位

#### 移除的观测项:
- ❌ `base_quat_w` - 四元数(部署不需要)
- ❌ `height_scan` - 高度扫描(无传感器)

#### 修改的观测项:
- 🔧 `projected_gravity` - 改用纯重力投影(不含加速度冲击)
- 🔧 `base_ang_vel` - 改用IMU陀螺仪数据(训练部署一致)

### 2. **观测维度对比**

| 模型 | 总维度 | 主要差异 |
|------|--------|----------|
| **旧模型** | ~55维 | 无torque反馈,无相位观测 |
| **新模型** | **74维** | +14(torque) +9(phase) -4(quat) |

---

## ⚠️ 常见部署错误: 维度不匹配

### 错误配置示例 (69维 - 错误!)

```python
# ❌ 错误: 这是69维,会导致模型推理失败!
observation = np.concatenate([
    ang_vel_body,      # 3维
    gravity_vec,       # 3维
    quaternion,        # ❌ 4维 - 训练时已移除!
    dof_pos,           # 14维
    dof_vel,           # 14维
    joint_torques,     # 14维
    commands,          # 3维
    actions,           # 14维
    # ❌ 缺少 adaptive_phase (9维)!
])
# 总计: 3+3+4+14+14+14+3+14 = 69维 ❌
```

### ✅ 正确配置 (74维)

```python
# ✅ 正确: 74维,与训练完全匹配
observation = np.concatenate([
    ang_vel_body,           # 3维 - IMU角速度
    gravity_vec,            # 3维 - 重力投影 (不要再加四元数!)
    # quaternion 已删除!     # 训练时移除了四元数
    dof_pos,                # 14维 - 关节位置
    dof_vel,                # 14维 - 关节速度
    joint_torques,          # 14维 - 关节转矩
    commands,               # 3维 - 速度命令
    actions,                # 14维 - 上一步动作
    adaptive_phase,         # ✅ 9维 - 步态相位 (必须添加!)
])
# 总计: 3+3+14+14+14+3+14+9 = 74维 ✅
```

**关键修改**:
1. ❌ **删除** `quaternion` (4维) - 训练时已用`projected_gravity`替代
2. ✅ **添加** `adaptive_phase` (9维) - 使用 AdaptivePhaseManager 生成

---

## 🚀 真机部署注意事项

### 📡 **1. IMU数据处理**

#### 角速度 (base_ang_vel)
```python
# 直接使用IMU陀螺仪输出
base_ang_vel = imu.get_angular_velocity()  # [roll_rate, pitch_rate, yaw_rate]
# 单位: rad/s
# 坐标系: OceanBDX body frame (X前, Y左, Z上)
```

#### 重力投影 (projected_gravity)
```python
# 方法1: 使用IMU姿态计算 (推荐)
quat = imu.get_quaternion()  # [w, x, y, z]
gravity_world = [0, 0, 9.81]
projected_gravity = rotate_vector_by_quaternion(gravity_world, quat)

# 方法2: 使用加速度计+低通滤波
accel_raw = imu.get_acceleration()
projected_gravity = low_pass_filter(accel_raw, cutoff=5Hz)  # 滤除运动加速度
```

**关键**: 
- ✅ 训练用的是纯重力投影(无运动加速度)
- ✅ 部署时必须滤除运动冲击,只保留重力分量

### ⚙️ **2. 关节数据**

#### 关节顺序 (必须严格匹配!)
```python
joint_order = [
    # 左腿 (5个)
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint",
    "leg_l4_joint", "leg_l5_joint",
    # 右腿 (5个)
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint",
    "leg_r4_joint", "leg_r5_joint",
    # 颈部 (4个)
    "neck_n1_joint", "neck_n2_joint",
    "neck_n3_joint", "neck_n4_joint"
]
```

#### 关节位置 (joint_pos_rel)
```python
# 相对于默认位置的偏移
joint_pos_rel = current_position - default_position

# default_position 需要从URDF/配置中读取
# 通常是机器人直立时的关节角度
```

#### 关节转矩 (joint_torques) - 🆕 新增!
```python
# 读取电机反馈的实际输出转矩
joint_torques = motor_controller.get_torque_feedback()

# 如果硬件不支持转矩反馈,可以用估计值:
joint_torques = Kp * (target_pos - current_pos) + Kd * (0 - current_vel)
```

### 🏃 **3. 自适应步态相位 (adaptive_phase)** - 🆕 核心!

这是新模型最重要的新增功能!

#### 部署代码示例:
```python
import numpy as np
import time

class AdaptivePhaseManager:
    def __init__(self):
        self.phase = 0.0  # 当前相位 [0, 1)
        self.last_time = time.time()
        
        # Disney BDX参考参数
        self.ref_velocity = 0.35  # m/s
        self.ref_period = 0.75    # s
        self.ref_stride = 0.131   # m
        self.ref_clearance = 0.037  # m
    
    def update(self, velocity_command):
        """根据速度命令更新相位"""
        # 计算当前速度(取前向速度为主)
        current_speed = abs(velocity_command[0])  # vx
        
        # 根据速度动态调整步态周期 (速度越快,周期越短)
        if current_speed < 0.1:
            period = 1.0  # 慢速/站立
        else:
            period = self.ref_period * (self.ref_velocity / current_speed) ** 0.5
            period = np.clip(period, 0.5, 1.5)  # 限制在合理范围
        
        # 更新相位
        dt = time.time() - self.last_time
        self.last_time = time.time()
        
        self.phase += dt / period
        self.phase = self.phase % 1.0  # 保持在[0,1)
        
        return self.get_observation(current_speed, period)
    
    def get_observation(self, speed, period):
        """生成9维相位观测 (与训练完全一致!)"""
        # 计算motion_time (累积时间)
        # 注意: 实际部署时可能需要维护一个累积计数器
        motion_time = self.phase * period  # 简化版本
        
        # 计算theta (与训练代码完全一致!)
        theta = np.pi * motion_time / 2.0
        
        # 1-6: 多频率sin/cos编码 (频率: 1.0x, 0.5x, 0.25x)
        sin_1x = np.sin(theta)
        cos_1x = np.cos(theta)
        sin_half = np.sin(theta / 2.0)
        cos_half = np.cos(theta / 2.0)
        sin_quarter = np.sin(theta / 4.0)
        cos_quarter = np.cos(theta / 4.0)
        
        # 7: 归一化步频 (phase_rate = 1/period)
        phase_rate = (1.0 / period) / 2.0  # 归一化 (max=2.0Hz)
        
        # 8: 归一化期望步幅 (两步距离)
        desired_stride = self.interpolate_stride(speed)
        stride_norm = np.clip(desired_stride / 0.5, 0.0, 1.0)  # max=0.5m
        
        # 9: 归一化抬脚高度
        desired_clearance = self.interpolate_clearance(speed)
        clearance_norm = np.clip(desired_clearance / 0.1, 0.0, 1.0)  # max=0.1m
        
        return np.array([
            sin_1x, cos_1x, sin_half, cos_half, sin_quarter, cos_quarter,
            phase_rate, stride_norm, clearance_norm
        ], dtype=np.float32)
    
    def interpolate_stride(self, speed):
        """根据速度插值步幅 (参考training_curriculum.py)"""
        # 速度-步幅映射表
        speed_points = [0.0, 0.1, 0.25, 0.35, 0.5, 0.6, 0.74]
        stride_points = [0.0, 0.08, 0.2, 0.262, 0.325, 0.36, 0.37]
        return np.interp(speed, speed_points, stride_points)
    
    def interpolate_clearance(self, speed):
        """根据速度插值抬脚高度"""
        speed_points = [0.0, 0.1, 0.25, 0.35, 0.5, 0.6, 0.74]
        clearance_points = [0.0, 0.025, 0.03, 0.037, 0.045, 0.055, 0.07]
        return np.interp(speed, speed_points, clearance_points)
```

#### 使用示例:
```python
phase_manager = AdaptivePhaseManager()

# 在控制循环中
while True:
    # 1. 获取速度命令
    velocity_cmd = get_velocity_command()  # [vx, vy, wz]
    
    # 2. 更新相位管理器
    phase_obs = phase_manager.update(velocity_cmd)  # 9维
    
    # 3. 构建完整观测
    obs = np.concatenate([
        imu.get_angular_velocity(),      # 3维
        compute_projected_gravity(),      # 3维
        get_joint_pos_rel(),              # 14维
        get_joint_vel_rel(),              # 14维
        get_joint_torques(),              # 14维
        velocity_cmd,                     # 3维
        last_action,                      # 14维
        phase_obs                         # 9维
    ])  # 总计74维
    
    # 4. 模型推理
    action = model.predict(obs)
    
    # 5. 执行动作
    execute_action(action)
```

---

## 🎯 关键部署检查清单

### ✅ **数据一致性检查**

- [ ] IMU坐标系与训练一致 (X前, Y左, Z上)
- [ ] 关节顺序完全匹配 (10腿+4颈)
- [ ] 重力投影是纯重力(不含运动加速度)
- [ ] 关节位置是相对偏移(不是绝对角度)
- [ ] 速度命令单位是 m/s 和 rad/s
- [ ] 观测维度严格是74维

### ✅ **相位管理器检查**

- [ ] 相位更新频率与控制频率一致 (建议50Hz)
- [ ] 速度-周期映射关系合理
- [ ] 相位在[0,1)循环
- [ ] 多频率编码正确计算
- [ ] 归一化参数与训练一致

### ✅ **性能验证**

- [ ] 观测计算延迟 < 2ms
- [ ] 相位更新延迟 < 0.5ms
- [ ] 模型推理延迟 < 5ms
- [ ] 总控制周期 < 20ms (50Hz)

---

## 🔧 常见问题排查

### 问题1: 机器人抖动或振动

**可能原因**:
- IMU数据未滤波,包含运动加速度噪声
- 关节转矩反馈有噪声

**解决方案**:
```python
# 对projected_gravity应用低通滤波
gravity_filtered = low_pass_filter(accel_raw, cutoff=5Hz)

# 对joint_torques应用滑动平均
torque_filtered = moving_average(torque_raw, window=3)
```

### 问题2: 步态不稳定

**可能原因**:
- 相位更新频率不对
- 速度命令突变

**解决方案**:
```python
# 确保相位更新与控制循环同步
phase_manager.update(velocity_cmd)  # 每个控制周期调用一次

# 对速度命令应用平滑
velocity_cmd_smooth = smooth_command(velocity_cmd, alpha=0.9)
```

### 问题3: 前倾或后仰

**可能原因**:
- IMU安装方向错误
- 重力投影坐标系不对

**解决方案**:
```python
# 检查IMU坐标系
# OceanBDX: X前, Y左, Z上
# 直立时重力应该是: [0, 0, ~9.81]

# 如果坐标系相反,需要转换
if imu_x_backward:
    base_ang_vel[0] *= -1  # 反转roll rate
    projected_gravity[0] *= -1  # 反转gx
```

---

## 📊 最佳checkpoint推荐

基于训练曲线分析:

| Checkpoint | Iteration | 总奖励 | Episode长度 | 推荐等级 |
|------------|-----------|--------|-------------|----------|
| model_7000.pt | 7000 | +65 | 1650步 | ⭐⭐⭐⭐ |
| **model_7500.pt** | **7500** | **+67** | **1670步** | ⭐⭐⭐⭐⭐ **最佳** |
| model_8000.pt | 8000 | +60 | 1600步 | ⭐⭐⭐ |
| model_9000.pt | 9000 | -10 | 1400步 | ⭐ 不推荐 |

**建议**: 使用 `model_7500.pt` 进行部署测试

---

## 🚀 快速部署代码模板

```python
import numpy as np
import onnxruntime as ort  # 或 torch

class OceanBDXController:
    def __init__(self, model_path):
        # 加载模型
        self.session = ort.InferenceSession(model_path)
        
        # 初始化相位管理器
        self.phase_manager = AdaptivePhaseManager()
        
        # 初始化last_action
        self.last_action = np.zeros(14, dtype=np.float32)
        
        # 默认关节位置 (从URDF读取)
        self.default_joint_pos = np.array([...])  # 14个关节的默认位置
    
    def get_observation(self, imu_data, joint_data, velocity_cmd):
        """构建74维观测向量"""
        # 1. IMU数据 (6维)
        base_ang_vel = imu_data['gyro']  # 3维
        projected_gravity = self.compute_gravity(imu_data)  # 3维
        
        # 2. 关节数据 (42维)
        joint_pos_rel = joint_data['position'] - self.default_joint_pos  # 14维
        joint_vel_rel = joint_data['velocity']  # 14维
        joint_torques = joint_data['torque']  # 14维
        
        # 3. 命令和动作 (17维)
        velocity_commands = velocity_cmd  # 3维
        last_actions = self.last_action  # 14维
        
        # 4. 相位观测 (9维)
        adaptive_phase = self.phase_manager.update(velocity_cmd)
        
        # 拼接
        obs = np.concatenate([
            base_ang_vel,
            projected_gravity,
            joint_pos_rel,
            joint_vel_rel,
            joint_torques,
            velocity_commands,
            last_actions,
            adaptive_phase
        ], dtype=np.float32)
        
        assert obs.shape == (74,), f"观测维度错误: {obs.shape}"
        return obs
    
    def compute_gravity(self, imu_data):
        """计算重力投影"""
        quat = imu_data['quaternion']  # [w,x,y,z]
        accel = imu_data['acceleration']
        
        # 使用低通滤波提取重力分量
        gravity = self.low_pass_filter(accel, cutoff=5.0)
        return gravity
    
    def predict(self, obs):
        """模型推理"""
        obs_batch = obs.reshape(1, -1)
        action = self.session.run(None, {'obs': obs_batch})[0]
        action = action.squeeze()
        
        # 保存用于下一步
        self.last_action = action.copy()
        
        return action
    
    def control_loop(self):
        """主控制循环 (50Hz)"""
        while True:
            # 1. 读取传感器
            imu_data = self.read_imu()
            joint_data = self.read_joints()
            velocity_cmd = self.get_velocity_command()
            
            # 2. 构建观测
            obs = self.get_observation(imu_data, joint_data, velocity_cmd)
            
            # 3. 模型推理
            action = self.predict(obs)
            
            # 4. 执行动作
            self.execute_action(action)
            
            # 5. 等待下一个控制周期
            time.sleep(0.02)  # 50Hz = 20ms
```

---

## ✅ 部署前检查清单

在部署到真机之前,请逐项确认:

### 1️⃣ 观测维度检查

- [ ] **总维度是74维** (不是69维!)
- [ ] **删除了** `quaternion` (4维)
- [ ] **添加了** `adaptive_phase` (9维)
- [ ] 观测顺序与训练一致: `[ang_vel(3), gravity(3), pos(14), vel(14), torque(14), cmd(3), action(14), phase(9)]`

### 2️⃣ AdaptivePhaseManager 实现

- [ ] 已实现 `AdaptivePhaseManager` 类
- [ ] `get_observation()` 返回9维数组
- [ ] 多频率编码使用 `theta`, `theta/2`, `theta/4` (不是 `2*theta`, `4*theta`)
- [ ] 步频/步幅/抬脚高度使用正确的归一化范围

### 3️⃣ IMU数据处理

- [ ] `ang_vel_body`: 直接使用IMU陀螺仪输出
- [ ] `gravity_vec`: 使用低通滤波(~5Hz)提取纯重力分量
- [ ] **不使用** `quaternion` 作为观测输入

### 4️⃣ 关节数据

- [ ] 关节顺序: 5个左腿 + 5个右腿 + 4个颈部 (共14个)
- [ ] `dof_pos`: 相对位置 (current - default)
- [ ] `dof_vel`: 关节速度
- [ ] `joint_torques`: 电机转矩反馈 (或PD估计值)

### 5️⃣ 模型文件

- [ ] 使用 `model_7500.pt` checkpoint (iter 7405, reward +67)
- [ ] 已转换为 ONNX 格式 (如果用ONNX Runtime)
- [ ] 模型输入shape: `[1, 74]`
- [ ] 模型输出shape: `[1, 14]`

### 6️⃣ 控制参数

- [ ] 控制频率: 50Hz (20ms周期)
- [ ] 动作缩放: `scale=0.5` (训练时的配置)
- [ ] 速度命令单位: m/s (vx, vy) 和 rad/s (wz)

### 7️⃣ 安全检查

- [ ] 实现了紧急停止机制
- [ ] 测试了低速模式 (vx < 0.2 m/s)
- [ ] 验证了姿态保护 (pitch/roll < 45°)
- [ ] 确认了关节限位保护

---

## 📝 总结

### 🎯 核心改进

1. ✅ **新增关节转矩反馈** - 提升接触感知
2. ✅ **自适应步态相位** - 速度自适应,防止作弊
3. ✅ **改进IMU数据** - 纯重力投影,训练部署一致
4. ✅ **移除不必要观测** - 简化部署,减少传感器依赖

### ⚠️ 关键注意

1. **观测维度**: 必须严格74维
2. **相位管理**: 必须实现AdaptivePhaseManager
3. **重力投影**: 必须滤除运动加速度
4. **关节顺序**: 必须与训练时一致

### 🚀 推荐checkpoint

**model_7500.pt** - 性能最佳,稳定性最好

---

**部署成功后,机器人应该能够:**
- ✅ 稳定直立站立 (>30秒)
- ✅ 跟随速度命令行走 (0-0.74 m/s)
- ✅ 自适应调整步态 (速度越快步频越高)
- ✅ 保持平衡 (Roll/Pitch < 45°)
- ✅ 摔倒率 < 10%

祝部署顺利! 🎉
