# 🤖 OceanBDX 部署配置参数汇总

## 📦 模型文件

**推荐使用的checkpoint**:
- 文件路径: `/home/ocean/oceanbdx/oceanbdx/logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/model_7500.pt`
- 训练迭代: ~7405 iterations
- 性能指标: Reward +67, Episode Length 1670, Termination Rate 9.27%

## 🎯 关键配置参数

### 1. Base Position (机体初始位置)

```yaml
# 训练时的base初始位置
init_state:
  pos: [0.0, 0.0, 0.4]  # X, Y, Z (米)
  rot: [1.0, 0.0, 0.0, 0.0]  # 四元数 [w, x, y, z]
```

**关键点**:
- ✅ **Base高度**: `0.4米` (离地高度)
- ✅ **目标高度**: `0.35米` (reward中的target_height)
- ⚠️ **如果部署时base_position错误**,会导致:
  - 腿部关节角度计算错误
  - 机器人蜷缩或过度伸展
  - 重心不稳定

### 2. 关节默认位置 (Default Joint Positions)

**这是最关键的参数!** 部署时必须与训练完全一致!

```python
# 训练时的关节默认位置 (单位: 弧度)
default_joint_pos = {
    # 右腿 (5个关节)
    "leg_r1_joint": -0.13,   # 髋关节外展/内收
    "leg_r2_joint": -0.07,   # 髋关节屈伸
    "leg_r3_joint": -0.2,    # 膝关节屈伸
    "leg_r4_joint": -0.052,  # 踝关节屈伸
    "leg_r5_joint": 0.05,    # 踝关节内外翻
    
    # 左腿 (5个关节)
    "leg_l1_joint": 0.13,    # 髋关节外展/内收
    "leg_l2_joint": 0.07,    # 髋关节屈伸
    "leg_l3_joint": 0.2,     # 膝关节屈伸
    "leg_l4_joint": 0.052,   # 踝关节屈伸
    "leg_l5_joint": -0.05,   # 踝关节内外翻
    
    # 颈部 (4个关节)
    "neck_n1_joint": 0.0,
    "neck_n2_joint": 0.0,
    "neck_n3_joint": 0.0,
    "neck_n4_joint": 0.0
}
```

### 3. 关节顺序 (Joint Order)

**部署时观测和动作的关节顺序必须严格遵循**:

```python
joint_order = [
    # 左腿 (5个)
    "leg_l1_joint",  # 索引 0
    "leg_l2_joint",  # 索引 1
    "leg_l3_joint",  # 索引 2
    "leg_l4_joint",  # 索引 3
    "leg_l5_joint",  # 索引 4
    
    # 右腿 (5个)
    "leg_r1_joint",  # 索引 5
    "leg_r2_joint",  # 索引 6
    "leg_r3_joint",  # 索引 7
    "leg_r4_joint",  # 索引 8
    "leg_r5_joint",  # 索引 9
    
    # 颈部 (4个)
    "neck_n1_joint",  # 索引 10
    "neck_n2_joint",  # 索引 11
    "neck_n3_joint",  # 索引 12
    "neck_n4_joint"   # 索引 13
]
```

### 4. 控制参数

```yaml
# 控制频率
decimation: 4          # 降采样因子
sim_dt: 0.005          # 仿真时间步 (200Hz)
control_freq: 50Hz     # 控制频率 = 200Hz / 4 = 50Hz
control_period: 0.02s  # 20ms

# 动作缩放
action_scale: 0.5      # 动作输出乘以0.5后作为关节位置偏移

# 电机参数 (腿部)
legs:
  stiffness: 50.0      # Kp
  damping: 4.0         # Kd
  effort_limit: 50.0   # 最大力矩 (N·m)
  velocity_limit: 15.0 # 最大速度 (rad/s)
  saturation_effort: 90.0  # 饱和力矩
  friction: 0.8

# 电机参数 (颈部)
neck:
  stiffness: 15.0
  damping: 1.5
  effort_limit: 10.0
  velocity_limit: 10.0
  saturation_effort: 8.0
  friction: 0.3
```

### 5. 观测空间配置

**总维度: 74维**

| 序号 | 观测项 | 维度 | 计算方法 |
|------|--------|------|----------|
| 1 | `base_ang_vel` | 3 | IMU陀螺仪输出 |
| 2 | `projected_gravity` | 3 | 重力投影到机体坐标系 |
| 3 | `joint_pos_rel` | 14 | `current_pos - default_pos` |
| 4 | `joint_vel_rel` | 14 | 关节速度 |
| 5 | `joint_torques` | 14 | 关节转矩反馈 |
| 6 | `velocity_commands` | 3 | 速度命令 [vx, vy, wz] |
| 7 | `last_actions` | 14 | 上一步的动作输出 |
| 8 | `adaptive_phase` | 9 | 步态相位观测 |

**关键点**:
- ⚠️ `joint_pos_rel` 必须是相对位置! 
  - ✅ 正确: `current_pos - default_pos`
  - ❌ 错误: 直接使用 `current_pos`

### 6. 动作空间配置

**总维度: 14维**

```python
# 模型输出action (范围约[-1, 1])
model_output = model.predict(observation)  # shape: [14]

# 转换为目标关节位置
target_joint_pos = default_joint_pos + model_output * 0.5

# 按照关节顺序赋值
for i, joint_name in enumerate(joint_order):
    robot.set_joint_position_target(joint_name, target_joint_pos[i])
```

**关键点**:
- ✅ 动作是**相对偏移**,不是绝对位置
- ✅ 缩放因子 `scale=0.5`
- ✅ 基准位置是 `default_joint_pos`

### 7. 速度命令范围

```python
# 训练时使用的速度范围 (课程学习动态调整)
# Stage 0 (0-20%):   [0.0, 0.0]     - 站立
# Stage 1 (20-45%):  [-0.35, 0.0]   - 低速前进
# Stage 2 (45-75%):  [-0.5, 0.0]    - 中速前进
# Stage 3 (75-100%): [-0.74, 0.0]   - 高速前进

# 注意: 负值表示向前 (因为硬件坐标系X+指向后方)
```

**部署建议**:
- 初始测试: `vx = -0.1 m/s` (慢速)
- 正常行走: `vx = -0.35 m/s`
- 最大速度: `vx = -0.74 m/s`

---

## 🚨 常见部署错误分析

### 问题: 机器人腿蜷缩在一起

**可能原因**:

#### 1️⃣ Base Position 高度错误

```python
# ❌ 错误: base离地太低
base_height = 0.2  # 太低!

# ✅ 正确: 训练时的高度
base_height = 0.4  # 离地0.4米
```

**影响**: 如果base高度错误,会导致:
- 腿部需要过度弯曲才能触地
- 关节角度超出训练范围
- 机器人蜷缩或跪倒

#### 2️⃣ 关节默认位置错误

```python
# ❌ 错误: 使用零位或URDF默认值
default_joint_pos = {
    "leg_r1_joint": 0.0,  # 错误!
    "leg_r2_joint": 0.0,  # 错误!
    ...
}

# ✅ 正确: 使用训练时的默认值
default_joint_pos = {
    "leg_r1_joint": -0.13,  # 训练时的值
    "leg_r2_joint": -0.07,  # 训练时的值
    ...
}
```

**影响**: 关节默认位置错误会导致:
- `joint_pos_rel` 计算错误
- 模型观测输入不匹配
- 动作输出对应到错误的关节角度

#### 3️⃣ 观测计算错误

```python
# ❌ 错误: 使用绝对位置
joint_pos_obs = current_joint_pos  # 错误!

# ✅ 正确: 使用相对位置
joint_pos_obs = current_joint_pos - default_joint_pos  # 正确!
```

#### 4️⃣ 动作执行错误

```python
# ❌ 错误: 直接使用模型输出
target_pos = model_output  # 错误!

# ✅ 正确: 添加默认位置和缩放
target_pos = default_joint_pos + model_output * 0.5  # 正确!
```

---

## ✅ 部署检查清单

在部署到真机前,请逐项确认:

### Base 配置
- [ ] Base初始高度设置为 `0.4米`
- [ ] Base在重力方向保持直立 (pitch≈0, roll≈0)
- [ ] 没有固定base (fix_root_link=false)

### 关节配置
- [ ] 14个关节默认位置与训练配置完全一致
- [ ] 关节顺序为: 5左腿 + 5右腿 + 4颈部
- [ ] 关节位置单位为弧度 (不是角度!)

### 观测计算
- [ ] `joint_pos_rel = current_pos - default_pos` (相对位置!)
- [ ] 总观测维度为74维 (不是69维!)
- [ ] 包含 `adaptive_phase` (9维)
- [ ] 不包含 `quaternion` (已移除)

### 动作执行
- [ ] `target = default_pos + action * 0.5`
- [ ] 动作缩放因子为0.5
- [ ] 关节顺序与观测一致

### 控制参数
- [ ] 控制频率: 50Hz (20ms周期)
- [ ] 腿部PD: Kp=50, Kd=4
- [ ] 颈部PD: Kp=15, Kd=1.5

---

## 📝 部署代码示例

```python
import numpy as np

class OceanBDXDeployment:
    def __init__(self):
        # 关节默认位置 (与训练完全一致!)
        self.default_joint_pos = np.array([
            # 左腿
            0.13, 0.07, 0.2, 0.052, -0.05,
            # 右腿
            -0.13, -0.07, -0.2, -0.052, 0.05,
            # 颈部
            0.0, 0.0, 0.0, 0.0
        ])
        
        # Base目标高度
        self.target_base_height = 0.4  # 米
        
        # 动作缩放
        self.action_scale = 0.5
        
        # 控制频率
        self.control_dt = 0.02  # 50Hz
        
    def get_joint_pos_rel(self, current_joint_pos):
        """计算相对关节位置观测"""
        return current_joint_pos - self.default_joint_pos
    
    def apply_action(self, model_output):
        """将模型输出转换为目标关节位置"""
        target_joint_pos = self.default_joint_pos + model_output * self.action_scale
        return target_joint_pos
    
    def check_base_height(self, current_base_height):
        """检查base高度是否合理"""
        if abs(current_base_height - self.target_base_height) > 0.1:
            print(f"⚠️ Base高度异常: {current_base_height:.3f}m (期望: {self.target_base_height}m)")
            return False
        return True
```

---

## 📞 故障排查

### 现象: 腿蜷缩在一起

**检查步骤**:

1. **验证Base高度**
   ```python
   current_height = robot.get_base_position()[2]
   print(f"当前Base高度: {current_height}")
   # 应该在 0.35-0.45米 范围
   ```

2. **验证关节默认位置**
   ```python
   print("关节默认位置:")
   for i, name in enumerate(joint_order):
       print(f"  {name}: {default_joint_pos[i]:.3f} rad")
   # 对比上面的配置表
   ```

3. **验证观测计算**
   ```python
   joint_pos_rel = current_pos - default_pos
   print(f"关节相对位置范围: [{joint_pos_rel.min():.3f}, {joint_pos_rel.max():.3f}]")
   # 应该在 [-1.0, 1.0] 范围内
   ```

4. **验证动作输出**
   ```python
   print(f"模型输出范围: [{model_output.min():.3f}, {model_output.max():.3f}]")
   # 应该在 [-2.0, 2.0] 范围内
   
   target_pos = default_pos + model_output * 0.5
   print(f"目标关节位置范围: [{target_pos.min():.3f}, {target_pos.max():.3f}]")
   # 应该在合理的关节限位内
   ```

---

## 📄 需要拷贝的文件

### 1. 模型文件
- `logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/model_7500.pt`

### 2. 配置文件 (参考用)
- `logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/params/env.yaml`
- `MODEL_DEPLOYMENT_GUIDE.md` (观测空间文档)
- `DEPLOYMENT_CONFIG.md` (本文件)

### 3. 关键参数提取

从 `env.yaml` 中提取:
- ✅ Base初始位置: `init_state.pos`
- ✅ 关节默认位置: `init_state.joint_pos`
- ✅ 动作缩放: `actions.joint_pos.scale`
- ✅ 控制频率: `decimation` 和 `sim.dt`
- ✅ PD参数: `actuators.legs` 和 `actuators.neck`
