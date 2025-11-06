# 🔍 OceanBDX 关节映射关系验证文档

**生成时间**: 2025-11-03  
**目的**: 确认训练模型与部署代码的关节顺序映射关系

---

## 📋 1. 训练模型的关节顺序 (Model Order)

根据训练配置 `DEPLOYMENT_CONFIG.md`，训练时使用的关节顺序为：

```python
# 训练模型的关节顺序 (总共14个关节)
joint_order_training = [
    # 索引 0-4: 左腿 (Left Leg)
    "leg_l1_joint",  # 0  - 髋关节外展/内收
    "leg_l2_joint",  # 1  - 髋关节屈伸
    "leg_l3_joint",  # 2  - 膝关节屈伸
    "leg_l4_joint",  # 3  - 踝关节屈伸
    "leg_l5_joint",  # 4  - 踝关节内外翻
    
    # 索引 5-9: 右腿 (Right Leg)
    "leg_r1_joint",  # 5  - 髋关节外展/内收
    "leg_r2_joint",  # 6  - 髋关节屈伸
    "leg_r3_joint",  # 7  - 膝关节屈伸
    "leg_r4_joint",  # 8  - 踝关节屈伸
    "leg_r5_joint",  # 9  - 踝关节内外翻
    
    # 索引 10-13: 颈部 (Neck)
    "neck_n1_joint",  # 10
    "neck_n2_joint",  # 11
    "neck_n3_joint",  # 12
    "neck_n4_joint"   # 13
]
```

**关键特征**: **左腿优先** (Left-first order)

---

## 📋 2. 部署代码的关节顺序 (Deployment Order / base.yaml)

部署代码 `base.yaml` 中定义的关节顺序为：

```yaml
# base.yaml 的关节顺序 (总共14个关节)
joint_names: [
    # 索引 0-4: 右腿 (Right Leg)
    "leg_r1_joint",  # 0  - 髋关节外展/内收
    "leg_r2_joint",  # 1  - 髋关节屈伸
    "leg_r3_joint",  # 2  - 膝关节屈伸
    "leg_r4_joint",  # 3  - 踝关节屈伸
    "leg_r5_joint",  # 4  - 踝关节内外翻
    
    # 索引 5-9: 左腿 (Left Leg)
    "leg_l1_joint",  # 5  - 髋关节外展/内收
    "leg_l2_joint",  # 6  - 髋关节屈伸
    "leg_l3_joint",  # 7  - 膝关节屈伸
    "leg_l4_joint",  # 8  - 踝关节屈伸
    "leg_l5_joint",  # 9  - 踝关节内外翻
    
    # 索引 10-13: 颈部 (Neck)
    "neck_n1_joint",  # 10
    "neck_n2_joint",  # 11
    "neck_n3_joint",  # 12
    "neck_n4_joint"   # 13
]
```

**关键特征**: **右腿优先** (Right-first order)

---

## 🔄 3. 当前配置的 joint_mapping

当前 `robot_lab/config.yaml` 中的映射配置：

```yaml
joint_mapping: [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10, 11, 12, 13]
```

### 映射逻辑说明

代码中的使用方式：
```cpp
// 读取机器人状态时:
state->motor_state.q[i] = robot_state_msg.motor_state[joint_mapping[i]].q;

// 发送控制命令时:
robot_command_msg.motor_command[joint_mapping[i]].q = command->motor_command.q[i];
```

其中：
- `i` = **模型索引** (训练时的关节顺序，0-13)
- `joint_mapping[i]` = **硬件/ROS索引** (base.yaml的关节顺序，0-13)

---

## 📊 4. 完整的映射关系表

| 模型索引 | 训练关节名 | joint_mapping[i] | 部署关节名 | 是否匹配 |
|---------|-----------|-----------------|-----------|---------|
| **0** | leg_l1_joint | **5** | leg_l1_joint | ✅ |
| **1** | leg_l2_joint | **6** | leg_l2_joint | ✅ |
| **2** | leg_l3_joint | **7** | leg_l3_joint | ✅ |
| **3** | leg_l4_joint | **8** | leg_l4_joint | ✅ |
| **4** | leg_l5_joint | **9** | leg_l5_joint | ✅ |
| **5** | leg_r1_joint | **0** | leg_r1_joint | ✅ |
| **6** | leg_r2_joint | **1** | leg_r2_joint | ✅ |
| **7** | leg_r3_joint | **2** | leg_r3_joint | ✅ |
| **8** | leg_r4_joint | **3** | leg_r4_joint | ✅ |
| **9** | leg_r5_joint | **4** | leg_r5_joint | ✅ |
| **10** | neck_n1_joint | **10** | neck_n1_joint | ✅ |
| **11** | neck_n2_joint | **11** | neck_n2_joint | ✅ |
| **12** | neck_n3_joint | **12** | neck_n3_joint | ✅ |
| **13** | neck_n4_joint | **13** | neck_n4_joint | ✅ |

**结论**: 映射关系正确！✅

---

## 🔍 5. 验证示例

### 示例 1: 左腿第一个关节 (L1)

- **训练模型**: 索引 0, 关节名 `leg_l1_joint`
- **映射查找**: `joint_mapping[0]` = 5
- **部署代码**: 索引 5, 关节名 `leg_l1_joint` (base.yaml)
- **结果**: ✅ 正确匹配

### 示例 2: 右腿第一个关节 (R1)

- **训练模型**: 索引 5, 关节名 `leg_r1_joint`
- **映射查找**: `joint_mapping[5]` = 0
- **部署代码**: 索引 0, 关节名 `leg_r1_joint` (base.yaml)
- **结果**: ✅ 正确匹配

### 示例 3: 颈部第一个关节 (N1)

- **训练模型**: 索引 10, 关节名 `neck_n1_joint`
- **映射查找**: `joint_mapping[10]` = 10
- **部署代码**: 索引 10, 关节名 `neck_n1_joint` (base.yaml)
- **结果**: ✅ 正确匹配

---

## 📋 6. default_dof_pos 映射验证

当前 `robot_lab/config.yaml` 中的默认位置配置：

```yaml
default_dof_pos: [0.13, 0.07, 0.2, 0.052, -0.05,    # 模型索引 0-4: 左腿
                  -0.13, -0.07, -0.2, -0.052, 0.05,  # 模型索引 5-9: 右腿
                  0.0, 0.0, 0.0, 0.0]                # 模型索引 10-13: 颈部
```

### 映射后的实际关节位置

当模型输出 action = [0, 0, ..., 0] 时（保持默认姿态），实际发送到机器人的关节位置为：

| 部署索引 | 关节名 | 模型索引 | default_dof_pos[模型索引] | 实际位置 |
|---------|-------|---------|-------------------------|---------|
| **0** | leg_r1_joint | 5 | -0.13 | -0.13 rad |
| **1** | leg_r2_joint | 6 | -0.07 | -0.07 rad |
| **2** | leg_r3_joint | 7 | -0.2 | -0.2 rad |
| **3** | leg_r4_joint | 8 | -0.052 | -0.052 rad |
| **4** | leg_r5_joint | 9 | 0.05 | 0.05 rad |
| **5** | leg_l1_joint | 0 | 0.13 | 0.13 rad |
| **6** | leg_l2_joint | 1 | 0.07 | 0.07 rad |
| **7** | leg_l3_joint | 2 | 0.2 | 0.2 rad |
| **8** | leg_l4_joint | 3 | 0.052 | 0.052 rad |
| **9** | leg_l5_joint | 4 | -0.05 | -0.05 rad |
| **10** | neck_n1_joint | 10 | 0.0 | 0.0 rad |
| **11** | neck_n2_joint | 11 | 0.0 | 0.0 rad |
| **12** | neck_n3_joint | 12 | 0.0 | 0.0 rad |
| **13** | neck_n4_joint | 13 | 0.0 | 0.0 rad |

### 与训练配置对比

根据 `DEPLOYMENT_CONFIG.md`，训练时的默认位置为：

```python
# 训练时的关节默认位置
default_joint_pos = {
    "leg_r1_joint": -0.13,   # ✅ 匹配
    "leg_r2_joint": -0.07,   # ✅ 匹配
    "leg_r3_joint": -0.2,    # ✅ 匹配
    "leg_r4_joint": -0.052,  # ✅ 匹配
    "leg_r5_joint": 0.05,    # ✅ 匹配
    
    "leg_l1_joint": 0.13,    # ✅ 匹配
    "leg_l2_joint": 0.07,    # ✅ 匹配
    "leg_l3_joint": 0.2,     # ✅ 匹配
    "leg_l4_joint": 0.052,   # ✅ 匹配
    "leg_l5_joint": -0.05,   # ✅ 匹配
    
    "neck_n1_joint": 0.0,    # ✅ 匹配
    "neck_n2_joint": 0.0,    # ✅ 匹配
    "neck_n3_joint": 0.0,    # ✅ 匹配
    "neck_n4_joint": 0.0     # ✅ 匹配
}
```

**结论**: default_dof_pos 配置正确！✅

---

## ⚙️ 7. 其他关键参数验证

### action_scale

```yaml
# 当前配置
action_scale: [0.5, 0.5, 0.5, 0.5, 0.5,  # 左腿
               0.5, 0.5, 0.5, 0.5, 0.5,  # 右腿
               0.5, 0.5, 0.5, 0.5]       # 颈部

# 训练配置
action_scale: 0.5  # 统一的缩放因子
```

✅ **匹配**

### PD 控制参数

```yaml
# 当前配置
rl_kp: [50, 50, 50, 50, 50,  # 左腿
        50, 50, 50, 50, 50,  # 右腿
        15, 15, 15, 15]       # 颈部

rl_kd: [4, 4, 4, 4, 4,        # 左腿
        4, 4, 4, 4, 4,        # 右腿
        1.5, 1.5, 1.5, 1.5]   # 颈部

# 训练配置
legs: Kp=50, Kd=4
neck: Kp=15, Kd=1.5
```

✅ **匹配**

---

## 🎯 8. 潜在问题排查

### 问题描述

用户报告：机器人蜷缩姿态变了但还是不对，颈部有明显运动。

### 可能原因分析

#### ✅ 已排除的问题

1. **joint_mapping 错误** - 已验证映射关系正确
2. **default_dof_pos 全零** - 已修正为训练值
3. **action_scale 错误** - 已修正为统一的0.5

#### ⚠️ 需要进一步检查的问题

1. **观测输入是否正确**
   - 观测维度是否为74维？
   - 各项观测是否按正确顺序排列？
   - adaptive_phase 是否正确计算？

2. **模型输出范围是否正常**
   - 模型输出的 action 范围是否在 [-2, 2] 左右？
   - 是否有异常的极大值导致关节过度运动？

3. **Gazebo 物理参数**
   - Base 初始高度是否为 0.4米？
   - 重力是否正确设置？
   - 关节摩擦力是否合理？

4. **颈部运动异常**
   - 颈部的 action_scale 是否应该更小？(当前0.5)
   - 颈部的 Kp/Kd 是否过高导致过度响应？
   - 颈部关节的默认位置是否应该非零？

---

## 🔧 9. 建议的调试步骤

### 步骤 1: 验证观测输入

在代码中添加调试输出，打印观测向量的维度和前几个值：

```cpp
// 在 rl_sdk.cpp 的 ComputeObservation() 函数中
torch::Tensor obs = torch::cat(obs_list, 1);
std::cout << "观测维度: " << obs.sizes() << std::endl;
std::cout << "观测前10个值: " << obs.slice(1, 0, 10) << std::endl;
```

**期望结果**: 应该看到 `[1, 74]` 的维度。

### 步骤 2: 验证模型输出

打印模型输出的 action 值：

```cpp
// 在 rl_sdk.cpp 的 Forward() 函数中
torch::Tensor actions = this->model.forward({clamped_obs}).toTensor();
std::cout << "动作输出范围: [" << actions.min().item<float>() 
          << ", " << actions.max().item<float>() << "]" << std::endl;
```

**期望结果**: 动作应该在 `[-2, 2]` 范围内。

### 步骤 3: 验证目标关节位置

打印最终发送给机器人的目标位置：

```cpp
// 在 rl_sdk.cpp 的 ComputeOutput() 函数中
std::cout << "目标关节位置: " << output_dof_pos << std::endl;
```

**期望结果**: 关节位置应该在合理范围内（不应该有极端值）。

### 步骤 4: 暂时禁用颈部控制

测试是否是颈部导致的问题：

```yaml
# 临时修改 action_scale
action_scale: [0.5, 0.5, 0.5, 0.5, 0.5,  # 左腿
               0.5, 0.5, 0.5, 0.5, 0.5,  # 右腿
               0.0, 0.0, 0.0, 0.0]       # 颈部暂时禁用
```

**期望结果**: 如果腿部姿态正常，说明问题在颈部。

---

## ✅ 10. 配置验证清单

请训练AI确认以下配置是否与训练时完全一致：

- [ ] **关节顺序**: 训练模型使用 [L1-L5, R1-R5, N1-N4] 顺序
- [ ] **default_dof_pos 值**: 
  - L1=0.13, L2=0.07, L3=0.2, L4=0.052, L5=-0.05
  - R1=-0.13, R2=-0.07, R3=-0.2, R4=-0.052, R5=0.05
  - N1-N4=0.0
- [ ] **action_scale**: 统一为 0.5（所有关节）
- [ ] **观测空间**: 总维度74维，包含 adaptive_phase (9维)
- [ ] **控制频率**: 50Hz (decimation=4, dt=0.005)
- [ ] **PD参数**: 腿部 Kp=50/Kd=4, 颈部 Kp=15/Kd=1.5
- [ ] **Base初始高度**: 0.4米

---

## 📝 11. 总结

**当前映射配置状态**: ✅ **理论上正确**

```yaml
joint_mapping: [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10, 11, 12, 13]
default_dof_pos: [0.13, 0.07, 0.2, 0.052, -0.05,
                  -0.13, -0.07, -0.2, -0.052, 0.05,
                  0.0, 0.0, 0.0, 0.0]
action_scale: [0.5, 0.5, 0.5, 0.5, 0.5,
               0.5, 0.5, 0.5, 0.5, 0.5,
               0.5, 0.5, 0.5, 0.5]
```

**如果机器人姿态仍然不对，请检查**:
1. 观测输入的计算是否正确（特别是 adaptive_phase）
2. 模型输出的数值范围是否正常
3. Gazebo 的物理参数设置
4. 是否需要调整颈部的控制参数

---

**文档版本**: v1.0  
**生成者**: AI Assistant  
**验证状态**: 待训练AI确认
