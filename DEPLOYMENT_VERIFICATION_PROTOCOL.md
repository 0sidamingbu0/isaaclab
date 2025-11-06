# 🔍 OceanBDX 模型部署验证协议

## 目标
通过标准测试用例，确认部署代码的观测输入和模型输出是否与训练时完全一致。

---

## 📋 验证步骤

### 第一步：生成训练时的标准测试数据

运行以下脚本生成训练环境的标准输入/输出：

```bash
cd /home/ocean/oceanbdx/oceanbdx
python generate_test_cases.py
```

这将生成 `test_cases.json`，包含：
1. 标准测试姿态下的观测向量（74维）
2. 模型对应的输出动作（14维）
3. 关节名称与索引的映射关系

---

### 第二步：部署代码验证

**部署 AI 需要做的事情**：

1. **加载模型**：`model_7500.pt`

2. **读取测试用例**：`test_cases.json`

3. **对于每个测试用例，执行以下验证**：

#### ✅ 验证 A：观测向量构建
```python
# 从 test_cases.json 读取 test_case_1
robot_state = test_case_1["robot_state"]
expected_obs = test_case_1["observation_vector"]

# 部署代码构建观测向量
actual_obs = your_compute_observation(robot_state)

# 验证每个维度
for i in range(74):
    diff = abs(actual_obs[i] - expected_obs[i])
    if diff > 0.01:  # 允许 1% 误差
        print(f"❌ 观测维度 {i} 不匹配: expected={expected_obs[i]:.4f}, actual={actual_obs[i]:.4f}")
```

#### ✅ 验证 B：模型输出动作
```python
# 使用相同观测向量进行推理
expected_action = test_case_1["model_output"]
actual_action = your_model_inference(expected_obs)

# 验证每个关节动作
for i in range(14):
    diff = abs(actual_action[i] - expected_action[i])
    if diff > 0.01:
        print(f"❌ 动作维度 {i} 不匹配: expected={expected_action[i]:.4f}, actual={actual_action[i]:.4f}")
```

#### ✅ 验证 C：关节映射
```python
# 从 test_cases.json 读取关节映射
joint_order_training = test_case_1["joint_order_training"]
joint_order_deployment = test_case_1["joint_order_deployment"]

# 打印你的关节顺序
print("训练时关节顺序:", joint_order_training)
print("部署时关节顺序:", your_joint_order)

# 确认是否需要映射
if your_joint_order != joint_order_training:
    print("⚠️ 需要使用 joint_mapping 重新排列")
```

---

## 📊 测试用例说明

### Test Case 1: 默认站立姿态
- **用途**：验证初始姿态时的观测和输出
- **机器人状态**：所有关节在 `default_dof_pos`
- **预期输出**：模型应该输出接近零的动作（因为已经在目标姿态）

### Test Case 2: 前倾姿态
- **用途**：验证 gravity_vec 的方向和数值
- **机器人状态**：base 向前倾斜 15 度
- **预期输出**：模型应该输出向后倾的动作来纠正

### Test Case 3: 左腿抬起
- **用途**：验证关节顺序和左右腿的正确性
- **机器人状态**：左腿关节偏离 default 位置
- **预期输出**：模型应该只调整左腿关节

### Test Case 4: 右腿抬起
- **用途**：验证关节顺序和左右腿的正确性
- **机器人状态**：右腿关节偏离 default 位置
- **预期输出**：模型应该只调整右腿关节

---

## 🔧 部署 AI 需要提供的信息

请在验证完成后，提供以下信息：

### 1. 观测向量构建细节
```python
# 你的 ComputeObservation() 函数
def ComputeObservation(robot_state):
    """
    请提供：
    1. 每个观测项的计算方式
    2. 数据源（IMU 哪个字段、关节哪个字段）
    3. 坐标系定义（X/Y/Z 方向）
    """
    # 示例：
    ang_vel = imu.gyroscope  # [wx, wy, wz] in body frame
    gravity_vec = ???  # 如何计算？
    dof_pos = joint_positions - default_positions
    # ...
    return observation
```

### 2. 关节顺序映射
```python
# 你的机器人关节顺序
YOUR_JOINT_ORDER = [
    "leg_r1_joint",  # 索引 0
    "leg_r2_joint",  # 索引 1
    # ... 请列出完整的 14 个关节
]

# 你的 joint_mapping（如果有）
JOINT_MAPPING = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10, 11, 12, 13]
```

### 3. 关节角度正方向定义
```
请提供关节正方向定义（或 URDF 链接）：
- leg_l1_joint (hip roll): 正值表示？（外展/内收）
- leg_l2_joint (hip yaw): 正值表示？（外旋/内旋）
- leg_l3_joint (hip pitch): 正值表示？（前摆/后摆）
- leg_l4_joint (knee): 正值表示？（弯曲/伸直）
- leg_l5_joint (ankle): 正值表示？（背屈/跖屈）
```

### 4. IMU 坐标系定义
```
请提供 IMU 坐标系定义：
- X 轴方向：前/后/左/右/上/下
- Y 轴方向：前/后/左/右/上/下
- Z 轴方向：前/后/左/右/上/下
- 机器人直立时，重力向量应该是：[?, ?, ?]
```

---

## ⚠️ 常见问题排查

### 问题 1：gravity_vec 数值异常
**症状**：模型输出极端动作（±2）
**排查**：
```python
# 检查 gravity_vec 的计算
print(f"gravity_vec: {gravity_vec}")
print(f"magnitude: {np.linalg.norm(gravity_vec)}")  # 应该接近 9.81

# 检查坐标系
# 机器人直立时，应该是：
# - Isaac Lab 训练: [0, 0, +9.81]  (Z 向上)
# 或者
# - Isaac Lab 训练: [0, 0, -9.81]  (Z 向下)
```

### 问题 2：左右腿动作相反
**症状**：机器人倒向一侧
**排查**：
```python
# 测试：让左腿 L3 关节偏离 +0.1
test_state = default_state.copy()
test_state["leg_l3_joint"] += 0.1

action = model(compute_observation(test_state))

# 检查哪个关节的动作最大
print("动作最大的关节:", np.argmax(np.abs(action)))
# 应该是 L3 (索引 2)，而不是 R3 (索引 7)
```

### 问题 3：关节角度符号相反
**症状**：机器人"反向"运动
**排查**：
```python
# 检查 dof_pos 的计算
dof_pos_rel = current_pos - default_pos

# 如果关节正方向定义相反，需要反转
# dof_pos_rel[joint_idx] *= -1
```

---

## 📤 提交验证报告

请运行 `python verify_deployment.py` 并提交生成的 `verification_report.txt`，包含：

1. ✅ 所有测试用例的通过/失败状态
2. ✅ 观测向量每个维度的误差
3. ✅ 模型输出动作的误差
4. ✅ 你的关节顺序和映射配置
5. ✅ 你的 IMU 坐标系定义

---

## 🎯 验证成功标准

- [ ] Test Case 1-4 全部通过
- [ ] 观测向量误差 < 1%
- [ ] 模型输出误差 < 1%
- [ ] 关节映射正确
- [ ] gravity_vec 方向和数值正确

通过后即可进行真实机器人测试！
