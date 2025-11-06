# ✅ gravity_vec 坐标系确认报告

**日期**：2025-11-03  
**结论**：已通过 Isaac Lab 源码验证

---

## 🔍 Isaac Lab 源码证据

### 1. 世界坐标系重力定义

**文件**：`source/isaaclab/isaaclab/sim/simulation_cfg.py`

```python
gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
```

**结论**：Isaac Lab 使用标准的 **ROS/Gazebo 坐标系**：
- X: 前 (Forward)
- Y: 左 (Left)
- Z: 上 (Up)
- **世界重力向量**: `[0, 0, -9.81]` (向下)

---

### 2. projected_gravity_b 的计算

**文件**：`source/isaaclab/isaaclab/assets/rigid_object/rigid_object_data.py`

```python
@property
def projected_gravity_b(self) -> torch.Tensor:
    """Projection of the gravity direction on base frame."""
    return math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)
```

**其中**：
```python
# GRAVITY_VEC_W 初始化
gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)
self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_physx_view.count, 1)
```

**❗ 关键发现**：`GRAVITY_VEC_W` 是**归一化**后的重力方向！

---

## 🚨 训练时的实际数值

基于源码，训练时：

### 世界坐标系：
```python
gravity_world = [0, 0, -9.81]  # 向下
gravity_dir = normalize([0, 0, -9.81]) = [0, 0, -1]  # 归一化
```

### 机器人直立时：
```python
base_quat = [1, 0, 0, 0]  # 无旋转
projected_gravity_b = quat_apply_inverse([1,0,0,0], [0,0,-1])
                    = [0, 0, -1]  # 向下
```

### 机器人前倾 15° 时：
```python
projected_gravity_b ≈ [-0.26, 0, -0.97]  # 归一化向量
```

---

## ❌ 我之前的回答错误

我之前说：
> 直立时：`[0, 0, +9.81]`（Z 向上为正）

**这是错误的！** 正确的是：

### 训练时的 projected_gravity_b：

| 机器人姿态 | projected_gravity_b | 说明 |
|-----------|---------------------|------|
| **直立** | `[0, 0, -1]` | 归一化，Z 向下 |
| **前倾 15°** | `[-0.26, 0, -0.97]` | 归一化 |
| **左倾 15°** | `[0, -0.26, -0.97]` | 归一化 |

**⚠️ 重要**：
1. 是**归一化的方向向量**，magnitude = 1，不是 9.81
2. Z 分量是**负值**（向下），不是正值

---

## 🔧 部署代码修正

### 错误的实现（你可能用的）：

```cpp
// ❌ 错误 1：使用 +9.81
torch::Tensor gravity_world = torch::tensor({0.0, 0.0, +9.81});

// ❌ 错误 2：没有归一化
gravity_vec = quat_rotate_inverse(base_quat, gravity_world);
// 结果：[0, 0, +9.81]  magnitude = 9.81
```

### ✅ 正确的实现：

```cpp
// 方法 A：匹配训练（推荐）
torch::Tensor gravity_world = torch::tensor({0.0, 0.0, -9.81});
torch::Tensor gravity_dir = gravity_world / torch::norm(gravity_world);  // 归一化
gravity_vec = quat_rotate_inverse(base_quat, gravity_dir);
// 直立时结果：[0, 0, -1]  magnitude = 1  ✅

// 方法 B：等价简化
torch::Tensor gravity_dir = torch::tensor({0.0, 0.0, -1.0});  // 已归一化
gravity_vec = quat_rotate_inverse(base_quat, gravity_dir);
// 直立时结果：[0, 0, -1]  magnitude = 1  ✅
```

---

## 📊 验证你的日志

### 你的 Step 0 日志：
```
gravity_vec: [0, -0, -1]  magnitude = 1
```

**分析**：
- ✅ magnitude = 1（归一化正确）
- ✅ Z = -1（方向正确）
- ✅ **这个值是正确的！**

### 你的 Step 100 日志：
```
gravity_vec: [-9.186, 0.449, 3.412]  magnitude = 9.81
```

**分析**：
- ❌ magnitude = 9.81（没有归一化）
- ❌ 这说明你的代码在运行过程中**没有归一化**

---

## 🎯 问题根源

### 问题不是 Step 0 的 gravity_vec！

Step 0 的 `[0, 0, -1]` 是**完全正确的**！

### 真正的问题是：

**模型在正确的初始观测下，输出了极端动作！**

这意味着：

#### 可能性 1：模型文件问题
- 模型损坏
- 加载了错误版本
- 导出时有问题

#### 可能性 2：其他观测项错误
- `adaptive_phase` 计算错误？
- `dof_vel_scaled` 有问题？
- 观测拼接顺序错误？

#### 可能性 3：模型本身的问题
- 训练时模型没有收敛好
- 在 default_dof_pos 确实需要输出大动作来开始行走

---

## 🧪 下一步测试

### 测试 1：确认模型加载
```python
# 在 Python 中加载模型并测试
import torch
model = torch.jit.load("model_7500.pt")
model.eval()

# 构建 Step 0 的观测
obs = torch.tensor([[
    0, 0, 0,  # ang_vel
    0, 0, -1,  # gravity_vec (正确值!)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # dof_pos_rel
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # dof_vel
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # torques
    0, 0, 0,  # commands
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # actions
    0, 1, 0, 1, 0, 1, 0.6667, 0, 0.37  # phase
]], dtype=torch.float32)

action = model(obs)
print("Model output:", action)
# 应该是小值，不是全 ±2
```

### 测试 2：归一化修复
修改你的代码，确保 Step 100+ 的 gravity_vec 也是归一化的：

```cpp
// 在每一步计算观测时
gravity_vec = quat_rotate_inverse(base_quat, torch::tensor({0, 0, -1.0}));
// 不要使用 {0, 0, -9.81}，否则会失去归一化
```

### 测试 3：对比训练环境
使用 `generate_standard_test_cases.py` 生成的测试用例，在训练环境测试模型输出，看是否也是极端值。

---

## 📋 总结

| 项目 | 你的理解 | 实际训练值 | 状态 |
|------|---------|-----------|------|
| 世界重力 | `[0, 0, -9.81]` | `[0, 0, -9.81]` | ✅ 正确 |
| gravity_dir | - | `[0, 0, -1]` | ⚠️ 需归一化 |
| 直立时 gravity_vec | `[0, 0, -1]` | `[0, 0, -1]` | ✅ Step 0 正确 |
| 倾斜时 gravity_vec | magnitude=9.81 | magnitude=1 | ❌ 需修复归一化 |
| Step 0 模型输出 | 极端值 ±2 | 应该是小值 | ❌ **这是真正的问题** |

---

## 🚀 修复建议

1. **修复归一化**：确保所有步的 gravity_vec 都是 magnitude=1
2. **验证模型**：在 Python 中测试模型是否正常
3. **检查其他观测**：特别是 adaptive_phase 的计算
4. **联系训练方**：确认模型在 Step 0 的预期输出

**你的 Step 0 gravity_vec 是对的！问题在别处！** 🎯
