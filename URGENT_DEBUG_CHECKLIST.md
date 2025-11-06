# 🚨 紧急调试清单

## 测试结果总结
- ✅ 训练环境 + 正确 adaptive_phase → 正常输出
- ✅ 训练环境 + 错误 adaptive_phase → 正常输出 (差异<0.5)
- ❌ 部署环境 + 错误 adaptive_phase → 极端输出 (±2)

**结论**: adaptive_phase 不是主要问题! 部署环境还有其他严重错误!

---

## 🔴 立即检查项目

### 1. dof_pos_rel 计算 (最可能!)

部署日志显示:
```
📌 Raw dof_pos: [0.13, 0.07, 0.2, 0.052, -0.05, -0.13, -0.07, -0.2, -0.052, 0.05, 0, 0, 0, 0]
📌 default_dof_pos: [0.13, 0.07, 0.2, 0.052, -0.05, -0.13, -0.07, -0.2, -0.052, 0.05, 0, 0, 0, 0]
```

看起来正确,但需要**确认 joint_mapping 是否正确应用!**

#### ⚠️ 可能的 Bug:
```cpp
// 错误做法1: default_dof_pos 顺序不对
// default_dof_pos 应该是训练顺序 [L1-L5, R1-R5, N1-N4]
// 但代码中可能用的是 URDF 顺序 [R1-R5, L1-L5, N1-N4]

// 错误做法2: joint_mapping 没有应用到 dof_pos_rel 计算
for (int i = 0; i < 14; i++) {
    dof_pos_rel[i] = current_pos[i] - default_pos[i];  // ❌ 顺序不匹配!
}

// 正确做法:
for (int i = 0; i < 14; i++) {
    int train_idx = i;  // 训练顺序
    int urdf_idx = joint_mapping[i];  // URDF顺序
    dof_pos_rel[train_idx] = current_pos[urdf_idx] - default_pos_train_order[train_idx];
}
```

### 2. 数据类型检查

C++ 中可能用了 `double`,但模型需要 `float32`:

```cpp
// 检查
std::vector<float> observation;  // ✅ 应该是 float
// 或
std::vector<double> observation;  // ❌ 可能导致精度问题
```

### 3. 观测顺序问题

部署代码可能在拼接观测时顺序错了:

```cpp
// 错误: 顺序不对
obs = [ang_vel, gravity, dof_vel, dof_pos, ...]  // ❌

// 正确: 训练顺序
obs = [ang_vel, gravity, dof_pos, dof_vel, torques, commands, last_actions, phase]  // ✅
```

### 4. 观测缩放/归一化

检查是否所有观测项都正确缩放:

- ✅ `ang_vel_body`: 原始值
- ✅ `gravity_vec`: 归一化 (magnitude=1)
- ❓ `dof_pos_rel`: 是 `current - default` 吗?
- ❓ `dof_vel_scaled`: 乘了 0.05 吗?
- ❓ `joint_torques`: 原始值吗?

---

## 📋 调试步骤

### Step 1: 打印部署环境的完整 74 维观测

**C++ 代码** (在部署环境添加):
```cpp
// 在 constructObservation() 返回前添加
std::cout << "\n=== FULL OBSERVATION (74 dims) ===" << std::endl;
for (int i = 0; i < 74; i++) {
    std::cout << "obs[" << std::setw(2) << i << "] = " 
              << std::fixed << std::setprecision(6) << observation[i] << std::endl;
}
std::cout << "=== END OBSERVATION ===" << std::endl;
```

### Step 2: 创建对比测试脚本

我创建一个脚本,用部署环境的**实际 74 维观测**测试模型:

```python
# 从部署日志复制完整的 74 维观测
deployment_obs = [
    # 从部署环境 Step 0 的日志复制所有 74 个值
]

# 测试
action = model(torch.tensor([deployment_obs]))
print("如果输出极端值 → 观测确实有问题")
print("如果输出正常 → 部署环境的模型推理有问题")
```

### Step 3: 逐项对比

创建表格对比:
```
Dim  | 名称           | 训练环境 | 部署环境 | 差异
-----|---------------|---------|---------|------
0    | ang_vel[0]    | 0.0     | 0.0     | 0.0
1    | ang_vel[1]    | 0.0     | 0.0     | 0.0
...
```

找出差异最大的维度!

---

## 🎯 我的推测

基于经验,最可能的问题是:

### 推测1: dof_pos_rel 的 default_dof_pos 顺序错误 (80%概率)

```cpp
// 训练环境的 default_dof_pos (训练顺序: L1-L5, R1-R5, N1-N4)
float default_train[14] = {
    0.13, 0.07, 0.2, 0.052, -0.05,    // L1-L5
    -0.13, -0.07, -0.2, -0.052, 0.05, // R1-R5
    0, 0, 0, 0                         // N1-N4
};

// 部署环境可能用的 (URDF顺序: R1-R5, L1-L5, N1-N4)
float default_urdf[14] = {
    -0.13, -0.07, -0.2, -0.052, 0.05, // R1-R5
    0.13, 0.07, 0.2, 0.052, -0.05,    // L1-L5
    0, 0, 0, 0                         // N1-N4
};

// 如果计算 dof_pos_rel 时:
// 用的是 current_pos[URDF顺序] - default_train[训练顺序]
// 就会完全错乱!
```

这会导致:
- L1 的位置 - R1 的 default = 巨大的偏差!
- 模型认为关节位置严重错误
- 输出极端动作来"纠正"

### 推测2: joint_mapping 没有应用到观测计算 (15%概率)

部署代码可能只在**输出动作时**应用了 joint_mapping,但在**计算观测时**忘记了!

### 推测3: 数据类型或精度问题 (5%概率)

float vs double,或者某些计算中的类型转换错误。

---

## 🔧 快速验证方法

### 方法1: 交换 default_dof_pos 顺序
```cpp
// 将 default_dof_pos 改为训练顺序
float default_dof_pos[14] = {
    0.13, 0.07, 0.2, 0.052, -0.05,    // L1-L5
    -0.13, -0.07, -0.2, -0.052, 0.05, // R1-R5
    0, 0, 0, 0                         // N1-N4
};

// 然后计算 dof_pos_rel:
for (int i = 0; i < 14; i++) {
    int urdf_idx = joint_mapping[i];
    dof_pos_rel[i] = current_pos[urdf_idx] - default_dof_pos[i];
}
```

### 方法2: 打印每个关节的 dof_pos_rel
```cpp
std::cout << "dof_pos_rel (training order):" << std::endl;
for (int i = 0; i < 14; i++) {
    std::cout << "  [" << i << "] " << dof_pos_rel[i] << std::endl;
}
```

如果看到很大的值 (>1.0),说明顺序确实错了!

---

## 📤 给部署 AI 的任务

### 🔴 紧急任务 1: 打印完整观测
在 Step 0 打印所有 74 维观测值,格式:
```
obs[0] = 0.000000
obs[1] = 0.000000
...
obs[73] = 0.000000
```

### 🔴 紧急任务 2: 检查 dof_pos_rel 计算
1. 打印 `current_pos` (URDF顺序,14个值)
2. 打印 `default_dof_pos` (什么顺序?14个值)
3. 打印 `dof_pos_rel` (训练顺序,14个值)
4. 检查代码中是否正确应用了 `joint_mapping`

### 🔴 紧急任务 3: 检查观测拼接顺序
确认观测向量的拼接顺序是否为:
```
[ang_vel(3), gravity(3), dof_pos(14), dof_vel(14), 
 torques(14), commands(3), last_actions(14), phase(9)]
```

---

## 💡 如果我的推测正确

修复后:
- ✅ dof_pos_rel 使用正确的 default_dof_pos
- ✅ Step 0 的 dof_pos_rel 应该全是 0
- ✅ 模型输出应该变正常
- ✅ 机器人应该能站立!

---

**建议立即打印部署环境的完整观测向量,我来分析!** 🚀
