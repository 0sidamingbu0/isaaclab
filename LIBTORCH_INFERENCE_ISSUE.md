# 🎯 新发现: 问题不在观测,在模型推理!

## 日期
2025-11-04

## 关键证据

### ✅ 已验证: 模型和观测都正常
1. **test_model_output.py**: 正确观测 [0.67,0,0.37] → 正常输出 ✅
2. **test_deployment_observation.py**: 错误观测 [0,0,0] → 仍正常输出 ✅  
3. **compare_observations.py**: 两种观测差异小,影响小 ✅

### ❌ 异常: 部署环境极端输出
- 相同的观测 (甚至更差的观测)
- 训练环境 (PyTorch JIT) → 正常输出
- 部署环境 (LibTorch C++) → 极端输出 ❌

---

## 🎯 结论: 问题在 C++/LibTorch 推理实现!

### 不是观测问题
- ✅ adaptive_phase 虽然错误,但影响很小
- ✅ gravity_vec 归一化问题已知,但 Step 0 是正确的
- ✅ 其他观测项 (dof_pos_rel, etc.) Step 0 都是0

### 是推理问题
- ❌ C++ LibTorch 模型加载可能有问题
- ❌ 观测向量类型/格式可能不匹配
- ❌ 模型推理调用可能不正确
- ❌ 输出解析可能有误

---

## 🔍 需要部署 AI 立即检查

### 1. 模型加载 (rl_sdk.cpp)
```cpp
// 检查模型加载代码
torch::jit::script::Module model = torch::jit::load(model_path);
model.eval();

// ⚠️ 可能的问题:
// - 模型路径错误
// - 模型版本不匹配
// - 模型加载选项不对
```

### 2. 观测向量构造
```cpp
// 检查观测向量类型
std::vector<float> observation(74, 0.0f);
// ... 填充观测 ...

// 转换为 Tensor
auto obs_tensor = torch::from_blob(
    observation.data(),
    {1, 74},
    torch::kFloat32  // ⚠️ 必须是 float32!
);

// ⚠️ 可能的问题:
// - 使用了 double 而不是 float
// - Tensor shape 不对
// - 内存布局不对 (row-major vs column-major)
```

### 3. 模型推理
```cpp
// 检查推理调用
std::vector<torch::jit::IValue> inputs;
inputs.push_back(obs_tensor);

auto output = model.forward(inputs);
auto action_tensor = output.toTensor();

// ⚠️ 可能的问题:
// - forward() 参数传递错误
// - 没有设置 eval() 模式
// - Tensor 设备不匹配 (CPU vs GPU)
```

### 4. 输出提取
```cpp
// 检查动作提取
auto action_accessor = action_tensor.accessor<float, 2>();
for (int i = 0; i < 14; i++) {
    float action = action_accessor[0][i];
    // ...
}

// ⚠️ 可能的问题:
// - accessor 类型错误 (double vs float)
// - 索引顺序错误
// - 输出维度错误
```

---

## 🔧 建议调试步骤

### Step 1: 打印 LibTorch 中间值
```cpp
// 在 rl_sdk.cpp 的推理代码中添加:

std::cout << "观测 Tensor shape: " << obs_tensor.sizes() << std::endl;
std::cout << "观测 Tensor dtype: " << obs_tensor.dtype() << std::endl;
std::cout << "观测前5维: ";
for (int i = 0; i < 5; i++) {
    std::cout << obs_tensor[0][i].item<float>() << " ";
}
std::cout << std::endl;

auto output = model.forward(inputs);

std::cout << "输出 Tensor shape: " << output.toTensor().sizes() << std::endl;
std::cout << "输出 Tensor dtype: " << output.toTensor().dtype() << std::endl;
std::cout << "输出前5维: ";
for (int i = 0; i < 5; i++) {
    std::cout << output.toTensor()[0][i].item<float>() << " ";
}
std::cout << std::endl;
```

### Step 2: 对比 Python 和 C++ 的中间结果
```python
# Python (训练环境)
import torch
model = torch::jit.load("policy.pt")
obs = torch.zeros(1, 74)
obs[0, 71:74] = torch.tensor([0.6667, 0.0, 0.37])
print("Python obs:", obs[:, :5])
action = model(obs)
print("Python action:", action[:, :5])
```

```cpp
// C++ (部署环境)
// 打印相同的中间值,逐项对比
```

### Step 3: 检查模型导出过程
```python
# 检查模型导出时的选项
model = ...
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "policy.pt")

# ⚠️ 可能需要额外选项:
# torch.jit.save(scripted_model, "policy.pt", _use_new_zipfile_serialization=True)
```

### Step 4: 使用相同的观测测试
在 C++ 中硬编码一个已知的观测向量:
```cpp
// 使用 test_model_output.py 中的完全相同的观测
std::vector<float> obs = {
    0.0, 0.0, 0.0,  // ang_vel
    0.0, 0.0, -1.0,  // gravity
    // ... (完整74维)
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37  // adaptive_phase
};

// 推理并打印输出
// 应该得到: [-0.08, 0.33, -0.30, ...]
// 如果不匹配 → LibTorch 推理有问题!
```

---

## 🎯 预期发现

### 场景A: 观测Tensor有问题
```
Python obs: [0.0, 0.0, 0.0, 0.0, 0.0]
C++ obs:    [一些奇怪的值或NaN]
→ 观测向量构造有误
```

### 场景B: 模型输出有问题
```
Python obs:    [0.0, 0.0, 0.0, 0.0, 0.0]
C++ obs:       [0.0, 0.0, 0.0, 0.0, 0.0]  ✅
Python action: [-0.08, 0.33, -0.30, 0.71, -0.44]
C++ action:    [-1.96, -0.75, -1.10, 0.77, 1.23]  ❌
→ LibTorch 推理实现有误
```

### 场景C: 数据类型问题
```
Python dtype: torch.float32
C++ dtype:    torch.float64  ❌
→ 类型不匹配导致计算错误
```

---

## 📋 其他可能的问题

### 1. 模型设备不匹配
```cpp
// 确保模型和数据在同一设备
model.to(torch::kCPU);
obs_tensor = obs_tensor.to(torch::kCPU);
```

### 2. 批处理维度
```cpp
// 确保输入是 [1, 74] 不是 [74]
auto obs_tensor = torch::from_blob(
    observation.data(),
    {1, 74},  // ← 注意: 必须有 batch dimension!
    torch::kFloat32
);
```

### 3. eval() 模式
```cpp
// 确保模型在 eval 模式
model.eval();
torch::NoGradGuard no_grad;  // 禁用梯度计算
```

### 4. 线程安全
```cpp
// 如果多线程调用,需要加锁
std::lock_guard<std::mutex> lock(model_mutex);
auto output = model.forward(inputs);
```

---

## 🎯 最终验证

如果修复后 C++ 输出仍然不对,创建一个最简单的测试:

```cpp
// minimal_test.cpp
#include <torch/script.h>
#include <iostream>

int main() {
    // 加载模型
    torch::jit::script::Module model = torch::jit::load("policy.pt");
    model.eval();
    
    // 创建全0观测
    auto obs = torch::zeros({1, 74}, torch::kFloat32);
    obs[0][71] = 0.6667;
    obs[0][73] = 0.37;
    
    // 推理
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(obs);
    auto output = model.forward(inputs).toTensor();
    
    // 打印
    std::cout << "Output: " << output << std::endl;
    
    // 期望: 前5维约为 [-0.08, 0.33, -0.30, 0.71, -0.44]
    return 0;
}
```

编译运行:
```bash
g++ minimal_test.cpp -o test \
    -I/path/to/libtorch/include \
    -L/path/to/libtorch/lib \
    -ltorch -ltorch_cpu -lc10
./test
```

如果这个简单测试输出正常 → 问题在 rl_sdk 的集成代码  
如果这个简单测试输出异常 → 问题在 LibTorch 环境或模型

---

**建议部署 AI 立即检查 C++/LibTorch 推理代码!** 🚀
