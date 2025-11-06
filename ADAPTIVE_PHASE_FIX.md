# 🔧 adaptive_phase 最后3维修复方案

## 验证日期
2025-11-03

## 问题确认
**部署代码的 adaptive_phase 最后3维全是 0,应该是 [0.6667, 0.0, 0.37]!**

---

## ✅ 验证结果

运行 `verify_adaptive_phase_simple.py` 确认:

### 场景对比
```
场景1 - 初始化默认值:
  adaptive_phase[-3:] = [0.6667, 0.5240, 0.3700]

场景2 - 静止速度 (speed=0.0):
  adaptive_phase[-3:] = [0.6250, 0.0000, 0.0000]

场景3 - 参考速度 (speed=0.35):
  adaptive_phase[-3:] = [0.6667, 0.5240, 0.3700]

训练实际 (test_model_output.py):
  adaptive_phase[-3:] = [0.6667, 0.0000, 0.3700]  ← 混合值!
```

### 混合值解释
训练环境在静止命令时:
- **第7维 0.6667**: 保持默认 phase_rate (1.0/0.75 / 2.0) - 保持步态节奏
- **第8维 0.0**: 步幅归零 (静止不移动)
- **第9维 0.37**: 保持默认 clearance (0.037 / 0.1) - 准备随时抬脚

这是合理的设计逻辑!

---

## 🔧 修复代码

### C++ 实现 (rl_sdk.cpp)

找到 adaptive_phase 计算部分 (应该在 `constructObservation()` 函数中):

```cpp
// ========== 当前错误的代码 (第71-73维全是0) ==========
// obs[65] = std::sin(theta);
// obs[66] = std::cos(theta);
// obs[67] = std::sin(theta / 2.0f);
// obs[68] = std::cos(theta / 2.0f);
// obs[69] = std::sin(theta / 4.0f);
// obs[70] = std::cos(theta / 4.0f);
// obs[71] = 0.0f;  // ❌ 错误!
// obs[72] = 0.0f;  // ❌ 错误!
// obs[73] = 0.0f;  // ❌ 错误!

// ========== 修复方案1: 使用训练验证的固定值 (最安全) ==========
// 前6维: sin/cos 编码 (保持不变)
obs[65] = std::sin(theta);
obs[66] = std::cos(theta);
obs[67] = std::sin(theta / 2.0f);
obs[68] = std::cos(theta / 2.0f);
obs[69] = std::sin(theta / 4.0f);
obs[70] = std::cos(theta / 4.0f);

// 后3维: 归一化步态参数 (修复!)
// 对于静止命令,使用训练环境验证的值
float cmd_vel = std::sqrt(commands[0]*commands[0] + commands[1]*commands[1]);

if (cmd_vel < 0.001f) {
    // 静止命令 - 使用混合默认值
    obs[71] = 0.6667f;  // phase_rate_norm = (1.0/0.75) / 2.0
    obs[72] = 0.0f;     // stride_norm = 0.0 (不移动)
    obs[73] = 0.37f;    // clearance_norm = 0.037 / 0.1
} else {
    // 有速度命令 - 动态计算 (后续实现)
    // 暂时先使用默认值
    obs[71] = 0.6667f;
    obs[72] = 0.0f;
    obs[73] = 0.37f;
}
```

### 方案2: 完整动态计算 (可选,更复杂)

如果需要支持不同速度命令,实现完整的 `AdaptiveGaitTable`:

```cpp
struct GaitParams {
    float period;     // 周期 (s)
    float stride;     // 步幅 (m, 两步距离)
    float clearance;  // 抬脚高度 (m)
};

GaitParams interpolateGaitParams(float speed) {
    // 速度-步态映射表 (从训练环境复制)
    const std::vector<std::pair<float, GaitParams>> GAIT_TABLE = {
        {0.0f,  {0.8f,  0.0f,   0.0f}},
        {0.1f,  {0.8f,  0.08f,  0.025f}},
        {0.25f, {0.8f,  0.2f,   0.03f}},
        {0.35f, {0.75f, 0.262f, 0.037f}},
        {0.5f,  {0.65f, 0.325f, 0.045f}},
        {0.6f,  {0.6f,  0.36f,  0.055f}},
        {0.74f, {0.5f,  0.37f,  0.07f}},
    };
    
    // Clamp速度
    speed = std::clamp(speed, 0.0f, 0.74f);
    
    // 线性插值
    for (size_t i = 0; i < GAIT_TABLE.size() - 1; i++) {
        if (speed >= GAIT_TABLE[i].first && speed <= GAIT_TABLE[i+1].first) {
            float alpha = (speed - GAIT_TABLE[i].first) / 
                         (GAIT_TABLE[i+1].first - GAIT_TABLE[i].first);
            
            GaitParams result;
            result.period = GAIT_TABLE[i].second.period * (1.0f - alpha) + 
                           GAIT_TABLE[i+1].second.period * alpha;
            result.stride = GAIT_TABLE[i].second.stride * (1.0f - alpha) + 
                           GAIT_TABLE[i+1].second.stride * alpha;
            result.clearance = GAIT_TABLE[i].second.clearance * (1.0f - alpha) + 
                              GAIT_TABLE[i+1].second.clearance * alpha;
            return result;
        }
    }
    
    return GAIT_TABLE[0].second;
}

// 在 constructObservation() 中使用
float cmd_vel = std::sqrt(commands[0]*commands[0] + commands[1]*commands[1]);
GaitParams params = interpolateGaitParams(cmd_vel);

// 归一化
const float MAX_PHASE_RATE = 2.0f;
const float MAX_STRIDE = 0.5f;
const float MAX_CLEARANCE = 0.1f;

float phase_rate = 1.0f / (params.period + 1e-8f);
float phase_rate_norm = std::clamp(phase_rate / MAX_PHASE_RATE, 0.0f, 1.0f);
float stride_norm = std::clamp(params.stride / MAX_STRIDE, 0.0f, 1.0f);
float clearance_norm = std::clamp(params.clearance / MAX_CLEARANCE, 0.0f, 1.0f);

obs[71] = phase_rate_norm;
obs[72] = stride_norm;
obs[73] = clearance_norm;
```

---

## 🎯 推荐修复流程

### Step 1: 快速验证 (5分钟)
使用方案1的固定值 `[0.6667, 0.0, 0.37]`:

```cpp
obs[71] = 0.6667f;
obs[72] = 0.0f;
obs[73] = 0.37f;
```

**立即测试**:
1. 重新编译部署代码
2. 运行 sim2sim,查看 Step 0 的模型输出
3. **预期**: 模型输出应该从极端值 (±2) 变为合理值 (接近 test_model_output.py 的结果)
4. **预期**: 机器人应该能站立,不再摔倒

### Step 2: 验证效果 (10分钟)
如果 Step 1 有效:
1. 测试静止命令 - 机器人应该稳定站立
2. 测试前进命令 - 机器人应该尝试行走 (可能还需要其他修复)
3. 对比部署和训练的动作输出

### Step 3: 完整实现 (可选)
如果需要支持动态速度:
1. 实现 `interpolateGaitParams()` 函数
2. 测试不同速度命令的效果
3. 验证与训练环境的一致性

---

## 📊 验证清单

修复后,检查以下内容:

### ✅ Step 0 观测正确性
```
期望: adaptive_phase[-3:] = [0.6667, 0.0, 0.37]
修复前: [0.0, 0.0, 0.0]  ❌
修复后: [0.6667, 0.0, 0.37]  ✅
```

### ✅ Step 0 模型输出改善
```
修复前: 10/14 极端值 (±2)
修复后: 应该接近训练测试的输出
  期望: [-0.08, 0.33, -0.30, 0.71, -0.44, 0.16, -0.23, -0.18, 0.23, 0.00, -0.96, 0.07, 0.42, -0.09]
```

### ✅ 机器人行为改善
```
修复前: 站起后立即摔倒
修复后: 应该能稳定站立
```

---

## 🐛 如果修复后仍有问题

### 情况1: 模型输出仍然极端
- 检查其他观测项是否也有问题 (特别是 gravity_vec 归一化)
- 打印完整的 74 维观测,逐项对比训练测试

### 情况2: 机器人站立但不稳
- 这可能是正常的 (模型需要微调平衡)
- 检查控制频率、PD参数等其他因素

### 情况3: 前进命令无效
- 实现方案2的完整动态计算
- 验证 commands 的计算是否正确

---

## 📋 相关文件

- `ADAPTIVE_PHASE_BUG_ANALYSIS.md` - 详细问题分析
- `verify_adaptive_phase_simple.py` - 验证脚本
- `test_model_output.py` - 模型测试脚本
- `MODEL_TEST_RESULTS_ANALYSIS.md` - 模型测试结果分析

---

## 🎯 预期效果

修复这个 bug 后,结合之前的 gravity_vec 归一化修复:

1. ✅ Step 0 观测完全正确
2. ✅ 模型输出合理动作 (不再全是±2)
3. ✅ 机器人能稳定站立
4. ✅ 响应速度命令 (可能需要进一步调试)

**这两个 bug (gravity_vec + adaptive_phase) 很可能是导致机器人摔倒的主要原因!**

---

**建议立即修复并测试!** 🚀
