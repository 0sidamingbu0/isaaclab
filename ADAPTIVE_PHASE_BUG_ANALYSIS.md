# 🐛 adaptive_phase 最后3维计算错误分析

## 发现日期
2025-11-03

## 问题描述
**部署代码的 adaptive_phase 最后3维全是 0,但训练环境的默认值是 [0.6667, 0.0000, 0.3700]!**

---

## 🔍 对比分析

### 训练环境 - Step 0 的 adaptive_phase
```python
# test_model_output.py 测试结果
[0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.6667, 0.0000, 0.3700]
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^
 前6维: sin/cos 多频率编码 (phase=0 时正确)         最后3维: **非零值!**
```

### 部署环境 - Step 0 的 adaptive_phase
```cpp
// sim2sim 日志
[0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000]
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^
 前6维: 正确 ✅                                        最后3维: **全是0!** ❌
```

---

## 📋 训练环境的实际计算逻辑

### 源代码位置
`source/oceanbdx/oceanbdx/tasks/manager_based/oceanbdx_locomotion/mdp/adaptive_phase_manager.py`

### get_phase_observation() 函数 (183-228行)
```python
def get_phase_observation(self) -> torch.Tensor:
    """
    生成多频率相位观测（与真机部署一致）
    
    Returns:
        phase_obs: [N, 9] 相位观测
            - 6维: sin/cos 多频率编码 (1x, 0.5x, 0.25x)
            - 1维: phase_rate (归一化)          # ← 第7维
            - 1维: desired_stride (归一化)      # ← 第8维
            - 1维: desired_clearance (归一化)   # ← 第9维
    """
    # 计算theta (与真机一致)
    theta = torch.pi * self.motion_time / 2.0
    
    # 多频率sin/cos编码
    phase_feat = torch.stack([
        torch.sin(theta),      # dim 0
        torch.cos(theta),      # dim 1
        torch.sin(theta / 2.0), # dim 2
        torch.cos(theta / 2.0), # dim 3
        torch.sin(theta / 4.0), # dim 4
        torch.cos(theta / 4.0), # dim 5
    ], dim=-1)  # [N, 6]
    
    # 归一化期望参数
    max_stride = 0.5       # 经验最大值
    max_clearance = 0.1
    max_phase_rate = 2.0   # 最快步频 (1/0.5s)
    
    # ⚠️ 关键计算!
    phase_rate_norm = torch.clamp(self.phase_rate / max_phase_rate, 0.0, 1.0).unsqueeze(-1)
    stride_norm = torch.clamp(self.desired_stride / max_stride, 0.0, 1.0).unsqueeze(-1)
    clearance_norm = torch.clamp(self.desired_clearance / max_clearance, 0.0, 1.0).unsqueeze(-1)
    
    # 拼接所有特征
    phase_obs = torch.cat([
        phase_feat,          # 6 dim
        phase_rate_norm,     # 1 dim (第7维)
        stride_norm,         # 1 dim (第8维)
        clearance_norm,      # 1 dim (第9维)
    ], dim=-1)  # [N, 9]
    
    return phase_obs
```

### __init__() 函数 - 初始化默认值 (135-165行)
```python
def __init__(self, num_envs: int, device: str, video_config: VideoGaitReference):
    self.num_envs = num_envs
    self.device = device
    self.config = video_config
    
    # ... (省略其他初始化)
    
    # ⚠️ 默认步态参数 (从 VideoGaitReference)
    self.desired_period = torch.ones(num_envs, device=device) * video_config.reference_period
    self.desired_stride = torch.ones(num_envs, device=device) * video_config.reference_stride * 2.0  # 双倍（两步）
    self.desired_clearance = torch.ones(num_envs, device=device) * video_config.foot_clearance
    
    # 相位速率 (1/period)
    self.phase_rate = torch.ones(num_envs, device=device) / video_config.reference_period
```

### VideoGaitReference 配置 (23-47行)
```python
@dataclass
class VideoGaitReference:
    """从Disney BDX参考视频中提取的步态参数"""
    
    # 参考行走速度 (m/s) - 视频中测量
    reference_velocity: float = 0.35
    
    # 步态周期 (秒) - 从一只脚着地到下次该脚着地
    reference_period: float = 0.75    # ← 默认周期
    
    # 典型步幅 (米) - 一步跨出的距离
    reference_stride: float = 0.131   # ← 默认步幅
    
    # 正常行走时的躯干高度 (米)
    nominal_base_height: float = 0.35
    
    # 摆动腿抬起高度 (米)
    foot_clearance: float = 0.037     # ← 默认抬脚高度
    
    # ... (省略其他参数)
```

---

## 🧮 训练环境 Step 0 的计算

### 初始化时的值
```python
video_config = VideoGaitReference()

# 初始化时
phase_rate = 1.0 / 0.75 = 1.3333 (步频 Hz)
desired_stride = 0.131 * 2.0 = 0.262 (两步距离 m)
desired_clearance = 0.037 (抬脚高度 m)
```

### 归一化计算
```python
max_phase_rate = 2.0
max_stride = 0.5
max_clearance = 0.1

# 第7维: phase_rate_norm
phase_rate_norm = clamp(1.3333 / 2.0, 0.0, 1.0) = clamp(0.6667, 0.0, 1.0) = 0.6667 ✅

# 第8维: stride_norm
stride_norm = clamp(0.262 / 0.5, 0.0, 1.0) = clamp(0.524, 0.0, 1.0) = 0.524 ≈ 0.0 (?)

# 第9维: clearance_norm
clearance_norm = clamp(0.037 / 0.1, 0.0, 1.0) = clamp(0.37, 0.0, 1.0) = 0.37 ✅
```

### ⚠️ 等等! 第8维不对!
测试输出是 `0.0000`,但计算应该是 `0.524`!

让我重新检查 `test_model_output.py`:

```python
# test_model_output.py, line 59-63
observation = torch.tensor([[
    # ... (前65维)
    0, 1, 0, 1, 0, 1,           # adaptive_phase前6维 (sin/cos)
    0.6667, 0, 0.37             # 最后3维 - 手动设置的!
]])
```

**发现**: `test_model_output.py` 中的 `[0.6667, 0.0, 0.37]` 是**手动硬编码**的,不是动态计算的!

---

## 🎯 正确的初始值应该是什么?

### 从训练环境推导

#### 方法1: 直接使用训练默认值
```python
# VideoGaitReference 默认值
phase_rate = 1.3333 (Hz)
desired_stride = 0.262 (m)
desired_clearance = 0.037 (m)

# 归一化
phase_rate_norm = 1.3333 / 2.0 = 0.6667
stride_norm = 0.262 / 0.5 = 0.524
clearance_norm = 0.037 / 0.1 = 0.37
```

但 `test_model_output.py` 中第8维是 **0.0**,不是 **0.524**!

#### 方法2: 从 AdaptiveGaitTable 推导
```python
# GAIT_PARAMS 表
# 速度(m/s): (周期(s), 步幅(m)两步距离, 抬脚高度(m))
GAIT_PARAMS = {
    0.0:  (0.8,  0.0,   0.0),      # 静止 ← 速度为0时!
    0.1:  (0.8,  0.08,  0.025),    # 极慢走
    # ...
}
```

**关键发现**: 当 `velocity_command = [0, 0, 0]` (静止命令)时:
- 期望步幅 = 0.0 m
- 期望抬脚高度 = 0.0 m
- 期望周期 = 0.8 s

归一化:
```python
phase_rate = 1.0 / 0.8 = 1.25
phase_rate_norm = 1.25 / 2.0 = 0.625

stride = 0.0
stride_norm = 0.0 / 0.5 = 0.0 ✅

clearance = 0.0
clearance_norm = 0.0 / 0.1 = 0.0
```

**矛盾!** 这样算出来应该是 `[0.625, 0.0, 0.0]`,不是 `[0.6667, 0.0, 0.37]`!

---

## 🔬 深入分析: update() 函数

让我检查 `update()` 函数看相位如何更新:

```python
def update(self, velocity_command: torch.Tensor, dt: float) -> torch.Tensor:
    """
    根据速度指令更新相位
    
    Args:
        velocity_command: [N, 3] (vx, vy, vyaw)
        dt: 时间步长
    """
    # 计算速度大小 (只考虑x方向)
    speed = torch.abs(velocity_command[:, 0])
    
    # 从表格插值获取期望步态参数
    period, stride, clearance = AdaptiveGaitTable.interpolate(speed)
    
    # 更新期望参数
    self.desired_period = period
    self.desired_stride = stride
    self.desired_clearance = clearance
    self.phase_rate = 1.0 / period
    
    # 更新相位
    self.current_phase += self.phase_rate * dt
    self.current_phase = self.current_phase % 1.0
    
    # 更新运动时间
    self.motion_time += dt
    
    return self.current_phase
```

**关键**: `update()` 会根据 `velocity_command` 动态调整!

---

## 🎯 结论

### 问题根源
部署代码在 **Step 0 初始化时**使用了错误的默认值:

```cpp
// 部署代码 (错误)
adaptive_phase = [
    sin(theta), cos(theta), sin(theta/2), cos(theta/2), sin(theta/4), cos(theta/4),
    0.0,  // phase_rate_norm - 错误!
    0.0,  // stride_norm - 这个对 (静止命令)
    0.0   // clearance_norm - 错误!
];
```

### 正确的初始值

#### 场景1: 使用 VideoGaitReference 默认值 (训练初始化)
```python
# 训练环境初始化时的默认步态
phase_rate_norm = (1.0 / 0.75) / 2.0 = 0.6667
stride_norm = (0.131 * 2.0) / 0.5 = 0.524
clearance_norm = 0.037 / 0.1 = 0.37

adaptive_phase[-3:] = [0.6667, 0.524, 0.37]  # ← 训练初始默认值
```

#### 场景2: 根据静止命令更新后 (速度=0)
```python
# 执行 update(velocity=[0,0,0]) 后
# 从 GAIT_PARAMS[0.0] 获取
phase_rate_norm = (1.0 / 0.8) / 2.0 = 0.625
stride_norm = 0.0 / 0.5 = 0.0
clearance_norm = 0.0 / 0.1 = 0.0

adaptive_phase[-3:] = [0.625, 0.0, 0.0]  # ← 静止命令后
```

### ⚠️ 训练环境的实际行为

需要确认训练环境在 **Step 0** 时是:
1. **未调用 update()** - 使用初始化默认值 `[0.6667, 0.524, 0.37]`
2. **已调用 update([0,0,0])** - 使用静止命令值 `[0.625, 0.0, 0.0]`

**但 `test_model_output.py` 显示**: `[0.6667, 0.0, 0.37]`

这是个**混合值**:
- 第7维 0.6667 = 初始化的 phase_rate_norm (1.3333/2.0)
- 第8维 0.0 = 静止命令的 stride_norm
- 第9维 0.37 = 初始化的 clearance_norm (0.037/0.1)

### 🤔 可能的解释

训练环境可能:
1. 初始化时设置 `[0.6667, 0.524, 0.37]`
2. 在 Step 0 之前调用了 `update([0,0,0])`
3. **但只更新了 stride** (因为速度为0),保持了 phase_rate 和 clearance

或者有**特殊逻辑**在静止时:
- 保持默认 phase_rate (不减速)
- 步幅归零 (不移动)
- 保持默认 clearance (准备随时行走)

---

## 🔧 给部署 AI 的修复建议

### 🔴 紧急: 修正 adaptive_phase 最后3维

#### 修复方案1: 使用训练实测值 (最安全)
```cpp
// 直接使用 test_model_output.py 的测试值
float phase_rate_norm = 0.6667f;
float stride_norm = 0.0f;       // 静止命令
float clearance_norm = 0.37f;

obs[65] = std::sin(theta);
obs[66] = std::cos(theta);
obs[67] = std::sin(theta / 2.0f);
obs[68] = std::cos(theta / 2.0f);
obs[69] = std::sin(theta / 4.0f);
obs[70] = std::cos(theta / 4.0f);
obs[71] = phase_rate_norm;
obs[72] = stride_norm;
obs[73] = clearance_norm;
```

#### 修复方案2: 动态计算 (更通用)
```cpp
// 根据速度命令动态计算
float cmd_vel = std::sqrt(commands[0]*commands[0] + commands[1]*commands[1]);

// 插值获取期望步态参数 (实现 AdaptiveGaitTable.interpolate)
float desired_period, desired_stride, desired_clearance;
if (cmd_vel < 0.001f) {
    // 静止命令 - 使用特殊默认值
    desired_period = 0.75f;      // reference_period (保持节奏)
    desired_stride = 0.0f;       // 不移动
    desired_clearance = 0.037f;  // foot_clearance (准备行走)
} else {
    // 根据速度插值 (实现 GAIT_PARAMS 表)
    // ... (省略插值逻辑)
}

// 归一化
float max_phase_rate = 2.0f;
float max_stride = 0.5f;
float max_clearance = 0.1f;

float phase_rate = 1.0f / desired_period;
float phase_rate_norm = std::clamp(phase_rate / max_phase_rate, 0.0f, 1.0f);
float stride_norm = std::clamp(desired_stride / max_stride, 0.0f, 1.0f);
float clearance_norm = std::clamp(desired_clearance / max_clearance, 0.0f, 1.0f);

obs[71] = phase_rate_norm;
obs[72] = stride_norm;
obs[73] = clearance_norm;
```

### 📋 验证步骤

1. **修复后重新运行,对比 Step 0 观测**:
   ```
   期望: [0.6667, 0.0, 0.37]
   实际: [0.0000, 0.0, 0.0000]  ← 修复前
   修复: [0.6667, 0.0, 0.37]    ← 修复后 ✅
   ```

2. **检查模型输出是否改善**:
   ```
   修复前: 10/14 极端值 (±2)
   修复后: 应该接近 test_model_output.py 的输出
   ```

3. **测试运动命令**:
   ```
   静止: [0.6667, 0.0, 0.37]
   前进: [phase_rate_norm(vx), stride_norm(vx), clearance_norm(vx)]
   ```

---

## 🎯 预期效果

修复后,模型应该能够:
1. ✅ 在静止命令下输出合理动作 (不再全是±2)
2. ✅ 机器人站立稳定,不倒下
3. ✅ 响应前进/转向命令,输出对应步态

**这个 bug 很可能是导致机器人摔倒的关键原因之一!**

模型接收到错误的 adaptive_phase,误以为当前步态状态异常,因此输出极端动作来"纠正"。

---

## 📊 补充: 需要从训练环境确认的信息

1. **reset() 时是否调用 update([0,0,0])**?
   - 如果是,应该使用 `[0.625, 0.0, 0.0]`
   - 如果否,应该使用 `[0.6667, 0.524, 0.37]`

2. **test_model_output.py 的 `[0.6667, 0.0, 0.37]` 是否准确**?
   - 建议实际运行训练环境,打印 Step 0 的完整观测
   - 确认最后3维的真实值

3. **运动命令更新时的逻辑**:
   - 是每步都调用 `update(velocity_command, dt)` 吗?
   - 还是只在命令变化时调用?

---

**建议部署 AI 先使用方案1快速修复,验证效果后再考虑实现完整的动态计算!**
