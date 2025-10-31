# 训练监控脚本使用指南

## 🚀 快速开始

### 1. 基础使用 (自动查找最新日志)
```bash
python scripts/monitor_training.py
```

### 2. 指定日志目录
```bash
python scripts/monitor_training.py --log_dir logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-12-03
```

### 3. 自定义检查间隔
```bash
# 每10秒检查一次
python scripts/monitor_training.py --interval 10
```

### 4. 保存监控报告
```bash
python scripts/monitor_training.py --save_report
```

## 📊 功能特性

### ✅ 实时监控
- 📈 平均奖励趋势
- 📏 Episode长度变化
- ⚠️ 摔倒率统计
- ⚡ 训练速度 (steps/s)
- ⏰ 预计剩余时间

### 🎓 课程追踪
- 当前训练阶段 (Stage 0-3)
- 训练进度百分比
- 阶段切换自动提醒

### 🎯 阶段特定指标
**Stage 0 (0-20%): 站立稳定**
- 高度跟踪 (目标: >0.3)
- 姿态控制 (目标: >-0.5)
- 验证速度跟踪关闭

**Stage 1 (20-45%): 学习行走**
- 速度跟踪 (目标: >0.5)
- 交替步态 (目标: >0.1)
- 重心转移 (目标: >0.1)

**Stage 2 (45-75%): 优化步态**
- 速度跟踪 (目标: >1.0)
- 交替步态 (目标: >0.3)
- 步长跟踪 (目标: >0.2)

**Stage 3 (75-100%): 精细调节**
- 速度跟踪 (目标: >1.5)
- 动作平滑 (目标: >-0.03)
- 能耗优化

### 🚨 智能报警
- ⚠️ 奖励异常下降 (>20%)
- ⚫ 训练卡死检测 (5分钟无数据)
- 🔴 摔倒率过高 (>50%)

### 📈 趋势分析
- Episode长度变化趋势
- 高度控制改善情况
- 姿态控制收敛状态

## 🔧 高级选项

### 调整报警阈值
```bash
# 奖励下降30%才报警
python scripts/monitor_training.py --alert_threshold 0.3
```

### 完整命令示例
```bash
python scripts/monitor_training.py \
  --log_dir logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-12-03 \
  --interval 20 \
  --alert_threshold 0.2 \
  --save_report
```

## 📝 输出示例

```
================================================================================
🎓 训练监控 - Iteration 250/10000 (进度: 2.5%, Stage 0)
================================================================================
⏰ 时间: 已训练 12分30秒, 预计剩余 8小时15分
⚡ 性能: 上次更新 5秒前

📊 核心指标:
  ├─ 平均奖励: -75.32 (↑ 8.5%)
  ├─ Episode长度: 48.5步 (↑ 15%)
  └─ 摔倒率: 12.3%

🎯 关键奖励 (Stage 0 - 站立稳定):
  ├─ 高度跟踪: 0.2800 ✅ (目标: >0.30)
  ├─ 姿态控制: -0.5200 ✅ (目标: >-0.50)
  ├─ 速度跟踪: 0.0000 ✅ (目标: >0.00)
  └─ 交替步态: 0.0000 ✅ (目标: >0.00)

📈 趋势分析:
  ✅ Episode长度稳定增长 (+35% in last 50 iters)
  ✅ 高度控制明显改善
  ⚠️ 姿态控制收敛较慢,可能需要调整权重

💾 最新checkpoint: model_250.pt

================================================================================
[30秒后自动刷新... 按Ctrl+C退出]
================================================================================
```

## 💡 使用建议

### 1. 在新终端运行
建议在独立终端窗口运行监控脚本,与训练进程分开:
```bash
# 终端1: 运行训练
python scripts/rsl_rl/train.py --task Isaac-Ocean-BDX-Locomotion-v0 ...

# 终端2: 运行监控 (新开)
python scripts/monitor_training.py
```

### 2. 关键检查点
- **Iteration 100-200**: 初期学习效果
- **Iteration 2000**: Stage 0结束,验证站立
- **Iteration 4500**: Stage 1结束,验证行走
- **Iteration 7500**: Stage 2结束,验证步态
- **Iteration 10000**: 训练完成

### 3. 异常处理
如果出现警报:
- **奖励下降**: 检查是否阶段切换,或权重配置问题
- **训练卡死**: 检查GPU/内存是否溢出
- **摔倒率高**: 正常初期现象,持续>500 iter需关注

## 🐛 故障排除

### 未找到日志文件
```bash
# 手动指定日志目录
python scripts/monitor_training.py --log_dir logs/rsl_rl/oceanbdx_locomotion/<你的日志目录>
```

### TensorBoard未安装
```bash
pip install tensorboard
```

### 权限错误
```bash
chmod +x scripts/monitor_training.py
```

## 📦 依赖项
- Python 3.8+
- tensorboard (可选,推荐安装)

安装依赖:
```bash
pip install tensorboard
```

---

**Happy Training! 🚀**
