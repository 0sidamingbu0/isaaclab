#!/usr/bin/env python3
"""
根据部署日志分析 dof_pos_rel 是否正确
"""

# 从部署日志提取的数据 (Step 0)
raw_dof_pos = [0.13, 0.07, 0.2, 0.052, -0.05, -0.13, -0.07, -0.2, -0.052, 0.05, 0, 0, 0, 0]
default_dof_pos = [0.13, 0.07, 0.2, 0.052, -0.05, -0.13, -0.07, -0.2, -0.052, 0.05, 0, 0, 0, 0]

# 部署日志显示的 dof_pos_rel (应该全是0)
dof_pos_rel_deployment = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print("=" * 80)
print("🔍 分析部署环境的 dof_pos_rel 计算")
print("=" * 80)

print("\n从部署日志 Step 0 提取:")
print(f"raw_dof_pos:     {raw_dof_pos}")
print(f"default_dof_pos: {default_dof_pos}")
print(f"dof_pos_rel:     {dof_pos_rel_deployment}")

# 验证
dof_pos_rel_calculated = [raw - default for raw, default in zip(raw_dof_pos, default_dof_pos)]
print(f"\n手动计算 (raw - default):")
print(f"dof_pos_rel:     {dof_pos_rel_calculated}")

if dof_pos_rel_calculated == dof_pos_rel_deployment:
    print("\n✅ dof_pos_rel 计算正确: 全是 0")
else:
    print(f"\n❌ dof_pos_rel 计算错误!")
    print(f"期望: {dof_pos_rel_calculated}")
    print(f"实际: {dof_pos_rel_deployment}")

# 检查顺序
print("\n" + "=" * 80)
print("🔍 检查训练顺序问题")
print("=" * 80)

print("\n问题: raw_dof_pos 和 default_dof_pos 的顺序是什么?")

print("\n假设1: 都是训练顺序 [L1-L5, R1-R5, N1-N4]")
print("  raw_dof_pos[0] = 0.13  → L1 = 0.13")
print("  raw_dof_pos[5] = -0.13 → R1 = -0.13")
print("  这和 default_dof_pos 一致")
print("  → dof_pos_rel = 0 ✅")

print("\n假设2: raw 是 URDF 顺序 [R1-R5, L1-L5, N1-N4], default 是训练顺序")
print("  raw_dof_pos[0] = 0.13  → R1 (URDF) = 0.13")
print("  default_dof_pos[0] = 0.13 → L1 (训练) = 0.13")
print("  但 R1 的 default 应该是 -0.13!")
print("  → 巧合: 0.13 - 0.13 = 0, 但逻辑错了!")

print("\n⚠️ 关键问题:")
print("部署日志显示 raw_dof_pos 和 default_dof_pos **数值完全相同**")
print("但可能是:")
print("  1. 真的都是训练顺序, dof_pos_rel=0 正确 ✅")
print("  2. raw 是 URDF 顺序, default 是训练顺序, 但数值巧合相同 ❌")

print("\n" + "=" * 80)
print("🔍 如何验证?")
print("=" * 80)

print("\n方法: 让机器人移动一点,再看 dof_pos_rel")
print("  如果 Step 50 的 dof_pos_rel 仍然看起来合理 → 可能正确")
print("  如果 Step 50 的 dof_pos_rel 变成很大的值 → 顺序错误!")

print("\n从之前的日志 (Step 50):")
print("  Left leg  (L1-L5): [0.3063, 0.1514, -0.6750, 0.1801, -0.9340]")
print("  Right leg (R1-R5): [-0.3069, -0.6799, 0.6986, -0.1535, -0.8831]")
print("  Neck      (N1-N4): [-0.6448, 0.9417, 0.4789, 0.7657]")

print("\n这些值看起来合理 (<1.0 范围内), 可能顺序是对的")

print("\n" + "=" * 80)
print("🎯 新的推测")
print("=" * 80)

print("\n既然:")
print("  1. dof_pos_rel 看起来合理")
print("  2. gravity_vec 在 Step 0 正确")
print("  3. adaptive_phase 错误但影响小")
print("  4. 训练环境用相同观测输出正常")
print("  5. 部署环境用相同观测输出极端")

print("\n那么问题可能在:")
print("  ❌ 不是观测计算错误")
print("  ✅ 可能是:")
print("     1. 模型加载/推理方式不同")
print("     2. 数据类型不一致 (float32 vs float64)")
print("     3. 内存布局/字节序不同")
print("     4. LibTorch C++ 推理有 bug")
print("     5. 观测向量传递给模型时出错")

print("\n" + "=" * 80)
print("🔧 建议测试")
print("=" * 80)

print("\n测试1: 部署环境中保存观测到文件")
print("```cpp")
print("// 保存观测到文件")
print("std::ofstream outfile(\"observation_step0.txt\");")
print("for (int i = 0; i < 74; i++) {")
print("    outfile << observation[i] << std::endl;")
print("}")
print("outfile.close();")
print("```")

print("\n测试2: Python 加载这个文件测试模型")
print("```python")
print("obs = np.loadtxt('observation_step0.txt')")
print("action = model(torch.from_numpy(obs).unsqueeze(0).float())")
print("print(action)  # 看是否输出极端值")
print("```")

print("\n如果 Python 测试输出正常 → LibTorch C++ 推理有问题!")
print("如果 Python 测试也极端 → 观测确实有错!")

print("\n✅ 这个测试能直接定位问题!")
