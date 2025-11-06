#!/usr/bin/env python3
"""
测试模型输出是否正常
使用训练环境直接测试模型在 default_dof_pos 时的输出
"""

import torch
import argparse
from pathlib import Path


def test_model_at_default_pose(model_path: str):
    """测试模型在默认姿态时的输出"""
    
    print("=" * 80)
    print("🧪 模型输出测试 - 默认站立姿态")
    print("=" * 80)
    
    # 加载模型
    print(f"\n📦 加载模型: {model_path}")
    try:
        model = torch.jit.load(model_path)
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 构建观测向量 (74 维)
    # 对应默认站立姿态：dof_pos = default，速度全零，静止命令
    
    print("\n" + "=" * 80)
    print("📋 构建测试观测向量")
    print("=" * 80)
    
    observation = torch.tensor([[
        # 1. ang_vel_body (3) - 无旋转
        0.0, 0.0, 0.0,
        
        # 2. gravity_vec (3) - 直立，归一化
        0.0, 0.0, -1.0,
        
        # 3. dof_pos_rel (14) - 全零（因为在 default_dof_pos）
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        
        # 4. dof_vel_scaled (14) - 全零（静止）
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        
        # 5. joint_torques (14) - 全零
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        
        # 6. commands (3) - 静止命令
        0.0, 0.0, 0.0,
        
        # 7. last_actions (14) - 全零（初始）
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        
        # 8. adaptive_phase (9) - 初始相位
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37
    ]], dtype=torch.float32)
    
    print(f"观测维度: {observation.shape}")
    print(f"观测范围: [{observation.min():.4f}, {observation.max():.4f}]")
    
    # 显示观测内容
    obs_np = observation.squeeze().numpy()
    print("\n观测内容分解:")
    print(f"  ang_vel_body:    {obs_np[0:3]}")
    print(f"  gravity_vec:     {obs_np[3:6]}")
    print(f"  dof_pos_rel:     {obs_np[6:20]}")
    print(f"  dof_vel_scaled:  {obs_np[20:34]}")
    print(f"  joint_torques:   {obs_np[34:48]}")
    print(f"  commands:        {obs_np[48:51]}")
    print(f"  last_actions:    {obs_np[51:65]}")
    print(f"  adaptive_phase:  {obs_np[65:74]}")
    
    # 模型推理
    print("\n" + "=" * 80)
    print("🤖 模型推理")
    print("=" * 80)
    
    with torch.no_grad():
        actions = model(observation)
    
    actions_np = actions.squeeze().numpy()
    
    print(f"\n动作输出维度: {actions.shape}")
    print(f"动作范围: [{actions.min():.4f}, {actions.max():.4f}]")
    
    # 详细显示每个关节的动作
    print("\n动作输出详情 (训练顺序):")
    joint_names = [
        "L1", "L2", "L3", "L4", "L5",  # 左腿
        "R1", "R2", "R3", "R4", "R5",  # 右腿
        "N1", "N2", "N3", "N4"         # 颈部
    ]
    
    print("\n  左腿 (L1-L5):")
    for i in range(5):
        print(f"    {joint_names[i]}: {actions_np[i]:7.4f}")
    
    print("\n  右腿 (R1-R5):")
    for i in range(5, 10):
        print(f"    {joint_names[i]}: {actions_np[i]:7.4f}")
    
    print("\n  颈部 (N1-N4):")
    for i in range(10, 14):
        print(f"    {joint_names[i]}: {actions_np[i]:7.4f}")
    
    # 分析结果
    print("\n" + "=" * 80)
    print("📊 结果分析")
    print("=" * 80)
    
    # 统计极端值
    extreme_count = sum(abs(actions_np) > 1.5)
    
    print(f"\n✅ 合理性检查:")
    print(f"  - 动作范围: [{actions_np.min():.4f}, {actions_np.max():.4f}]")
    print(f"  - 动作平均值: {actions_np.mean():.4f}")
    print(f"  - 动作标准差: {actions_np.std():.4f}")
    print(f"  - 极端值数量 (|action| > 1.5): {extreme_count} / 14")
    
    if extreme_count > 10:
        print("\n❌ 异常！超过 10 个关节输出极端值")
        print("   这在静止状态下不正常")
    elif extreme_count > 5:
        print("\n⚠️  警告！有 5+ 个关节输出较大动作")
        print("   可能模型训练时需要大动作来维持平衡")
    else:
        print("\n✅ 正常！大部分动作值较小")
        print("   模型在默认姿态下表现稳定")
    
    # 计算目标位置
    print("\n" + "=" * 80)
    print("🎯 目标关节位置 (default + action * 0.5)")
    print("=" * 80)
    
    default_dof_pos = [
        0.13, 0.07, 0.2, 0.052, -0.05,    # 左腿
        -0.13, -0.07, -0.2, -0.052, 0.05,  # 右腿
        0.0, 0.0, 0.0, 0.0                 # 颈部
    ]
    
    action_scale = 0.5
    target_pos = [default_dof_pos[i] + actions_np[i] * action_scale for i in range(14)]
    
    print("\n  关节 | Default  | Action   | Target   | Deviation")
    print("  " + "-" * 60)
    for i in range(14):
        deviation = target_pos[i] - default_dof_pos[i]
        print(f"  {joint_names[i]:4s} | {default_dof_pos[i]:8.3f} | {actions_np[i]:8.3f} | "
              f"{target_pos[i]:8.3f} | {deviation:8.3f}")
    
    max_deviation = max(abs(target_pos[i] - default_dof_pos[i]) for i in range(14))
    print(f"\n  最大偏离: {max_deviation:.3f} rad")
    
    if max_deviation > 0.5:
        print("\n  ⚠️  警告：有关节偏离 default 超过 0.5 rad")
    
    return actions_np


def test_model_with_velocity_command(model_path: str, vx: float = 0.5):
    """测试模型在给定速度命令时的输出"""
    
    print("\n" + "=" * 80)
    print(f"🧪 模型输出测试 - 前进命令 (vx={vx})")
    print("=" * 80)
    
    # 加载模型
    print(f"\n📦 加载模型: {model_path}")
    model = torch.jit.load(model_path)
    model.eval()
    
    # 构建观测向量
    observation = torch.tensor([[
        0.0, 0.0, 0.0,  # ang_vel
        0.0, 0.0, -1.0,  # gravity
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dof_pos_rel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # dof_vel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # torques
        vx, 0.0, 0.0,  # commands - 前进命令!
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # actions
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37  # phase
    ]], dtype=torch.float32)
    
    print(f"观测维度: {observation.shape}")
    print(f"速度命令: vx={vx}, vy=0, vyaw=0")
    
    # 模型推理
    with torch.no_grad():
        actions = model(observation)
    
    actions_np = actions.squeeze().numpy()
    
    print(f"\n动作范围: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"动作平均值: {actions_np.mean():.4f}")
    
    # 显示动作
    joint_names = ["L1", "L2", "L3", "L4", "L5", "R1", "R2", "R3", "R4", "R5", "N1", "N2", "N3", "N4"]
    print("\n动作输出:")
    for i in range(14):
        print(f"  {joint_names[i]}: {actions_np[i]:7.4f}")
    
    extreme_count = sum(abs(actions_np) > 1.5)
    print(f"\n极端值数量 (|action| > 1.5): {extreme_count} / 14")
    
    if extreme_count > 10:
        print("❌ 异常！前进命令下也不应该全是极端值")
    else:
        print("✅ 正常！模型在前进命令下输出合理的步态动作")
    
    return actions_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试模型输出")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径 (e.g., logs/rsl_rl/.../model_10000.pt)")
    parser.add_argument("--test-velocity", action="store_true", help="是否测试速度命令")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ 模型文件不存在: {args.model}")
        exit(1)
    
    # 测试 1：默认姿态，静止命令
    print("\n" + "=" * 80)
    print("📋 测试 1: 默认站立姿态 + 静止命令")
    print("=" * 80)
    actions_static = test_model_at_default_pose(args.model)
    
    # 测试 2：默认姿态，前进命令
    if args.test_velocity:
        print("\n" + "=" * 80)
        print("📋 测试 2: 默认站立姿态 + 前进命令")
        print("=" * 80)
        actions_forward = test_model_with_velocity_command(args.model, vx=0.5)
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)
    print("\n💡 如果静止命令下输出大量极端值，可能的原因:")
    print("  1. 模型训练时的 default_dof_pos 与此不同")
    print("  2. 观测向量拼接顺序不对")
    print("  3. 某些观测项的缩放/归一化不对")
    print("  4. 模型导出时有问题")
    print("\n📤 请将此测试结果发给部署 AI 对比")
