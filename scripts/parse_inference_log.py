#!/usr/bin/env python3
"""
解析 Isaac Lab 推理日志文件
用于对比不同环境(Isaac Lab vs Gazebo)的推理结果

使用方法:
    python scripts/parse_inference_log.py inference_log_20251105_115608.txt
    
    # 对比两个日志
    python scripts/parse_inference_log.py isaac_log.txt gazebo_log.txt --compare
"""

import argparse
import numpy as np
import sys


def parse_log_file(filename):
    """解析日志文件,返回步数、观测、动作和信息的列表"""
    steps = []
    observations = []
    actions = []
    infos = []
    
    with open(filename, 'r') as f:
        current_step = None
        current_obs = None
        current_act = None
        current_info = None
        
        for line in f:
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('#') or line.startswith('='):
                continue
            
            if line.startswith('STEP '):
                current_step = int(line.split()[1])
            
            elif line.startswith('OBS '):
                obs_values = [float(x) for x in line.split()[1:]]
                current_obs = np.array(obs_values)
            
            elif line.startswith('ACT '):
                act_values = [float(x) for x in line.split()[1:]]
                current_act = np.array(act_values)
            
            elif line.startswith('INFO '):
                # 解析INFO行
                info_dict = {}
                parts = line.split()[1:]  # 跳过 'INFO'
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        info_dict[key] = value
                current_info = info_dict
                
                # 当收集完一个完整的step时保存
                if current_step is not None and current_obs is not None and current_act is not None:
                    steps.append(current_step)
                    observations.append(current_obs)
                    actions.append(current_act)
                    infos.append(current_info)
                    
                    # 重置
                    current_step = None
                    current_obs = None
                    current_act = None
                    current_info = None
    
    return steps, observations, actions, infos


def print_summary(filename, steps, observations, actions, infos):
    """打印日志摘要统计"""
    print(f"\n{'='*80}")
    print(f"📊 日志文件分析: {filename}")
    print(f"{'='*80}")
    print(f"总步数: {len(steps)}")
    
    if len(steps) == 0:
        print("⚠️  日志为空!")
        return
    
    print(f"步数范围: {steps[0]} - {steps[-1]}")
    
    # 观测统计
    obs_array = np.array(observations)
    print(f"\n📈 观测统计 (维度: {obs_array.shape[1]}):")
    print(f"  最小值: {obs_array.min():.6f}")
    print(f"  最大值: {obs_array.max():.6f}")
    print(f"  均值: {obs_array.mean():.6f}")
    print(f"  标准差: {obs_array.std():.6f}")
    
    # 关键观测分量
    ang_vel = obs_array[:, 0:3]
    gravity = obs_array[:, 3:6]
    dof_pos = obs_array[:, 6:20]
    
    print(f"\n  角速度范围: [{ang_vel.min():.3f}, {ang_vel.max():.3f}]")
    print(f"  重力向量范围: [{gravity.min():.6f}, {gravity.max():.6f}]")
    print(f"  关节位置范围: [{dof_pos.min():.3f}, {dof_pos.max():.3f}]")
    
    # 动作统计
    act_array = np.array(actions)
    print(f"\n🎯 动作统计 (维度: {act_array.shape[1]}):")
    print(f"  最小值: {act_array.min():.6f}")
    print(f"  最大值: {act_array.max():.6f}")
    print(f"  均值: {act_array.mean():.6f}")
    print(f"  标准差: {act_array.std():.6f}")
    
    # 检查是否有极端值
    extreme_threshold = 10.0
    extreme_actions = np.abs(act_array) > extreme_threshold
    if extreme_actions.any():
        print(f"\n⚠️  发现 {extreme_actions.sum()} 个极端动作值 (|value| > {extreme_threshold})!")
        extreme_steps = np.where(extreme_actions.any(axis=1))[0]
        print(f"  出现在步数: {[steps[i] for i in extreme_steps[:10]]}...")
    else:
        print(f"\n✅ 所有动作值在正常范围内 (|value| <= {extreme_threshold})")
    
    # INFO 统计
    if infos:
        print(f"\n🔍 姿态信息统计:")
        tilts = [float(info['tilt']) for info in infos if 'tilt' in info]
        rolls = [float(info['roll']) for info in infos if 'roll' in info]
        pitches = [float(info['pitch']) for info in infos if 'pitch' in info]
        heights = [float(info['height']) for info in infos if 'height' in info]
        
        if tilts:
            print(f"  倾斜角 (Tilt): 均值={np.mean(tilts):.2f}°, 范围=[{np.min(tilts):.2f}°, {np.max(tilts):.2f}°]")
        if rolls:
            print(f"  侧倾角 (Roll): 均值={np.mean(rolls):.2f}°, 范围=[{np.min(rolls):.2f}°, {np.max(rolls):.2f}°]")
        if pitches:
            print(f"  俯仰角 (Pitch): 均值={np.mean(pitches):.2f}°, 范围=[{np.min(pitches):.2f}°, {np.max(pitches):.2f}°]")
        if heights:
            print(f"  高度 (Height): 均值={np.mean(heights):.3f}m, 范围=[{np.min(heights):.3f}m, {np.max(heights):.3f}m]")


def compare_logs(file1, file2):
    """对比两个日志文件"""
    print(f"\n{'='*80}")
    print(f"🔍 对比两个日志文件")
    print(f"{'='*80}")
    
    steps1, obs1, act1, info1 = parse_log_file(file1)
    steps2, obs2, act2, info2 = parse_log_file(file2)
    
    print(f"\n文件1: {file1} - {len(steps1)} 步")
    print(f"文件2: {file2} - {len(steps2)} 步")
    
    # 找到共同的步数
    common_steps = min(len(steps1), len(steps2))
    if common_steps == 0:
        print("\n⚠️  没有可对比的数据!")
        return
    
    print(f"\n对比前 {common_steps} 步...")
    
    # 转换为数组
    obs1_array = np.array(obs1[:common_steps])
    obs2_array = np.array(obs2[:common_steps])
    act1_array = np.array(act1[:common_steps])
    act2_array = np.array(act2[:common_steps])
    
    # 计算差异
    obs_diff = obs1_array - obs2_array
    act_diff = act1_array - act2_array
    
    print(f"\n📊 观测差异:")
    print(f"  最大绝对差异: {np.abs(obs_diff).max():.6f}")
    print(f"  均方根误差: {np.sqrt((obs_diff**2).mean()):.6f}")
    print(f"  平均绝对差异: {np.abs(obs_diff).mean():.6f}")
    
    print(f"\n🎯 动作差异:")
    print(f"  最大绝对差异: {np.abs(act_diff).max():.6f}")
    print(f"  均方根误差: {np.sqrt((act_diff**2).mean()):.6f}")
    print(f"  平均绝对差异: {np.abs(act_diff).mean():.6f}")
    
    # 找出差异最大的步数
    max_act_diff_idx = np.abs(act_diff).max(axis=1).argmax()
    print(f"\n⚠️  动作差异最大的步数: {steps1[max_act_diff_idx]}")
    print(f"  文件1动作: {act1_array[max_act_diff_idx][:5]} ...")
    print(f"  文件2动作: {act2_array[max_act_diff_idx][:5]} ...")
    print(f"  差异: {act_diff[max_act_diff_idx][:5]} ...")
    
    # 逐步对比前几步
    print(f"\n📋 前5步详细对比:")
    for i in range(min(5, common_steps)):
        print(f"\n  步数 {steps1[i]}:")
        print(f"    观测差异: max={np.abs(obs_diff[i]).max():.6f}, mean={np.abs(obs_diff[i]).mean():.6f}")
        print(f"    动作差异: max={np.abs(act_diff[i]).max():.6f}, mean={np.abs(act_diff[i]).mean():.6f}")
        print(f"    文件1动作[0:3]: [{act1_array[i,0]:.3f}, {act1_array[i,1]:.3f}, {act1_array[i,2]:.3f}]")
        print(f"    文件2动作[0:3]: [{act2_array[i,0]:.3f}, {act2_array[i,1]:.3f}, {act2_array[i,2]:.3f}]")


def main():
    parser = argparse.ArgumentParser(description="解析和对比 Isaac Lab 推理日志")
    parser.add_argument('logfile1', help='第一个日志文件')
    parser.add_argument('logfile2', nargs='?', help='第二个日志文件 (用于对比)')
    parser.add_argument('--compare', action='store_true', help='对比两个日志文件')
    parser.add_argument('--steps', type=int, help='只显示前N步')
    
    args = parser.parse_args()
    
    # 解析第一个文件
    try:
        steps1, obs1, act1, info1 = parse_log_file(args.logfile1)
        print_summary(args.logfile1, steps1, obs1, act1, info1)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {args.logfile1}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 解析文件失败: {e}")
        sys.exit(1)
    
    # 如果提供了第二个文件,进行对比
    if args.logfile2:
        try:
            steps2, obs2, act2, info2 = parse_log_file(args.logfile2)
            print_summary(args.logfile2, steps2, obs2, act2, info2)
            compare_logs(args.logfile1, args.logfile2)
        except FileNotFoundError:
            print(f"❌ 文件不存在: {args.logfile2}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 解析文件失败: {e}")
            sys.exit(1)
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
