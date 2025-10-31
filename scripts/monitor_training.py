#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
训练监控脚本 - 实时追踪训练进度和关键指标
支持TensorBoard日志解析、课程进度追踪、智能报警
"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ TensorBoard未安装，部分功能将不可用")
    print("   安装: pip install tensorboard")


class TrainingMonitor:
    """训练监控器 - 实时追踪和分析训练进度"""
    
    def __init__(
        self,
        log_dir: str,
        interval: int = 30,
        alert_threshold: float = 0.2,
        save_report: bool = False,
    ):
        """
        初始化监控器
        
        Args:
            log_dir: 日志目录路径
            interval: 检查间隔(秒)
            alert_threshold: 奖励下降报警阈值(0-1)
            save_report: 是否保存监控报告
        """
        self.log_dir = Path(log_dir)
        self.interval = interval
        self.alert_threshold = alert_threshold
        self.save_report = save_report
        
        # 查找TensorBoard事件文件
        self.event_file = self._find_latest_event_file()
        if not self.event_file:
            print(f"❌ 未找到日志文件: {log_dir}")
            sys.exit(1)
        
        # 初始化数据存储
        self.history: Dict[str, List[Tuple[int, float]]] = {}
        self.last_iter = -1
        self.last_update_time = time.time()
        self.start_time = time.time()
        
        # 课程阶段定义
        self.stages = {
            0: {"range": (0.0, 0.20), "name": "站立稳定"},
            1: {"range": (0.20, 0.45), "name": "学习行走"},
            2: {"range": (0.45, 0.75), "name": "优化步态"},
            3: {"range": (0.75, 1.00), "name": "精细调节"},
        }
        self.current_stage = 0
        
        print(f"✅ 监控器已启动: {self.log_dir}")
        print(f"📁 日志文件: {self.event_file.name}")
        print(f"⏱️  检查间隔: {interval}秒\n")
    
    def _find_latest_event_file(self) -> Optional[Path]:
        """查找最新的TensorBoard事件文件"""
        event_files = list(self.log_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            return None
        return max(event_files, key=lambda p: p.stat().st_mtime)
    
    def _get_current_stage(self, progress: float) -> int:
        """根据进度返回当前阶段"""
        for stage_id, stage_info in self.stages.items():
            if stage_info["range"][0] <= progress < stage_info["range"][1]:
                return stage_id
        return 3  # 最后阶段
    
    def _load_tensorboard_data(self) -> Dict[str, List[Tuple[int, float]]]:
        """加载TensorBoard数据"""
        if not TENSORBOARD_AVAILABLE:
            return {}
        
        try:
            ea = event_accumulator.EventAccumulator(str(self.event_file))
            ea.Reload()
            
            data = {}
            # 获取所有标量数据
            for tag in ea.Tags()["scalars"]:
                events = ea.Scalars(tag)
                data[tag] = [(e.step, e.value) for e in events]
            
            return data
        except Exception as e:
            print(f"⚠️ 读取TensorBoard数据失败: {e}")
            return {}
    
    def _get_latest_value(self, tag: str) -> Optional[Tuple[int, float]]:
        """获取指定标签的最新值"""
        if tag in self.history and self.history[tag]:
            return self.history[tag][-1]
        return None
    
    def _get_change_rate(self, tag: str, lookback: int = 50) -> Optional[float]:
        """计算指标变化率"""
        if tag not in self.history or len(self.history[tag]) < 2:
            return None
        
        data = self.history[tag]
        if len(data) < lookback:
            old_val = data[0][1]
            new_val = data[-1][1]
        else:
            old_val = data[-lookback][1]
            new_val = data[-1][1]
        
        if abs(old_val) < 1e-6:
            return None
        return (new_val - old_val) / abs(old_val)
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            return f"{int(seconds/60)}分{int(seconds%60)}秒"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}小时{minutes}分"
    
    def _print_header(self, iteration: int, progress: float, stage: int):
        """打印监控头部"""
        print("\n" + "="*80)
        print(f"🎓 训练监控 - Iteration {iteration}/10000 (进度: {progress*100:.1f}%, Stage {stage})")
        print("="*80)
        
        # 时间统计
        elapsed = time.time() - self.start_time
        if iteration > 0:
            eta = (elapsed / iteration) * (10000 - iteration)
            print(f"⏰ 时间: 已训练 {self._format_time(elapsed)}, 预计剩余 {self._format_time(eta)}")
        
        # 性能统计
        time_since_update = time.time() - self.last_update_time
        print(f"⚡ 性能: 上次更新 {int(time_since_update)}秒前")
        print()
    
    def _print_core_metrics(self):
        """打印核心指标"""
        print("📊 核心指标:")
        
        # 平均奖励
        reward = self._get_latest_value("Train/mean_reward")
        if reward:
            change = self._get_change_rate("Train/mean_reward", 50)
            change_str = f"(↑ {change*100:+.1f}%)" if change and change > 0 else f"(↓ {abs(change)*100:.1f}%)" if change else ""
            print(f"  ├─ 平均奖励: {reward[1]:.2f} {change_str}")
        
        # Episode长度
        ep_len = self._get_latest_value("Train/mean_episode_length")
        if ep_len:
            change = self._get_change_rate("Train/mean_episode_length", 50)
            change_str = f"(↑ {change*100:+.1f}%)" if change and change > 0 else f"(↓ {abs(change)*100:.1f}%)" if change else ""
            print(f"  ├─ Episode长度: {ep_len[1]:.1f}步 {change_str}")
        
        # 终止率
        base_orient_term = self._get_latest_value("Episode_Termination/base_orientation")
        base_height_term = self._get_latest_value("Episode_Termination/base_height")
        if base_orient_term or base_height_term:
            term_rate = 0
            if base_orient_term:
                term_rate += base_orient_term[1]
            if base_height_term:
                term_rate += base_height_term[1]
            print(f"  └─ 摔倒率: {term_rate*100:.1f}%")
        print()
    
    def _print_stage_rewards(self, stage: int):
        """打印当前阶段的关键奖励"""
        stage_name = self.stages[stage]["name"]
        print(f"🎯 关键奖励 (Stage {stage} - {stage_name}):")
        
        # Stage 0: 站立稳定
        if stage == 0:
            rewards_to_check = [
                ("Episode_Reward/base_height_tracking", 0.3, "高度跟踪"),
                ("Episode_Reward/orientation_penalty", -0.5, "姿态控制"),
                ("Episode_Reward/velocity_tracking", 0.0, "速度跟踪"),
                ("Episode_Reward/feet_alternating_contact", 0.0, "交替步态"),
            ]
        # Stage 1: 学习行走
        elif stage == 1:
            rewards_to_check = [
                ("Episode_Reward/velocity_tracking", 0.5, "速度跟踪"),
                ("Episode_Reward/base_height_tracking", 0.4, "高度跟踪"),
                ("Episode_Reward/feet_alternating_contact", 0.1, "交替步态"),
                ("Episode_Reward/weight_transfer", 0.1, "重心转移"),
            ]
        # Stage 2: 优化步态
        elif stage == 2:
            rewards_to_check = [
                ("Episode_Reward/velocity_tracking", 1.0, "速度跟踪"),
                ("Episode_Reward/feet_alternating_contact", 0.3, "交替步态"),
                ("Episode_Reward/stride_length_tracking", 0.2, "步长跟踪"),
                ("Episode_Reward/foot_clearance", 0.1, "抬脚高度"),
            ]
        # Stage 3: 精细调节
        else:
            rewards_to_check = [
                ("Episode_Reward/velocity_tracking", 1.5, "速度跟踪"),
                ("Episode_Reward/feet_alternating_contact", 0.4, "交替步态"),
                ("Episode_Reward/action_smoothness", -0.03, "动作平滑"),
                ("Episode_Reward/joint_torque_penalty", -0.0001, "能耗优化"),
            ]
        
        for tag, target, name in rewards_to_check:
            val = self._get_latest_value(tag)
            if val:
                value = val[1]
                # 判断是否达标
                if target >= 0:
                    status = "✅" if value >= target else "⚠️"
                else:
                    status = "✅" if value >= target else "⚠️"
                print(f"  ├─ {name}: {value:.4f} {status} (目标: >{target:.2f})")
        print()
    
    def _check_alerts(self):
        """检查并打印警报"""
        alerts = []
        
        # 检查奖励异常下降
        reward_change = self._get_change_rate("Train/mean_reward", 50)
        if reward_change and reward_change < -self.alert_threshold:
            alerts.append(f"🚨 奖励下降超过{self.alert_threshold*100:.0f}%: {reward_change*100:.1f}%")
        
        # 检查摔倒率过高
        base_orient_term = self._get_latest_value("Episode_Termination/base_orientation")
        if base_orient_term and base_orient_term[1] > 0.5:
            alerts.append(f"⚠️ 摔倒率过高: {base_orient_term[1]*100:.1f}%")
        
        # 检查训练卡死
        time_since_update = time.time() - self.last_update_time
        if time_since_update > 300:  # 5分钟没更新
            alerts.append(f"⚫ 训练可能卡死: {int(time_since_update/60)}分钟无新数据")
        
        if alerts:
            print("⚠️ 警报:")
            for alert in alerts:
                print(f"  {alert}")
            print()
    
    def _print_trends(self):
        """打印趋势分析"""
        print("📈 趋势分析:")
        
        # Episode长度趋势
        ep_len_change = self._get_change_rate("Train/mean_episode_length", 50)
        if ep_len_change:
            if ep_len_change > 0.1:
                print(f"  ✅ Episode长度稳定增长 (+{ep_len_change*100:.0f}% in last 50 iters)")
            elif ep_len_change < -0.1:
                print(f"  ⚠️ Episode长度下降 ({ep_len_change*100:.0f}% in last 50 iters)")
        
        # 高度跟踪趋势
        height_change = self._get_change_rate("Episode_Reward/base_height_tracking", 50)
        if height_change:
            if height_change > 0.2:
                print(f"  ✅ 高度控制明显改善")
            elif height_change < -0.2:
                print(f"  ⚠️ 高度控制退化")
        
        # 姿态控制趋势
        orient_change = self._get_change_rate("Episode_Reward/orientation_penalty", 50)
        if orient_change:
            if orient_change > 0.2:  # 注意: 惩罚值是负的,增长=改善
                print(f"  ✅ 姿态控制改善中")
            elif orient_change < -0.2:
                print(f"  ⚠️ 姿态控制退化,可能需要调整权重")
        
        print()
    
    def _save_checkpoint_info(self):
        """显示最新checkpoint信息"""
        checkpoint_dir = self.log_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("model_*.pt"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"💾 最新checkpoint: {latest.name}")
                print()
    
    def run(self):
        """运行监控循环"""
        try:
            while True:
                # 加载最新数据
                self.history = self._load_tensorboard_data()
                
                # 获取当前iteration
                if "Train/mean_reward" in self.history and self.history["Train/mean_reward"]:
                    current_iter = self.history["Train/mean_reward"][-1][0]
                    
                    # 检查是否有新数据
                    if current_iter > self.last_iter:
                        self.last_iter = current_iter
                        self.last_update_time = time.time()
                        
                        # 计算训练进度
                        progress = current_iter / 10000
                        stage = self._get_current_stage(progress)
                        
                        # 检测阶段切换
                        if stage != self.current_stage:
                            print(f"\n{'='*80}")
                            print(f"🎉 Stage切换: {self.current_stage} → {stage} ({self.stages[stage]['name']})")
                            print(f"{'='*80}\n")
                            self.current_stage = stage
                        
                        # 清屏并打印监控信息
                        os.system('clear' if os.name != 'nt' else 'cls')
                        self._print_header(current_iter, progress, stage)
                        self._print_core_metrics()
                        self._print_stage_rewards(stage)
                        self._check_alerts()
                        self._print_trends()
                        self._save_checkpoint_info()
                        
                        print("="*80)
                        print(f"[{self.interval}秒后自动刷新... 按Ctrl+C退出]")
                        print("="*80)
                
                # 等待下次检查
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n\n✅ 监控已停止")
            if self.save_report:
                self._generate_report()
    
    def _generate_report(self):
        """生成训练报告"""
        report_path = self.log_dir / "training_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 训练监控报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 训练统计\n\n")
            f.write(f"- 总迭代数: {self.last_iter}/10000\n")
            f.write(f"- 训练时长: {self._format_time(time.time() - self.start_time)}\n")
            f.write(f"- 当前阶段: Stage {self.current_stage}\n\n")
            
            # 添加关键指标
            f.write(f"## 关键指标\n\n")
            for tag in ["Train/mean_reward", "Train/mean_episode_length"]:
                val = self._get_latest_value(tag)
                if val:
                    f.write(f"- {tag}: {val[1]:.2f}\n")
        
        print(f"\n📝 报告已保存: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练监控脚本")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="日志目录路径 (默认: 自动查找最新)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="检查间隔(秒)",
    )
    parser.add_argument(
        "--alert_threshold",
        type=float,
        default=0.2,
        help="奖励下降报警阈值",
    )
    parser.add_argument(
        "--save_report",
        action="store_true",
        help="退出时保存监控报告",
    )
    
    args = parser.parse_args()
    
    # 查找日志目录
    if args.log_dir is None:
        log_base = Path("logs/rsl_rl/oceanbdx_locomotion")
        if log_base.exists():
            # 查找最新的日志目录
            log_dirs = [d for d in log_base.iterdir() if d.is_dir()]
            if log_dirs:
                args.log_dir = str(max(log_dirs, key=lambda p: p.stat().st_mtime))
            else:
                print("❌ 未找到日志目录")
                sys.exit(1)
        else:
            print(f"❌ 日志基础目录不存在: {log_base}")
            sys.exit(1)
    
    # 创建并运行监控器
    monitor = TrainingMonitor(
        log_dir=args.log_dir,
        interval=args.interval,
        alert_threshold=args.alert_threshold,
        save_report=args.save_report,
    )
    monitor.run()


if __name__ == "__main__":
    main()
