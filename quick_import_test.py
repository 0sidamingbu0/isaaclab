#!/usr/bin/env python3
"""快速验证所有导入是否正常"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source/oceanbdx"))

print("=" * 80)
print("🧪 快速导入测试")
print("=" * 80)

try:
    print("\n1️⃣ 测试训练课程模块...")
    from oceanbdx.tasks.manager_based.oceanbdx_locomotion.mdp import (
        TrainingCurriculum,
        get_current_stage,
    )
    print("✅ TrainingCurriculum 导入成功")
    print(f"   - get_current_stage(0.15) = {get_current_stage(0.15)}")
    print(f"   - get_current_stage(0.50) = {get_current_stage(0.50)}")
    print(f"   - get_current_stage(0.85) = {get_current_stage(0.85)}")
    
    print("\n2️⃣ 测试奖励函数...")
    from oceanbdx.tasks.manager_based.oceanbdx_locomotion.config import mdp
    print("✅ config.mdp 导入成功")
    print(f"   - mdp.reward_velocity_tracking_exp: {hasattr(mdp, 'reward_velocity_tracking_exp')}")
    print(f"   - mdp.reward_feet_alternating_contact: {hasattr(mdp, 'reward_feet_alternating_contact')}")
    print(f"   - mdp.adaptive_gait_phase_observation: {hasattr(mdp, 'adaptive_gait_phase_observation')}")
    
    print("\n3️⃣ 测试环境类...")
    from oceanbdx.tasks.manager_based.oceanbdx_locomotion import OceanBDXLocomotionEnv
    print("✅ OceanBDXLocomotionEnv 导入成功")
    
    print("\n" + "=" * 80)
    print("🎉 所有导入测试通过！可以开始训练")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
