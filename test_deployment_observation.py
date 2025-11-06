"""
请训练方用这个观测向量测试模型，看输出是否正常
"""
import torch

# Step 0 的完整 74 维观测（从部署环境 log 提取）
observation = torch.tensor([[
    # 1. ang_vel_body (3)
    0.0, 0.0, 0.0,
    
    # 2. gravity_vec (3)
    0.0, -0.0, -1.0,
    
    # 3. dof_pos_rel (14) - Training order: [L1-L5, R1-R5, N1-N4]
    0.0, 0.0, 0.0, 0.0, 0.0,  # L1-L5
    0.0, 0.0, 0.0, 0.0, 0.0,  # R1-R5
    0.0, 0.0, 0.0, 0.0,        # N1-N4
    
    # 4. dof_vel_scaled (14)
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    
    # 5. joint_torques (14)
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    
    # 6. commands (3)
    0.0, 0.0, 0.0,
    
    # 7. last_actions (14)
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    
    # 8. adaptive_phase (9)
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.6667, 0.0, 0.37
]], dtype=torch.float32)

print("观测维度:", observation.shape)
print("观测范围:", f"[{observation.min().item():.4f}, {observation.max().item():.4f}]")

# 加载模型
model_path = "logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt"
model = torch.jit.load(model_path)
model.eval()

# 推理
with torch.no_grad():
    action = model(observation)

print("\n模型输出维度:", action.shape)
print("动作范围:", f"[{action.min().item():.4f}, {action.max().item():.4f}]")
print("\n动作输出 (Training order):")
print(f"  Left leg  (L1-L5): {action[0, :5].tolist()}")
print(f"  Right leg (R1-R5): {action[0, 5:10].tolist()}")
print(f"  Neck      (N1-N4): {action[0, 10:14].tolist()}")

print("\n期望输出 (你们之前的测试结果):")
print(f"  Left leg  (L1-L5): [-0.0827, 0.3333, -0.2954, 0.7120, -0.4442]")
print(f"  Right leg (R1-R5): [0.1568, -0.2259, -0.1768, 0.2263, 0.0045]")
print(f"  Neck      (N1-N4): [-0.9644, 0.0712, 0.4182, -0.0887]")

print("\n部署环境实际输出:")
print(f"  Left leg  (L1-L5): [-2.0, -0.79, -1.19, 0.78, 1.21]")
print(f"  Right leg (R1-R5): [-1.01, 1.37, 2.0, 1.80, -1.34]")
print(f"  Neck      (N1-N4): [0.40, -1.40, 1.54, 0.75]")

print("\n❓ 问题：相同的观测，为什么输出不同？")
print("请在训练环境运行此脚本，对比输出结果！")
