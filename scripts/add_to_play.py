"""
将此代码片段添加到 scripts/rsl_rl/play.py 的主循环中，
用于输出关节角度信息
"""

# 在 while simulation_app.is_running(): 循环中添加以下代码

    timestep += 1
    
    # === 添加这段代码用于输出关节角度 ===
    if timestep % 100 == 0:  # 每100步输出一次
        robot = env.scene["robot"]
        joint_positions = robot.data.joint_pos[0]  # 第一个环境的关节角度
        joint_velocities = robot.data.joint_vel[0]  # 关节速度
        
        print(f"\n=== 时间步 {timestep} 的关节状态 ===")
        for i, (name, pos, vel) in enumerate(zip(robot.data.joint_names, joint_positions, joint_velocities)):
            pos_deg = pos.item() * 180.0 / 3.14159
            print(f"{name}: {pos.item():.4f} rad ({pos_deg:.2f}°), 速度: {vel.item():.4f} rad/s")
        
        # 检查是否稳定
        max_vel = torch.max(torch.abs(joint_velocities)).item()
        if max_vel < 0.05:
            print(f"\n🎯 机器人稳定! 最大关节速度: {max_vel:.4f} rad/s")
            print("稳定状态的关节角度配置:")
            print("joint_pos={")
            for name, pos in zip(robot.data.joint_names, joint_positions):
                print(f'    "{name}": {pos.item():.6f},  # {pos.item() * 180.0 / 3.14159:.2f}°')
            print("}")
    # === 添加代码结束 ===