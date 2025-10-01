#!/usr/bin/env python3
"""
将IMU四元数转换为机体坐标系中的重力投影向量
"""

import numpy as np

def quaternion_to_projected_gravity(quat_wxyz):
    """
    从四元数计算重力在机体坐标系中的投影 - 适配OceanBDX IMU坐标系
    
    OceanBDX IMU坐标系定义：
    - 完全直立: [0, 0, +9.81] (Z向上为正)
    - 前倾: [-9.81, 0, +xxx] (X向前)  
    - 左倾: [0, -9.81, +xxx] (Y向左)
    
    Args:
        quat_wxyz: 四元数 [qw, qx, qy, qz] (标量部分在前)
        
    Returns:
        projected_gravity: 重力在机体坐标系的投影 [gx, gy, gz] - OceanBDX坐标系
    """
    qw, qx, qy, qz = quat_wxyz
    
    # 世界坐标系中的重力向量 (OceanBDX定义：向上为正)
    gravity_world = np.array([0.0, 0.0, +9.81])
    
    # 四元数旋转矩阵 (从世界到机体)
    # 适配OceanBDX的坐标系定义
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    # 将世界坐标系的重力向量转换到机体坐标系
    projected_gravity = R @ gravity_world
    
    return projected_gravity


def imu_to_model_input(imu_data):
    """
    将IMU原始数据转换为模型输入格式
    
    Args:
        imu_data: {
            "acceleration": [ax, ay, az],    # m/s²
            "gyroscope": [wx, wy, wz],       # rad/s  
            "quaternion": [qw, qx, qy, qz]   # 姿态四元数
        }
        
    Returns:
        dict: 模型需要的IMU相关输入
    """
    
    # 1. 角速度直接使用陀螺仪数据
    base_ang_vel = np.array(imu_data["gyroscope"])
    
    # 2. 从四元数计算重力投影
    projected_gravity = quaternion_to_projected_gravity(imu_data["quaternion"])
    
    # 3. 线速度需要额外估计 (简化版本，实际需要更复杂的滤波)
    # 这里展示概念，实际部署需要速度估计算法
    base_lin_vel = np.array([0.0, 0.0, 0.0])  # 需要从加速度积分或其他传感器获取
    
    return {
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel, 
        "projected_gravity": projected_gravity
    }


# 测试示例
if __name__ == "__main__":
    # 模拟IMU数据
    test_imu = {
        "acceleration": [0.1, 0.2, -9.8],
        "gyroscope": [0.01, -0.02, 0.1],
        "quaternion": [1.0, 0.0, 0.0, 0.0]  # 无旋转状态
    }
    
    result = imu_to_model_input(test_imu)
    print("IMU转换结果:")
    print(f"base_ang_vel: {result['base_ang_vel']}")
    print(f"projected_gravity: {result['projected_gravity']}")
    
    # 验证: 无旋转时，重力投影应该是 [0, 0, -9.81]
    print(f"验证 - 无旋转时重力投影: {result['projected_gravity']}")
    
    # 测试机体倾斜45度(绕x轴)
    import math
    angle = math.pi / 4  # 45度
    test_imu_tilted = {
        "acceleration": [0.1, 0.2, -9.8],
        "gyroscope": [0.01, -0.02, 0.1],
        "quaternion": [math.cos(angle/2), math.sin(angle/2), 0.0, 0.0]  # 绕x轴旋转45度
    }
    
    result_tilted = imu_to_model_input(test_imu_tilted)
    print(f"\n倾斜45度时重力投影: {result_tilted['projected_gravity']}")
    print(f"预期值约为: [0, 6.94, -6.94]")