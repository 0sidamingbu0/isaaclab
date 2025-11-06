#!/bin/bash

# OceanBDX 键盘控制 - 快速启动脚本
# 使用方法: ./run_keyboard_control.sh [ISAACLAB_PATH]

set -e

# 默认 Isaac Lab 路径 (需要修改为你的实际路径)
ISAACLAB_PATH="${1:-$HOME/IsaacLab}"

# 当前脚本所在目录 (oceanbdx 工程根目录)
OCEANBDX_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 脚本和 checkpoint 的绝对路径
SCRIPT_PATH="${OCEANBDX_PATH}/scripts/play_keyboard_control.py"
CHECKPOINT_PATH="${OCEANBDX_PATH}/logs/rsl_rl/oceanbdx_locomotion/2025-10-31_10-39-00/exported/policy.pt"

# 检查 Isaac Lab 路径
if [ ! -f "${ISAACLAB_PATH}/isaaclab.sh" ]; then
    echo "❌ 错误: Isaac Lab 路径不正确!"
    echo "   找不到: ${ISAACLAB_PATH}/isaaclab.sh"
    echo ""
    echo "使用方法:"
    echo "  $0 /path/to/IsaacLab"
    echo ""
    echo "示例:"
    echo "  $0 ~/IsaacLab"
    echo "  $0 /home/ocean/IsaacLab"
    exit 1
fi

# 检查脚本文件
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "❌ 错误: 找不到键盘控制脚本!"
    echo "   ${SCRIPT_PATH}"
    exit 1
fi

# 检查 checkpoint 文件
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "⚠️  警告: 找不到默认 checkpoint!"
    echo "   ${CHECKPOINT_PATH}"
    echo ""
    echo "将使用脚本中的默认 checkpoint 路径"
    CHECKPOINT_ARG=""
else
    CHECKPOINT_ARG="--checkpoint ${CHECKPOINT_PATH}"
fi

echo "=================================="
echo "🤖 OceanBDX 键盘控制"
echo "=================================="
echo "Isaac Lab: ${ISAACLAB_PATH}"
echo "Script:    ${SCRIPT_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "=================================="
echo ""

# 进入 Isaac Lab 目录并运行
cd "${ISAACLAB_PATH}"

echo "🚀 启动中..."
echo ""

# 运行脚本
./isaaclab.sh -p "${SCRIPT_PATH}" ${CHECKPOINT_ARG} --num_envs 1 "$@"
