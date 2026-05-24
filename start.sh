#!/bin/bash
# green-computing 启动脚本
# 数据中心能效优化系统 - 三层HRL (Hour/Minute/Second)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  green-computing - 数据中心能效优化"
echo "  三层HRL: Hour(SAC) → Minute(TD3) → Second(TD3)"
echo "=============================================="
echo

VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install -q torch numpy 2>/dev/null

echo "[启动] 运行主程序..."
python3 main.py "$@"