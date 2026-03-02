#!/bin/bash
# 启动脚本 - 前台运行 Time-Art 服务
# 使用方式: ./start.sh [uv|python]

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
Time-Art 启动脚本 (前台运行)

使用方式:
    ./start.sh [命令] [选项]

命令:
    uv       使用 uv 运行 (默认)
    python   使用 python 直接运行

选项:
    --port PORT    指定端口号 (默认: 9999)
    --debug        启用调试模式
    --no-reload    禁用自动重载

示例:
    ./start.sh                    # 使用 uv 运行
    ./start.sh python            # 使用 python 运行
    ./start.sh uv --port 8888    # 指定端口
    ./start.sh python --debug    # 调试模式

EOF
}

# 默认参数
RUNNER="uv"
PORT=9999
DEBUG=""
RELOAD=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        uv|python)
            RUNNER="$1"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --no-reload)
            RELOAD=""
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查端口是否被占用
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

if check_port $PORT; then
    log_warn "端口 $PORT 已被占用，请先关闭占用该端口的进程"
    log_info "使用以下命令查看占用进程: lsof -i :$PORT"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 启动服务
log_info "启动 Time-Art 服务..."
log_info "运行方式: $RUNNER"
log_info "监听端口: $PORT"
log_info "日志目录: $PROJECT_ROOT/logs"
echo ""

# 设置 FLASK 环境变量
export FLASK_APP=main.py

if [ "$RUNNER" = "uv" ]; then
    # 使用 uv 运行
    if ! command -v uv &> /dev/null; then
        log_error "uv 未安装，请先安装: pip install uv"
        exit 1
    fi

    uv run python main.py --port $PORT $DEBUG $RELOAD
else
    # 使用 python 运行
    if [ ! -d ".venv" ]; then
        log_error "虚拟环境不存在，请先运行: uv sync 或 python -m venv .venv"
        exit 1
    fi

    # 激活虚拟环境并运行
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    else
        log_error "找不到虚拟环境激活脚本"
        exit 1
    fi

    python main.py --port $PORT $DEBUG $RELOAD
fi
