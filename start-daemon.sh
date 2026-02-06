#!/bin/bash
# 启动脚本 - 后台运行 Time-Art 服务
# 使用方式: ./start-daemon.sh [uv|python]

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
Time-Art 启动脚本 (后台运行)

使用方式:
    ./start-daemon.sh [命令] [选项]

命令:
    uv       使用 uv 运行 (默认)
    python   使用 python 直接运行
    stop     停止后台服务
    restart  重启后台服务
    status   查看服务状态

选项:
    --port PORT    指定端口号 (默认: 9999)

示例:
    ./start-daemon.sh              # 使用 uv 后台运行
    ./start-daemon.sh python       # 使用 python 后台运行
    ./start-daemon.sh stop         # 停止服务
    ./start-daemon.sh status       # 查看状态

EOF
}

# PID 文件路径
PID_FILE="$PROJECT_ROOT/logs/app.pid"
LOG_FILE="$PROJECT_ROOT/logs/app.log"

# 默认参数
RUNNER="uv"
PORT=9999
ACTION="start"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        uv|python)
            RUNNER="$1"
            shift
            ;;
        start|stop|restart|status)
            ACTION="$1"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
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

# 创建日志目录
mkdir -p logs

# 获取进程 ID
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    fi
}

# 检查进程是否运行
is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# 停止服务
stop_service() {
    if is_running; then
        local pid=$(get_pid)
        log_info "停止 Time-Art 服务 (PID: $pid)..."
        kill "$pid"

        # 等待进程结束
        local count=0
        while is_running && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done

        if is_running; then
            log_warn "进程未响应，强制停止..."
            kill -9 "$pid"
            rm -f "$PID_FILE"
        else
            log_success "服务已停止"
            rm -f "$PID_FILE"
        fi
    else
        log_warn "服务未运行"
    fi
}

# 启动服务
start_service() {
    # 检查端口是否被占用
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_error "端口 $PORT 已被占用"
        log_info "使用以下命令查看占用进程: lsof -i :$PORT"
        exit 1
    fi

    # 检查是否已在运行
    if is_running; then
        log_warn "服务已在运行 (PID: $(get_pid))"
        exit 0
    fi

    log_info "启动 Time-Art 后台服务..."
    log_info "运行方式: $RUNNER"
    log_info "监听端口: $PORT"
    log_info "日志文件: $LOG_FILE"
    log_info "PID 文件: $PID_FILE"
    echo ""

    # 设置 FLASK 环境变量
    export FLASK_APP=main.py

    # 启动命令
    local cmd=""
    if [ "$RUNNER" = "uv" ]; then
        if ! command -v uv &> /dev/null; then
            log_error "uv 未安装，请先安装: pip install uv"
            exit 1
        fi
        cmd="uv run python main.py --port $PORT"
    else
        if [ ! -d ".venv" ]; then
            log_error "虚拟环境不存在，请先运行: uv sync 或 python -m venv .venv"
            exit 1
        fi
        cmd="python main.py --port $PORT"
    fi

    # 后台启动并重定向日志
    nohup $cmd >> "$LOG_FILE" 2>&1 &
    local pid=$!

    # 保存 PID
    echo $pid > "$PID_FILE"

    # 等待启动
    sleep 2

    if is_running; then
        log_success "服务启动成功 (PID: $pid)"
        echo ""
        log_info "查看日志: tail -f $LOG_FILE"
        log_info "停止服务: ./start-daemon.sh stop"
        log_info "查看状态: ./start-daemon.sh status"
    else
        log_error "服务启动失败，请查看日志: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# 查看服务状态
show_status() {
    if is_running; then
        local pid=$(get_pid)
        log_success "服务正在运行"
        echo ""
        echo "  PID:      $pid"
        echo "  端口:     $PORT"
        echo "  日志文件: $LOG_FILE"
        echo ""

        # 显示进程信息
        ps -p "$pid" -o pid,etime,cmd || true

        # 显示最近的日志
        if [ -f "$LOG_FILE" ]; then
            echo ""
            log_info "最近的日志:"
            tail -n 5 "$LOG_FILE"
        fi
    else
        log_warn "服务未运行"
        rm -f "$PID_FILE"
    fi
}

# 执行对应的命令
case $ACTION in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        sleep 1
        start_service
        ;;
    status)
        show_status
        ;;
esac
