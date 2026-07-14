#!/bin/bash
# ============================================================
# GSM8K 数据选择实验 - 后台启动脚本
#
# 用法:
#   bash run_experiments.sh                    # 后台运行实验
#   bash run_experiments.sh --dry_run          # 只查看实验计划
#   bash run_experiments.sh --resume <exp_id>  # 从指定实验恢复
#   bash run_experiments.sh --status           # 查看实验运行状态
#   bash run_experiments.sh --tail             # 实时查看最新日志
#   bash run_experiments.sh --stop             # 停止实验
# ============================================================

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/root/experiments"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/experiment_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/experiment.pid"

case "${1:-}" in
    --dry_run)
        echo "=== 查看实验计划（不运行）==="
        cd "${BASE_DIR}"
        python scripts/run_experiments.py --dry_run
        ;;

    --status)
        if [ -f "${PID_FILE}" ]; then
            PID=$(cat "${PID_FILE}")
            if kill -0 "${PID}" 2>/dev/null; then
                echo "=== 实验正在运行 ==="
                echo "PID: ${PID}"
                echo "日志文件: $(ls -t ${LOG_DIR}/experiment_*.log 2>/dev/null | head -1)"
                echo ""
                echo "最近输出:"
                tail -5 "$(ls -t ${LOG_DIR}/experiment_*.log 2>/dev/null | head -1)" 2>/dev/null
            else
                echo "=== 实验已结束 ==="
                echo "PID 文件存在但进程已退出 (PID: ${PID})"
                echo "上次运行日志: $(ls -t ${LOG_DIR}/experiment_*.log 2>/dev/null | head -1)"
                rm -f "${PID_FILE}"
            fi
        else
            echo "=== 没有正在运行的实验 ==="
            echo "最近的日志文件:"
            ls -t "${LOG_DIR}"/experiment_*.log 2>/dev/null | head -3 || echo "  (无日志文件)"
        fi
        ;;

    --tail)
        LATEST_LOG=$(ls -t "${LOG_DIR}"/experiment_*.log 2>/dev/null | head -1)
        if [ -n "${LATEST_LOG}" ]; then
            echo "=== 实时查看日志: ${LATEST_LOG} ==="
            echo "按 Ctrl+C 退出查看（实验继续后台运行）"
            sleep 1
            tail -f "${LATEST_LOG}"
        else
            echo "没有找到日志文件"
            exit 1
        fi
        ;;

    --stop)
        if [ -f "${PID_FILE}" ]; then
            PID=$(cat "${PID_FILE}")
            echo "=== 停止实验 (PID: ${PID}) ==="
            kill "${PID}" 2>/dev/null && echo "已发送停止信号" || echo "进程不存在"
            rm -f "${PID_FILE}"
        else
            echo "没有找到正在运行的实验"
        fi
        ;;

    --resume)
        RESUME_ID="${2:-}"
        if [ -z "${RESUME_ID}" ]; then
            echo "用法: bash run_experiments.sh --resume <exp_id>"
            echo "例如: bash run_experiments.sh --resume heuristic_quality_r0.10_s123"
            exit 1
        fi
        echo "=== 从实验 ${RESUME_ID} 恢复 ==="
        echo "日志将保存到: ${LOG_FILE}"
        echo ""
        echo "启动命令:"
        echo "  nohup python scripts/run_experiments.py --resume ${RESUME_ID} > ${LOG_FILE} 2>&1 &"
        echo ""
        nohup python "${BASE_DIR}/scripts/run_experiments.py" \
            --resume "${RESUME_ID}" \
            > "${LOG_FILE}" 2>&1 &
        echo $! > "${PID_FILE}"
        echo "实验已在后台启动 (PID: $!)"
        echo "查看日志: bash run_experiments.sh --tail"
        ;;

    *)
        echo "=== 启动 GSM8K 数据选择实验 ==="
        echo "日志将保存到: ${LOG_FILE}"
        echo ""
        echo "启动命令:"
        echo "  nohup python scripts/run_experiments.py > ${LOG_FILE} 2>&1 &"
        echo ""
        echo "实验将在后台运行，关闭终端不会影响实验。"
        echo ""
        echo "可用命令:"
        echo "  bash run_experiments.sh --status   # 查看运行状态"
        echo "  bash run_experiments.sh --tail     # 实时查看日志"
        echo "  bash run_experiments.sh --stop     # 停止实验"
        echo ""

        # 后台启动实验
        cd "${BASE_DIR}"
        nohup python scripts/run_experiments.py \
            > "${LOG_FILE}" 2>&1 &
        echo $! > "${PID_FILE}"
        echo "实验已在后台启动 (PID: $!)"
        echo ""
        echo "查看进度: bash run_experiments.sh --tail"
        ;;
esac
