#!/bin/bash
# =============================================================================
# Data-Whisperer 综合实验脚本
#
# 对三种数据选择策略（datawhisperer, random, heuristic）在三种数据比例
# （5%, 10%, 20%）下进行实验，每种实验重复 3 次取平均值。
#
# 模型：Qwen2.5-3B-Instruct
# 数据集：gsm8k
#
# 流程：数据选择 → 微调 → 评估 → 汇总
# =============================================================================

# 不使用 set -e，因为 Data Whisperer 失败不应影响 random/heuristic 实验
# set -e

BASE_PATH="$(cd "$(dirname "$0")/.." && pwd)"
echo "Base path: ${BASE_PATH}"

# =============================================================================
# Conda 环境配置
# =============================================================================
# 检测当前 conda 环境，如果 torch 有问题则切换到 myconda 环境
CONDA_BASE=$(conda info --base 2>/dev/null)
CUR_CONDA_ENV=$(conda info --envs 2>/dev/null | grep '*' | awk '{print $1}')
echo "Current conda env: ${CUR_CONDA_ENV}"

# 测试当前环境的 torch 是否可用
python -c "import torch; print('torch OK')" 2>/dev/null && TORCH_OK=true || TORCH_OK=false

if [ "${TORCH_OK}" = false ]; then
    echo "Current conda env (${CUR_CONDA_ENV}) torch not working, trying myconda..."
    # 尝试使用 myconda 环境
    if [ -f "${CONDA_BASE}/envs/myconda/bin/python" ]; then
        PYTHON_CMD="${CONDA_BASE}/envs/myconda/bin/python"
        echo "Using myconda python: ${PYTHON_CMD}"
    else
        PYTHON_CMD="python"
        echo "Warning: myconda not found, using default python"
    fi
else
    PYTHON_CMD="python"
fi

# =============================================================================
# 实验配置
# =============================================================================
MODEL_TYPE="qwen"
MODEL_NAME="Qwen2.5-3B-Instruct"

# 模型路径：支持相对路径和绝对路径
MODEL_PATH="$(cd "${BASE_PATH}/../models/Qwen2.5-3B-Instruct" && pwd)"
echo "Model path: ${MODEL_PATH}"


# 如果使用绝对路径，注释上面那行，取消注释下面这行：
# MODEL_PATH="/obs/pretrained_models/Qwen/Qwen2.5-3B-Instruct"

DATASET="gsm8k"
TRAIN_PATH="${BASE_PATH}/data/${DATASET}/train.json"
TEST_PATH="${BASE_PATH}/data/${DATASET}/test.json"

DATA_RATIOS=(0.20)
METHODS=("datawhisperer")
REPEATS=3

RESULTS_DIR="${BASE_PATH}/results/experiments/gsm8k_qwen2.5_3b"
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"

# =============================================================================
# 辅助函数
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# =============================================================================
# 主实验循环
# =============================================================================
mkdir -p "${RESULTS_DIR}"
echo "method,ratio,repeat,accuracy" > "${SUMMARY_FILE}"

for method in "${METHODS[@]}"; do
    for ratio in "${DATA_RATIOS[@]}"; do
        for repeat in $(seq 1 ${REPEATS}); do
            log ""
            log "=========================================="
            log "实验: method=${method}, ratio=${ratio}, repeat=${repeat}"
            log "=========================================="

            EXP_DIR="${RESULTS_DIR}/${method}/ratio_${ratio}/repeat_${repeat}"
            mkdir -p "${EXP_DIR}"

            # ---- Step 1: 数据选择 ----
            log "[Step 1/3] 数据选择..."

            # Data Whisperer：评分结果按 repeat 共享（同一 repeat 的不同 ratio 共用同一份评分）
            DW_SCORED_ARG=""
            if [ "${method}" = "datawhisperer" ]; then
                DW_DIR="${RESULTS_DIR}/datawhisperer/repeat_${repeat}"
                DW_SCORED_FILE="${DW_DIR}/dat_whisperer.json"

                # 先检查旧路径（兼容之前运行的结果：datawhisperer/ratio_0.05/repeat_1/dat_whisperer.json）
                if [ ! -f "${DW_SCORED_FILE}" ]; then
                    OLD_DW_FILE="${RESULTS_DIR}/datawhisperer/ratio_0.05/repeat_${repeat}/dat_whisperer.json"
                    if [ -f "${OLD_DW_FILE}" ]; then
                        log "发现已有评分结果：${OLD_DW_FILE}，复制到共享目录"
                        mkdir -p "${DW_DIR}"
                        cp "${OLD_DW_FILE}" "${DW_SCORED_FILE}"
                    fi
                fi

                if [ ! -f "${DW_SCORED_FILE}" ]; then
                    log "Data Whisperer 评分结果不存在（repeat=${repeat}），开始运行 pruning.py..."
                    # 先安装 rouge_score（pruning.py 导入时需要）
                    pip install rouge-score -q 2>/dev/null || true
                    mkdir -p "${DW_DIR}"
                    export PYTHONPATH="${BASE_PATH}/pruning:${PYTHONPATH}"
                    python "${BASE_PATH}/pruning/pruning.py" \
                        --model_type "${MODEL_TYPE}" \
                        --model_path "${MODEL_PATH}" \
                        --model_name "${MODEL_NAME}" \
                        --data_path "${TRAIN_PATH}" \
                        --dataset "${DATASET}" \
                        --method "datawhisperer" \
                        --metric "exact_match" \
                        --output_filtered_path "${DW_DIR}" \
                        --k_folds 2 \
                        --batch_train 10 \
                        --batch_test 5 \
                        --parallel_batches 1 \
                        --max_token 8192 \
                        --gpu_index 0 \
                        2>&1 | tee "${DW_DIR}/pruning.log" || true
                else
                    log "Data Whisperer 评分结果已存在（${DW_SCORED_FILE}），跳过 pruning.py"
                fi
                DW_SCORED_ARG="--dw_scored_file ${DW_SCORED_FILE}"
            fi


            python "${BASE_PATH}/scripts/run_selection.py" \
                --method "${method}" \
                --full_data "${TRAIN_PATH}" \
                --keep_ratio "${ratio}" \
                --output_dir "${EXP_DIR}" \
                --repeat "${repeat}" \
                --heuristic "quality_score" \
                --keep_long "true" \
                --similarity_threshold 0.8 \
                ${DW_SCORED_ARG} \
                2>&1 | tee -a "${EXP_DIR}/selection.log"



            SUBSET_FILE="${EXP_DIR}/subset.json"
            if [ ! -f "${SUBSET_FILE}" ]; then
                log "ERROR: 数据选择失败，未生成子集文件，跳过此实验"
                continue
            fi

            # ---- Step 2: 微调（使用不同随机种子） ----
            log "[Step 2/3] 微调模型..."
            ${PYTHON_CMD} "${BASE_PATH}/scripts/run_finetune.py" \
                --model_path "${MODEL_PATH}" \
                --subset_file "${SUBSET_FILE}" \
                --output_dir "${EXP_DIR}/finetune" \
                --dataset "${DATASET}" \
                --num_epochs 5 \
                --batch_size 2 \
                --learning_rate 2e-5 \
                --max_length 512 \
                --seed $((42 + repeat)) \
                2>&1 | tee "${EXP_DIR}/finetune.log" || true


            # 微调后的 LoRA 权重保存在 finetune/lora_model 目录下
            LORA_PATH="${EXP_DIR}/finetune/lora_model"


            log "[Step 3/3] 评估模型..."
            ${PYTHON_CMD} "${BASE_PATH}/scripts/run_evaluation.py" \
                --model_path "${MODEL_PATH}" \
                --lora_path "${LORA_PATH}" \
                --test_file "${TEST_PATH}" \
                --output_dir "${EXP_DIR}/evaluation" \
                --dataset "${DATASET}" \
                2>&1 | tee "${EXP_DIR}/evaluation.log" || true


            # ---- 记录结果 ----
            EVAL_FILE="${EXP_DIR}/evaluation/results.json"
            if [ -f "${EVAL_FILE}" ]; then
                accuracy=$(${PYTHON_CMD} -c "import json; print(json.load(open('${EVAL_FILE}'))['accuracy'])")
                echo "${method},${ratio},${repeat},${accuracy}" >> "${SUMMARY_FILE}"
                log "结果: accuracy=${accuracy}"
            fi

        done
    done
done

# =============================================================================
# 汇总结果
# =============================================================================
log ""
log "=========================================="
log "实验完成！汇总结果："
log "=========================================="

${PYTHON_CMD} -c "
import pandas as pd
df = pd.read_csv('${SUMMARY_FILE}')
print('\\n=== 各方法在各比例下的平均准确率 ===')
print(df.groupby(['method', 'ratio'])['accuracy'].agg(['mean', 'std']).round(4))
print()
pivot = df.pivot_table(values='accuracy', index='method', columns='ratio', aggfunc=['mean', 'std']).round(4)
print(pivot)
"


log "汇总文件: ${SUMMARY_FILE}"
log "所有实验结果保存在: ${RESULTS_DIR}"
