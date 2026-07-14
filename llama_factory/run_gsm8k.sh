#!/bin/bash
# ============================================================
# Data Whisperer → Llama-Factory: GSM8K Fine-tuning Pipeline
# 
# 使用 Qwen2.5-3B-Instruct + LoRA 微调 GSM8K 数据集
# 支持数据选择策略: none, random, heuristic, datawhisperer
# ============================================================

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Base directory: ${BASE_DIR}"

# ============================================================
# Step 0: 配置路径和参数（请根据实际情况修改）
# ============================================================

MODEL_PATH="/mnt/models/Qwen2.5-3B-Instruct"

# ---------- 数据选择策略配置 ----------
# 可选: none (不使用), random, heuristic, datawhisperer
SELECT_METHOD="none"

# --- 通用参数 ---
# 保留比例 (0.0 ~ 1.0)
SELECT_KEEP_RATIO=0.5

# --- Random 参数 ---
SELECT_SEED=42

# --- Heuristic 参数 ---
# 策略: length, answer_length, quality_score, dedup
SELECT_HEURISTIC="length"
# keep_long=true: 保留长样本, false: 保留短样本
SELECT_KEEP_LONG=true
# 去重阈值 (仅 dedup 策略)
SELECT_SIMILARITY_THRESHOLD=0.8

# --- DataWhisperer 参数 ---
# 验证集路径（留空则使用 k-fold）
SELECT_VAL_PATH=""
# k-fold 折数
SELECT_K_FOLDS=2
# 训练 batch 大小
SELECT_BATCH_TRAIN=5
# 验证 batch 大小
SELECT_BATCH_TEST=8
# 并行 batch 数
SELECT_PARALLEL_BATCHES=5
# 最大 token 数
SELECT_MAX_TOKEN=8192
# 模型类型 (llama3_8b, qwen, mistral)
SELECT_MODEL_TYPE="qwen"
# 模型名称 (用于选择 attention layer)
SELECT_MODEL_NAME="Qwen2.5-3B-Instruct"
# 评估指标
SELECT_METRIC="exact_match"
# attention 层索引（留空则自动选择）
SELECT_ATTN_LAYER=""

# ============================================================
# Step 1: 数据格式转换
#   将 Data Whisperer 的 GSM8K 数据转换为 Llama-Factory 格式
# ============================================================
echo ""
echo "========================================"
echo "Step 1: Converting GSM8K data format..."
echo "========================================"
python "${BASE_DIR}/scripts/convert_gsm8k.py"

# ============================================================
# Step 2: 数据选择（可选）
#   支持 Random / Heuristic / DataWhisperer 策略对训练数据进行筛选
# ============================================================
if [ "${SELECT_METHOD}" != "none" ]; then
    echo ""
    echo "========================================"
    echo "Step 2: Selecting data (method=${SELECT_METHOD})..."
    echo "========================================"
    
    if [ "${SELECT_METHOD}" = "random" ]; then
        python "${BASE_DIR}/scripts/select_data.py" \
            --input_path "${BASE_DIR}/data/gsm8k_train.json" \
            --output_path "${BASE_DIR}/data/gsm8k_train_selected.json" \
            --method random \
            --keep_ratio "${SELECT_KEEP_RATIO}" \
            --seed "${SELECT_SEED}"
    
    elif [ "${SELECT_METHOD}" = "heuristic" ]; then
        HEURISTIC_ARGS="--heuristic ${SELECT_HEURISTIC}"
        if [ "${SELECT_KEEP_LONG}" = true ]; then
            HEURISTIC_ARGS="${HEURISTIC_ARGS} --keep_long"
        else
            HEURISTIC_ARGS="${HEURISTIC_ARGS} --no-keep_long"
        fi
        HEURISTIC_ARGS="${HEURISTIC_ARGS} --similarity_threshold ${SELECT_SIMILARITY_THRESHOLD}"
        
        python "${BASE_DIR}/scripts/select_data.py" \
            --input_path "${BASE_DIR}/data/gsm8k_train.json" \
            --output_path "${BASE_DIR}/data/gsm8k_train_selected.json" \
            --method heuristic \
            --keep_ratio "${SELECT_KEEP_RATIO}" \
            ${HEURISTIC_ARGS}
    
    elif [ "${SELECT_METHOD}" = "datawhisperer" ]; then
        # 使用工程中已有的 DataWhisperer 管线（pruning/pruning.py）
        # 原始 GSM8K 数据在 data/gsm8k/train.json（JSON 数组，question/answer 字段）
        PROJECT_DIR="$(cd "${BASE_DIR}/.." && pwd)"
        
        DW_OUTPUT_DIR="${BASE_DIR}/data/datawhisperer_output"
        mkdir -p "${DW_OUTPUT_DIR}"
        
        DW_ARGS=""
        if [ -n "${SELECT_VAL_PATH}" ]; then
            DW_ARGS="${DW_ARGS} --val_path ${SELECT_VAL_PATH}"
        fi
        if [ -n "${SELECT_ATTN_LAYER}" ]; then
            DW_ARGS="${DW_ARGS} --attn_layer ${SELECT_ATTN_LAYER}"
        fi
        
        export PYTHONPATH="${PROJECT_DIR}/pruning:${PYTHONPATH}"
        cd "${PROJECT_DIR}"
        python "${PROJECT_DIR}/pruning/pruning.py" \
            --model_path "${MODEL_PATH}" \
            --model_type "${SELECT_MODEL_TYPE}" \
            --model_name "${SELECT_MODEL_NAME}" \
            --data_path "${PROJECT_DIR}/data/gsm8k/train.json" \
            --dataset gsm8k \
            --method datawhisperer \
            --parallel_batches "${SELECT_PARALLEL_BATCHES}" \
            --batch_train "${SELECT_BATCH_TRAIN}" \
            --batch_test "${SELECT_BATCH_TEST}" \
            --max_token "${SELECT_MAX_TOKEN}" \
            --k_folds "${SELECT_K_FOLDS}" \
            --metric "${SELECT_METRIC}" \
            --output_filtered_path "${DW_OUTPUT_DIR}" \
            ${DW_ARGS}
        
        # DataWhisperer 输出 dat_whisperer.json（已按 score 降序排列）
        # 按 keep_ratio 截取 top-k，并转换为 Llama-Factory JSONL 格式
        python -c "
import json, os

with open('${DW_OUTPUT_DIR}/dat_whisperer.json', 'r') as f:
    data = json.load(f)

keep_ratio = ${SELECT_KEEP_RATIO}
keep_count = max(1, int(len(data) * keep_ratio))
selected = data[:keep_count]

print(f'[DataWhisperer] Selected {keep_count}/{len(data)} samples (keep_ratio={keep_ratio})')
print(f'  Score range: {selected[-1][\"score\"]:.4f} ~ {selected[0][\"score\"]:.4f}')

os.makedirs('${BASE_DIR}/data', exist_ok=True)
with open('${BASE_DIR}/data/gsm8k_train_selected.json', 'w') as f:
    for item in selected:
        record = {
            'instruction': item['question'],
            'output': item['answer'],
            'score': item['score']
        }
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f'Converted to Llama-Factory format: {len(selected)} samples')
"
        cd "${BASE_DIR}"
    fi
    
    # 使用筛选后的数据替换原训练数据
    cp "${BASE_DIR}/data/gsm8k_train_selected.json" "${BASE_DIR}/data/gsm8k_train.json"
    echo "Replaced training data with selected subset."
else
    echo ""
    echo "========================================"
    echo "Step 2: Skipping data selection (SELECT_METHOD=none)..."
    echo "========================================"
fi

# ============================================================
# Step 3: 复制数据文件和 dataset_info.json 到 Llama-Factory
#   Llama-Factory 默认从 data/ 目录读取数据集
# ============================================================
echo ""
echo "========================================"
echo "Step 3: Copying data files to Llama-Factory..."
echo "========================================"
cp "${BASE_DIR}/data/gsm8k_train.json" "/LLaMA-Factory/data/gsm8k_train.json"
cp "${BASE_DIR}/data/gsm8k_test.json" "/LLaMA-Factory/data/gsm8k_test.json"
cp "${BASE_DIR}/data/dataset_info.json" "/LLaMA-Factory/data/dataset_info.json"

# ============================================================
# Step 4: 启动 LoRA 微调
#   参数说明:
#   - model: Qwen2.5-3B-Instruct
#   - method: LoRA (lora_target: all)
#   - lr_scheduler: cosine with warmup_ratio=0.1
#   - batch_size: 8
#   - learning_rate: 2e-5
#   - epochs: 5
# ============================================================
echo ""
echo "========================================"
echo "Step 4: Starting LoRA fine-tuning..."
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset: GSM8K"
echo "Method: LoRA"
echo "Batch size: 8"
echo "Learning rate: 2e-5"
echo "Epochs: 5"
echo "LR scheduler: cosine (warmup_ratio=0.1)"
echo "========================================"
echo ""

cd "/LLaMA-Factory"
llamafactory-cli train "${BASE_DIR}/configs/train_qwen_gsm8k.yaml"

# ============================================================
# Step 5: 合并 LoRA 权重
# ============================================================
echo ""
echo "========================================"
echo "Step 5: Merging LoRA weights..."
echo "========================================"
llamafactory-cli export \
    --model_name_or_path ${MODEL_PATH} \
    --adapter_name_or_path /mnt/Data-Whisperer/llama_factory/output/qwen2.5-3b-gsm8k-lora \
    --template qwen \
    --finetuning_type lora \
    --export_dir /mnt/Data-Whisperer/llama_factory/output/qwen2.5-3b-gsm8k-lora-merged \
    --export_device auto

# ============================================================
# Step 6: 评估 GSM8K 测试集（Exact Match）
# ============================================================
echo ""
echo "========================================"
echo "Step 6: Evaluating on GSM8K test set..."
echo "========================================"
cd "${BASE_DIR}"
python "${BASE_DIR}/scripts/eval_gsm8k.py" \
    --model_name_or_path /mnt/Data-Whisperer/llama_factory/output/qwen2.5-3b-gsm8k-lora-merged \
    --batch_size 8

echo ""
echo "========================================"
echo "Pipeline completed!"
echo "========================================"
echo "Results saved to: ${BASE_DIR}/results/gsm8k_eval_results.json"
