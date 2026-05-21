#!/usr/bin/env bash

set -e

# ==========================
# Configuration
# ==========================

MODEL=("/nlsasfs/home/aidrive/saipan/Model/Olmo-3-7B-Think")
BATCH_SIZE=500
RANGE=("range_1_10")
DTYPE=("float" "int")
TASKS=("mixed")
LOOK_BACK=("look_back1")
REASONING=("TRUE")
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_DIR="logs/${DATE}/${TIME}"

mkdir -p "$LOG_DIR"

# ==========================
# Run evaluation
# ==========================
for run in {1..5}; do
    for model in "${MODEL[@]}"; do
        for range in "${RANGE[@]}"; do
            for dtype in "${DTYPE[@]}"; do
                for look_back in "${LOOK_BACK[@]}"; do
                    for task in "${TASKS[@]}"; do
                        for reasoning in "${REASONING[@]}"; do

                            if [[ "$range" == "range_0_1" ]]; then
                                if [[ "$dtype" == "int" ]]; then
                                    continue
                                fi
                            fi

                            if [[ "$look_back" == "look_back2" || "$look_back" == "look_back3" ]]; then
                                if [[ "$task" == "add" || "$task" == "subtract" || "$task" == "multiply" || "$task" == "divide" ]]; then
                                    continue
                                fi
                            fi

                            DATASET="dataset/${range}/${dtype}/${look_back}/algo_reasoning_dataset_${task}.jsonl"
                            OUTPUT="rerun_${run}/${model##*/}/${range}/${dtype}/${look_back}/${task}/results"
                            LOG_FILE="$LOG_DIR/${model##*/}_${range}_${dtype}_${look_back}_${task}.txt"

                            echo "Model: $model"
                            echo "Dataset: $DATASET"

                            python test_batch_rerun.py \
                            --model "$model" \
                            --output "$OUTPUT" \
                            --batch_size "$BATCH_SIZE" \
                            --dataset "$DATASET" \
                            --reasoning "$reasoning" \
                            2>&1 | tee "$LOG_FILE"

                            echo "Run completed."
                        
                        done
                    done
                done
            done
        done
    done
done