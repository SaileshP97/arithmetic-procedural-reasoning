#!/bin/bash

NUM_STEPS_LIST=(5 15 25 35 45 55 65 75 85 95)
SAMPLES_PER_NUM_STEP=100
SEED=434
RANGE=("0 1" "1 10" "10 100")
DTYPE=("float" "int")
LOOK_BACK=(4 5 6 7)
#TASKS=("mixed" "add" "subtract" "multiply" "divide")
TASKS=("mixed")

for range in "${RANGE[@]}"; do
    for dtype in "${DTYPE[@]}"; do
        for task in "${TASKS[@]}"; do
            for look_back in "${LOOK_BACK[@]}"; do

                if [[ "$range" == "0 1" ]]; then
                    if [[ "$dtype" == "int" ]]; then
                        continue
                    fi
                fi

                safe_range="range_${range// /_}"
                name="dataset/${safe_range}/${dtype}/look_back${look_back}/algo_reasoning_dataset_${task}"

                python dataset_generator.py \
                    --num_steps_list "${NUM_STEPS_LIST[@]}" \
                    --samples_per_step "$SAMPLES_PER_NUM_STEP" \
                    --seed "$SEED" \
                    --tasks "$task" \
                    --input_range $range \
                    --input_type "$dtype" \
                    --look_back "$look_back" \
                    --name "$name" \

            done
        done
    done
done