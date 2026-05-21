#!/usr/bin/env python3
"""
Dataset Generator for Step-wise Arithmetic Reasoning Experiments
---------------------------------------------------------------
Creates a reproducible dataset of algorithm prompts + inputs + outputs.

Usage:
    python dataset_generator.py
"""

import os
import random
import numpy as np
import pandas as pd
import uuid
import argparse
from datetime import datetime, timezone


# ===============================
# 1. Reproducibility
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# ===============================
# 2. Algorithm Generator
# ===============================
def generate_algo(num_steps, tasks=None):
    steps = tasks if tasks is not None else ["add", "subtract", "multiply", "divide"]
    prompt = "Algorithm:\nfunction(x, y):\nStep 1: Add both inputs.\n"

    steps_in_algo = random.choices(steps, k=num_steps - 1)

    for idx, step in enumerate(steps_in_algo):
        step_num = idx + 2
        prev = idx + 1

        if step == "add":
            prompt += f"Step {step_num}: Add second input to step {prev} output.\n"
        elif step == "subtract":
            prompt += f"Step {step_num}: Subtract second input from step {prev} output.\n"
        elif step == "multiply":
            prompt += f"Step {step_num}: Multiply step {prev} output by second input.\n"
        elif step == "divide":
            prompt += f"Step {step_num}: Divide step {prev} output by second input.\n"

    prompt += "Return output of last step."
    return prompt

def generate_algo_new(num_steps, tasks=None, look_back=1, seed=None):
    if seed is not None:
        random.seed(seed)

    steps = tasks if tasks != ["mixed"] else ["add", "subtract", "multiply", "divide"]
    
    prompt = "function(x, y):\n"
    prompt += "Let S1 = x\n"
    prompt += "Let S2 = y\n"
    prompt += "Step 1: S3 = S1 + S2\n"

    current_step = 2
    history = ["S2", "S3"]

    for _ in range(num_steps - 1):

        op = random.choice(steps)

        candidates = history.copy()  # Fix 1: copy, don't alias
        a = random.choice(candidates)
        candidates.remove(a)
        b = random.choice(candidates)

        result = f"S{current_step + 2}"  # Fix 3: correct numbering

        if op == "add":
            expr = f"{a} + {b}"
        elif op == "subtract":
            expr = f"{a} - {b}"
        elif op == "multiply":
            expr = f"{a} * {b}"
        elif op == "divide":
            expr = f"{a} / {b}"

        prompt += f"Step {current_step}: {result} = {expr}\n"

        if len(history)==(look_back+1):
            history.pop(1)

        history.append(result)
        current_step += 1

    prompt += f"Final Step: Return {history[-1]}"
    return prompt


# ===============================
# 3. Ground Truth Executor
# ===============================
import numpy as np

def exec_algo(prompt, inputs):
    lines = prompt.strip().split("\n")

    vars_dict = {}

    vars_dict["S1"] = inputs[0]
    vars_dict["S2"] = inputs[1]

    for line in lines:
        line = line.strip()

        if "=" not in line:
            continue

        try:
            expr_part = line.split(":")[-1].strip()
            left, right = expr_part.split("=")

            left = left.strip()
            right = right.strip()

            first_var = right.split()[0].strip()
            task = right.split()[1].strip()
            second_var = right.split()[2].strip()
            value = eval(f"vars_dict['{first_var}'] {task} vars_dict['{second_var}']")

            vars_dict[left] = value

        except ZeroDivisionError:
            vars_dict[left] = np.nan
                    
        except Exception as e:
            pass

    return [str(v) for v in vars_dict.values()]


# ===============================
# 4. Dataset Generator
# ===============================
def generate_dataset(num_steps_list, samples_per_step, tasks, input_range, input_type, look_back=1, seed=432):
    set_seed(seed)

    rows = []

    for steps in num_steps_list:
        for _ in range(samples_per_step):

            #algo = generate_algo(steps, tasks=tasks)
            algo = generate_algo_new(steps, tasks=tasks, look_back=look_back)
            if input_type == "int":
                x, y = random.choices(range(input_range[0], input_range[1]), k=2)
            else:
                x, y = round(random.uniform(input_range[0], input_range[1]), 3), round(random.uniform(input_range[0], input_range[1]), 3)
                
            int_ans = exec_algo(algo, (x, y))

            rows.append({
                "id": str(uuid.uuid4()),
                "num_steps": steps,
                "algorithm": algo,
                "x": x,
                "y": y,
                "intermediate_values": int_ans,
                "expected_output": int_ans[-1],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    return pd.DataFrame(rows)


# ===============================
# 5. Save Dataset
# ===============================
def save_dataset(df, name):
    df.to_json(f"{name}.jsonl", orient="records", lines=True)

# ===============================
# 6. Main
# ===============================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps_list", nargs="+", type=int,
                        default=range(5, 100, 10), help="List of number of steps in the algorithm")
    parser.add_argument("--samples_per_step", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", type=str,
                        default="mixed", help="List of tasks to include in the algorithms")
    parser.add_argument("--input_range", nargs=2, type=int, default=[1, 10],
                        help="Range of input values")
    parser.add_argument("--input_type", type=str, choices=["int", "float"], default="int",
                        help="Type of input values")
    parser.add_argument("--look_back", type=int, default=1)
    parser.add_argument("--name", type=str, default="dataset/algo_reasoning_dataset")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.name), exist_ok=True)

    print("Generating dataset...")
    df = generate_dataset(num_steps_list=args.num_steps_list, 
                          samples_per_step=args.samples_per_step, 
                          tasks=args.tasks,
                          input_range=args.input_range,
                          input_type=args.input_type,
                          look_back=args.look_back,
                          seed=args.seed)

    print(f"Saving dataset: {args.name}")
    save_dataset(df, args.name)

    print("Done ✅")
    print(df.head())

if __name__ == "__main__":
    main()