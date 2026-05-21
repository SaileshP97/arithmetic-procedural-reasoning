#!/usr/bin/env python3
"""
Stepwise Arithmetic Benchmark Runner
=====================================
A single script to run and save all experiments for the algo-reasoning benchmark.

Supports:
  - HuggingFace Hub models or local model paths
  - vLLM (fast, GPU-optimized) or HuggingFace Transformers (CPU/GPU)
  - Multi-GPU via tensor parallelism (vLLM)
  - CPU inference (HuggingFace backend)
  - Auto-resumes interrupted runs
  - Generates a summary CSV and unified parquet after all experiments

Usage examples:

  # vLLM + GPU (recommended for speed)
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --backend vllm

  # HuggingFace Transformers + CPU
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --backend hf --device cpu

  # Local model path, multi-GPU
  python run_benchmark.py --model ./models/DeepSeek-R1-Distill-Qwen-1.5B --backend vllm --tensor_parallel_size 4

  # Run only specific tasks/ranges
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --tasks add subtract --ranges range_0_1 range_1_10

  # No reasoning tokens
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --reasoning false

  # Also produce a unified parquet (all experiments in one file)
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --build_parquet

  # Only rebuild the parquet/summary from existing results (no inference)
  python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --summary_only --build_parquet
"""

import argparse
import json
import os
import re
import sys
import time
from glob import glob
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────
# Division-by-zero / NaN keyword detection
# ──────────────────────────────────────────────
NAN_KEYWORDS = [
    "<answer>NaN</answer>",
    "<answer>undefined</answer>",
    "Division by zero error",
    "division by zero error",
    "<answer>undefined (division by zero)</answer>",
    "This is undefined",
]

# ──────────────────────────────────────────────
# Experiment matrix  (mirrors bash script)
# ──────────────────────────────────────────────
ALL_RANGES    = ["range_0_1", "range_1_10", "range_10_100"]
ALL_DTYPES    = ["float", "int"]
ALL_LOOKBACKS = [f"look_back{i}" for i in range(1, 8)]
ALL_TASKS     = ["mixed", "add", "subtract", "multiply", "divide"]


def is_valid_combination(range_: str, dtype: str, lookback: str, task: str) -> bool:
    if range_ == "range_0_1" and dtype == "int":
        return False
    if lookback != "look_back1" and task != "mixed":
        return False
    return True


def build_experiment_list(ranges, dtypes, lookbacks, tasks):
    return [
        (r, d, lb, t)
        for r, d, lb, t in product(ranges, dtypes, lookbacks, tasks)
        if is_valid_combination(r, d, lb, t)
    ]


# ──────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────
def load_prompts(prompt_path: str) -> dict:
    with open(prompt_path, "r") as f:
        return json.load(f)


def build_prompt(prompts: dict, algo: str, x, y, reasoning: bool) -> str:
    mode = "reasoning" if reasoning else "no_reasoning"
    return prompts["base"].format(
        behavior=prompts["modes"][mode]["behavior"],
        output_rules=prompts["modes"][mode]["output_rules"],
        algo=algo, x=x, y=y,
    )


# ──────────────────────────────────────────────
# Feature engineering  (mirrors build_dataset.py)
# ──────────────────────────────────────────────
EXAMPLE_PLACEHOLDERS = ["FINAL_RESULT"]


def normalized_answer_position(text: str):
    """
    Position of the first genuine <answer> tag as a percentage of total
    output length (0–100). NaN if no valid answer found.

    Tells you *where* in the response the model committed to an answer —
    early (near 0) means the model answered quickly; late (near 100) means
    it only answered after long deliberation or self-correction.
    """
    if not isinstance(text, str) or not text:
        return np.nan
    for match in re.finditer(r"<answer>(.*?)<\\?/answer>", text):
        val = match.group(1).strip()
        if val in EXAMPLE_PLACEHOLDERS:
            continue
        try:
            float(val)          # must be numeric
        except ValueError:
            continue
        return (match.start() / len(text)) * 100
    return np.nan


def count_steps(text: str) -> int:
    """
    Number of *distinct* step labels (e.g. 'Step 1:', 'Step 3:') found in
    the output.  Measures how many reasoning steps the model wrote out.
    """
    if not isinstance(text, str):
        return 0
    return len(set(re.findall(r"Step\s+\d+:", text)))


def steps_diff_label(diff: float) -> str:
    """
    Categorical label comparing predicted vs expected step count.
      'over'  — model wrote >2 more steps than the dataset says are needed
      'under' — model wrote >2 fewer steps
      'exact' — within ±2 steps of the ground-truth count
    """
    if diff > 2:
        return "over"
    if diff < -2:
        return "under"
    return "exact"


def enrich_row(row: dict, text: str) -> dict:
    """Add all feature-engineering columns to a result row."""
    pred = row.get("model_prediction")

    # Normalise model_prediction: treat sentinel strings as NaN
    if pred in (None, "null", "None", "nan", "", "TypeError"):
        pred = np.nan
    else:
        try:
            pred = float(pred)
        except (ValueError, TypeError):
            pred = np.nan

    predicted_steps = count_steps(text)
    num_steps       = row.get("num_steps", 0) or 0
    diff            = predicted_steps - num_steps

    row["model_prediction"]    = pred
    row["answered"]            = not (isinstance(pred, float) and np.isnan(pred))
    row["answer_pos_norm"]     = normalized_answer_position(text)
    row["predicted_steps"]     = predicted_steps
    row["steps_diff"]          = diff
    row["comparison"]          = steps_diff_label(diff)
    return row


# ──────────────────────────────────────────────
# Answer extraction
# ──────────────────────────────────────────────
def extract_answers(text: str, gt):
    """Return (first_pred, correct, correct_any)."""
    matches = re.findall(r"<answer>(.*?)<\\?/answer>", text, re.DOTALL)

    genuine    = []
    first_pred = None

    for match in matches:
        val = match.strip()
        if val in EXAMPLE_PLACEHOLDERS:
            continue
        try:
            numeric = float(val)
        except ValueError:
            continue
        if np.isinf(numeric) or np.isnan(numeric):
            continue
        if gt is not None and not np.isnan(float(gt)) and gt != 42 and numeric == 42:
            continue
        genuine.append(numeric)
        if first_pred is None:
            first_pred = numeric

    gt_val = float(gt) if gt is not None else None

    if gt_val is not None and np.isnan(gt_val):
        correct_any = any(k in text for k in NAN_KEYWORDS)
        return first_pred, correct_any, correct_any

    correct_any = any(abs(gt_val - v) < 1e-3 for v in genuine) if gt_val is not None else False
    correct     = (first_pred is not None and abs(first_pred - gt_val) < 1e-3) if gt_val is not None else False
    return first_pred, correct, correct_any


# ──────────────────────────────────────────────
# Backend: vLLM
# ──────────────────────────────────────────────
def load_vllm_engine(model_name, max_tokens, tensor_parallel_size, gpu_memory_utilization, dtype_str):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        sys.exit("[ERROR] vLLM not installed. Run: pip install vllm  OR use --backend hf")
    llm = LLM(
        model=model_name,
        dtype=dtype_str,
        max_model_len=32768,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sp = SamplingParams(temperature=0, max_tokens=min(max_tokens, 32000))
    return llm, sp


def generate_vllm(llm, sp, prompts):
    outputs = llm.generate(prompts, sp)
    return (
        [o.outputs[0].text for o in outputs],
        [len(o.prompt_token_ids) for o in outputs],
        [len(o.outputs[0].token_ids) for o in outputs],
    )


# ──────────────────────────────────────────────
# Backend: HuggingFace Transformers
# ──────────────────────────────────────────────
def load_hf_model(model_name, device, torch_dtype_str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("[ERROR] transformers not installed. Run: pip install transformers torch")

    import torch
    dtype_map   = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(torch_dtype_str, torch.float16)

    print(f"  Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"  Loading model (dtype={torch_dtype_str}, device={device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return tokenizer, model


def generate_hf(tokenizer, model, prompts, max_tokens, device):
    import torch
    texts, in_toks, out_toks = [], [], []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        n_in  = inputs["input_ids"].shape[-1]
        n_out = output.shape[-1] - n_in
        texts.append(tokenizer.decode(output[0][n_in:], skip_special_tokens=True))
        in_toks.append(n_in)
        out_toks.append(n_out)
    return texts, in_toks, out_toks


# ──────────────────────────────────────────────
# Core experiment loop  (model loaded once, reused)
# ──────────────────────────────────────────────
def run_experiment_with_engine(
    *,
    dataset_path: str,
    output_path: str,
    engine,
    backend: str,
    prompts: dict,
    reasoning: bool,
    batch_size: int,
    model_tag: str,
    range_: str,
    dtype_: str,
    lookback_: str,
    task_: str,
    hf_device: str = "cuda",
    max_tokens: int = 32000,
):
    if not os.path.exists(dataset_path):
        print(f"    [SKIP] Dataset not found: {dataset_path}")
        return None

    df = pd.read_json(dataset_path, lines=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results   = []
    start_idx = 0
    if os.path.exists(output_path):
        try:
            done_df   = pd.read_json(output_path, lines=True)
            start_idx = len(done_df)
            results   = done_df.to_dict("records")
        except Exception:
            pass
        if start_idx >= len(df):
            acc = pd.DataFrame(results)["correct"].mean() if results else float("nan")
            print(f"    [DONE] Already complete. Accuracy={acc:.4f}")
            return acc

    remaining_df = df.iloc[start_idx:].reset_index(drop=True)

    for batch_start in tqdm(range(0, len(remaining_df), batch_size), desc="    batches", leave=False):
        batch_df      = remaining_df.iloc[batch_start: batch_start + batch_size]
        batch_prompts = [
            build_prompt(prompts, row["algorithm"], row["x"], row["y"], reasoning)
            for _, row in batch_df.iterrows()
        ]

        if backend == "vllm":
            llm, sp = engine
            texts, in_toks, out_toks = generate_vllm(llm, sp, batch_prompts)
        else:
            tokenizer, model = engine
            texts, in_toks, out_toks = generate_hf(tokenizer, model, batch_prompts, max_tokens, hf_device)

        for (_, row), text, n_in, n_out in zip(batch_df.iterrows(), texts, in_toks, out_toks):
            gt = row["expected_output"]
            first_pred, correct, correct_any = extract_answers(text, gt)

            record = {
                # ── Identity ──────────────────────────────────────────────
                "id":               row["id"],
                "model":            model_tag,
                "range":            range_,
                "dtype":            dtype_,
                "look_back":        lookback_,
                "task":             task_,
                # ── Input ─────────────────────────────────────────────────
                "num_steps":        row["num_steps"],
                "x":                row["x"],
                "y":                row["y"],
                # ── Ground truth ──────────────────────────────────────────
                "expected_output":  float(gt) if not pd.isna(gt) else "nan",
                # ── Raw prediction ────────────────────────────────────────
                "model_prediction": first_pred,
                "predicted_text":   text,
                # ── Correctness ───────────────────────────────────────────
                "correct":          correct,
                "correct@any":      correct_any,
                # ── Token counts ──────────────────────────────────────────
                "input_tokens":     n_in,
                "output_tokens":    n_out,
                "total_tokens":     n_in + n_out,
            }

            # ── Feature engineering (from build_dataset.py) ───────────────
            record = enrich_row(record, text)

            results.append(record)

        pd.DataFrame(results).to_json(output_path, orient="records", lines=True)

    acc = pd.DataFrame(results)["correct"].mean()
    print(f"    Accuracy = {acc:.4f}  ({len(results)} samples)")
    return acc


# ──────────────────────────────────────────────
# Summary CSV
# ──────────────────────────────────────────────
def generate_summary(results_root: str, model_tag: str):
    files = (
        glob(f"{results_root}/{model_tag}/*/*/*/*/*/*.jsonl") +
        glob(f"{results_root}/{model_tag}/*/*/*/*/*.jsonl")
    )
    rows = []
    for f in files:
        parts = f.replace(results_root + "/", "").split("/")
        if len(parts) < 6:
            continue
        _, range_, dtype_, lookback_, task_, _ = parts[:6]
        try:
            df = pd.read_json(f, lines=True)
            rows.append({
                "model":          model_tag,
                "range":          range_,
                "dtype":          dtype_,
                "lookback":       lookback_,
                "task":           task_,
                "n_samples":      len(df),
                "accuracy":       round(df["correct"].mean(), 4),
                "accuracy@any":   round(df["correct@any"].mean(), 4) if "correct@any" in df else float("nan"),
                # extra aggregates available because we now store these columns
                "answered_rate":  round(df["answered"].mean(), 4) if "answered" in df else float("nan"),
                "avg_out_tokens": round(df["output_tokens"].mean(), 1) if "output_tokens" in df else float("nan"),
                "avg_steps_pred": round(df["predicted_steps"].mean(), 2) if "predicted_steps" in df else float("nan"),
                "pct_over":       round((df["comparison"] == "over").mean(), 4) if "comparison" in df else float("nan"),
                "pct_under":      round((df["comparison"] == "under").mean(), 4) if "comparison" in df else float("nan"),
                "pct_exact":      round((df["comparison"] == "exact").mean(), 4) if "comparison" in df else float("nan"),
            })
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["range", "dtype", "lookback", "task"])


# ──────────────────────────────────────────────
# Unified parquet  (mirrors build_dataset.py)
# ──────────────────────────────────────────────
def build_parquet(results_root: str, model_tag: str, out_path: str):
    """
    Flatten all per-experiment JSONL files into one parquet.
    predicted_text (large) is excluded to keep the file manageable —
    pass keep_text=True if you need it.
    """
    files = (
        glob(f"{results_root}/{model_tag}/*/*/*/*/*/*.jsonl") +
        glob(f"{results_root}/{model_tag}/*/*/*/*/*.jsonl")
    )

    DROP_COLS = {"predicted_text", "intermediate_values"}

    frames = []
    for f in tqdm(files, desc="  Building parquet"):
        try:
            df = pd.read_json(f, lines=True)
        except Exception as e:
            print(f"  [WARN] {f}: {e}")
            continue

        # Drop heavy columns that shouldn't be in the unified file
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # Back-fill experiment-dimension columns from the path if missing
        # (files written by older runner versions may not have them)
        parts = f.replace(results_root + "/", "").split("/")
        if len(parts) >= 6:
            _, range_, dtype_, lookback_, task_, _ = parts[:6]
            for col, val in [
                ("model",     model_tag),
                ("range",     range_),
                ("dtype",     dtype_),
                ("look_back", lookback_),
                ("task",      task_),
            ]:
                if col not in df.columns:
                    df[col] = val

        frames.append(df)

    if not frames:
        print("  No data found — nothing written.")
        return

    unified = pd.concat(frames, ignore_index=True)

    # Canonical column order
    id_cols      = ["id", "model", "range", "dtype", "look_back", "task"]
    input_cols   = ["num_steps", "x", "y", "expected_output"]
    pred_cols    = ["model_prediction", "correct", "correct@any", "answered"]
    token_cols   = ["input_tokens", "output_tokens", "total_tokens"]
    feature_cols = ["answer_pos_norm", "predicted_steps", "steps_diff", "comparison"]

    ordered = id_cols + input_cols + pred_cols + token_cols + feature_cols
    rest    = [c for c in unified.columns if c not in ordered]
    unified = unified[[c for c in ordered if c in unified.columns] + rest]

    unified.to_parquet(out_path, index=False)
    print(f"\n  ✓ Parquet saved → {out_path}  ({len(unified):,} rows, {len(unified.columns)} columns)")
    print(f"  Columns: {list(unified.columns)}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Stepwise Arithmetic Benchmark — single-script runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--model", required=True,
                   help="HuggingFace model ID or path to local model directory")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                   help="Inference backend. Default: vllm")

    vg = p.add_argument_group("vLLM options")
    vg.add_argument("--tensor_parallel_size", type=int, default=1)
    vg.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    vg.add_argument("--vllm_dtype", default="float16",
                    choices=["float16", "bfloat16", "float32", "auto"])

    hg = p.add_argument_group("HuggingFace Transformers options")
    hg.add_argument("--device", default="cuda",
                    help="cuda / cpu / mps / auto")
    hg.add_argument("--hf_dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])

    eg = p.add_argument_group("Experiment matrix")
    eg.add_argument("--ranges",    nargs="+", default=ALL_RANGES,   choices=ALL_RANGES)
    eg.add_argument("--dtypes",    nargs="+", default=ALL_DTYPES,   choices=ALL_DTYPES)
    eg.add_argument("--lookbacks", nargs="+", default=ALL_LOOKBACKS)
    eg.add_argument("--tasks",     nargs="+", default=ALL_TASKS,    choices=ALL_TASKS)
    eg.add_argument("--reasoning", default="true",
                    help="true/false — whether to generate chain-of-thought. Default: true")

    pg = p.add_argument_group("Paths")
    pg.add_argument("--dataset_root", default="dataset")
    pg.add_argument("--results_root", default="results")
    pg.add_argument("--prompt_path",  default="prompts/prompts.json")

    rg = p.add_argument_group("Runtime")
    rg.add_argument("--batch_size",    type=int, default=500)
    rg.add_argument("--max_tokens",    type=int, default=32000)
    rg.add_argument("--summary_only",  action="store_true",
                    help="Skip inference; only regenerate summary/parquet from existing results.")
    rg.add_argument("--build_parquet", action="store_true",
                    help="After inference, also write a unified parquet file.")

    return p.parse_args()


def main():
    args      = parse_args()
    reasoning = args.reasoning.strip().lower() in ("true", "1", "yes")
    model_tag = Path(args.model).name

    print("=" * 60)
    print("  Stepwise Arithmetic Benchmark")
    print("=" * 60)
    print(f"  Model       : {args.model}  (tag: {model_tag})")
    print(f"  Backend     : {args.backend}")
    print(f"  Reasoning   : {reasoning}")
    print(f"  Ranges      : {args.ranges}")
    print(f"  Dtypes      : {args.dtypes}")
    print(f"  Lookbacks   : {args.lookbacks}")
    print(f"  Tasks       : {args.tasks}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Results dir : {args.results_root}/{model_tag}/")
    print("=" * 60)

    experiments = build_experiment_list(args.ranges, args.dtypes, args.lookbacks, args.tasks)
    print(f"\n  Total experiments: {len(experiments)}\n")

    if not args.summary_only:
        if not os.path.exists(args.prompt_path):
            sys.exit(f"[ERROR] Prompt file not found: {args.prompt_path}")
        prompts = load_prompts(args.prompt_path)

        print("Loading model (this may take a while)...")
        t0 = time.time()
        if args.backend == "vllm":
            engine = load_vllm_engine(
                args.model, args.max_tokens,
                args.tensor_parallel_size, args.gpu_memory_utilization, args.vllm_dtype,
            )
        else:
            engine = load_hf_model(args.model, args.device, args.hf_dtype)
        print(f"  Model loaded in {time.time() - t0:.1f}s\n")

        all_accs = {}
        for i, (range_, dtype_, lookback_, task_) in enumerate(experiments, 1):
            dataset_path = (
                f"{args.dataset_root}/{range_}/{dtype_}/{lookback_}/"
                f"algo_reasoning_dataset_{task_}.jsonl"
            )
            output_path = (
                f"{args.results_root}/{model_tag}/{range_}/{dtype_}/{lookback_}/"
                f"{task_}/results.jsonl"
            )
            print(f"[{i:3d}/{len(experiments)}] {range_} | {dtype_} | {lookback_} | {task_}")
            acc = run_experiment_with_engine(
                dataset_path=dataset_path,
                output_path=output_path,
                engine=engine,
                backend=args.backend,
                prompts=prompts,
                reasoning=reasoning,
                batch_size=args.batch_size,
                model_tag=model_tag,
                range_=range_,
                dtype_=dtype_,
                lookback_=lookback_,
                task_=task_,
                hf_device=args.device if args.backend == "hf" else "cuda",
                max_tokens=args.max_tokens,
            )
            if acc is not None:
                all_accs[f"{range_}|{dtype_}|{lookback_}|{task_}"] = acc

        if all_accs:
            print(f"\nAll experiments complete. Mean accuracy: {np.mean(list(all_accs.values())):.4f}\n")

    # ── Summary CSV ──
    print("Generating summary CSV...")
    summary = generate_summary(args.results_root, model_tag)
    if summary is not None:
        summary_path = f"{args.results_root}/{model_tag}/summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Summary saved → {summary_path}")
        print("\n" + summary.to_string(index=False))
    else:
        print("  No results found to summarize.")

    # ── Unified parquet ──
    if args.build_parquet:
        print("\nBuilding unified parquet...")
        parquet_path = f"{args.results_root}/{model_tag}/unified_results.parquet"
        build_parquet(args.results_root, model_tag, parquet_path)


if __name__ == "__main__":
    main()