# Stepwise Arithmetic Benchmark

A benchmark for evaluating LLMs on **multi-step arithmetic reasoning**. Models are tested on chained calculations across varying number ranges, data types, look-back depths, and arithmetic operations — with or without chain-of-thought reasoning.

The key challenge is *look-back*: each step depends on a result computed several steps earlier, requiring the model to track intermediate values across a growing context. `look_back1` means the current step depends only on the previous result; `look_back7` means it depends on a result computed seven steps ago.

---

## Repository Structure

```
.
├── dataset/                         # Benchmark datasets (JSONL)
│   └── <range>/<dtype>/<lookback>/
│       └── algo_reasoning_dataset_<task>.jsonl
├── prompts/
│   └── prompts.json                 # Prompt templates (reasoning / no_reasoning modes)
├── run_benchmark.py                 # Main runner — inference + results + parquet
├── dataset_generator.py            # Generate new or custom datasets
└── results/                         # Auto-created on first run
    └── <model>/
        ├── summary.csv              # Aggregated metrics across all experiments
        ├── unified_results.parquet  # All rows in one file (--build_parquet)
        └── <range>/<dtype>/<lookback>/<task>/
            └── results.jsonl        # Per-experiment results
```

---

## Installation

```bash
pip install pandas numpy tqdm pyarrow

# vLLM backend — fast GPU inference (recommended)
pip install vllm

# HuggingFace Transformers backend — CPU or GPU
pip install transformers torch accelerate
```

For gated models (e.g. Llama), log in first:
```bash
huggingface-cli login
```

---

## Quick Start

```bash
# Full benchmark, HuggingFace model, vLLM backend (GPU)
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct

# Local model, CPU
python run_benchmark.py --model ./models/my-model --backend hf --device cpu

# Full benchmark + unified parquet output
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --build_parquet
```

Results are saved under `results/<model-name>/`. A `summary.csv` is always generated at the end.

---

## Experiment Matrix

The benchmark varies four dimensions:

| Dimension | Values | Description |
|-----------|--------|-------------|
| **Range** | `range_0_1`, `range_1_10`, `range_10_100` | Magnitude of input numbers |
| **Dtype** | `float`, `int` | Whether inputs are floating-point or integer |
| **Look-back** | `look_back1` … `look_back7` | How many steps back the current step depends on |
| **Task** | `add`, `subtract`, `multiply`, `divide`, `mixed` | Arithmetic operation(s) used |

Not all combinations are valid:
- `range_0_1` + `int` is skipped (integers in (0,1) don't exist)
- `look_back2` through `look_back7` only run the `mixed` task

The script enforces these constraints automatically.

---

## All Options

### Model & Backend

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace model ID (e.g. `Qwen/Qwen2.5-7B-Instruct`) or path to a local directory |
| `--backend` | `vllm` | `vllm` — fast, GPU-only; `hf` — HuggingFace Transformers, CPU/GPU |

### vLLM Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `--gpu_memory_utilization` | `0.9` | Fraction of GPU memory to allocate |
| `--vllm_dtype` | `float16` | `float16`, `bfloat16`, `float32`, or `auto` |

### HuggingFace Transformers Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda` | `cuda`, `cpu`, `mps`, or `auto` |
| `--hf_dtype` | `float16` | `float16`, `bfloat16`, or `float32` |

### Experiment Matrix

| Flag | Default | Description |
|------|---------|-------------|
| `--ranges` | all | One or more of `range_0_1`, `range_1_10`, `range_10_100` |
| `--dtypes` | all | `float`, `int`, or both |
| `--lookbacks` | all | Any subset of `look_back1` … `look_back7` |
| `--tasks` | all | Any subset of `add`, `subtract`, `multiply`, `divide`, `mixed` |
| `--reasoning` | `true` | `true` enables chain-of-thought; `false` asks for a direct answer |

### Runtime & Paths

| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | `500` | Samples per inference batch |
| `--max_tokens` | `32000` | Maximum tokens to generate per sample |
| `--dataset_root` | `dataset` | Root folder containing dataset files |
| `--results_root` | `results` | Root folder for saving results |
| `--prompt_path` | `prompts/prompts.json` | Path to prompt template file |
| `--build_parquet` | off | Also write a unified `unified_results.parquet` after inference |
| `--summary_only` | off | Skip inference entirely; only regenerate `summary.csv` and (optionally) the parquet |

---

## Usage Examples

```bash
# Multi-GPU (4 GPUs via tensor parallelism)
python run_benchmark.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --backend vllm \
    --tensor_parallel_size 4

# Local model, CPU only
python run_benchmark.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --backend hf \
    --device cpu \
    --hf_dtype float32

# Quick subset — one range, two tasks, two lookbacks
python run_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --ranges range_1_10 \
    --tasks add multiply \
    --lookbacks look_back1 look_back2

# No chain-of-thought
python run_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reasoning false

# Run everything and produce a unified parquet at the end
python run_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --build_parquet

# Rebuild summary and parquet from existing results, no inference
python run_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --summary_only \
    --build_parquet
```

---

## Output Files

### Per-experiment: `results.jsonl`

One file per experiment combination. Each row is one sample with the following fields:

**Identity**

| Field | Description |
|-------|-------------|
| `id` | Sample ID |
| `model` | Model name/tag |
| `range` | Number range used |
| `dtype` | `float` or `int` |
| `look_back` | Look-back depth |
| `task` | Arithmetic operation |

**Input**

| Field | Description |
|-------|-------------|
| `num_steps` | Number of reasoning steps required |
| `x`, `y` | Input operands |
| `expected_output` | Ground truth answer (`"nan"` for division-by-zero cases) |

**Prediction & Correctness**

| Field | Description |
|-------|-------------|
| `model_prediction` | First valid numeric answer extracted from model output (`NaN` if none found) |
| `correct` | `True` if `model_prediction` matches ground truth within ±0.001 |
| `correct@any` | `True` if *any* answer tag in the output matches ground truth — more lenient than `correct` |
| `answered` | `True` if the model produced any valid numeric answer at all |
| `predicted_text` | Full raw model output |

**Token Counts**

| Field | Description |
|-------|-------------|
| `input_tokens` | Number of tokens in the prompt |
| `output_tokens` | Number of tokens generated |
| `total_tokens` | Sum of both |

**Reasoning Features**

| Field | Description |
|-------|-------------|
| `answer_pos_norm` | Position of the first valid `<answer>` tag as a percentage of total output length (0–100). Low values mean the model answered early; high values mean it answered only after extended reasoning or self-correction |
| `predicted_steps` | Number of distinct `Step N:` labels found in the output |
| `steps_diff` | `predicted_steps − num_steps` — positive means the model wrote more steps than needed, negative means fewer |
| `comparison` | `"over"` if `steps_diff > 2`, `"under"` if `steps_diff < −2`, `"exact"` otherwise |

---

### Aggregated: `summary.csv`

One row per experiment combination, with the following columns:

| Column | Description |
|--------|-------------|
| `model`, `range`, `dtype`, `lookback`, `task` | Experiment dimensions |
| `n_samples` | Number of evaluated samples |
| `accuracy` | Fraction where `model_prediction` is correct (strict) |
| `accuracy@any` | Fraction where any answer in the output is correct (lenient) |
| `answered_rate` | Fraction of samples where the model produced any numeric answer |
| `avg_out_tokens` | Mean number of generated tokens |
| `avg_steps_pred` | Mean number of reasoning steps written |
| `pct_over` | Fraction of samples where the model over-elaborated (wrote too many steps) |
| `pct_under` | Fraction of samples where the model under-elaborated (wrote too few steps) |
| `pct_exact` | Fraction of samples where step count was within ±2 of ground truth |

---

### Unified: `unified_results.parquet`

Produced when `--build_parquet` is passed. All per-experiment JSONL files concatenated into a single parquet, with `predicted_text` and `intermediate_values` dropped to keep the file size manageable. Useful for cross-model or cross-dimension analysis in notebooks.

---

## Notes

- **Auto-resume** — if a run is interrupted, re-running the same command picks up from where it left off.
- **Division by zero** — samples where the expected output is undefined (NaN) are evaluated by checking whether the model's output contains recognised phrases such as `<answer>NaN</answer>` or `"division by zero"`.
- **Model is loaded once** — the model stays in memory across all experiment combinations, so there is no repeated load/unload overhead.