# Stepwise Arithmetic Benchmark

A benchmark for evaluating LLMs on multi-step arithmetic reasoning tasks. Models are tested across varying number ranges, data types, look-back depths, and arithmetic operations — with or without chain-of-thought reasoning.

---

## Repository Structure

```
.
├── dataset/                    # Benchmark datasets (JSONL)
│   └── <range>/<dtype>/<lookback>/algo_reasoning_dataset_<task>.jsonl
├── prompts/
│   └── prompts.json            # Prompt templates
├── run_benchmark.py            # ← Main benchmark runner (single script)
├── dataset_generator.py        # Generate new datasets
└── results/                    # Output directory (auto-created)
    └── <model>/<range>/<dtype>/<lookback>/<task>/results.jsonl
```

---

## Installation

```bash
pip install pandas numpy tqdm

# For vLLM backend (GPU, recommended):
pip install vllm

# For HuggingFace Transformers backend (CPU or GPU):
pip install transformers torch accelerate
```

---

## Quick Start

```bash
# Run full benchmark with a HuggingFace model (vLLM, GPU)
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct

# Run with a local model (HuggingFace Transformers, CPU)
python run_benchmark.py --model ./models/my-model --backend hf --device cpu
```

Results are saved to `results/<model-name>/` and a `summary.csv` is generated at the end.

---

## Options

### Model & Backend

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace model ID or path to a local model directory |
| `--backend` | `vllm` | `vllm` (fast, GPU-only) or `hf` (Transformers, CPU/GPU) |

### vLLM Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `--gpu_memory_utilization` | `0.9` | Fraction of GPU memory to allocate |
| `--vllm_dtype` | `float16` | Model dtype: `float16`, `bfloat16`, `float32`, `auto` |

### HuggingFace Transformers Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda` | `cuda`, `cpu`, `mps`, or `auto` |
| `--hf_dtype` | `float16` | Torch dtype: `float16`, `bfloat16`, `float32` |

### Experiment Matrix

| Flag | Default | Description |
|------|---------|-------------|
| `--ranges` | all | `range_0_1`, `range_1_10`, `range_10_100` |
| `--dtypes` | all | `float`, `int` |
| `--lookbacks` | all | `look_back1` … `look_back7` |
| `--tasks` | all | `mixed`, `add`, `subtract`, `multiply`, `divide` |
| `--reasoning` | `true` | `true` to enable chain-of-thought, `false` for direct answers |

### Runtime

| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | `500` | Samples per inference batch |
| `--max_tokens` | `32000` | Maximum tokens to generate |
| `--dataset_root` | `dataset` | Root folder containing dataset files |
| `--results_root` | `results` | Root folder for saving results |
| `--prompt_path` | `prompts/prompts.json` | Path to prompt templates |
| `--summary_only` | `false` | Skip inference; regenerate `summary.csv` only |

---

## Usage Examples

```bash
# Multi-GPU inference (4 GPUs)
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

# Run a quick subset (single range, single task)
python run_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --ranges range_1_10 \
    --tasks add multiply \
    --lookbacks look_back1 look_back2

# No chain-of-thought
python run_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reasoning false

# Regenerate summary CSV from existing results (no inference)
python run_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --summary_only
```

---

## Output

Each experiment writes a JSONL file to:
```
results/<model>/<range>/<dtype>/<lookback>/<task>/results.jsonl
```

Each row contains:

| Field | Description |
|-------|-------------|
| `id` | Sample ID |
| `expected_output` | Ground truth answer |
| `model_prediction` | First numeric answer extracted from model output |
| `correct` | Whether `model_prediction` matches ground truth (±1e-3) |
| `correct@any` | Whether *any* answer in the output matches ground truth |
| `input_tokens` | Number of prompt tokens |
| `output_tokens` | Number of generated tokens |
| `predicted_text` | Full model output text |

After all experiments, a summary is saved to:
```
results/<model>/summary.csv
```

---

## Notes

- **Auto-resume**: interrupted runs are automatically resumed from where they stopped.
- **Division by zero**: samples with undefined expected output (NaN) are evaluated by checking whether the model correctly identifies the undefined result (e.g., outputs `<answer>NaN</answer>` or states "division by zero").
- **Experiment matrix**: not all combinations are valid. `range_0_1` excludes integer dtype; lookbacks beyond `look_back1` only run the `mixed` task. The script handles these constraints automatically.
- **HuggingFace login**: for gated models (e.g. Llama), authenticate first with `huggingface-cli login`.