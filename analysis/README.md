# Camera-ready analyses (EACL 2027)

New analyses promised in the ARR August 2026 rebuttal for
"When LLMs Stop Following Steps".

| ID | Analysis | Requested by | Outputs |
|----|----------|--------------|---------|
| C2 | Compare intermediate states with the reference executor: correct-prefix length, first mismatch position | JsBn (W1), TMAk | `outputs/c2_*`, `figures/c2_*` |
| C8 | Error type at the first erroneous step | TMAk | `outputs/c8_*`, `figures/c8_error_types.pdf` |
| C5 | Statistics of the intermediate values (max magnitude, min non-zero, non-finite) and their relation to FAA | JsBn | `outputs/c5_*`, `figures/c5_faa_vs_magnitude.pdf` |

## Reproduce

```bash
cd analysis/scripts
python3 test_parser.py            # parser unit tests (formats seen in the generations)
python3 extract_all.py            # ~25 min, 8 cores; reads the SSD, writes outputs/per_example.parquet
python3 report.py                 # tables (CSV + LaTeX) and figures (PDF)
```

Inputs (read-only):
- `/Volumes/Crucial/final_results/<model>/<range>/<dtype>/look_back<k>/<task>/results.jsonl`: 13 models, all 55 configurations
- `/Volumes/Crucial/Reasoning/reasoning_soket/results/{deepseek-v3.2,kimi-k2.5}/...`: partial traces
  (DeepSeek-V3.2: 13 configurations, 12,767 examples; Kimi-K2.5: 2 configurations, 777 examples)
- `/Volumes/Crucial/reasoning_cdac/dataset/`: procedure text (`algorithm`) joined by example `id`
- `/Volumes/Crucial/openreview_attachments/extracted/unified_results.parquet`: the scoring used in the paper
  (attached to the ARR submission); `correct` from this file is used for FAA so that numbers match the paper

## Definitions

**State extraction.** For each variable `S_k` (k ≥ 3) we take the *first* numeric value the model commits to
in its output (mirroring First-Answer Accuracy). A committed value ends an assignment clause:
`S3 = 6.049 + 7.183 = 13.232`, `S3 ≈ 13.232`, `S3: 13.232`, `S3 is (approximately) 13.232`,
a markdown table row, or an exact integer fraction (`S6 = 86/157`). Restatements of the procedure
(`S3 = S1 + S2`) and unfinished expressions (`S3 = 6.049 + 7.183`) are not values. Unicode/LaTeX
operators, bold markup and `a × 10^b` notation are normalized first. The last value per variable is
reported as a robustness check.

**Reference-matching prefix (C2).** Number of initial steps whose extracted value matches the reference
executor's value within 1% relative tolerance (0.1% and 5% in `c2_tolerance_sensitivity`). Evaluation
stops at the first non-finite reference value.

**Faithful prefix (C2).** Number of initial steps whose value is the correct result of the specified
operation applied to the model's *own* previous values (relative error ≤ 0.1%, or within one unit of the
last digit the model wrote). Accumulated rounding from earlier steps therefore does not count as an
error; only a step that does not follow from the model's own state does. `first_error_step` is the first
step that is not faithful.

**Error type at the first non-faithful step (C8).**
- *Skipped step*: no value for this variable, but values for later variables
- *Trace ends early*: no value for this or any later variable (truncation or stopping)
- *Operand order swapped*: the value equals the step with operands reversed (`S2 − S12` for `S13 = S12 − S2`)
- *Wrong operation*: the value equals a different operation on the correct operands
- *Wrong operand (state retrieval)*: the value equals an operation on a different pair of available
  variables (S1, S2, or the 8 most recent)
- *Arithmetic error*: correct operands and operation, wrong result

Examples with no extractable intermediate state at all (answer-only outputs) are reported separately as
"visible states (%)" and excluded from prefix and error-type statistics.

**Reference statistics (C5).** For S3..S_final of the reference execution: ⌊log10|v|⌋ of the largest and
of the smallest non-zero value, and whether any value is non-finite (nan/inf). Computed once per example.

## Caveats (state in the paper)

- Intermediate-state metrics measure what is *visible* in the output. Models that answer without a trace
  (most outputs of OA-120 via the API, MI-7, OL-7) have little or no visible state; their statistics cover
  only the visible subset.
- DS-685 and KK-1T traces exist only for a subset of configurations (see `outputs/coverage.csv`);
  `outputs/c2_common_subset.*` compares all 15 models on the configurations they share.
- For 4 configurations of QW-4 (range 10–100, look-back 2–3, 4,000 examples) the procedure text was not
  stored; operations and operands were reconstructed from the reference values
  (`procedure_source = inferred`).
- The tolerances are choices; sensitivity tables are provided.
- Parser validation: `scripts/test_parser.py` (format unit tests) and `scripts/inspect_cases.py`
  (prints trace context for random examples of a given model and error type, for manual checking).
