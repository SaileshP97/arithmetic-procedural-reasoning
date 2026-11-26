"""Condensed view of sampled traces for SFB validation: head, first Step lines, first S-assignments, and a middle snippet."""
import json, re, sys
import pandas as pd
s = pd.read_csv("manual_validation/sfb_validation_sample.csv")
for no in map(int, sys.argv[1:]):
    r = s[s.no == no].iloc[0]
    with open(r.file) as fh:
        t = next(json.loads(l) for l in fh if r.id in l)["predicted_text"] or ""
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    step_lines = [l for l in lines if re.search(r"Step\s*\d+", l)]
    val_lines = [l for l in lines if re.search(r"\bS\d+\s*(=|≈|:|is)\s*[-−(]?\d", l)]
    print(f"##### #{no} {r.model} T={r['T']} G={r['G']} sfb={r.sfb} markers={r.n_markers} values={r.n_values} chars={r.chars}")
    print("  HEAD:", t[:220].replace("\n", " ⏎ "))
    print(f"  STEP-LINES ({len(step_lines)}):", " || ".join(x[:90] for x in step_lines[:4]))
    print(f"  VALUE-LINES ({len(val_lines)}):", " || ".join(x[:90] for x in val_lines[:4]))
    m = len(t) // 2
    print("  MIDDLE:", t[m:m + 220].replace("\n", " ⏎ "))
    print("  TAIL:", t[-160:].replace("\n", " ⏎ "))
