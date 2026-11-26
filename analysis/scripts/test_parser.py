"""Unit tests for trace_parser.extract_states on formats observed in the stored generations."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parser import extract_states
from decimal import Decimal as D
cases = [
    ("Step 1: S3 = S1 + S2 = 6.049 + 7.183 = 13.232", {3: "13.232"}),
    ("Step 1: S3 = S1 + S2\nSo, S3 = 6.049 + 7.183 = 13.232", {3: "13.232"}),
    ("S6 = 7.183 / 13.232 ≈ 0.5403", {6: "0.5403"}),
    ("So S3 is 139. Step 2", {3: "139"}),
    ("S14 is approximately 5.49684211.", {14: "5.49684211"}),
    ("So S7 is about 0.514.", {7: "0.514"}),
    ("S3: 13.232", {3: "13.232"}),
    ("- S3 = 6.049 + 7.183 = 13.232", {3: "13.232"}),
    ("**Step 2:** S4 = S2 + S3 = 0.687 + 1.248 = **1.935**", {4: "1.935"}),
    ("| 7 | `S9 = S2 * S8` | **3.899296** |", {9: "3.899296"}),
    ("S17 = 7.183 × (−7.17044) ≈ −51.50227", {17: "-51.50227"}),
    ("So S6=86/157. Let me note", {6: str(D(86) / D(157))}),
    ("State: S1=5, S2=8, S3=13", {3: "13"}),
    ("S2 = 80 and S3 = 139", {3: "139"}),
    ("S20 = 1.2 × 10^5", {20: "1.2e5"}),
    ("Step 1: S3 = S1 + S2. That's 2.679 + 9.704, which equals 12.383.", {3: "12.383"}),
    ("Step 2: S4 = S3 / S2. So 44.242 divided by 20.43. Let me calculate that. 44.242 ÷ 20.43 ≈ 2.1655", {4: "2.1655"}),
    ("Step 30: S32 = S31 + S2. Adding 0.976 to 1.232 gives 2.208.", {32: "2.208"}),
    ("Step 2: S4 = S2 + S3. S2 is 7.183 and S3 is 13.232. Adding them: 7.183 + 13.232 = 20.415.", {3: "13.232", 4: "20.415"}),
    ("Step 1: S3 = S1 + S2\nStep 2: S4 = S2 * S3\nStep 3: S5 = S4 / S2", {}),
    # must NOT produce a value
    ("Step 1: S3 = S1 + S2", {}),
    ("S3 = 6.049 + 7.183", {}),
    ("S13 = S10 / S2 = (-1.4280109...) / 5.862.", {}),
    ("S6 ≈ let's calculate precisely.", {}),
]
bad = 0
for text, want in cases:
    got = {k: v for k, v in extract_states(text, 99)[0].items()}
    ok = set(got) == set(want) and all(abs(got[k] - D(want[k])) < D("1e-12") for k in want)
    bad += not ok
    print("ok  " if ok else "FAIL", repr(text)[:60], {k: str(v) for k, v in got.items()})
print("failures:", bad); sys.exit(bad > 0)
