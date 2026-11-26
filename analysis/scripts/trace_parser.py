"""Parse model-generated traces and compare intermediate states with the reference executor.

Used for the camera-ready analyses:
  C2  first mismatch position / correct-prefix length   (Reviewer JsBn W1, TMAk)
  C8  error type at the first mismatch                  (Reviewer TMAk)
  C5  intermediate-value statistics of the reference    (Reviewer JsBn)

Conventions
-----------
* Variables: S1 = x, S2 = y, Step k assigns one new variable (normally S{k+2}).
* Reference values come from each result file's own `intermediate_values`
  (index i -> S{i+1}); this is what the model was scored against.
* For every variable we keep the FIRST numeric value the model commits to
  (mirrors First-Answer Accuracy). A "last value" variant is also computed.
* A committed value is a number that ends an assignment clause, e.g.
  "S3 = 6.049 + 7.183 = 13.232", "S3 ≈ 13.232", "S3: 13.232", "S3 is 13.232",
  or a markdown table row "| 1 | S3 = S1 + S2 | 13.232 |".
  Restatements such as "S3 = S1 + S2" or unfinished "S3 = 6.049 + 7.183" are ignored.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, getcontext, Overflow, DivisionByZero

ctx = getcontext()
ctx.prec = 60
ctx.Emax = 10**8
ctx.Emin = -10**8

OPS = {"+", "-", "*", "/"}

# ---------------------------------------------------------------- text normalisation
_NORM = [
    (re.compile(r"\\approx"), "≈"),
    (re.compile(r"\\times|×|⋅|·|\\cdot"), "*"),
    (re.compile(r"÷|\\div"), "/"),
    (re.compile(r"[−–—]"), "-"),
    (re.compile(r"\\\(|\\\)|\\\[|\\\]|\$"), " "),
    (re.compile(r"\*\*|__|`"), " "),                      # markdown bold / code
    (re.compile(r"S_\{?(\d+)\}?"), r"S\1"),                # S_{3} -> S3
    (re.compile(r"(?<=\d),(?=\d{3}\b)"), ""),              # 1,234 -> 1234
    (re.compile(r"(\d)\s*\*\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?"), r"\1e\2"),  # 1.2 * 10^5
    (re.compile(r"(\d)\s*e\s*([-+]\d+)"), r"\1e\2"),
]


def normalise(text: str) -> str:
    for pat, rep in _NORM:
        text = pat.sub(rep, text)
    return text


NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_LEAD_NUM = re.compile(r"^\s*\(?\s*(" + NUM + r")\s*\)?(.*)$", re.S)
_ASSIGN = re.compile(r"\bS(\d+)\s*(:=|=|≈|~|:|->|→|\bis\b|\bequals\b|\bbecomes\b)\s*"
                     r"(?:(?:now|approximately|approx\.?|about|roughly|around|equal to|exactly|still|then)\s+)*")
# clause boundaries: newline, ';', ' and ', ', ', or a sentence-ending period
_BOUNDARY = re.compile(r"\n|;|\band\b|,\s|\.(?=\s|$)|\|")
_LAST_EQ = re.compile(r"[=≈]")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
# "Step k: S_t = S_a op S_b" header (the procedure line itself, followed by prose computation)
_STEP_HEADER = re.compile(r"Step\s*\d+\s*:?\s*S(\d+)\s*=\s*S\d+\s*[-+*/]\s*S\d+")
_STEP_MARK = re.compile(r"Step\s*\d+")
# a committed result inside prose: "= 12.383", "≈ 2.165", "equals 12.383", "gives (me) 2.208", "is approximately 0.98"
_RESULT = re.compile(r"(?:=|≈|~|\bequals\b|\bgives(?: me)?\b|\bgiving\b|\bresults in\b|\bwhich is\b|\bis\b|\bget\b)\s*"
                     r"(?:(?:approximately|approx\.?|about|roughly|around|exactly)\s+)?\(?\s*(" + NUM + r")")
_TABLE_VAR = re.compile(r"\bS(\d+)\s*=")


def _to_dec(s) -> Decimal | None:
    try:
        d = Decimal(str(s).strip())
    except (InvalidOperation, ValueError):
        return None
    return d


def _clause_value(clause: str) -> Decimal | None:
    """Return the committed numeric value at the end of an assignment clause, if any."""
    parts = _LAST_EQ.split(clause)
    tail = parts[-1]
    m = _LEAD_NUM.match(tail)
    if not m:
        return None
    rest = m.group(2).lstrip(" \t.…)")  # "(-1.428...) / 5.862": skip ellipsis / closing paren
    # an exact integer fraction written as the value, e.g. "S6 = 86/157"
    fm = re.fullmatch(r"\s*([-+]?\d+)\s*/\s*(\d+)\s*\.?\s*", tail)
    if fm and int(fm.group(2)) != 0:
        return Decimal(fm.group(1)) / Decimal(fm.group(2))
    if rest[:1] in {"+", "-", "*", "/", "^", "%"} or rest[:1].isdigit():
        return None  # unfinished expression such as "6.049 + 7.183"
    return _to_dec(m.group(1))


def extract_states(text: str, max_var: int) -> tuple[dict[int, Decimal], dict[int, Decimal], int]:
    """Return ({var: first value}, {var: last value}, position of first committed value per var unused)."""
    first: dict[int, Decimal] = {}
    last: dict[int, Decimal] = {}
    if not text:
        return first, last, 0
    t = normalise(text)

    # markdown table rows: | k | S9 = S2 * S8 | 3.8993 |
    for row in _TABLE_ROW.finditer(t):
        line = row.group(0)
        mv = _TABLE_VAR.search(line)
        if not mv:
            continue
        var = int(mv.group(1))
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        val = None
        for c in reversed(cells):
            m = re.fullmatch(r"\(?\s*(" + NUM + r")\s*\)?", c)
            if m:
                val = _to_dec(m.group(1))
                break
        if val is not None and 3 <= var <= max_var:
            first.setdefault(var, val)  # provisional; prose values below may come earlier
            last[var] = val

    found = []
    for m in _ASSIGN.finditer(t):
        var = int(m.group(1))
        if var < 3 or var > max_var:
            continue
        # "S4 = S2 + S3 = 0.687 + ..." : S3 here is an operand, not an assignment target
        j = m.start() - 1
        while j >= 0 and t[j] in " \t":
            j -= 1
        if j >= 0 and t[j] in "+-*/=≈(^":
            # a "-" or "*" that starts a line is a list bullet ("- S3 = 13.232"), not an operator
            k = j - 1
            while k >= 0 and t[k] in " \t":
                k -= 1
            if not (t[j] in "-*" and (k < 0 or t[k] == "\n")):
                continue
        start = m.end()
        b = _BOUNDARY.search(t, start)
        clause = t[start: b.start() if b else min(len(t), start + 200)]
        val = _clause_value(clause)
        if val is not None:
            found.append((m.start(), var, val))

    # step blocks written in prose:
    #   "Step 2: S4 = S3 / S2. So 44.242 divided by 20.43 ... ≈ 2.165"
    #   "Step 1: S3 = S1 + S2. That's 2.679 + 9.704, which equals 12.383."
    # The block runs to the next "Step k" marker; the value is the first committed result in it.
    assigned = {v for _, v, _ in found}
    for hm in _STEP_HEADER.finditer(t):
        var = int(hm.group(1))
        if var < 3 or var > max_var or var in assigned:
            continue
        nxt = _STEP_MARK.search(t, hm.end())
        block = t[hm.end(): nxt.start() if nxt else hm.end() + 600][:600]
        for rm in _RESULT.finditer(block):
            if re.search(r"S\d+\s*$", block[:rm.start()]):
                continue  # "S2 is 7.183": value of an operand, not the result of this step
            rest = block[rm.end(1):].lstrip(" \t.…)")
            if rest[:1] in {"+", "-", "*", "/", "^", "%"} or rest[:1].isdigit():
                continue  # unfinished expression
            val = _to_dec(rm.group(1))
            if val is not None:
                found.append((hm.end() + rm.start(), var, val))
                break
    found.sort()
    prose_first: dict[int, Decimal] = {}
    for _, var, val in found:
        prose_first.setdefault(var, val)
        last[var] = val
    # prose occurrences take precedence for "first" (tables usually appear as end-of-trace summaries)
    for var, val in prose_first.items():
        first[var] = val
    return first, last, len(found)


# ---------------------------------------------------------------- procedure + reference
_STEP = re.compile(r"Step\s+(\d+)\s*:\s*S(\d+)\s*=\s*S(\d+)\s*([-+*/])\s*S(\d+)")


def parse_algorithm(alg: str):
    """List of (step_no, target, a, op, b)."""
    return [(int(k), int(t), int(a), op, int(b)) for k, t, a, op, b in _STEP.findall(alg or "")]


def ref_values(iv) -> dict[int, Decimal | None]:
    out = {}
    for i, v in enumerate(iv or []):
        out[i + 1] = _to_dec(v)
    return out


def apply(op: str, a: Decimal | None, b: Decimal | None) -> Decimal | None:
    if a is None or b is None or not a.is_finite() or not b.is_finite():
        return None
    try:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return None if b == 0 else a / b
    except (Overflow, DivisionByZero, InvalidOperation):
        return None
    return None


def close(m: Decimal | None, r: Decimal | None, rel: float) -> bool:
    if m is None or r is None or not m.is_finite() or not r.is_finite():
        return False
    return abs(m - r) <= Decimal(str(rel)) * abs(r) + Decimal("1e-9")


LOCAL_REL = Decimal("1e-3")


def locally_consistent(m: Decimal | None, own: Decimal | None) -> bool:
    """Is the model's value the correct result of the step applied to its OWN previous values?

    Accepts relative error <= 0.1%, or an error within one unit of the last digit the
    model wrote (models round displayed values, e.g. 0.0126 for 0.012562).
    Integer-valued results written without decimals must match exactly.
    """
    if m is None or own is None or not m.is_finite() or not own.is_finite():
        return False
    diff = abs(m - own)
    if diff <= LOCAL_REL * abs(own) + Decimal("1e-12"):
        return True
    exp = m.as_tuple().exponent
    if isinstance(exp, int) and exp < 0:
        return diff <= Decimal(1).scaleb(exp)
    return own == own.to_integral_value() and diff == 0 or (own != own.to_integral_value() and diff < 1)


# ---------------------------------------------------------------- per-example evaluation
TOLS = (1e-3, 1e-2, 5e-2)
PRIMARY = 1e-2


def evaluate(text: str, steps, ref: dict, n_steps: int) -> dict:
    """Compare model states with the reference executor and with the procedure itself.

    ref_prefix_*      : number of initial steps whose value matches the reference (relative tol)
    faithful_prefix   : number of initial steps whose value is the correct result of the step
                        applied to the model's OWN previous values (separates execution errors
                        from accumulated rounding)
    error_type        : category of the first non-faithful step
    """
    steps = steps[:n_steps]
    targets = [t for _, t, *_ in steps]
    max_var = max(targets) if targets else n_steps + 2
    first, last, n_found = extract_states(text, max_var)
    out = {"n_parsed_states": sum(1 for t in targets if t in first), "n_assign_matches": n_found}

    # stop evaluation at the first non-finite reference value
    censor = None
    for k, t in enumerate(targets, 1):
        r = ref.get(t)
        if r is None or not r.is_finite():
            censor = k
            break
    usable = (censor - 1) if censor else len(targets)
    out["ref_censored_at"] = censor
    out["n_eval_steps"] = usable

    for tol in TOLS:
        for variant, states in (("first", first), ("last", last)):
            if variant == "last" and tol != PRIMARY:
                continue
            L = 0
            for t in targets[:usable]:
                if close(states.get(t), ref.get(t), tol):
                    L += 1
                else:
                    break
            out[f"ref_prefix_{variant}_{tol:g}"] = L
    Lr = out[f"ref_prefix_first_{PRIMARY:g}"]
    out["ref_first_mismatch_step"] = Lr + 1 if Lr < usable else None

    def own(v):
        return ref.get(v) if v <= 2 else first.get(v)

    F = 0
    etype = None
    for idx, (k, t, a, op, b) in enumerate(steps[:usable]):
        mv = first.get(t)
        if mv is None:
            later = any(tt in first for tt in targets[idx + 1:])
            etype = "skipped_step" if later else "trace_ends"
            break
        expected = apply(op, own(a), own(b))
        if locally_consistent(mv, expected):
            F += 1
            continue
        if op in "-/" and locally_consistent(mv, apply(op, own(b), own(a))):
            etype = "operand_order_swapped"   # e.g. executes S2 - S12 for "S13 = S12 - S2"
        elif any(locally_consistent(mv, apply(o, own(a), own(b))) or locally_consistent(mv, apply(o, own(b), own(a)))
                 for o in OPS if o != op):
            etype = "wrong_operation"
        else:
            cands = [1, 2] + list(range(max(3, t - 8), t))
            hit = any(locally_consistent(mv, apply(o, own(i), own(j)))
                      for i in cands for j in cands if (i, j) != (a, b) for o in OPS)
            etype = "wrong_operand" if hit else "arithmetic_error"
        break
    out["faithful_prefix"] = F
    # sensitivity: looser local tolerance (1% relative)
    F2 = 0
    for k, t, a, op, b in steps[:usable]:
        if close(first.get(t), apply(op, own(a), own(b)), 1e-2):
            F2 += 1
        else:
            break
    out["faithful_prefix_loose"] = F2
    out["first_error_step"] = F + 1 if F < usable else None
    if out["n_parsed_states"] == 0:
        etype = "no_intermediate_states"
    elif etype is None:
        etype = "all_faithful"
    out["error_type"] = etype
    return out


def ref_stats(ref: dict, targets) -> dict:
    """C5: statistics of the reference intermediate values S3..S_final."""
    mags, mins = [], []
    nonfinite = zero = 0
    first_nf = None
    for k, t in enumerate(targets, 1):
        r = ref.get(t)
        if r is None or not r.is_finite():
            nonfinite += 1
            if first_nf is None:
                first_nf = k
            continue
        if r == 0:
            zero += 1
            continue
        e = r.copy_abs().adjusted()  # floor(log10|r|)
        mags.append(e)
        mins.append(e)
    return {
        "ref_max_log10": max(mags) if mags else None,
        "ref_min_nonzero_log10": min(mins) if mins else None,
        "ref_n_nonfinite": nonfinite,
        "ref_first_nonfinite_step": first_nf,
        "ref_n_zero": zero,
    }
