#!/usr/bin/env python3
"""
Track C Day 1 — scalar derivative checker.

Fill in your hand-derived f'(x) expressions in the DERIVATIVES dict below,
then run this script. It checks each against (a) sympy's symbolic answer
and (b) a numerical finite-difference approximation at three sample points.

Mismatches print the diff so you know where to redo the calculation.

Run:
    pip install sympy            # if you don't have it
    python check.py

No GPU, no torch needed.
"""

import math

try:
    import sympy as sp
except ImportError:
    raise SystemExit("Install sympy first:  pip install sympy")


# ----------------------------------------------------------------------
# YOUR WORK GOES HERE.
#
# For each item, fill the value with a sympy expression for f'(x) that
# you derived by hand. Use `x` as the symbol. Available helpers:
#   sp.exp, sp.log, sp.tanh, sp.Piecewise, sp.Max, sp.sin, sp.cos, ...
#
# Leave as None to mark "not yet done"; the checker will skip those.
# ----------------------------------------------------------------------

x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True, nonnegative=True)   # for items 9
xj = sp.IndexedBase("xj")                          # placeholder, item 10 uses scalar form


DERIVATIVES: dict[str, dict] = {
    "1. f(x) = x^3 - 5x + 7": {
        "f":  x**3 - 5*x + 7,
        # df/dx = ?
        "your_df": None,
    },
    "2. f(x) = (2x + 1)^4": {
        "f":  (2*x + 1)**4,
        "your_df": None,
    },
    "3. f(x) = e^(2x)": {
        "f":  sp.exp(2*x),
        "your_df": None,
    },
    "4. f(x) = ln(x^2 + 1)": {
        "f":  sp.log(x**2 + 1),
        "your_df": None,
    },
    "5. f(x) = x * ln(x)": {
        "f":  x * sp.log(x),
        "your_df": None,
        "domain_min": 1e-6,    # avoid log(0) in numeric check
    },
    "6. f(x) = sigmoid(x) = 1 / (1 + e^(-x))": {
        "f":  1 / (1 + sp.exp(-x)),
        # Hint: should factor to σ(x) * (1 - σ(x)).
        "your_df": None,
    },
    "7. f(x) = tanh(x)": {
        "f":  sp.tanh(x),
        # Hint: 1 - tanh^2(x).
        "your_df": None,
    },
    "8. f(x) = ReLU(x) = max(0, x)": {
        "f":  sp.Max(0, x),
        # Hint: piecewise; convention says df/dx(0) = 0 (or 1; either subgradient is valid).
        # Avoid x=0 in numerical check.
        "your_df": None,
        "skip_point_zero": True,
    },
    "9. f(p) = -y*ln(p) - (1-y)*ln(1-p)  (BCE w.r.t. p, y constant)": {
        # We rename the variable to p inside this entry. Use `p` below; the
        # checker swaps it for x at evaluation time. Treat y as a sympy symbol.
        "var_name": "p",
        "extra_const": {"y": 1},   # plug y=1 when running numerical check
        "f":  -y * sp.log(sp.Symbol("p", positive=True)) - (1 - y) * sp.log(1 - sp.Symbol("p", positive=True)),
        "your_df": None,
        "domain_min": 1e-4,
        "domain_max": 1 - 1e-4,
    },
    "10. f(x) = e^x / (e^x + C)  (softmax for one element, C = sum over others)": {
        # Simplified to scalar form by letting C be a constant (sum of other exp(x_j)).
        # Pick C = 2.0 for the numerical check.
        "f":  sp.exp(x) / (sp.exp(x) + 2),
        # Hint: should simplify to s * (1 - s) where s = softmax value.
        "your_df": None,
    },
}


# ----------------------------------------------------------------------
# Checker.  Don't edit below unless you know why.
# ----------------------------------------------------------------------

def numerical_derivative(f_func, x0: float, h: float = 1e-5) -> float:
    """Central finite difference."""
    return (f_func(x0 + h) - f_func(x0 - h)) / (2 * h)


def to_callable(expr, var_sym):
    """Lambdify a sympy expr to a Python float function."""
    return sp.lambdify(var_sym, expr, modules=["math"])


def check_one(label: str, entry: dict) -> tuple[str, str]:
    """
    Return (status_symbol, detail_string).
    status: '✓' all good, '·' skipped, '✗' wrong, '?' not yet attempted.
    """
    if entry["your_df"] is None:
        return "?", "(not yet attempted)"

    var_name = entry.get("var_name", "x")
    var_sym = sp.Symbol(var_name, real=True)

    f_expr = entry["f"]
    user_df = entry["your_df"]

    # Plug constants if any
    consts = entry.get("extra_const", {})
    for c_name, c_val in consts.items():
        f_expr = f_expr.subs(sp.Symbol(c_name), c_val)
        user_df = user_df.subs(sp.Symbol(c_name), c_val)

    truth_df = sp.diff(f_expr, var_sym)

    # Symbolic check (simplify the difference)
    diff = sp.simplify(user_df - truth_df)
    sym_ok = (diff == 0)

    # Numerical check at a few points
    f_func = to_callable(f_expr, var_sym)
    user_df_func = to_callable(user_df, var_sym)
    truth_df_func = to_callable(truth_df, var_sym)

    sample_xs = [-1.7, 0.3, 1.5]
    if entry.get("skip_point_zero"):
        sample_xs = [p for p in sample_xs if abs(p) > 0.1]
    dom_min = entry.get("domain_min")
    dom_max = entry.get("domain_max")
    if dom_min is not None:
        sample_xs = [p for p in sample_xs if p >= dom_min]
    if dom_max is not None:
        sample_xs = [p for p in sample_xs if p <= dom_max]
    if not sample_xs:
        sample_xs = [(dom_min or 0.1) + 0.01]

    num_diffs = []
    for x0 in sample_xs:
        num = numerical_derivative(f_func, x0)
        usr = user_df_func(x0)
        tru = truth_df_func(x0)
        num_diffs.append((x0, num, usr, tru))

    num_ok = all(abs(n - u) < 1e-3 for (_, n, u, _) in num_diffs)

    if sym_ok and num_ok:
        return "✓", f"OK   user = {sp.nsimplify(user_df, rational=False)}"
    elif sym_ok and not num_ok:
        return "✗", f"SYMBOLIC OK but NUMERIC FAIL — bug in checker. Show this to claude."
    elif num_ok and not sym_ok:
        return "≈", f"numerically OK, symbolic form different. truth = {truth_df}, you = {user_df}"
    else:
        sample_str = " | ".join(
            f"x={p:+.2f}: num={n:+.4f}, you={u:+.4f}, truth={t:+.4f}"
            for (p, n, u, t) in num_diffs
        )
        return "✗", f"WRONG. truth = {truth_df}\n        {sample_str}"


def main():
    print("=" * 78)
    print("Track C Day 1 — scalar derivatives")
    print("=" * 78)

    results = []
    for label, entry in DERIVATIVES.items():
        sym, detail = check_one(label, entry)
        results.append((sym, label, detail))

    for sym, label, detail in results:
        print(f"\n{sym}  {label}")
        print(f"   {detail}")

    print("\n" + "=" * 78)
    todo = sum(1 for s, _, _ in results if s == "?")
    ok   = sum(1 for s, _, _ in results if s == "✓")
    near = sum(1 for s, _, _ in results if s == "≈")
    bad  = sum(1 for s, _, _ in results if s == "✗")
    print(f"  ✓ {ok}   ≈ {near}   ✗ {bad}   ? {todo}")
    if todo == 0 and bad == 0:
        print("\n  All derivatives match.  Done for tonight.")
    elif bad > 0:
        print("\n  Fix the ✗ items, then re-run.")


if __name__ == "__main__":
    main()
