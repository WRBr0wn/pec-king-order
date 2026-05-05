"""
Problem 3 — Inference by Enumeration (Batcave network).

DAG:  S → V,  S → C → A

Variables:
  S ∈ {1,2,3,4,5}  — seismic reading (prior uniform: P(S=i) = 0.2)
  V ∈ {1,2,3,4,5}  — visual reading
  C ∈ {T,F}         — crater visible
  A ∈ {T,F}         — alarm triggered

CPTs (from spec):

  P(V | S):
    S ∈ {2,3,4}: P(V=S|S) = 0.8;  P(V=S-1|S) = 0.1;  P(V=S+1|S) = 0.1
    S = 1:       P(V=1|S=1) = 0.9; P(V=2|S=1) = 0.1
    S = 5:       P(V=5|S=5) = 0.9; P(V=4|S=5) = 0.1

  P(C | S):
    P(C=T | S >= 4) = 0.98
    P(C=T | S <  4) = 0.02

  P(A | C):
    P(A=T | C=T) = 0.999
    P(A=T | C=F) = 0.0

Query: P(S=5 | A=T)

Formula (from spec):
  P(S=5 | A=T) ∝ P(S=5) · Σ_V Σ_C P(V|S=5) · P(C|S=5) · P(A=T|C)
  Normalise by Σ_{s=1}^{5} P(S=s) · Σ_V Σ_C P(V|S=s) · P(C|S=s) · P(A=T|C)

V is a leaf unconnected to A, so Σ_V P(V|S) = 1 for any S.
This means V drops out of the query and the formula reduces to:
  P(S=5 | A=T) ∝ P(S=5) · Σ_C P(C|S=5) · P(A=T|C)
The full enumeration over V is retained below for completeness and to
match the spec formula exactly.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# CPT definitions
# ---------------------------------------------------------------------------

S_VALUES = [1, 2, 3, 4, 5]
V_VALUES = [1, 2, 3, 4, 5]
# C: 0=False, 1=True
# A: 0=False, 1=True

P_S = {s: 0.2 for s in S_VALUES}   # uniform prior


def p_v_given_s(v: int, s: int) -> float:
    """P(V=v | S=s)."""
    if s == 1:
        if v == 1: return 0.9
        if v == 2: return 0.1
        return 0.0
    if s == 5:
        if v == 5: return 0.9
        if v == 4: return 0.1
        return 0.0
    # s in {2, 3, 4}
    if v == s:     return 0.8
    if v == s - 1: return 0.1
    if v == s + 1: return 0.1
    return 0.0


def p_c_given_s(c: int, s: int) -> float:
    """P(C=c | S=s).  c: 1=True, 0=False."""
    p_true = 0.98 if s >= 4 else 0.02
    return p_true if c == 1 else (1.0 - p_true)


def p_a_given_c(a: int, c: int) -> float:
    """P(A=a | C=c).  a,c: 1=True, 0=False."""
    p_true = 0.999 if c == 1 else 0.0
    return p_true if a == 1 else (1.0 - p_true)


# ---------------------------------------------------------------------------
# Joint factor for one S value (summing out V and C)
# ---------------------------------------------------------------------------

def _joint_s_alarm(s: int) -> float:
    """P(S=s) · Σ_V Σ_C P(V|S=s) · P(C|S=s) · P(A=T|C).

    Enumerates all V in {1..5} and C in {T,F} explicitly, matching the
    spec formula.
    """
    total = 0.0
    for v in V_VALUES:
        for c in [0, 1]:
            total += (
                p_v_given_s(v, s)
                * p_c_given_s(c, s)
                * p_a_given_c(1, c)   # A=T
            )
    return P_S[s] * total


# ---------------------------------------------------------------------------
# Query: P(S=5 | A=T)
# ---------------------------------------------------------------------------

def p_s5_given_alarm() -> float:
    """P(S=5 | A=T) by full enumeration over V and C."""
    numerator   = _joint_s_alarm(5)
    denominator = sum(_joint_s_alarm(s) for s in S_VALUES)
    return numerator / denominator


def full_posterior_s_given_alarm() -> dict[int, float]:
    """P(S=s | A=T) for all s ∈ {1..5}. Returns dict s → float."""
    joints      = {s: _joint_s_alarm(s) for s in S_VALUES}
    denominator = sum(joints.values())
    return {s: v / denominator for s, v in joints.items()}


# ---------------------------------------------------------------------------
# Print-friendly summary (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    posterior = full_posterior_s_given_alarm()
    print("P(S | A=T) by enumeration:")
    for s, p in sorted(posterior.items()):
        marker = " <-- query" if s == 5 else ""
        print(f"  P(S={s} | A=T) = {p:.6f}{marker}")
    print(f"\nP(S=5 | A=T) = {p_s5_given_alarm():.6f}")
