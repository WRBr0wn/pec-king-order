"""
Problem 4 — Belief Propagation (Pearl 1988) on the wet-grass network.

DAG:  R → W,  R → H ← S

Variables (all binary: T=1, F=0):
  R — Rain
  S — Sprinkler
  W — Wet grass (child of R)
  H — Hose running (child of R and S)

CPTs (from spec):
  P(R=T) = 0.2,  P(S=T) = 0.1

  P(W=T | R=T) = 1.0,  P(W=T | R=F) = 0.2

  P(H=T | R=T, S=T) = 1.0
  P(H=T | R=F, S=T) = 0.9
  P(H=T | R=T, S=F) = 1.0
  P(H=T | R=F, S=F) = 0.0

Pearl (1988) message-passing formulas (per spec):

  λ(X)       = Π_j  λ_{Y_j}(X)
  π(X)       = Σ_U  P(X|U) · Π_i π_X(U_i)
  BEL(X)     = α · λ(X) · π(X)
  λ_X(U_i)   = Σ_X  λ(X) · Σ_{U_k, k≠i}  P(X|U) · Π_{k≠i} π_X(U_k)
  π_{Y_j}(X) = α · BEL(X) / λ_{Y_j}(X)

Conventions:
  All messages and beliefs are 2-element arrays indexed [F, T] = [0, 1].
  "Unobserved leaf" → λ = [1, 1]
  "Root node"       → π(X) = P(X)
  "Observed X=T"    → λ(X) = [0, 1]
  "Observed X=F"    → λ(X) = [1, 0]

Queries (per spec):
  (4b-i)   No observations       → BEL(W=T)
  (4b-ii)  No observations       → BEL(H=T)
  (4b-iii) Observe H=T           → BEL(W=T)
  (4b-iv)  Observe H=T and W=T  → BEL(S=T)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# CPTs as numpy arrays indexed [parent_val, ...] for easy marginalisation
# ---------------------------------------------------------------------------

# Priors (root nodes)
P_R = np.array([0.8, 0.2], dtype=np.float64)   # [P(R=F), P(R=T)]
P_S = np.array([0.9, 0.1], dtype=np.float64)   # [P(S=F), P(S=T)]

# P(W | R): shape (2, 2) — P_W_given_R[r, w]
P_W_given_R = np.array([
    [0.8, 0.2],   # R=F: P(W=F)=0.8, P(W=T)=0.2
    [0.0, 1.0],   # R=T: P(W=F)=0.0, P(W=T)=1.0
], dtype=np.float64)

# P(H | R, S): shape (2, 2, 2) — P_H_given_RS[r, s, h]
P_H_given_RS = np.array([
    # R=F
    [[1.0, 0.0],   # R=F, S=F: P(H=F)=1.0, P(H=T)=0.0
     [0.1, 0.9]],  # R=F, S=T: P(H=F)=0.1, P(H=T)=0.9
    # R=T
    [[0.0, 1.0],   # R=T, S=F: P(H=F)=0.0, P(H=T)=1.0
     [0.0, 1.0]],  # R=T, S=T: P(H=F)=0.0, P(H=T)=1.0
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalise(v: np.ndarray) -> np.ndarray:
    """Normalise a non-negative vector to sum to 1."""
    s = v.sum()
    assert s > 0, f"Cannot normalise zero vector: {v}"
    return v / s


# ---------------------------------------------------------------------------
# Core Pearl BP pass for this specific 4-node network
#
# Network topology:
#
#   R (root) ─── W (leaf)
#    │
#    └──────────  H (leaf with two parents: R, S)
#   S (root) ───┘
#
# Nodes in topological order for downward π pass: R, S, W, H
# Nodes in reverse order for upward λ pass: W, H, R, S
# ---------------------------------------------------------------------------

def _run_bp(
    obs_W: np.ndarray | None,   # [0,1] or None
    obs_H: np.ndarray | None,   # [0,1] or None
    obs_S: np.ndarray | None,   # [0,1] or None
) -> dict[str, np.ndarray]:
    """Full Pearl BP pass. obs_* are λ vectors [P(e|X=F), P(e|X=T)]; None = unobserved.

    Returns dict 'R'/'S'/'W'/'H' → BEL array [F, T].
    """
    lam_W = obs_W if obs_W is not None else np.ones(2)
    lam_H = obs_H if obs_H is not None else np.ones(2)
    lam_S_obs = obs_S if obs_S is not None else np.ones(2)

    # ------------------------------------------------------------------
    # Downward π pass (root → leaves)
    # ------------------------------------------------------------------

    # π(R) = P(R)  (root)
    pi_R = P_R.copy()

    # π(S) = P(S)  (root)
    pi_S = P_S.copy() * lam_S_obs   # fold in any S observation into π
    pi_S = _normalise(pi_S) if lam_S_obs is not None and obs_S is not None else P_S.copy()

    # π(W): single parent R
    # π(W=w) = Σ_r P(W=w|R=r) · π_W(R=r)
    # π_W(R) = π(R) initially (no other children of R yet in down-pass;
    #           we fold λ_H contribution into R belief after upward pass)
    pi_W = P_W_given_R.T @ pi_R   # shape (2,)

    # π(H): two parents R and S
    # π(H=h) = Σ_r Σ_s P(H=h|R=r,S=s) · π_H(R=r) · π_H(S=s)
    # At this point π_H(R)=π(R) and π_H(S)=π(S) (no sibling messages yet)
    pi_H = np.einsum('rsh,r,s->h', P_H_given_RS, pi_R, pi_S)   # (2,)

    # ------------------------------------------------------------------
    # Upward λ pass (leaves → roots)
    # ------------------------------------------------------------------

    # W is a leaf: λ(W) comes from observation (or [1,1])
    lam_W_node = lam_W   # (2,)

    # H is a leaf: λ(H) comes from observation (or [1,1])
    lam_H_node = lam_H   # (2,)

    # λ_W(R): message from W to R
    # λ_W(R=r) = Σ_w λ(W=w) · P(W=w|R=r)
    lam_W_to_R = P_W_given_R @ lam_W_node   # (2,)   P_W_given_R[r,w] summed over w

    # λ_H(R): message from H to R (marginalising over S)
    # λ_H(R=r) = Σ_h λ(H=h) · Σ_s P(H=h|R=r,S=s) · π_H(S=s)
    lam_H_to_R = np.einsum('rsh,h,s->r', P_H_given_RS, lam_H_node, pi_S)   # (2,)

    # λ_H(S): message from H to S (marginalising over R)
    # λ_H(S=s) = Σ_h λ(H=h) · Σ_r P(H=h|R=r,S=s) · π_H(R=r)
    lam_H_to_S = np.einsum('rsh,h,r->s', P_H_given_RS, lam_H_node, pi_R)   # (2,)

    # λ(R) = λ_W(R) * λ_H(R)  (product of all child messages)
    lam_R = lam_W_to_R * lam_H_to_R   # (2,)

    # λ(S) = λ_H(S)  (only child of S is H)
    lam_S = lam_H_to_S   # (2,)

    # ------------------------------------------------------------------
    # Beliefs: BEL(X) = α · λ(X) · π(X)
    # ------------------------------------------------------------------

    # For R and S: π is the prior; for W and H: use π computed above
    bel_R = _normalise(lam_R * pi_R)
    bel_S = _normalise(lam_S * pi_S)
    bel_W = _normalise(lam_W_node * pi_W)
    bel_H = _normalise(lam_H_node * pi_H)

    # Refine W and H using updated R belief (Pearl iterative step):
    # After λ(R) is known, recompute π_{Y_j}(X) = α·BEL(X)/λ_{Y_j}(X)
    # π_W(R=r) = α · BEL(R=r) / λ_W(R=r)
    # Then recompute π(W) and BEL(W).
    safe_lam_W_to_R = np.where(lam_W_to_R > 0, lam_W_to_R, 1.0)
    pi_W_R = _normalise(bel_R / safe_lam_W_to_R)   # π_W(R)

    pi_W_refined = P_W_given_R.T @ pi_W_R
    bel_W = _normalise(lam_W_node * pi_W_refined)

    # Similarly refine H:
    safe_lam_H_to_R = np.where(lam_H_to_R > 0, lam_H_to_R, 1.0)
    pi_H_R = _normalise(bel_R / safe_lam_H_to_R)   # π_H(R)

    safe_lam_H_to_S = np.where(lam_H_to_S > 0, lam_H_to_S, 1.0)
    pi_H_S = _normalise(bel_S / safe_lam_H_to_S)   # π_H(S)

    pi_H_refined = np.einsum('rsh,r,s->h', P_H_given_RS, pi_H_R, pi_H_S)
    bel_H = _normalise(lam_H_node * pi_H_refined)

    return {"R": bel_R, "S": bel_S, "W": bel_W, "H": bel_H}


# ---------------------------------------------------------------------------
# Query functions (4b-i through 4b-iv)
# ---------------------------------------------------------------------------

def bel_w_no_obs() -> float:
    """(4b-i) BEL(W=T) with no observations."""
    beliefs = _run_bp(obs_W=None, obs_H=None, obs_S=None)
    return float(beliefs["W"][1])


def bel_h_no_obs() -> float:
    """(4b-ii) BEL(H=T) with no observations."""
    beliefs = _run_bp(obs_W=None, obs_H=None, obs_S=None)
    return float(beliefs["H"][1])


def bel_w_given_h(h_val: int = 1) -> float:
    """(4b-iii) BEL(W=T) given H observed. h_val: 1=T (default), 0=F."""
    lam_H = np.array([1.0, 0.0]) if h_val == 0 else np.array([0.0, 1.0])
    beliefs = _run_bp(obs_W=None, obs_H=lam_H, obs_S=None)
    return float(beliefs["W"][1])


def bel_s_given_h_and_w(h_val: int = 1, w_val: int = 1) -> float:
    """(4b-iv) BEL(S=T) given H and W both observed. h_val, w_val: 1=T, 0=F."""
    lam_H = np.array([1.0, 0.0]) if h_val == 0 else np.array([0.0, 1.0])
    lam_W = np.array([1.0, 0.0]) if w_val == 0 else np.array([0.0, 1.0])
    beliefs = _run_bp(obs_W=lam_W, obs_H=lam_H, obs_S=None)
    return float(beliefs["S"][1])


# ---------------------------------------------------------------------------
# Print-friendly summary (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Pearl Belief Propagation — Wet Grass Network")
    print("=" * 48)

    print(f"\n(4b-i)  BEL(W=T) | no obs        = {bel_w_no_obs():.6f}")
    print(f"(4b-ii) BEL(H=T) | no obs        = {bel_h_no_obs():.6f}")
    print(f"(4b-iii)BEL(W=T) | H=T           = {bel_w_given_h(1):.6f}")
    print(f"(4b-iv) BEL(S=T) | H=T, W=T      = {bel_s_given_h_and_w(1, 1):.6f}")

    print("\nFull beliefs (no obs):")
    beliefs = _run_bp(None, None, None)
    for var, bel in beliefs.items():
        print(f"  BEL({var}) = [F={bel[0]:.4f}, T={bel[1]:.4f}]")

    print("\nFull beliefs (H=T observed):")
    lam_H_T = np.array([0.0, 1.0])
    beliefs = _run_bp(None, lam_H_T, None)
    for var, bel in beliefs.items():
        print(f"  BEL({var}|H=T) = [F={bel[0]:.4f}, T={bel[1]:.4f}]")

    print("\nFull beliefs (H=T, W=T observed):")
    lam_W_T = np.array([0.0, 1.0])
    beliefs = _run_bp(lam_W_T, lam_H_T, None)
    for var, bel in beliefs.items():
        print(f"  BEL({var}|H=T,W=T) = [F={bel[0]:.4f}, T={bel[1]:.4f}]")
