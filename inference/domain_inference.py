"""
Problem 1 -- domain inference for Pec-King Order.

Three required belief computations (per spec):

  1. P(chicken_i beats chicken_j | ObservationHistory)
  2. P(chicken_i will be King in Coop_k | ObservationHistory)  -> KingBelief
  3. P(ability profile of chicken_j | ObservationHistory)       -> AbilityBelief

Inference uses variable elimination over the ability Bayes net
(prof said enumeration kills the machine).

Bayes net structure:
  Latent:   ability scores per chicken per coop
  Observed: win/loss outcomes for chickens present in that coop
  Structure: Ability_i -> BattleOutcome(i,j) <- Ability_j   (v-structure)

All functions accept raw state arrays so they work inside the simulation
loop without unpacking Python entity objects.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import chex


# ---------------------------------------------------------------------------
# 1. P(chicken_i beats chicken_j | AbilityBelief)
# ---------------------------------------------------------------------------

def p_i_beats_j(
    i: int,
    j: int,
    coop: int,
    ability_belief: chex.Array,   # float32, (M, N_coop, N_coop)
    N_coop: int,
) -> chex.Array:
    """P(i beats j in coop) by marginalising over ability beliefs.

    Variable elimination over a two-node factor graph:
      P(i beats j | obs) = sum_{a_i, a_j}
          P(i beats j | a_i, a_j) * P(a_i | obs) * P(a_j | obs)

    O(N_coop^2) per query.
    """
    p_ai = ability_belief[i, coop, :]   # (N_coop,)
    p_aj = ability_belief[j, coop, :]   # (N_coop,)

    vals = jnp.arange(1, N_coop + 1, dtype=jnp.int32)
    r = vals[:, None]
    c = vals[None, :]
    p_win_given_ab = jnp.where(
        r > c, jnp.float32(1.0),
        jnp.where(r < c, jnp.float32(0.0), jnp.float32(0.5)),
    )  # (N_coop, N_coop)

    joint = p_ai[:, None] * p_aj[None, :]
    return jnp.sum(p_win_given_ab * joint)


# ---------------------------------------------------------------------------
# 2. P(ability profile of chicken_j | ObservationHistory)
# ---------------------------------------------------------------------------

def p_ability_profile(
    j: int,
    ability_belief: chex.Array,   # float32, (M, N_coop, N_coop)
    N_coop: int,
) -> chex.Array:
    """Marginal ability belief for chicken j.

    Returns (N_coop, N_coop) where [c, v] = P(a_{j,c} == v+1 | obs).
    Under independence across coops, this IS the profile distribution.
    """
    return ability_belief[j]


# ---------------------------------------------------------------------------
# 3. Incremental ability belief update (called each step in step_env)
# ---------------------------------------------------------------------------

def incremental_ability_update(
    ability_belief: chex.Array,  # float32 (M, M, N_coop, N_coop)
    battle_pair: chex.Array,     # int32 (n_cages, N_coop, 2)
    cage_occupied: chex.Array,   # bool (n_cages, N_coop)
    battle_outcome: chex.Array,  # int8 (M, M, N_coop)
    observed_coop: chex.Array,   # int32 (M,) -- which coop each agent observed (-1 = none)
    N_coop: int,
    M: int,
) -> chex.Array:
    """Update ability beliefs from one round of battles.

    Iterates over actual cage battles only (at most M/2 per coop) instead of
    scanning all M^2 pairs. Vectorizes the Bayes update across all M observers
    at once, so the hot path is matrix ops not Python scalar math.

    For each occupied cage with pair (i, j) in coop c:
      - All observers who saw coop c get beliefs about i and j updated
      - Variable elimination over A_i -> Outcome <- A_j
    """
    # p_win[v, w] = P(i wins | a_i=v+1, a_j=w+1)
    vals = jnp.arange(1, N_coop + 1, dtype=jnp.int32)
    r, c = vals[:, None], vals[None, :]
    p_win = jnp.where(r > c, 1.0, jnp.where(r < c, 0.0, 0.5))
    p_lose = 1.0 - p_win

    n_cages = battle_pair.shape[0]
    new_belief = ability_belief

    for coop in range(N_coop):
        sees_coop = (observed_coop == coop)  # (M,) bool

        def cage_body(cage, belief):
            i = battle_pair[cage, coop, 0]
            j = battle_pair[cage, coop, 1]
            outcome = battle_outcome[i, j, coop]

            # +1 → p_win, -1 → p_lose; outcome=0 gated out by valid mask below
            p_lik = jnp.where(outcome > 0, p_win, p_lose)  # (N_coop, N_coop)

            prior_i = belief[:, i, coop, :]  # (M, N_coop)
            prior_j = belief[:, j, coop, :]  # (M, N_coop)

            # vectorized VE across all M observers
            # lik_i[m, v] = sum_w p_lik[v, w] * prior_j[m, w]
            lik_i = jnp.dot(prior_j, p_lik.T)  # (M, N_coop)
            lik_j = jnp.dot(prior_i, p_lik)     # (M, N_coop)

            # Add a small Laplace-smoothing floor before normalising so that
            # conflicting evidence never collapses the belief to all-zeros.
            # Without this, a wrong point-mass prior × a zero-likelihood column
            # yields a [0,0,0,0] posterior that can never recover (permanent collapse).
            post_i = prior_i * lik_i + jnp.float32(1e-8)
            post_i = post_i / post_i.sum(axis=1, keepdims=True)
            post_j = prior_j * lik_j + jnp.float32(1e-8)
            post_j = post_j / post_j.sum(axis=1, keepdims=True)

            # only update when cage occupied, outcome nonzero, observer saw coop
            valid = cage_occupied[cage, coop] & (outcome != 0)
            do_update = sees_coop[:, None] & valid  # (M, 1)

            belief = belief.at[:, i, coop, :].set(
                jnp.where(do_update, post_i, belief[:, i, coop, :])
            )
            belief = belief.at[:, j, coop, :].set(
                jnp.where(do_update, post_j, belief[:, j, coop, :])
            )
            return belief

        new_belief = jax.lax.fori_loop(0, n_cages, cage_body, new_belief)

    return new_belief


# ---------------------------------------------------------------------------
# 4. Derive king beliefs from ability beliefs (called at tournament close)
# ---------------------------------------------------------------------------

def derive_king_belief_from_ability(
    ability_belief: chex.Array,  # float32 (M, M, N_coop, N_coop)
    N_coop: int,
    M: int,
) -> chex.Array:
    """Compute per-observer king beliefs from ability beliefs via expected wins.

    For each observer m and coop c, computes expected wins for every chicken
    using einsum over the pairwise win probability matrix, then normalizes
    to a probability simplex.

    Returns float32 (M, M, N_coop, 2) -- [obs, subj, coop, 0/1]
    """
    vals = jnp.arange(1, N_coop + 1, dtype=jnp.int32)
    r, c = vals[:, None], vals[None, :]
    p_win_ab = jnp.where(r > c, 1.0, jnp.where(r < c, 0.0, 0.5))

    eye_mask = 1.0 - jnp.eye(M, dtype=jnp.float32)

    king_scores_list = []
    for coop in range(N_coop):
        bc = ability_belief[:, :, coop, :]  # (M_obs, M_subj, N_coop_vals)
        # batched matchup: matchups[obs, i, j] = P(i beats j | obs's beliefs)
        matchups = jnp.einsum('oia,ab,ojb->oij', bc, p_win_ab, bc)
        scores = (matchups * eye_mask[None, :, :]).sum(axis=2)
        king_scores_list.append(scores)

    king_probs = jnp.stack(king_scores_list, axis=2)  # (M_obs, M, N_coop)
    king_probs = king_probs / (king_probs.sum(axis=1, keepdims=True) + 1e-12)

    p_king = king_probs[:, :, :, None]
    p_not_king = 1.0 - p_king
    return jnp.concatenate([p_not_king, p_king], axis=-1).astype(jnp.float32)


# ---------------------------------------------------------------------------
# 5. Single-observer king belief helpers (used by run.py interactive mode)
#    Not JIT-compiled — correctness over vectorization.
# ---------------------------------------------------------------------------

def p_king_in_coop(
    coop: int,
    ability_belief: chex.Array,  # float32 (M, N_coop, N_coop) — one observer's beliefs
    N_coop: int,
    M: int,
) -> chex.Array:
    """P(chicken_i is King in coop | obs) for one observer.

    Expected wins normalized to a probability simplex over M chickens.
    Returns float32 (M,).
    """
    vals = jnp.arange(1, N_coop + 1, dtype=jnp.int32)
    r, c = vals[:, None], vals[None, :]
    p_win_ab = jnp.where(r > c, 1.0, jnp.where(r < c, 0.0, 0.5))
    bc = ability_belief[:, coop, :]                              # (M, N_coop)
    matchups = jnp.einsum('ia,ab,jb->ij', bc, p_win_ab, bc)    # (M, M)
    scores = (matchups * (1.0 - jnp.eye(M))).sum(axis=1)        # (M,)
    return scores / (scores.sum() + jnp.float32(1e-12))


def update_king_belief(
    ability_belief: chex.Array,  # float32 (M, N_coop, N_coop) — one observer
    N_coop: int,
    M: int,
) -> chex.Array:
    """Recompute king beliefs from ability beliefs for one observer.

    Returns float32 (M, N_coop, 2): [m, c, 1] = P(m is king in c).
    """
    return update_king_belief_vectorized(ability_belief, N_coop, M)


def update_king_belief_vectorized(
    ability_belief: chex.Array,  # float32 (M, N_coop, N_coop) — one observer
    N_coop: int,
    M: int,
) -> chex.Array:
    """Vectorized single-observer king belief via einsum.

    Returns float32 (M, N_coop, 2): [m, c, 1] = P(m is king in c).
    """
    vals = jnp.arange(1, N_coop + 1, dtype=jnp.int32)
    r, c = vals[:, None], vals[None, :]
    p_win_ab = jnp.where(r > c, 1.0, jnp.where(r < c, 0.0, 0.5))
    eye = 1.0 - jnp.eye(M, dtype=jnp.float32)

    king_scores = []
    for coop in range(N_coop):
        bc = ability_belief[:, coop, :]                              # (M, N_coop)
        matchups = jnp.einsum('ia,ab,jb->ij', bc, p_win_ab, bc)    # (M, M)
        scores = (matchups * eye).sum(axis=1)                        # (M,)
        king_scores.append(scores)

    king_probs = jnp.stack(king_scores, axis=1)  # (M, N_coop)
    # normalize per coop so values are a simplex over chickens
    king_probs = king_probs / (king_probs.sum(axis=0, keepdims=True) + jnp.float32(1e-12))

    p_king = king_probs[:, :, None]
    p_not_king = jnp.float32(1.0) - p_king
    return jnp.concatenate([p_not_king, p_king], axis=-1)  # (M, N_coop, 2)


def update_king_beliefs_all_agents(
    all_ability_beliefs: chex.Array,  # float32 (M, M, N_coop, N_coop)
    N_coop: int,
    M: int,
) -> chex.Array:
    """Per-observer king beliefs from per-observer ability beliefs.

    Thin wrapper around derive_king_belief_from_ability.
    Returns float32 (M, M, N_coop, 2).
    """
    return derive_king_belief_from_ability(all_ability_beliefs, N_coop, M)


# ---------------------------------------------------------------------------
# 6. Batch belief update from full battle history (post-sim, not JIT-compiled)
# ---------------------------------------------------------------------------

def update_ability_belief_from_history(
    battle_history: chex.Array,  # int32 (M, M, N_coop, T)
    N_coop: int,
    M: int,
    n_iters: int = 3,
    ability_pmf: "np.ndarray | None" = None,
) -> chex.Array:
    """Batch ability belief update from full battle history. Numpy iterative BP, not JIT.

    Returns float32 (M, N_coop, N_coop).
    """
    bh = np.array(battle_history)       # (M, M, N_coop, T)
    vals = np.arange(1, N_coop + 1)
    p_win = np.where(
        vals[:, None] > vals[None, :], 1.0,
        np.where(vals[:, None] < vals[None, :], 0.0, 0.5),
    )  # (N_coop, N_coop)

    if ability_pmf is not None:
        prior = np.array(ability_pmf, dtype=np.float64)
    else:
        prior = np.ones(N_coop, dtype=np.float64) / N_coop
    log_prior = np.log(np.clip(prior, 1e-30, None))

    # separate win / loss counts per (i, j, coop)
    wins_count = (bh > 0).sum(axis=-1).astype(np.float64)   # (M, M, N_coop)
    loss_count = (bh < 0).sum(axis=-1).astype(np.float64)   # (M, M, N_coop)

    belief = np.broadcast_to(prior, (M, N_coop, N_coop)).copy()
    eps = 0.01

    for _ in range(n_iters):
        new_belief = np.empty((M, N_coop, N_coop), dtype=np.float64)
        for coop in range(N_coop):
            for i in range(M):
                log_lik = np.zeros(N_coop, dtype=np.float64)
                for j in range(M):
                    if i == j:
                        continue
                    nw = wins_count[i, j, coop]
                    nl = loss_count[i, j, coop]
                    if nw == 0 and nl == 0:
                        continue
                    p_j = belief[j, coop, :]
                    lik_win  = p_win @ p_j
                    lik_lose = (1.0 - p_win) @ p_j
                    if nw > 0:
                        log_lik += nw * np.log(np.clip(lik_win,  1e-30, None))
                    if nl > 0:
                        log_lik += nl * np.log(np.clip(lik_lose, 1e-30, None))
                log_lik += log_prior
                log_lik -= log_lik.max()
                posterior = np.exp(log_lik)
                s = posterior.sum()
                new_belief[i, coop, :] = posterior / s if s > 1e-300 else prior
        belief = new_belief
        belief = (1.0 - eps) * belief + eps * prior[None, None, :]

    return jnp.array(belief, dtype=jnp.float32)


def update_ability_beliefs_all_agents(
    all_obs_wins: "np.ndarray",    # float64 (M, M, M, N_coop)
    all_obs_losses: "np.ndarray",  # float64 (M, M, M, N_coop)
    N_coop: int,
    M: int,
    n_iters: int = 3,
    ability_pmf: "np.ndarray | None" = None,
) -> chex.Array:
    """Per-observer ability belief update from per-observer win/loss counts.

    all_obs_wins[m, i, j, c]  = times observer m saw chicken i beat j in coop c.
    all_obs_losses[m, i, j, c] = times observer m saw chicken i lose to j in coop c.

    Vectorized log-likelihood across opponents j per iteration.
    Returns float32 (M, M, N_coop, N_coop).
    """
    vals = np.arange(1, N_coop + 1)
    p_win = np.where(
        vals[:, None] > vals[None, :], 1.0,
        np.where(vals[:, None] < vals[None, :], 0.0, 0.5),
    )  # (N_coop, N_coop)

    if ability_pmf is not None:
        prior = np.array(ability_pmf, dtype=np.float64)
    else:
        prior = np.ones(N_coop, dtype=np.float64) / N_coop
    log_prior = np.log(np.clip(prior, 1e-30, None))

    shared_belief = np.broadcast_to(prior, (M, N_coop, N_coop)).copy()
    all_beliefs   = np.broadcast_to(prior, (M, M, N_coop, N_coop)).copy()
    p_lose = 1.0 - p_win
    eps = 0.01

    for _ in range(n_iters):
        new_all = np.empty((M, M, N_coop, N_coop), dtype=np.float64)
        for coop in range(N_coop):
            sb_c = shared_belief[:, coop, :]         # (M, N_coop)
            lik_win_all  = p_win  @ sb_c.T           # (N_coop, M)
            lik_lose_all = p_lose @ sb_c.T           # (N_coop, M)
            log_lw = np.log(np.clip(lik_win_all,  1e-30, None))
            log_ll = np.log(np.clip(lik_lose_all, 1e-30, None))

            for i in range(M):
                NW = all_obs_wins[:, i, :, coop].copy()    # (M_obs, M_j)
                NL = all_obs_losses[:, i, :, coop].copy()  # (M_obs, M_j)
                NW[:, i] = 0.0
                NL[:, i] = 0.0
                # (M_obs, M_j) @ (M_j, N_coop) -> (M_obs, N_coop)
                log_lik = NW @ log_lw.T + NL @ log_ll.T
                log_lik += log_prior[None, :]
                log_lik -= log_lik.max(axis=1, keepdims=True)
                posterior = np.exp(log_lik)
                s = posterior.sum(axis=1, keepdims=True)
                good = (s > 1e-300).ravel()
                result = np.broadcast_to(prior, (M, N_coop)).copy()
                result[good] = posterior[good] / s[good]
                new_all[:, i, coop, :] = result

        all_beliefs   = new_all
        shared_belief = all_beliefs.mean(axis=0)
        shared_belief = (1.0 - eps) * shared_belief + eps * prior[None, None, :]

    return jnp.array(all_beliefs, dtype=jnp.float32)

