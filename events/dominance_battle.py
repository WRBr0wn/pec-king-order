"""
dominance_battle — execute all caged battles simultaneously.
battle_outcome   — single-pair outcome function (per spec).

Battle rule (EXACT from spec):
  - Higher ability wins.
  - Equal ability: fair coin flip, p = 0.5 each.
  - Returns +1 if left chicken wins, -1 if left chicken loses.

Updates per round:
  state.battle_outcome[i, j, coop]  =  +1 (i beat j) or -1 (i lost to j)
  state.battle_outcome[j, i, coop]  = -outcome[i,j,coop]   (antisymmetric)
  state.loss_count[loser, coop]     += 1
  state.win_count[winner, coop]     += 1
JAX note: all per-cage operations are expressed with jnp scatter (.at[].set/add)
and jax.lax.cond for the tie-break branch — no Python if inside jit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex

from envs.state import EnvState, EnvParams


# ---------------------------------------------------------------------------
# Core outcome function (per spec)
# ---------------------------------------------------------------------------

def battle_outcome(
    ability_L: chex.Array,   # int32 scalar — ability score of left chicken
    ability_R: chex.Array,   # int32 scalar — ability score of right chicken
    key: chex.PRNGKey,
) -> chex.Array:
    """Return +1 if left wins, -1 if left loses.

    Rule (EXACT from spec):
      ability_L > ability_R  →  +1
      ability_L < ability_R  →  -1
      ability_L == ability_R →  fair coin flip (p=0.5 each)
    """
    coin = jax.random.bernoulli(key, p=0.5)          # True ~50%
    tie_result = jnp.where(coin, jnp.int32(1), jnp.int32(-1))

    return jnp.where(
        ability_L > ability_R, jnp.int32(1),
        jnp.where(ability_L < ability_R, jnp.int32(-1), tie_result),
    )


# ---------------------------------------------------------------------------
# Single-cage battle (operates on one (i, j, coop) triple)
# ---------------------------------------------------------------------------

def _battle_one_cage(
    i: chex.Array,            # int32 scalar — left chicken index
    j: chex.Array,            # int32 scalar — right chicken index
    coop: int,                # static Python int (loop variable)
    abilities: chex.Array,    # int32, (M, N_coop)
    key: chex.PRNGKey,
) -> chex.Array:
    """Return int8 scalar outcome (+1/-1) for one cage battle in given coop.

    Uses jax.lax.cond to stay traceable under jit even though i/j are dynamic.
    """
    a_L = abilities[i, coop]
    a_R = abilities[j, coop]
    return battle_outcome(a_L, a_R, key).astype(jnp.int8)


# ---------------------------------------------------------------------------
# Main event: dominance_battle
# ---------------------------------------------------------------------------

def dominance_battle(
    key: chex.PRNGKey,
    state: EnvState,
    params: EnvParams,
) -> EnvState:
    """Execute all occupied cage battles across all coops in one vectorized pass.

    Replaces the original 128-iteration Python loop with array ops over
    (n_cages, N_coop) — all battles in all coops at once.
    """
    M, N_coop = params.M, params.N_coop
    n_cages = M // 2

    # --- Extract all cage pairs at once ---
    left  = state.battle_pair[:, :, 0]   # (n_cages, N_coop) int32
    right = state.battle_pair[:, :, 1]   # (n_cages, N_coop) int32
    occ   = state.cage_occupied          # (n_cages, N_coop) bool

    # Clamp -1 (unoccupied) to 0 for safe indexing; outcomes masked by occ later
    safe_left  = jnp.clip(left, 0, M - 1)
    safe_right = jnp.clip(right, 0, M - 1)

    # --- Gather abilities for all pairs ---
    coop_idx = jnp.broadcast_to(jnp.arange(N_coop)[None, :], (n_cages, N_coop))
    a_L = state.abilities[safe_left, coop_idx]   # (n_cages, N_coop)
    a_R = state.abilities[safe_right, coop_idx]  # (n_cages, N_coop)

    # --- Vectorized battle outcomes ---
    key, subkey = jax.random.split(key)
    coins = jax.random.bernoulli(subkey, p=0.5, shape=(n_cages, N_coop))
    tie_result = jnp.where(coins, jnp.int8(1), jnp.int8(-1))

    outcomes = jnp.where(
        a_L > a_R, jnp.int8(1),
        jnp.where(a_L < a_R, jnp.int8(-1), tie_result),
    )  # (n_cages, N_coop)
    outcomes = jnp.where(occ, outcomes, jnp.int8(0))

    # --- Scatter into (M, M, N_coop) battle_outcome ---
    flat_left  = safe_left.ravel()    # (n_cages * N_coop,)
    flat_right = safe_right.ravel()   # (n_cages * N_coop,)
    flat_coop  = coop_idx.ravel()     # (n_cages * N_coop,)
    flat_out   = outcomes.ravel()     # (n_cages * N_coop,) int8
    flat_occ   = occ.ravel()          # (n_cages * N_coop,) bool

    bo = jnp.zeros((M, M, N_coop), dtype=jnp.int8)
    bo = bo.at[flat_left, flat_right, flat_coop].add(flat_out)
    bo = bo.at[flat_right, flat_left, flat_coop].add(-flat_out)

    # --- Update loss_count / win_count ---
    i_wins = flat_occ & (flat_out == 1)
    j_wins = flat_occ & (flat_out == -1)

    new_win_count  = state.win_count
    new_loss_count = state.loss_count

    new_win_count = new_win_count.at[flat_left, flat_coop].add(
        jnp.where(i_wins, jnp.int32(1), jnp.int32(0))
    )
    new_loss_count = new_loss_count.at[flat_right, flat_coop].add(
        jnp.where(i_wins, jnp.int32(1), jnp.int32(0))
    )
    new_win_count = new_win_count.at[flat_right, flat_coop].add(
        jnp.where(j_wins, jnp.int32(1), jnp.int32(0))
    )
    new_loss_count = new_loss_count.at[flat_left, flat_coop].add(
        jnp.where(j_wins, jnp.int32(1), jnp.int32(0))
    )

    # --- Mark pairs as having battled (bool mask, never cancels out) ---
    new_has_battled = state.has_battled
    # For each occupied cage, mark (i,j) and (j,i) as battled in that coop
    battled_this_round = (bo != 0)  # (M, M, N_coop) bool
    new_has_battled = new_has_battled | battled_this_round

    return state.replace(
        battle_outcome=bo,
        loss_count=new_loss_count,
        win_count=new_win_count,
        has_battled=new_has_battled,
    )
