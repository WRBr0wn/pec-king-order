"""
close_tournament — detect end-of-tournament, assign kings, accumulate history.
new_tournament   — redistribute chickens, reset per-tournament counters.
tournament_is_done — predicate: no eligible pairs remain anywhere.

King assignment rule (per spec):
  1. Non-dominated chickens (0 losses in a region) -> receive crown.
  2. If none: chicken(s) with most wins -> receive crown.
  3. Ties (including cyclic dominance / "field") can produce multiple kings
     per coop — cannot be treated as a joint probability distribution.

King storage: state.king is a bool mask (M, N_coop) — True where a chicken
holds the crown.  Full king sets and one representative index per coop are
also returned in `info` for downstream use (run.py).

NewTournament placement (per spec):
  Crowned chickens go to the coop where they had the most cumulative battles
  (across all completed tournaments), with uniform random tiebreak.
  Uncrowned chickens are assigned uniformly at random.
  All loss_count, win_count, battle_pair, cage_occupied reset to zero/-1.
  cumulative_battles accumulates across tournaments and is never reset.
  BattleHistory continues to accumulate; abilities unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex

from envs.state import EnvState, EnvParams
from envs.spaces import ZONE_BATTLE, ZONE_SPECTATOR, ZONE_TRANSIT


# ---------------------------------------------------------------------------
# Predicate: is the tournament over?
# ---------------------------------------------------------------------------

def tournament_is_done(
    state: EnvState,
    params: EnvParams,
) -> chex.Array:
    """Return bool scalar — True when no eligible battle pairs remain.

    Eligible pair: two chickens in the same coop's BattleZone, neither
    exceeding k losses, who have not yet battled this tournament in that coop.

    Vectorized O(M^2 * N_coop) check using array operations — no Python
    loops over JAX scalars (per spec: "assessed at the beginning of a round").
    """
    M = params.M
    upper = jnp.triu(jnp.ones((M, M), dtype=jnp.bool_), k=1)  # (M, M)

    # eligible[m, c] = in BattleZone of c AND under loss limit
    eligible = (
        state.zone_array[:, :, ZONE_BATTLE].astype(jnp.bool_)  # (M, N_coop)
        & (state.loss_count < params.k)                          # (M, N_coop)
    )

    # pair_ok[i, j, c] = both eligible in c, haven't battled, i < j
    pair_ok = (
        eligible[:, None, :]   # (M, 1, N_coop)
        & eligible[None, :, :] # (1, M, N_coop)
        & ~state.has_battled   # (M, M, N_coop)
        & upper[:, :, None]    # (M, M, 1)
    )
    return ~pair_ok.any()


# ---------------------------------------------------------------------------
# King assignment helpers
# ---------------------------------------------------------------------------

def _kings_for_coop(
    coop: int,
    state: EnvState,
    params: EnvParams,
) -> chex.Array:
    """Return bool mask (M,) of kings for one coop.

    Rule:
      1. Chickens with 0 losses in this coop -> non-dominated -> kings.
      2. If none: chickens with max win_count in this coop -> kings.
    """
    losses = state.loss_count[:, coop]   # (M,)
    wins   = state.win_count[:, coop]    # (M,)

    participated   = (wins > 0) | (losses > 0)          # (M,) bool — at least one battle in this coop
    undefeated     = (losses == 0) & participated        # (M,) bool — never lost, but did compete

    # Use undefeated if any exist, else fall back to max-wins among participants
    any_undefeated = undefeated.any()

    # Exclude non-participants from max-wins fallback (treat as -1 so they can't win)
    wins_or_neg    = jnp.where(participated, wins, jnp.int32(-1))
    max_wins       = wins_or_neg.max()
    top_winners    = (wins == max_wins) & participated   # (M,) bool

    return jnp.where(any_undefeated, undefeated, top_winners)  # (M,) bool


# ---------------------------------------------------------------------------
# close_tournament
# ---------------------------------------------------------------------------

def close_tournament(
    state: EnvState,
    params: EnvParams,
) -> tuple[EnvState, dict]:
    """Assign kings for all coops and record them in state.king.

    battle_history is accumulated outside the JIT loop in run.py.

    Returns
    -------
    (updated_state, info)
    info["king_masks"] : bool, shape (N_coop, M) — full king sets per coop.
    info["king_repr"]  : int32, shape (N_coop,)  — one representative king index.
    """
    M, N_coop = params.M, params.N_coop

    masks = [_kings_for_coop(coop, state, params) for coop in range(N_coop)]
    king_masks_arr = jnp.stack(masks).astype(jnp.bool_)                     # (N_coop, M)
    king_repr_arr  = jnp.stack(
        [jnp.argmax(m).astype(jnp.int32) for m in masks]
    )                                                                         # (N_coop,) int32

    # Accumulate total battles this tournament into the never-reset counter
    new_cumulative = state.cumulative_battles + state.win_count + state.loss_count

    new_state = state.replace(
        king=king_masks_arr.T,  # (M, N_coop) bool — all crown-holders this tournament
        cumulative_battles=new_cumulative,
        tournament_count=state.tournament_count + jnp.int32(1),
        tournament_step=jnp.int32(0),
    )

    # Deposit crowns into sacks: one crown per coop where chicken is king
    crowns_earned = new_state.king.sum(axis=1).astype(jnp.int32)  # (M,)
    new_state = new_state.replace(sack=new_state.sack + crowns_earned)

    info = {
        "king_masks": king_masks_arr,
        "king_repr":  king_repr_arr,
    }
    return new_state, info


# ---------------------------------------------------------------------------
# new_tournament
# ---------------------------------------------------------------------------

def new_tournament(
    key: chex.PRNGKey,
    state: EnvState,
    params: EnvParams,
    king_masks: chex.Array,  # bool, (N_coop, M) — used for crowned placement
) -> EnvState:
    """Redistribute chickens and reset per-tournament counters.

    Placement (per spec):
      - Crowned chickens → coop with most cumulative battles (random tiebreak).
      - Uncrowned chickens → uniform random coop.

    Resets: loss_count, win_count, battle_pair, cage_occupied, battle_outcome.
    Does NOT reset: abilities, tournament_count, cumulative_battles.
    """
    M, N_coop = params.M, params.N_coop
    t = state.tournament_count  # already incremented by close_tournament

    # --- Crowned chickens: go to coop with most cumulative battles ---
    # state.king is (M, N_coop) bool; any crown at all makes a chicken "crowned"
    is_crowned = state.king.any(axis=1)  # (M,) bool

    # Tiebreak: add small uniform noise so argmax randomizes among equal coops
    key, subkey = jax.random.split(key)
    noise = jax.random.uniform(subkey, shape=(M, N_coop))
    best_coop = jnp.argmax(
        state.cumulative_battles.astype(jnp.float32) + noise, axis=1
    ).astype(jnp.int32)  # (M,) — most-battled coop per chicken

    # --- Uncrowned chickens: uniform random coop ---
    key, subkey = jax.random.split(key)
    rand_coop = jax.random.randint(subkey, shape=(M,), minval=0, maxval=N_coop)

    # Combine: crowned → best_coop, uncrowned → rand_coop
    dest_coop = jnp.where(is_crowned, best_coop, rand_coop)

    # --- Build new zone_array: all assigned to BattleZone of dest_coop ---
    new_zone = jnp.zeros((M, N_coop, 3), dtype=jnp.int32)
    # For each chicken m, set zone_array[m, dest_coop[m], ZONE_BATTLE] = 1
    new_zone = new_zone.at[
        jnp.arange(M),
        dest_coop,
        ZONE_BATTLE,
    ].set(jnp.int32(1))

    # --- Reset per-tournament counters ---
    new_loss_count    = jnp.zeros((M, N_coop), dtype=jnp.int32)
    new_win_count     = jnp.zeros((M, N_coop), dtype=jnp.int32)
    new_battle_pair   = jnp.full((M // 2, N_coop, 2), fill_value=-1, dtype=jnp.int32)
    new_cage_occupied = jnp.zeros((M // 2, N_coop), dtype=jnp.bool_)
    new_battle_outcome = jnp.zeros((M, M, N_coop), dtype=jnp.int8)
    new_has_battled    = jnp.zeros((M, M, N_coop), dtype=jnp.bool_)
    new_king          = jnp.zeros((M, N_coop), dtype=jnp.bool_)

    # Check if all tournaments are done
    done = (t >= jnp.int32(params.T)).astype(jnp.bool_)

    return state.replace(
        zone_array=new_zone,
        loss_count=new_loss_count,
        win_count=new_win_count,
        # cumulative_battles intentionally NOT reset — accumulates across all tournaments
        battle_pair=new_battle_pair,
        cage_occupied=new_cage_occupied,
        battle_outcome=new_battle_outcome,
        has_battled=new_has_battled,
        king=new_king,
        action_array=jnp.zeros((M, 1), dtype=jnp.int32),
        done=done,
    )
