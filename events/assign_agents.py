"""
assign_agents_to_cage  — pair eligible BattleZone chickens into cages.
relocate_zone          — move unpaired BattleZone chickens to SpectatorZone.

Preconditions for cage assignment (per spec):
  1. Chicken must be in the coop's BattleZone (zone_array[m, c, 0] == 1).
  2. The pair must NOT have battled in this coop this tournament
     (has_battled[i, j, coop] == False).
  3. Both chickens must have < k losses in this coop this tournament
     (loss_count[m, c] < k).
  Pairing order: fewest losses first; random tiebreak within same loss count.

Odd chicken out (or any unpaired eligible chicken) -> relocate_zone.
Chickens already in SpectatorZone are unaffected by both events.

All JAX hot-path logic uses jax.lax primitives (no Python if/for).
Python-level loops only appear over the static N_coop dimension.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex

from envs.state import EnvState, EnvParams
from envs.spaces import ZONE_BATTLE, ZONE_SPECTATOR, ZONE_TRANSIT


# ---------------------------------------------------------------------------
# Eligibility helpers
# ---------------------------------------------------------------------------

def _eligible_mask(
    state: EnvState,
    params: EnvParams,
    coop: int,
) -> chex.Array:
    """Return bool mask of shape (M,) — True if chicken is eligible to battle in coop.

    Eligibility:
      - currently in BattleZone of this coop
      - loss_count[m, coop] < k
    The "not battled this pair this tournament" constraint is enforced
    during pairing (checked against has_battled), not here.
    """
    M = params.M
    in_battle_zone = state.zone_array[:, coop, ZONE_BATTLE].astype(jnp.bool_)  # (M,)
    under_loss_limit = state.loss_count[:, coop] < params.k                    # (M,)
    return in_battle_zone & under_loss_limit  # (M,)


def _already_battled(
    state: EnvState,
    coop: int,
    i: chex.Array,
    j: chex.Array,
) -> chex.Array:
    """True if chickens i and j have already battled in this coop this tournament.

    Uses the has_battled bool mask which can't cancel out like net win/loss can.
    """
    return state.has_battled[i, j, coop]


# ---------------------------------------------------------------------------
# Sorting eligible chickens by loss count (fewest first, random tiebreak)
# ---------------------------------------------------------------------------

def _sort_by_losses(
    eligible: chex.Array,   # bool, (M,)
    loss_count_coop: chex.Array,  # int32, (M,)  loss_count[:, coop]
    key: chex.PRNGKey,
) -> chex.Array:
    """Sort eligible chickens fewest-losses-first with uniform random tiebreak.

    Ineligible chickens get sort key 1e9 so they land at the back.
    Returns int32 (M,) — all M indices.
    """
    M = eligible.shape[0]
    noise = jax.random.uniform(key, shape=(M,), minval=0.0, maxval=1.0)
    # Ineligible chickens get key = k+1 (guaranteed last)
    sort_key = jnp.where(
        eligible,
        loss_count_coop.astype(jnp.float32) + noise,
        jnp.full(M, fill_value=1e9, dtype=jnp.float32),
    )
    return jnp.argsort(sort_key)  # ascending


# ---------------------------------------------------------------------------
# Greedy pairing via fori_loop (pure JAX, JIT-safe)
# ---------------------------------------------------------------------------

def _greedy_pair(
    sorted_indices: chex.Array,   # int32, (M,)  eligible chickens in order
    eligible: chex.Array,         # bool,  (M,)
    has_battled_coop: chex.Array, # bool, (M, M) — True if pair fought this tournament
    M: int,
) -> chex.Array:
    """Greedy pairing: for each eligible chicken in sorted order, find the first
    unpaired eligible opponent they haven't fought this tournament.

    fori_loop so it compiles under jit. Returns int32 (M//2, 2); [-1,-1] = empty.
    """
    n_cages = M // 2
    # inverse permutation: position_of[agent_id] = position in sorted order
    position_of = jnp.zeros(M, dtype=jnp.int32).at[sorted_indices].set(
        jnp.arange(M, dtype=jnp.int32)
    )
    agent_ids = jnp.arange(M, dtype=jnp.int32)

    init = (
        jnp.zeros(M, dtype=jnp.bool_),                # used
        jnp.full((n_cages, 2), -1, dtype=jnp.int32),  # pairs
        jnp.int32(0),                                   # pair_idx
    )

    def body(pos, carry):
        used, pairs, pair_idx = carry
        ci = sorted_indices[pos]
        skip = ~eligible[ci] | used[ci]

        # eligible, not used, hasn't fought ci, not ci itself
        valid = eligible & ~used & ~has_battled_coop[ci] & (agent_ids != ci)

        # pick the valid opponent earliest in sorted order
        scores = jnp.where(valid, position_of, M)
        cj = jnp.argmin(scores)
        has_match = scores[cj] < M

        do_pair = ~skip & has_match

        pairs = pairs.at[pair_idx, 0].set(jnp.where(do_pair, ci, pairs[pair_idx, 0]))
        pairs = pairs.at[pair_idx, 1].set(jnp.where(do_pair, cj, pairs[pair_idx, 1]))
        used = used.at[ci].set(used[ci] | do_pair)
        used = used.at[cj].set(used[cj] | do_pair)
        pair_idx = pair_idx + do_pair.astype(jnp.int32)

        return (used, pairs, pair_idx)

    _, pairs, _ = jax.lax.fori_loop(0, M, body, init)
    return pairs


# ---------------------------------------------------------------------------
# Main event: assign_agents_to_cage
# ---------------------------------------------------------------------------

def assign_agents_to_cage(
    key: chex.PRNGKey,
    state: EnvState,
    params: EnvParams,
) -> EnvState:
    """Pair eligible BattleZone chickens into cages for all coops.

    Updates battle_pair and cage_occupied. Call relocate_zone after to
    move the unpaired chickens out of BattleZone.
    """
    M, N_coop = params.M, params.N_coop

    new_battle_pair   = state.battle_pair    # (M//2, N_coop, 2)  — will be rebuilt
    new_cage_occupied = state.cage_occupied  # (M//2, N_coop)

    # Reset cage assignments for this round
    new_battle_pair   = jnp.full((M // 2, N_coop, 2), fill_value=-1, dtype=jnp.int32)
    new_cage_occupied = jnp.zeros((M // 2, N_coop), dtype=jnp.bool_)

    for coop in range(N_coop):
        key, subkey = jax.random.split(key)

        eligible = _eligible_mask(state, params, coop)             # (M,)
        sorted_idx = _sort_by_losses(
            eligible=eligible,
            loss_count_coop=state.loss_count[:, coop],
            key=subkey,
        )                                                           # (M,)

        has_battled_coop = state.has_battled[:, :, coop]             # (M, M) bool

        pairs = _greedy_pair(
            sorted_indices=sorted_idx,
            eligible=eligible,
            has_battled_coop=has_battled_coop,
            M=M,
        )                                                           # (M//2, 2)

        occupied = (pairs[:, 0] != -1).astype(jnp.bool_)           # (M//2,)

        new_battle_pair   = new_battle_pair.at[:, coop, :].set(pairs)
        new_cage_occupied = new_cage_occupied.at[:, coop].set(occupied)

    return state.replace(
        battle_pair=new_battle_pair,
        cage_occupied=new_cage_occupied,
    )


# ---------------------------------------------------------------------------
# Companion event: relocate_zone
# ---------------------------------------------------------------------------

def relocate_zone(
    state: EnvState,
    params: EnvParams,
) -> EnvState:
    """Move unpaired BattleZone chickens to SpectatorZone.

    A chicken is "unpaired" if it is eligible to battle (in BattleZone,
    under loss limit) but was not assigned to a cage this round.

    Spec note: once relocated to SpectatorZone, a chicken will not re-enter
    BattleZone for this coop unless it moves to another coop via relocate_region.

    Updates state.zone_array only.
    """
    M, N_coop = params.M, params.N_coop

    # Determine which chickens were assigned to a cage this round
    # battle_pair[:, :, 0] gives the first-of-pair indices per cage per coop
    # shape (M//2, N_coop); -1 = empty slot
    assigned_mask = jnp.zeros((M, N_coop), dtype=jnp.int32)

    for coop in range(N_coop):
        pairs = state.battle_pair[:, coop, :]  # (M//2, 2)
        for slot in range(2):
            idxs = pairs[:, slot]              # (M//2,) chicken indices
            valid = idxs >= 0                  # (M//2,) bool
            # scatter: mark each valid index as assigned in this coop
            assigned_mask = assigned_mask.at[idxs, coop].add(
                valid.astype(jnp.int32)
            )

    new_zone = state.zone_array  # (M, N_coop, 3)

    for coop in range(N_coop):
        in_battle  = state.zone_array[:, coop, ZONE_BATTLE].astype(jnp.bool_)   # (M,)
        unassigned = in_battle & (assigned_mask[:, coop] == 0)   # ALL unpaired BattleZone chickens

        # Clear BattleZone bit, set SpectatorZone bit for unassigned chickens
        was_battle    = new_zone[:, coop, ZONE_BATTLE]
        was_spectator = new_zone[:, coop, ZONE_SPECTATOR]

        new_battle    = jnp.where(unassigned, jnp.int32(0), was_battle)
        new_spectator = jnp.where(unassigned, jnp.int32(1), was_spectator)

        new_zone = new_zone.at[:, coop, ZONE_BATTLE].set(new_battle)
        new_zone = new_zone.at[:, coop, ZONE_SPECTATOR].set(new_spectator)

    return state.replace(zone_array=new_zone)
