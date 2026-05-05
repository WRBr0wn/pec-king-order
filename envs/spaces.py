"""
ObservationSpace and ActionSpace for Pec-King Order.

Action encoding (per spec):
  0               — no action
  1  .. N_coop    — a_watch coop 0 .. N_coop-1   (stay + observe)
  N_coop+1 .. 2*N_coop — a_move  coop 0 .. N_coop-1   (move + no observation)
  Total distinct values: 2*N_coop + 1  →  Discrete(2*N_coop + 1)

Observation vector layout (fixed-size per agent, flattened):
  [0  : M*M)          last_battle_obs  — battle outcomes seen this round
                       values ∈ {-1, 0, +1}  (M*M elements)
  [M*M : M*M+N)       my_ability       — own ability scores per coop
                       values ∈ [1, N_coop]  (N_coop elements)
  [M*M+N : M*M+N+2)   my_location      — [coop_idx, zone_idx]
                       coop_idx ∈ [0, N_coop), zone_idx ∈ {0,1,2}  (2 elements)

  Total dim = M*M + N_coop + 2  (4102 for default M=64, N_coop=4)
"""

from __future__ import annotations

import jax.numpy as jnp
import chex
from gymnax.environments import spaces

from envs.state import EnvParams


# ---------------------------------------------------------------------------
# Action constants — import these instead of using magic numbers
# ---------------------------------------------------------------------------

# Zone indices (axis-2 of zone_array)
ZONE_BATTLE    = 0
ZONE_SPECTATOR = 1
ZONE_TRANSIT   = 2

# Action index boundaries (functions of N_coop — use helpers below)
ACTION_NO_OP = 0


def a_watch_index(coop: int, N_coop: int) -> int:
    """Integer action for a_watch targeting *coop* (0-indexed).

    Range: [1, N_coop].
    """
    assert 0 <= coop < N_coop, f"coop {coop} out of range [0, {N_coop})"
    return 1 + coop


def a_move_index(coop: int, N_coop: int) -> int:
    """Integer action for a_move toward *coop* (0-indexed).

    Range: [N_coop+1, 2*N_coop].
    """
    assert 0 <= coop < N_coop, f"coop {coop} out of range [0, {N_coop})"
    return N_coop + 1 + coop


def action_type(action: chex.Array, N_coop: int) -> tuple[chex.Array, chex.Array]:
    """Decode a scalar action integer into (action_kind, target_coop).

    Returns
    -------
    action_kind : int32 scalar
        0 = no-op, 1 = a_watch, 2 = a_move
    target_coop : int32 scalar
        0-indexed coop target.  -1 for no-op.
    """
    is_no_op  = action == ACTION_NO_OP
    is_watch  = (action >= 1) & (action <= N_coop)
    # a_move otherwise

    kind = jnp.where(is_no_op, 0, jnp.where(is_watch, 1, 2)).astype(jnp.int32)
    coop = jnp.where(
        is_no_op,
        jnp.int32(-1),
        jnp.where(is_watch, (action - 1).astype(jnp.int32),
                  (action - N_coop - 1).astype(jnp.int32)),
    )
    return kind, coop


def action_to_one_hot(action: chex.Array, N_coop: int) -> chex.Array:
    """Convert a scalar action integer to a 2*N_coop one-hot vector.

    The one-hot length is 2*N_coop (no-op maps to all-zeros).
    Indices 0..N_coop-1  → a_watch coops 0..N_coop-1
    Indices N_coop..2*N_coop-1 → a_move coops 0..N_coop-1

    Parameters
    ----------
    action : scalar int32 in [0, 2*N_coop]
    N_coop : int

    Returns
    -------
    one_hot : float32, shape (2*N_coop,)
    """
    # Shift by 1 so that action=0 maps to index -1 (out-of-range → all zeros).
    shifted = (action - 1).astype(jnp.int32)
    return jnp.where(
        action == ACTION_NO_OP,
        jnp.zeros(2 * N_coop, dtype=jnp.float32),
        (jnp.arange(2 * N_coop, dtype=jnp.int32) == shifted).astype(jnp.float32),
    )


# ---------------------------------------------------------------------------
# Observation vector helpers
# ---------------------------------------------------------------------------

def obs_dim(params: EnvParams) -> int:
    """Total length of the flattened per-agent observation vector."""
    return params.M * params.M + params.N_coop + 2


def make_observation(
    last_battle_obs: chex.Array,   # (M, M) int8
    my_ability: chex.Array,        # (N_coop,) int32
    my_location: chex.Array,       # (2,) int32  [coop_idx, zone_idx]
) -> chex.Array:
    """Pack per-agent observation components into a single flat float32 vector.

    Layout matches the slice offsets documented in this module's docstring.
    All inputs are cast to float32 for uniformity with gymnax Box spaces.

    Parameters
    ----------
    last_battle_obs : int8, shape (M, M)
        Battle outcomes observed this round; values ∈ {-1, 0, +1}.
    my_ability : int32, shape (N_coop,)
        Agent's own ability scores; values ∈ [1, N_coop].
    my_location : int32, shape (2,)
        [coop_idx, zone_idx].

    Returns
    -------
    obs : float32, shape (obs_dim,)
    """
    return jnp.concatenate([
        last_battle_obs.ravel().astype(jnp.float32),
        my_ability.ravel().astype(jnp.float32),
        my_location.ravel().astype(jnp.float32),
    ])


def split_observation(
    obs: chex.Array,   # (obs_dim,) float32
    params: EnvParams,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Inverse of make_observation — split flat vector back into components.

    Returns
    -------
    last_battle_obs : float32, shape (M, M)
    my_ability      : float32, shape (N_coop,)
    my_location     : float32, shape (2,)
    """
    M, N = params.M, params.N_coop
    battle_flat  = obs[:M * M]
    ability_flat = obs[M * M : M * M + N]
    location_flat = obs[M * M + N : M * M + N + 2]
    return (
        battle_flat.reshape(M, M),
        ability_flat,
        location_flat,
    )


# ---------------------------------------------------------------------------
# Space constructors (called by PecKingOrder.observation_space / action_space)
# ---------------------------------------------------------------------------

def observation_space(params: EnvParams) -> spaces.Box:
    """gymnax Box space for a single agent's observation vector.

    Shape  : (obs_dim,)  where obs_dim = M*M + N_coop + 2
    low    : -1.0  (battle outcomes can be -1; ability/location are non-negative
                    but sharing one bound is fine for gymnax Box)
    high   :  N_coop  (max ability score; battle obs ≤ 1 ≤ N_coop)
    dtype  : float32
    """
    dim = obs_dim(params)
    return spaces.Box(
        low=-1.0,
        high=float(params.N_coop),
        shape=(dim,),
        dtype=jnp.float32,
    )


def action_space(params: EnvParams) -> spaces.Discrete:
    """gymnax Discrete space for a single agent's action.

    n = 2*N_coop + 1  (0=no-op, 1..N=a_watch, N+1..2N=a_move)
    """
    return spaces.Discrete(2 * params.N_coop + 1)
