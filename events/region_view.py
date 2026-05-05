"""
region_view — distribute battle observations to chickens after each round.

Observation rules (per spec):
  BattleZone chickens         → receive all outcomes from their own coop
  SpectatorZone + a_watch     → receive all outcomes from their chosen coop
  SpectatorZone + a_move      → NO observation (they are in TransitZone)
  TransitZone chickens        → NO observation (in transit)

The event reads the current round's battle_outcome and action_array from
EnvState and returns an updated state.battle_outcome (unchanged) plus a
per-chicken observation matrix that callers use to update LastObservation
in AgentComponents (done in entity_update / the main step loop).

Returns
-------
obs_received : int8, shape (M, M, M)
    obs_received[m, :, :] is the (M, M) battle-outcome matrix that
    chicken m observed this round.  Zeros for chickens with no observation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex

from envs.state import EnvState, EnvParams
from envs.spaces import ZONE_BATTLE, ZONE_SPECTATOR, ZONE_TRANSIT, a_watch_index


def region_view(
    state: EnvState,
    params: EnvParams,
) -> chex.Array:
    """Compute per-chicken observation matrices for the current round.

    Parameters
    ----------
    state  : current EnvState (after dominance_battle has written battle_outcome)
    params : EnvParams

    Returns
    -------
    obs_received : int8, shape (M, M, M)
        obs_received[m] is the (M, M) battle outcome slice chicken m observed.
        Slice is zeros if the chicken has no observation this round.
    """
    M, N_coop = params.M, params.N_coop

    # Start with all-zero observations
    obs_received = jnp.zeros((M, M, M), dtype=jnp.int8)

    for coop in range(N_coop):
        coop_outcomes = state.battle_outcome[:, :, coop]  # (M, M) int8

        # --- BattleZone chickens: observe their own coop unconditionally ---
        in_battle_zone = state.zone_array[:, coop, ZONE_BATTLE].astype(jnp.bool_)
        # in_battle_zone shape: (M,)
        # For each chicken m with in_battle_zone[m]==True, set obs_received[m] = coop_outcomes
        obs_received = jnp.where(
            in_battle_zone[:, None, None],   # broadcast to (M, M, M)
            # Tile coop_outcomes to match (M, M, M): same slice for all observers
            jnp.broadcast_to(coop_outcomes[None], (M, M, M)),
            obs_received,
        )

        # --- SpectatorZone + a_watch targeting this coop ---
        # A chicken may be in SpectatorZone of *any* coop and watch any coop's
        # monitor — physical location does not restrict the target coop.
        # action = 1 + coop  (a_watch_index)
        watch_action = a_watch_index(coop, N_coop)
        action_flat  = state.action_array[:, 0]                                           # (M,) int32
        in_any_spectator = state.zone_array[:, :, ZONE_SPECTATOR].any(axis=1).astype(jnp.bool_)  # (M,)
        watching_coop    = (action_flat == watch_action).astype(jnp.bool_)                # (M,)

        is_watcher = in_any_spectator & watching_coop  # (M,)

        obs_received = jnp.where(
            is_watcher[:, None, None],
            jnp.broadcast_to(coop_outcomes[None], (M, M, M)),
            obs_received,
        )

    return obs_received  # (M, M, M)
