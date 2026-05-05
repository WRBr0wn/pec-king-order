"""
relocate_region — move SpectatorZone a_move chickens into TransitZone.

Preconditions (per spec):
  - Chicken must be in SpectatorZone of their current coop.
  - Chicken must have chosen a_move action (action > N_coop).
  - Must occur BEFORE battles begin (actions are resolved pre-battle).

Arrival rule (per spec):
  Chickens arriving via TransitZone enter the destination coop's BattleZone
  if eligible (< k losses there, has not yet filled quota for that coop).
  If not eligible -> straight to SpectatorZone.

Two-phase logic:
  Phase 1 (this event, pre-battle):
    a_move chickens: clear current coop SpectatorZone bit, set TransitZone bit.
    Their destination coop is encoded in the action integer: a_move(coop) = N+1+coop.

  Phase 2 (entity_update, post-battle):
    TransitZone chickens: clear TransitZone bit, set destination zone bit
    (BattleZone or SpectatorZone depending on eligibility).

Updates state.zone_array only.  No observation is recorded (spec: a_move
chickens receive no observation that round).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex

from envs.state import EnvState, EnvParams
from envs.spaces import ZONE_BATTLE, ZONE_SPECTATOR, ZONE_TRANSIT


# ---------------------------------------------------------------------------
# Phase 1: depart current coop SpectatorZone → TransitZone
# ---------------------------------------------------------------------------

def relocate_region(
    state: EnvState,
    params: EnvParams,
) -> EnvState:
    """Move SpectatorZone a_move chickens into TransitZone (pre-battle phase).

    For each chicken m:
      If in SpectatorZone of any coop AND action is a_move(dest):
        - Clear SpectatorZone[m, current_coop] = 0
        - Set   TransitZone[m, dest_coop]      = 1
        (TransitZone is keyed to the *destination* coop for arrival logic.)

    Parameters
    ----------
    state  : current EnvState (action_array already set)
    params : EnvParams

    Returns
    -------
    Updated EnvState with new zone_array.
    """
    M, N_coop = params.M, params.N_coop
    action_flat = state.action_array[:, 0]  # (M,) int32

    new_zone = state.zone_array  # (M, N_coop, 3)

    for src_coop in range(N_coop):
        in_spectator = state.zone_array[:, src_coop, ZONE_SPECTATOR].astype(jnp.bool_)  # (M,)

        for dest_coop in range(N_coop):
            if dest_coop == src_coop:
                # Moving to the same coop has no effect
                continue

            a_move_val = N_coop + 1 + dest_coop                       # a_move action integer
            choosing_move = (action_flat == a_move_val)               # (M,) bool

            is_departing = in_spectator & choosing_move               # (M,) bool

            # Clear SpectatorZone in src_coop
            new_spectator_src = jnp.where(
                is_departing,
                jnp.int32(0),
                new_zone[:, src_coop, ZONE_SPECTATOR],
            )
            new_zone = new_zone.at[:, src_coop, ZONE_SPECTATOR].set(new_spectator_src)

            # Set TransitZone keyed to dest_coop
            new_transit_dest = jnp.where(
                is_departing,
                jnp.int32(1),
                new_zone[:, dest_coop, ZONE_TRANSIT],
            )
            new_zone = new_zone.at[:, dest_coop, ZONE_TRANSIT].set(new_transit_dest)

    return state.replace(zone_array=new_zone)


# ---------------------------------------------------------------------------
# Phase 2: arrive at destination coop (called from entity_update post-battle)
# ---------------------------------------------------------------------------

def arrive_from_transit(
    state: EnvState,
    params: EnvParams,
) -> EnvState:
    """Resolve TransitZone chickens into their destination coop.

    Called during entity_update after battles resolve.

    Arrival rule (per spec):
      - If eligible (loss_count[m, dest] < k): enter BattleZone.
      - Otherwise: enter SpectatorZone.

    Clears TransitZone bit; sets BattleZone or SpectatorZone bit.
    """
    M, N_coop = params.M, params.N_coop
    new_zone = state.zone_array  # (M, N_coop, 3)

    for dest_coop in range(N_coop):
        in_transit = state.zone_array[:, dest_coop, ZONE_TRANSIT].astype(jnp.bool_)  # (M,)
        eligible   = state.loss_count[:, dest_coop] < params.k                       # (M,)

        goes_to_battle    = in_transit & eligible      # (M,)
        goes_to_spectator = in_transit & ~eligible     # (M,)

        # Clear TransitZone
        new_transit = jnp.where(in_transit, jnp.int32(0), new_zone[:, dest_coop, ZONE_TRANSIT])
        new_zone = new_zone.at[:, dest_coop, ZONE_TRANSIT].set(new_transit)

        # Set BattleZone for eligible arrivals
        new_battle = jnp.where(
            goes_to_battle,
            jnp.int32(1),
            new_zone[:, dest_coop, ZONE_BATTLE],
        )
        new_zone = new_zone.at[:, dest_coop, ZONE_BATTLE].set(new_battle)

        # Set SpectatorZone for ineligible arrivals
        new_spectator = jnp.where(
            goes_to_spectator,
            jnp.int32(1),
            new_zone[:, dest_coop, ZONE_SPECTATOR],
        )
        new_zone = new_zone.at[:, dest_coop, ZONE_SPECTATOR].set(new_spectator)

    return state.replace(zone_array=new_zone)
