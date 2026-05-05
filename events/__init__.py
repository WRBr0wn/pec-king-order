"""
Event system for Pec-King Order.

Events are pure functions:
    (EnvState, EntitySet, PRNGKey, EnvParams) -> (EnvState, EntitySet)

Step order (per spec):
  1. assign_agents_to_cage  — pair eligible BattleZone chickens
     OR relocate_zone       — unpaired BattleZone chickens -> SpectatorZone
  2. SpectatorZone chickens choose a_move or a_watch (action resolution)
  3. Concurrent:
       dominance_battle     — all caged pairs fight simultaneously
       region_view          — a_watch chickens receive observation
       relocate_region      — a_move chickens enter TransitZone (no observation)
  4. entity_update          — write all component and state array changes

  Between steps: check tournament reset conditions
       close_tournament     — assign kings, accumulate BattleHistory
       new_tournament       — redistribute chickens, reset per-tournament state
"""

from events.assign_agents    import assign_agents_to_cage, relocate_zone
from events.dominance_battle import dominance_battle, battle_outcome
from events.region_view      import region_view
from events.relocate         import relocate_region
from events.tournament       import close_tournament, new_tournament, tournament_is_done

__all__ = [
    "assign_agents_to_cage",
    "relocate_zone",
    "dominance_battle",
    "battle_outcome",
    "region_view",
    "relocate_region",
    "close_tournament",
    "new_tournament",
    "tournament_is_done",
]
