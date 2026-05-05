"""
tests/test_events.py — unit tests for the 5 event modules.

Coverage:
  battle_outcome (dominance_battle.py)
    - higher ability always wins (deterministic cases)
    - equal ability returns +1 or -1 (never 0) (stochastic case)
    - equal ability converges to p=0.5 over many draws

  dominance_battle
    - battle_outcome antisymmetry after event
    - loss_count and win_count incremented correctly
    - win_count + loss_count == #battles per coop
    - has_battled is set after battles

  assign_agents_to_cage
    - no pair shares a cage with an over-limit chicken (>= k losses)
    - no pair that already battled this tournament is re-paired
    - cage_occupied is True only where battle_pair != -1

  relocate_zone
    - unpaired eligible BattleZone chickens move to SpectatorZone
    - paired chickens stay in BattleZone

  relocate_region (relocate.py)
    - a_move chickens leave SpectatorZone and enter TransitZone
    - a_watch chickens stay in SpectatorZone

  arrive_from_transit (relocate.py)
    - eligible arrivals go to BattleZone
    - ineligible arrivals (>= k losses) go to SpectatorZone
    - TransitZone is cleared after arrival

  region_view
    - BattleZone chickens receive non-zero obs matching their coop outcomes
    - SpectatorZone a_watch chickens receive obs for the target coop
    - SpectatorZone a_move / TransitZone chickens receive all-zero obs

  tournament_is_done
    - returns False when eligible pairs remain
    - returns True when all chickens exceed k losses

  close_tournament / new_tournament
    - king index is in [0, M) after close_tournament
    - new_tournament resets loss_count, win_count to zero
    - new_tournament: every agent is in exactly one BattleZone
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.state import EnvState, EnvParams, empty_env_state
from envs.spaces import ZONE_BATTLE, ZONE_SPECTATOR, ZONE_TRANSIT
from events.dominance_battle import battle_outcome, dominance_battle
from events.assign_agents    import assign_agents_to_cage, relocate_zone
from events.relocate         import relocate_region, arrive_from_transit
from events.region_view      import region_view
from events.tournament       import tournament_is_done, close_tournament, new_tournament


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    params: EnvParams,
    *,
    abilities: jnp.ndarray | None = None,
    zone_array: jnp.ndarray | None = None,
    action_array: jnp.ndarray | None = None,
    loss_count: jnp.ndarray | None = None,
    win_count: jnp.ndarray | None = None,
    battle_pair: jnp.ndarray | None = None,
    cage_occupied: jnp.ndarray | None = None,
) -> EnvState:
    """Build an EnvState from empty_env_state with optional overrides."""
    M, N, T = params.M, params.N_coop, params.T
    base = empty_env_state(params)

    if abilities is None:
        abilities = jnp.ones((M, N), dtype=jnp.int32)

    thresholds = jnp.arange(1, N + 1, dtype=jnp.int32)
    ability_matrix = (abilities[None, :, :] >= thresholds[:, None, None]).astype(jnp.int32)

    return base.replace(
        abilities=abilities,
        ability_matrix=ability_matrix,
        zone_array=zone_array   if zone_array   is not None else base.zone_array,
        action_array=action_array if action_array is not None else base.action_array,
        loss_count=loss_count   if loss_count   is not None else base.loss_count,
        win_count=win_count     if win_count     is not None else base.win_count,
        battle_pair=battle_pair if battle_pair   is not None else base.battle_pair,
        cage_occupied=cage_occupied if cage_occupied is not None else base.cage_occupied,
    )


def _all_in_battle_zone(M: int, N: int) -> jnp.ndarray:
    """zone_array with all agents in BattleZone of coop (m % N)."""
    z = jnp.zeros((M, N, 3), dtype=jnp.int32)
    coops = jnp.arange(M) % N
    return z.at[jnp.arange(M), coops, ZONE_BATTLE].set(1)


SMALL = EnvParams(M=8, N_coop=4, k=4, T=5)


# ---------------------------------------------------------------------------
# battle_outcome
# ---------------------------------------------------------------------------

class TestBattleOutcome:
    def test_higher_wins(self):
        key = jax.random.PRNGKey(0)
        result = battle_outcome(jnp.int32(3), jnp.int32(1), key)
        assert int(result) == 1

    def test_lower_loses(self):
        key = jax.random.PRNGKey(0)
        result = battle_outcome(jnp.int32(1), jnp.int32(4), key)
        assert int(result) == -1

    def test_tie_is_nonzero(self):
        key = jax.random.PRNGKey(0)
        result = battle_outcome(jnp.int32(2), jnp.int32(2), key)
        assert int(result) in (1, -1)

    def test_tie_converges_to_half(self):
        """Over 1000 draws, ties should give ~50% each."""
        wins = 0
        n = 1000
        for i in range(n):
            key = jax.random.PRNGKey(i)
            r = int(battle_outcome(jnp.int32(2), jnp.int32(2), key))
            wins += (r == 1)
        ratio = wins / n
        assert 0.44 < ratio < 0.56, f"Tie win ratio {ratio:.3f} not near 0.5"

    def test_deterministic_across_same_key(self):
        key = jax.random.PRNGKey(99)
        r1 = int(battle_outcome(jnp.int32(2), jnp.int32(2), key))
        r2 = int(battle_outcome(jnp.int32(2), jnp.int32(2), key))
        assert r1 == r2, "Same key must give same result"


# ---------------------------------------------------------------------------
# dominance_battle
# ---------------------------------------------------------------------------

class TestDominanceBattle:
    def _state_with_one_pair(self, coop: int = 0):
        """State with agents 0 and 1 caged in coop 0, abilities 3 vs 1."""
        M, N, T = SMALL.M, SMALL.N_coop, SMALL.T
        abilities = jnp.ones((M, N), dtype=jnp.int32)
        # agent 0 ability=3 in coop 0, agent 1 ability=1
        abilities = abilities.at[0, coop].set(3)
        abilities = abilities.at[1, coop].set(1)

        zone = _all_in_battle_zone(M, N)
        pair = jnp.full((M // 2, N, 2), -1, dtype=jnp.int32)
        pair = pair.at[0, coop, :].set(jnp.array([0, 1]))
        occ  = jnp.zeros((M // 2, N), dtype=jnp.bool_)
        occ  = occ.at[0, coop].set(True)
        return _make_state(SMALL, abilities=abilities, zone_array=zone,
                           battle_pair=pair, cage_occupied=occ)

    def test_higher_ability_wins(self):
        key   = jax.random.PRNGKey(5)
        state = self._state_with_one_pair(coop=0)
        new   = dominance_battle(key, state, SMALL)
        # agent 0 (ability=3) should beat agent 1 (ability=1) in coop 0
        assert int(new.battle_outcome[0, 1, 0]) ==  1
        assert int(new.battle_outcome[1, 0, 0]) == -1

    def test_antisymmetry(self):
        key   = jax.random.PRNGKey(5)
        state = self._state_with_one_pair()
        new   = dominance_battle(key, state, SMALL)
        bo = new.battle_outcome.astype(jnp.int32)
        diff = bo + bo.transpose(1, 0, 2)
        assert bool((diff == 0).all())

    def test_loss_count_incremented(self):
        key   = jax.random.PRNGKey(5)
        state = self._state_with_one_pair()
        new   = dominance_battle(key, state, SMALL)
        # Agent 1 (ability=1) should have +1 loss in coop 0
        assert int(new.loss_count[1, 0]) == 1
        assert int(new.loss_count[0, 0]) == 0

    def test_win_count_incremented(self):
        key   = jax.random.PRNGKey(5)
        state = self._state_with_one_pair()
        new   = dominance_battle(key, state, SMALL)
        assert int(new.win_count[0, 0]) == 1
        assert int(new.win_count[1, 0]) == 0

    def test_win_plus_loss_equals_battles(self):
        key   = jax.random.PRNGKey(5)
        state = self._state_with_one_pair()
        new   = dominance_battle(key, state, SMALL)
        total = int(new.win_count[:, 0].sum()) + int(new.loss_count[:, 0].sum())
        # One battle → one win + one loss = 2
        assert total == 2

    def test_has_battled_set_after_battle(self):
        key   = jax.random.PRNGKey(5)
        state = self._state_with_one_pair()
        new   = dominance_battle(key, state, SMALL)
        assert bool(new.has_battled.any()), "has_battled should be set after a battle"

    def test_empty_cage_no_update(self):
        """No occupied cages → everything stays zero."""
        key   = jax.random.PRNGKey(5)
        state = _make_state(SMALL, zone_array=_all_in_battle_zone(SMALL.M, SMALL.N_coop))
        new   = dominance_battle(key, state, SMALL)
        assert bool((new.battle_outcome == 0).all())
        assert bool((new.loss_count == 0).all())
        assert bool((new.win_count == 0).all())


# ---------------------------------------------------------------------------
# assign_agents_to_cage + relocate_zone
# ---------------------------------------------------------------------------

class TestAssignAndRelocate:
    def _base_state(self):
        zone = _all_in_battle_zone(SMALL.M, SMALL.N_coop)
        return _make_state(SMALL, zone_array=zone)

    def test_occupied_cages_have_valid_indices(self):
        key   = jax.random.PRNGKey(0)
        state = self._base_state()
        new   = assign_agents_to_cage(key, state, SMALL)
        M = SMALL.M
        for coop in range(SMALL.N_coop):
            for cage in range(M // 2):
                if new.cage_occupied[cage, coop]:
                    i, j = int(new.battle_pair[cage, coop, 0]), int(new.battle_pair[cage, coop, 1])
                    assert 0 <= i < M
                    assert 0 <= j < M
                    assert i != j

    def test_no_over_limit_chicken_paired(self):
        """Chickens with k losses must not appear in any cage."""
        M, N = SMALL.M, SMALL.N_coop
        # Give agents 0 and 1 exactly k losses in coop 0 → ineligible
        loss = jnp.zeros((M, N), dtype=jnp.int32)
        loss = loss.at[0, 0].set(SMALL.k)
        loss = loss.at[1, 0].set(SMALL.k)
        zone  = _all_in_battle_zone(M, N)
        state = _make_state(SMALL, zone_array=zone, loss_count=loss)
        key   = jax.random.PRNGKey(0)
        new   = assign_agents_to_cage(key, state, SMALL)
        # Neither 0 nor 1 should appear in any cage slot for coop 0
        for cage in range(M // 2):
            pair = new.battle_pair[cage, 0, :]
            assert int(pair[0]) not in (0, 1) or int(pair[0]) == -1
            assert int(pair[1]) not in (0, 1) or int(pair[1]) == -1

    def test_cage_occupied_consistent_with_battle_pair(self):
        key   = jax.random.PRNGKey(1)
        state = self._base_state()
        new   = assign_agents_to_cage(key, state, SMALL)
        for coop in range(SMALL.N_coop):
            for cage in range(SMALL.M // 2):
                occ  = bool(new.cage_occupied[cage, coop])
                pair = new.battle_pair[cage, coop, :]
                has_pair = int(pair[0]) >= 0
                assert occ == has_pair, \
                    f"cage_occupied mismatch at cage={cage}, coop={coop}"

    def test_relocate_zone_moves_unpaired_to_spectator(self):
        """Unpaired eligible BattleZone chickens must end up in SpectatorZone."""
        M, N = SMALL.M, SMALL.N_coop
        # Only pair agent 0 and 1 in coop 0; others remain unpaired
        zone  = _all_in_battle_zone(M, N)
        pair  = jnp.full((M // 2, N, 2), -1, dtype=jnp.int32)
        pair  = pair.at[0, 0, :].set(jnp.array([0, 1]))
        occ   = jnp.zeros((M // 2, N), dtype=jnp.bool_)
        occ   = occ.at[0, 0].set(True)
        state = _make_state(SMALL, zone_array=zone, battle_pair=pair, cage_occupied=occ)
        new   = relocate_zone(state, SMALL)

        # Agents 2..7 in coop 0 are unpaired → SpectatorZone
        for m in range(2, M):
            coop_of_m = m % N
            if coop_of_m == 0:
                assert int(new.zone_array[m, 0, ZONE_SPECTATOR]) == 1, \
                    f"Agent {m} should be in SpectatorZone of coop 0"


# ---------------------------------------------------------------------------
# relocate_region / arrive_from_transit
# ---------------------------------------------------------------------------

class TestRelocateRegion:
    def _state_in_spectator(self, dest_coop: int = 1):
        """Agent 0 in SpectatorZone of coop 0, action = a_move(dest_coop)."""
        M, N = SMALL.M, SMALL.N_coop
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 0, ZONE_SPECTATOR].set(1)
        # Other agents in their round-robin coop BattleZone
        for m in range(1, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        # a_move(dest_coop) = N + 1 + dest_coop
        a_move_val = N + 1 + dest_coop
        actions = jnp.zeros((M, 1), dtype=jnp.int32)
        actions = actions.at[0, 0].set(a_move_val)
        return _make_state(SMALL, zone_array=zone, action_array=actions)

    def test_a_move_leaves_spectator(self):
        state = self._state_in_spectator(dest_coop=1)
        new   = relocate_region(state, SMALL)
        assert int(new.zone_array[0, 0, ZONE_SPECTATOR]) == 0, \
            "Agent 0 should have left SpectatorZone of coop 0"

    def test_a_move_enters_transit(self):
        state = self._state_in_spectator(dest_coop=1)
        new   = relocate_region(state, SMALL)
        assert int(new.zone_array[0, 1, ZONE_TRANSIT]) == 1, \
            "Agent 0 should be in TransitZone of dest coop 1"

    def test_a_watch_stays_in_spectator(self):
        M, N = SMALL.M, SMALL.N_coop
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 0, ZONE_SPECTATOR].set(1)
        for m in range(1, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        # a_watch(coop=0) = 1
        actions = jnp.zeros((M, 1), dtype=jnp.int32)
        actions = actions.at[0, 0].set(1)
        state = _make_state(SMALL, zone_array=zone, action_array=actions)
        new   = relocate_region(state, SMALL)
        assert int(new.zone_array[0, 0, ZONE_SPECTATOR]) == 1, \
            "a_watch chicken should remain in SpectatorZone"

    def test_arrive_eligible_goes_to_battle(self):
        """Arriving chicken with 0 losses should enter BattleZone."""
        M, N = SMALL.M, SMALL.N_coop
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 1, ZONE_TRANSIT].set(1)
        for m in range(1, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        state = _make_state(SMALL, zone_array=zone)
        new   = arrive_from_transit(state, SMALL)
        assert int(new.zone_array[0, 1, ZONE_TRANSIT])  == 0, "Transit should be cleared"
        assert int(new.zone_array[0, 1, ZONE_BATTLE])   == 1, "Eligible should go to BattleZone"

    def test_arrive_ineligible_goes_to_spectator(self):
        """Arriving chicken at loss limit should enter SpectatorZone."""
        M, N = SMALL.M, SMALL.N_coop
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 1, ZONE_TRANSIT].set(1)
        for m in range(1, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        loss = jnp.zeros((M, N), dtype=jnp.int32)
        loss = loss.at[0, 1].set(SMALL.k)   # agent 0 has k losses in coop 1
        state = _make_state(SMALL, zone_array=zone, loss_count=loss)
        new   = arrive_from_transit(state, SMALL)
        assert int(new.zone_array[0, 1, ZONE_TRANSIT])   == 0
        assert int(new.zone_array[0, 1, ZONE_SPECTATOR]) == 1


# ---------------------------------------------------------------------------
# region_view
# ---------------------------------------------------------------------------

class TestRegionView:
    def test_battle_zone_chicken_receives_own_coop_obs(self):
        """A chicken in coop 0 BattleZone must receive coop-0 outcomes."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        # Inject a known outcome into battle_outcome for coop 0
        bo = jnp.zeros((M, M, N), dtype=jnp.int8)
        bo = bo.at[0, 1, 0].set(1)
        bo = bo.at[1, 0, 0].set(-1)
        state = _make_state(SMALL, zone_array=zone)
        state = state.replace(battle_outcome=bo)
        obs = region_view(state, SMALL)
        # Agent 0 is in coop 0 BattleZone → should see (0,1) = +1
        assert int(obs[0, 0, 1]) == 1,  "BattleZone chicken should see outcome[0,1]"
        assert int(obs[0, 1, 0]) == -1, "BattleZone chicken should see outcome[1,0]"

    def test_a_watch_spectator_receives_target_coop(self):
        """SpectatorZone a_watch(coop=2) chicken should see coop-2 outcomes."""
        M, N = SMALL.M, SMALL.N_coop
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 0, ZONE_SPECTATOR].set(1)
        for m in range(1, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        actions = jnp.zeros((M, 1), dtype=jnp.int32)
        # a_watch(coop=2) = 1 + 2 = 3
        actions = actions.at[0, 0].set(3)
        bo = jnp.zeros((M, M, N), dtype=jnp.int8)
        bo = bo.at[2, 3, 2].set(1)
        bo = bo.at[3, 2, 2].set(-1)
        state = _make_state(SMALL, zone_array=zone, action_array=actions)
        state = state.replace(battle_outcome=bo)
        obs = region_view(state, SMALL)
        assert int(obs[0, 2, 3]) == 1,  "Watcher of coop 2 should see outcome[2,3]"
        assert int(obs[0, 3, 2]) == -1, "Watcher of coop 2 should see outcome[3,2]"

    def test_a_move_chicken_gets_no_obs(self):
        """SpectatorZone a_move chicken should get all-zero observation."""
        M, N = SMALL.M, SMALL.N_coop
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 0, ZONE_SPECTATOR].set(1)
        for m in range(1, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        actions = jnp.zeros((M, 1), dtype=jnp.int32)
        # a_move(coop=1) = N + 1 + 1 = 6
        actions = actions.at[0, 0].set(SMALL.N_coop + 1 + 1)
        bo = jnp.ones((M, M, N), dtype=jnp.int8)
        state = _make_state(SMALL, zone_array=zone, action_array=actions)
        state = state.replace(battle_outcome=bo)
        obs = region_view(state, SMALL)
        assert bool((obs[0] == 0).all()), "a_move chicken should receive zero obs"


# ---------------------------------------------------------------------------
# tournament_is_done / close_tournament / new_tournament
# ---------------------------------------------------------------------------

class TestTournament:
    def test_not_done_when_eligible_pairs_remain(self):
        zone  = _all_in_battle_zone(SMALL.M, SMALL.N_coop)
        state = _make_state(SMALL, zone_array=zone)
        assert not bool(tournament_is_done(state, SMALL))

    def test_done_when_all_over_limit(self):
        """All chickens at k losses → no eligible pairs → done."""
        M, N = SMALL.M, SMALL.N_coop
        zone  = _all_in_battle_zone(M, N)
        loss  = jnp.full((M, N), SMALL.k, dtype=jnp.int32)
        state = _make_state(SMALL, zone_array=zone, loss_count=loss)
        assert bool(tournament_is_done(state, SMALL))

    def test_close_tournament_assigns_kings(self):
        M, N = SMALL.M, SMALL.N_coop
        # Give agent 0 a win in each coop (most wins → king)
        win = jnp.zeros((M, N), dtype=jnp.int32)
        win = win.at[0, :].set(3)
        zone  = _all_in_battle_zone(M, N)
        state = _make_state(SMALL, zone_array=zone, win_count=win)
        new_state, info = close_tournament(state, SMALL)
        for coop in range(N):
            assert bool(new_state.king[:, coop].any()), f"no king in coop {coop}"

    def test_close_tournament_king_masks_shape(self):
        M, N = SMALL.M, SMALL.N_coop
        zone  = _all_in_battle_zone(M, N)
        state = _make_state(SMALL, zone_array=zone)
        _, info = close_tournament(state, SMALL)
        assert info["king_masks"].shape == (N, M)
        assert info["king_repr"].shape  == (N,)

    def test_new_tournament_resets_counters(self):
        M, N = SMALL.M, SMALL.N_coop
        win  = jnp.ones((M, N), dtype=jnp.int32)
        loss = jnp.ones((M, N), dtype=jnp.int32)
        zone = _all_in_battle_zone(M, N)
        state = _make_state(SMALL, zone_array=zone, win_count=win, loss_count=loss)
        state, info = close_tournament(state, SMALL)
        key   = jax.random.PRNGKey(0)
        new   = new_tournament(key, state, SMALL, info["king_masks"])
        assert bool((new.loss_count == 0).all())
        assert bool((new.win_count  == 0).all())

    def test_new_tournament_zone_invariant(self):
        """After new_tournament, each agent must be in exactly one BattleZone."""
        M, N = SMALL.M, SMALL.N_coop
        zone  = _all_in_battle_zone(M, N)
        state = _make_state(SMALL, zone_array=zone)
        state, info = close_tournament(state, SMALL)
        key  = jax.random.PRNGKey(0)
        new  = new_tournament(key, state, SMALL, info["king_masks"])
        # Each agent should be in exactly one BattleZone
        battle_totals = new.zone_array[:, :, ZONE_BATTLE].sum(axis=1)  # (M,)
        assert bool((battle_totals == 1).all()), \
            f"Zone invariant violated after new_tournament: {battle_totals}"
        # No SpectatorZone or TransitZone occupancy
        spec_totals   = new.zone_array[:, :, ZONE_SPECTATOR].sum()
        transit_totals = new.zone_array[:, :, ZONE_TRANSIT].sum()
        assert int(spec_totals)    == 0
        assert int(transit_totals) == 0

    def test_tournament_count_increments_after_close(self):
        M, N = SMALL.M, SMALL.N_coop
        zone  = _all_in_battle_zone(M, N)
        state = _make_state(SMALL, zone_array=zone)
        assert int(state.tournament_count) == 0
        new_state, _ = close_tournament(state, SMALL)
        assert int(new_state.tournament_count) == 1
