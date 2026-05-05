"""
tests/test_preconditions.py — Event Precondition Enforcement.

Verifies spec requirements for event gating:
  - assign_agents_to_cage: no pair battles more than once per coop per tournament
  - assign_agents_to_cage: agents with >= k losses never paired in that coop
  - assign_agents_to_cage: pairing order is fewest-losses-first
  - assign_agents_to_cage: odd agent out goes to SpectatorZone
  - relocate_zone: only unpaired BattleZone chickens move to SpectatorZone
  - relocate_region: only SpectatorZone a_move chickens enter TransitZone
  - region_view: BattleZone chickens observe only their coop; a_move sees nothing
  - Arrival rule: TransitZone arrivals -> BattleZone if eligible, SpectatorZone if not

Uses M=8, N_coop=2 for speed; fixed PRNGKey(42) for reproducibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.state import EnvParams, EnvState, empty_env_state
from envs.spaces import ZONE_BATTLE, ZONE_SPECTATOR, ZONE_TRANSIT
from events.assign_agents import assign_agents_to_cage, relocate_zone
from events.dominance_battle import dominance_battle
from events.relocate import relocate_region, arrive_from_transit
from events.region_view import region_view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL = EnvParams(M=8, N_coop=2, k=4, T=5)


def _make_state(
    params: EnvParams,
    *,
    abilities=None, zone_array=None, action_array=None,
    loss_count=None, win_count=None, battle_pair=None,
    cage_occupied=None,
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
        zone_array=zone_array if zone_array is not None else base.zone_array,
        action_array=action_array if action_array is not None else base.action_array,
        loss_count=loss_count if loss_count is not None else base.loss_count,
        win_count=win_count if win_count is not None else base.win_count,
        battle_pair=battle_pair if battle_pair is not None else base.battle_pair,
        cage_occupied=cage_occupied if cage_occupied is not None else base.cage_occupied,
    )


def _all_in_battle_zone(M: int, N: int) -> jnp.ndarray:
    """zone_array with all agents in BattleZone of coop (m % N)."""
    z = jnp.zeros((M, N, 3), dtype=jnp.int32)
    coops = jnp.arange(M) % N
    return z.at[jnp.arange(M), coops, ZONE_BATTLE].set(1)


# ---------------------------------------------------------------------------
# assign_agents_to_cage: no duplicate battles per coop per tournament
# ---------------------------------------------------------------------------

class TestNoDuplicateBattles:
    """No pair should battle more than once per coop per tournament."""

    def test_no_repeat_pairing_across_rounds(self):
        """Run 2 rounds of assign+battle; check no pair is re-paired in round 2."""
        M, N = SMALL.M, SMALL.N_coop
        key = jax.random.PRNGKey(42)
        zone = _all_in_battle_zone(M, N)
        state = _make_state(SMALL, zone_array=zone)

        # Round 1: assign and battle
        key, k1, k2 = jax.random.split(key, 3)
        state = assign_agents_to_cage(k1, state, SMALL)
        round1_pairs = np.array(state.battle_pair)  # (M//2, N, 2)
        state = dominance_battle(k2, state, SMALL)

        # All agents back to BattleZone for round 2
        state = state.replace(zone_array=_all_in_battle_zone(M, N))

        # Round 2: assign again
        key, k3 = jax.random.split(key)
        state = assign_agents_to_cage(k3, state, SMALL)
        round2_pairs = np.array(state.battle_pair)

        # Check: no pair from round 1 appears in round 2 for the same coop
        for coop in range(N):
            r1_set = set()
            for cage in range(M // 2):
                i, j = int(round1_pairs[cage, coop, 0]), int(round1_pairs[cage, coop, 1])
                if i >= 0 and j >= 0:
                    r1_set.add((min(i, j), max(i, j)))

            for cage in range(M // 2):
                i, j = int(round2_pairs[cage, coop, 0]), int(round2_pairs[cage, coop, 1])
                if i >= 0 and j >= 0:
                    pair = (min(i, j), max(i, j))
                    assert pair not in r1_set, \
                        f"Duplicate pair {pair} in coop {coop} across rounds 1 and 2"


# ---------------------------------------------------------------------------
# assign_agents_to_cage: agents with >= k losses never paired
# ---------------------------------------------------------------------------

class TestLossLimitEnforcement:
    """Chickens at or above k losses must not appear in any cage."""

    def test_over_limit_excluded(self):
        """Set agents 0,1 to k losses in coop 0; verify neither is paired there."""
        M, N = SMALL.M, SMALL.N_coop
        key = jax.random.PRNGKey(42)
        zone = _all_in_battle_zone(M, N)
        loss = jnp.zeros((M, N), dtype=jnp.int32)
        loss = loss.at[0, 0].set(SMALL.k)
        loss = loss.at[1, 0].set(SMALL.k)
        state = _make_state(SMALL, zone_array=zone, loss_count=loss)
        state = assign_agents_to_cage(key, state, SMALL)

        pairs_coop0 = np.array(state.battle_pair[:, 0, :])  # (M//2, 2)
        for cage in range(M // 2):
            i, j = int(pairs_coop0[cage, 0]), int(pairs_coop0[cage, 1])
            if i >= 0:
                assert i not in (0, 1), \
                    f"Agent {i} with k losses was paired in coop 0"
            if j >= 0:
                assert j not in (0, 1), \
                    f"Agent {j} with k losses was paired in coop 0"


# ---------------------------------------------------------------------------
# assign_agents_to_cage: fewest losses first pairing order
# ---------------------------------------------------------------------------

class TestPairingOrder:
    """Pairing should prioritize chickens with fewest losses."""

    def test_fewest_losses_paired_first(self):
        """Give agents different loss counts; verify lower-loss agents are paired."""
        M, N = SMALL.M, SMALL.N_coop
        key = jax.random.PRNGKey(42)

        # Put 4 agents in coop 0 BattleZone with varying losses
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 0, ZONE_BATTLE].set(1)  # 0 losses
        zone = zone.at[1, 0, ZONE_BATTLE].set(1)  # 0 losses
        zone = zone.at[2, 0, ZONE_BATTLE].set(1)  # 2 losses
        zone = zone.at[3, 0, ZONE_BATTLE].set(1)  # 3 losses
        for m in range(4, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)

        loss = jnp.zeros((M, N), dtype=jnp.int32)
        loss = loss.at[2, 0].set(2)
        loss = loss.at[3, 0].set(3)

        state = _make_state(SMALL, zone_array=zone, loss_count=loss)
        state = assign_agents_to_cage(key, state, SMALL)

        # Agents 0 and 1 (0 losses each) should be paired before 2 and 3
        pairs_coop0 = np.array(state.battle_pair[:, 0, :])
        paired_agents = set()
        for cage in range(M // 2):
            i, j = int(pairs_coop0[cage, 0]), int(pairs_coop0[cage, 1])
            if i >= 0:
                paired_agents.add(i)
                paired_agents.add(j)

        # At minimum, agents 0 and 1 should be paired (they have fewest losses)
        assert 0 in paired_agents and 1 in paired_agents, \
            f"Agents 0,1 (0 losses) should be paired first but got {paired_agents}"


# ---------------------------------------------------------------------------
# Odd agent out -> SpectatorZone
# ---------------------------------------------------------------------------

class TestOddAgentOut:
    """Unpaired eligible BattleZone chicken goes to SpectatorZone after relocate_zone."""

    def test_odd_agent_relocates(self):
        """Put 3 eligible agents in one coop; 1 must go to SpectatorZone."""
        M, N = SMALL.M, SMALL.N_coop
        key = jax.random.PRNGKey(42)

        # 3 agents in coop 0 BattleZone
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 0, ZONE_BATTLE].set(1)
        zone = zone.at[1, 0, ZONE_BATTLE].set(1)
        zone = zone.at[2, 0, ZONE_BATTLE].set(1)
        for m in range(3, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)

        state = _make_state(SMALL, zone_array=zone)
        state = assign_agents_to_cage(key, state, SMALL)
        state = relocate_zone(state, SMALL)

        # One of {0, 1, 2} should be in SpectatorZone of coop 0
        in_spec = np.array(state.zone_array[:3, 0, ZONE_SPECTATOR])
        assert int(in_spec.sum()) >= 1, \
            "With 3 agents, at least 1 must be moved to SpectatorZone (odd out)"


# ---------------------------------------------------------------------------
# relocate_zone: only unpaired BattleZone chickens move
# ---------------------------------------------------------------------------

class TestRelocateZonePrecondition:
    """Only unpaired BattleZone chickens should be moved to SpectatorZone."""

    def test_paired_stay_in_battle(self):
        """Paired agents must remain in BattleZone after relocate_zone."""
        M, N = SMALL.M, SMALL.N_coop
        # Place agents 0 and 1 BOTH in coop 0's BattleZone
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 0, ZONE_BATTLE].set(1)
        zone = zone.at[1, 0, ZONE_BATTLE].set(1)
        for m in range(2, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        pair = jnp.full((M // 2, N, 2), -1, dtype=jnp.int32)
        pair = pair.at[0, 0, :].set(jnp.array([0, 1]))
        occ = jnp.zeros((M // 2, N), dtype=jnp.bool_)
        occ = occ.at[0, 0].set(True)
        state = _make_state(SMALL, zone_array=zone, battle_pair=pair, cage_occupied=occ)
        new = relocate_zone(state, SMALL)

        # Agents 0 and 1 are paired -> must stay in BattleZone of coop 0
        assert int(new.zone_array[0, 0, ZONE_BATTLE]) == 1, \
            "Paired agent 0 should remain in BattleZone"
        assert int(new.zone_array[1, 0, ZONE_BATTLE]) == 1, \
            "Paired agent 1 should remain in BattleZone"


# ---------------------------------------------------------------------------
# relocate_region: only SpectatorZone a_move enter TransitZone
# ---------------------------------------------------------------------------

class TestRelocateRegionPrecondition:
    """Only SpectatorZone a_move chickens should enter TransitZone."""

    def test_battle_zone_agent_not_moved(self):
        """A BattleZone agent choosing a_move should NOT be moved to TransitZone."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        # Agent 0 in BattleZone of coop 0, action = a_move(coop 1)
        a_move_val = N + 1 + 1  # a_move to coop 1
        actions = jnp.zeros((M, 1), dtype=jnp.int32)
        actions = actions.at[0, 0].set(a_move_val)
        state = _make_state(SMALL, zone_array=zone, action_array=actions)
        new = relocate_region(state, SMALL)

        # Agent 0 should still be in BattleZone (not moved — only SpectatorZone agents move)
        assert int(new.zone_array[0, 0, ZONE_BATTLE]) == 1, \
            "BattleZone agent should not be moved by relocate_region"
        assert int(new.zone_array[0, 1, ZONE_TRANSIT]) == 0, \
            "BattleZone agent should not appear in TransitZone"

    def test_spectator_a_watch_stays(self):
        """SpectatorZone a_watch agent should NOT enter TransitZone."""
        M, N = SMALL.M, SMALL.N_coop
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 0, ZONE_SPECTATOR].set(1)
        for m in range(1, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        # a_watch(coop=0) = 1
        actions = jnp.zeros((M, 1), dtype=jnp.int32)
        actions = actions.at[0, 0].set(1)
        state = _make_state(SMALL, zone_array=zone, action_array=actions)
        new = relocate_region(state, SMALL)

        assert int(new.zone_array[0, 0, ZONE_SPECTATOR]) == 1, \
            "a_watch agent should remain in SpectatorZone"

    def test_spectator_a_move_enters_transit(self):
        """SpectatorZone a_move agent should enter TransitZone of dest coop."""
        M, N = SMALL.M, SMALL.N_coop
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 0, ZONE_SPECTATOR].set(1)
        for m in range(1, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        # a_move(coop=1) = N + 1 + 1
        actions = jnp.zeros((M, 1), dtype=jnp.int32)
        actions = actions.at[0, 0].set(N + 1 + 1)
        state = _make_state(SMALL, zone_array=zone, action_array=actions)
        new = relocate_region(state, SMALL)

        assert int(new.zone_array[0, 0, ZONE_SPECTATOR]) == 0, \
            "a_move agent should leave SpectatorZone"
        assert int(new.zone_array[0, 1, ZONE_TRANSIT]) == 1, \
            "a_move agent should enter TransitZone of dest coop"


# ---------------------------------------------------------------------------
# region_view: observation rules
# ---------------------------------------------------------------------------

class TestRegionViewPreconditions:
    """BattleZone sees own coop; a_move sees nothing."""

    def test_battle_zone_sees_own_coop(self):
        """BattleZone chicken in coop 0 should see coop 0 outcomes."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        bo = jnp.zeros((M, M, N), dtype=jnp.int8)
        bo = bo.at[0, 1, 0].set(1)
        bo = bo.at[1, 0, 0].set(-1)
        state = _make_state(SMALL, zone_array=zone)
        state = state.replace(battle_outcome=bo)
        obs = region_view(state, SMALL)

        # Agent 0 in coop 0 BattleZone should see the outcome
        assert int(obs[0, 0, 1]) == 1, \
            "BattleZone chicken should see battle outcome in own coop"

    def test_a_move_sees_nothing(self):
        """a_move chicken should receive all-zero observation."""
        M, N = SMALL.M, SMALL.N_coop
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 0, ZONE_SPECTATOR].set(1)
        for m in range(1, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        # a_move(coop=1) = N + 1 + 1
        actions = jnp.zeros((M, 1), dtype=jnp.int32)
        actions = actions.at[0, 0].set(N + 1 + 1)
        bo = jnp.ones((M, M, N), dtype=jnp.int8)
        state = _make_state(SMALL, zone_array=zone, action_array=actions)
        state = state.replace(battle_outcome=bo)
        obs = region_view(state, SMALL)

        assert bool((obs[0] == 0).all()), \
            "a_move chicken should receive zero observation"


# ---------------------------------------------------------------------------
# Arrival rule: eligible -> BattleZone, ineligible -> SpectatorZone
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# assign_agents_to_cage: JIT compilation
# ---------------------------------------------------------------------------

class TestGreedyPairJIT:
    """_greedy_pair must work under jax.jit."""

    def test_jit_compiles(self):
        """assign_agents_to_cage compiles and runs under jit."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        state = _make_state(SMALL, zone_array=zone)
        key = jax.random.PRNGKey(99)

        @jax.jit
        def go(k, s):
            return assign_agents_to_cage(k, s, SMALL)

        new = go(key, state)
        pairs = np.array(new.battle_pair)
        assert (pairs[:, :, 0] >= 0).any(), "JIT pairing produced no pairs"


class TestArrivalRule:
    """TransitZone arrivals go to BattleZone if eligible, SpectatorZone if not."""

    def test_eligible_arrival_goes_to_battle(self):
        """Agent with 0 losses arriving via TransitZone enters BattleZone."""
        M, N = SMALL.M, SMALL.N_coop
        zone = jnp.zeros((M, N, 3), dtype=jnp.int32)
        zone = zone.at[0, 1, ZONE_TRANSIT].set(1)
        for m in range(1, M):
            zone = zone.at[m, m % N, ZONE_BATTLE].set(1)
        state = _make_state(SMALL, zone_array=zone)
        new = arrive_from_transit(state, SMALL)

        assert int(new.zone_array[0, 1, ZONE_TRANSIT]) == 0, "Transit should be cleared"
        assert int(new.zone_array[0, 1, ZONE_BATTLE]) == 1, "Agent should be in BattleZone"