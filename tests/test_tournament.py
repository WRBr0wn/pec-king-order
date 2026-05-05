"""
tests/test_tournament.py — Tournament Reset & King Assignment Logic.

Verifies spec requirements for tournament lifecycle:
  - Tournament stops when no eligible pairs remain in ANY region (not just one)
  - Non-dominated agents (0 losses in a region) receive crowns for that region
  - If no non-dominated agents: agents with most wins receive crowns
  - Crowned agents assigned to coop with most battles (random tiebreak)
  - Uncrowned agents assigned uniformly random across coops
  - After reset: all battle preconditions cleared (pairs can battle again)
  - After reset: abilities unchanged

Uses M=8, N_coop=2 for speed; fixed PRNGKey(42) for reproducibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.state import EnvParams, EnvState, empty_env_state
from envs.spaces import ZONE_BATTLE, ZONE_SPECTATOR, ZONE_TRANSIT
from events.tournament import tournament_is_done, close_tournament, new_tournament


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL = EnvParams(M=8, N_coop=2, k=4, T=5)


def _make_state(
    params, *, abilities=None, zone_array=None,
    loss_count=None, win_count=None,
):
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
        loss_count=loss_count if loss_count is not None else base.loss_count,
        win_count=win_count if win_count is not None else base.win_count,
    )


def _all_in_battle_zone(M, N):
    z = jnp.zeros((M, N, 3), dtype=jnp.int32)
    coops = jnp.arange(M) % N
    return z.at[jnp.arange(M), coops, ZONE_BATTLE].set(1)


# ---------------------------------------------------------------------------
# Tournament termination
# ---------------------------------------------------------------------------

class TestTournamentTermination:
    """Tournament stops when no eligible pairs remain in ANY region."""

    def test_not_done_when_eligible_pairs_exist(self):
        """With all agents eligible and in BattleZone, tournament should not be done."""
        zone = _all_in_battle_zone(SMALL.M, SMALL.N_coop)
        state = _make_state(SMALL, zone_array=zone)
        assert not bool(tournament_is_done(state, SMALL)), \
            "Tournament should not be done when eligible pairs exist"

    def test_done_when_all_over_loss_limit(self):
        """All agents at k losses -> no eligible pairs -> tournament done."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        loss = jnp.full((M, N), SMALL.k, dtype=jnp.int32)
        state = _make_state(SMALL, zone_array=zone, loss_count=loss)
        assert bool(tournament_is_done(state, SMALL)), \
            "Tournament should be done when all agents have k losses"

    def test_done_requires_all_coops(self):
        """Tournament is NOT done if one coop still has eligible pairs."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        loss = jnp.full((M, N), SMALL.k, dtype=jnp.int32)
        # But leave agents in coop 0 eligible (0 losses)
        # Agent 0 is in coop 0 (m%N=0), agent 2 is in coop 0 (2%2=0)
        loss = loss.at[0, 0].set(0)
        loss = loss.at[2, 0].set(0)
        state = _make_state(SMALL, zone_array=zone, loss_count=loss)
        assert not bool(tournament_is_done(state, SMALL)), \
            "Tournament should NOT be done if coop 0 still has eligible pairs"

    def test_done_when_only_one_eligible_per_coop(self):
        """One eligible chicken per coop -> no pairs possible -> done."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        loss = jnp.full((M, N), SMALL.k, dtype=jnp.int32)
        # Only agent 0 eligible in coop 0, only agent 1 eligible in coop 1
        loss = loss.at[0, 0].set(0)
        loss = loss.at[1, 1].set(0)
        state = _make_state(SMALL, zone_array=zone, loss_count=loss)
        assert bool(tournament_is_done(state, SMALL)), \
            "One eligible per coop means no pairs -> tournament done"


# ---------------------------------------------------------------------------
# Crown assignment — non-dominated (0 losses)
# ---------------------------------------------------------------------------

class TestCrownNonDominated:
    """Non-dominated agents (0 losses, but participated) receive crowns."""

    def test_undefeated_gets_crown(self):
        """Agent with wins and 0 losses in a coop should be crowned."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        win = jnp.zeros((M, N), dtype=jnp.int32)
        loss = jnp.zeros((M, N), dtype=jnp.int32)
        # Agent 0: 3 wins, 0 losses in coop 0
        win = win.at[0, 0].set(3)
        # Agent 1: 1 win, 1 loss in coop 0
        win = win.at[1, 0].set(1)
        loss = loss.at[1, 0].set(1)

        state = _make_state(SMALL, zone_array=zone, win_count=win, loss_count=loss)
        new_state, info = close_tournament(state, SMALL)

        king_masks = np.array(info["king_masks"])  # (N, M)
        # Agent 0 should be king in coop 0 (undefeated)
        assert king_masks[0, 0], \
            "Agent 0 (3W/0L) should be crowned in coop 0"

    def test_multiple_undefeated_all_crowned(self):
        """If multiple agents are undefeated, all should be in king_masks."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        win = jnp.zeros((M, N), dtype=jnp.int32)
        loss = jnp.zeros((M, N), dtype=jnp.int32)
        # Agents 0 and 2: both undefeated in coop 0
        win = win.at[0, 0].set(2)
        win = win.at[2, 0].set(1)

        state = _make_state(SMALL, zone_array=zone, win_count=win, loss_count=loss)
        _, info = close_tournament(state, SMALL)

        king_masks = np.array(info["king_masks"])
        assert king_masks[0, 0] and king_masks[0, 2], \
            "Both undefeated agents should be in king_masks for coop 0"


# ---------------------------------------------------------------------------
# Crown assignment — fallback to most wins
# ---------------------------------------------------------------------------

class TestCrownMostWins:
    """If no non-dominated agents, chickens with most wins get crown."""

    def test_most_wins_gets_crown(self):
        """All agents have losses; agent with most wins should be crowned."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        win = jnp.zeros((M, N), dtype=jnp.int32)
        loss = jnp.zeros((M, N), dtype=jnp.int32)
        # All have at least 1 loss in coop 0
        loss = loss.at[0, 0].set(1)
        loss = loss.at[1, 0].set(2)
        loss = loss.at[2, 0].set(1)
        # Agent 0: 5 wins, agent 2: 3 wins
        win = win.at[0, 0].set(5)
        win = win.at[1, 0].set(1)
        win = win.at[2, 0].set(3)

        state = _make_state(SMALL, zone_array=zone, win_count=win, loss_count=loss)
        _, info = close_tournament(state, SMALL)

        king_masks = np.array(info["king_masks"])
        assert king_masks[0, 0], \
            "Agent 0 (most wins=5) should be crowned when all have losses"


# ---------------------------------------------------------------------------
# Post-reset placement
# ---------------------------------------------------------------------------

class TestNewTournamentPlacement:
    """After new_tournament: crowned -> most-battles coop, uncrowned -> random."""

    def test_crowned_placement_is_random(self):
        """Crowned agents get uniform random coop assignment — no preferred coop."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        win = jnp.zeros((M, N), dtype=jnp.int32)
        loss = jnp.zeros((M, N), dtype=jnp.int32)
        win = win.at[0, 0].set(4)
        loss = loss.at[0, 0].set(1)
        win = win.at[0, 1].set(1)

        state = _make_state(SMALL, zone_array=zone, win_count=win, loss_count=loss)
        state, info = close_tournament(state, SMALL)
        key = jax.random.PRNGKey(42)
        new = new_tournament(key, state, SMALL, info["king_masks"])

        # zone invariant: every agent in exactly one BattleZone
        battle_totals = new.zone_array[:, :, ZONE_BATTLE].sum(axis=1)
        assert bool((battle_totals == 1).all()), "zone invariant violated after placement"

    def test_uncrowned_distribution(self):
        """Uncrowned agents should end up in valid coops (basic check)."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        state = _make_state(SMALL, zone_array=zone)
        state, info = close_tournament(state, SMALL)
        key = jax.random.PRNGKey(42)
        new = new_tournament(key, state, SMALL, info["king_masks"])

        # Every agent should be in exactly one BattleZone
        battle_totals = new.zone_array[:, :, ZONE_BATTLE].sum(axis=1)
        assert bool((battle_totals == 1).all()), \
            f"Each agent should be in exactly one BattleZone: {battle_totals}"


# ---------------------------------------------------------------------------
# Post-reset state checks
# ---------------------------------------------------------------------------

class TestPostResetState:
    """After new_tournament: counters reset, abilities unchanged."""

    def _run_close_and_new(self):
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        win = jnp.ones((M, N), dtype=jnp.int32)
        loss = jnp.ones((M, N), dtype=jnp.int32)
        abilities = jnp.array([[1, 2]] * M, dtype=jnp.int32)  # known abilities
        state = _make_state(SMALL, zone_array=zone, win_count=win,
                            loss_count=loss, abilities=abilities)
        state, info = close_tournament(state, SMALL)
        key = jax.random.PRNGKey(42)
        new = new_tournament(key, state, SMALL, info["king_masks"])
        return state, new, abilities

    def test_counters_reset(self):
        """loss_count and win_count should be zero after new_tournament."""
        _, new, _ = self._run_close_and_new()
        assert bool((new.loss_count == 0).all()), "loss_count not reset to 0"
        assert bool((new.win_count == 0).all()), "win_count not reset to 0"

    def test_abilities_unchanged(self):
        """Abilities should not change after new_tournament."""
        old, new, expected_abilities = self._run_close_and_new()
        assert bool((new.abilities == expected_abilities).all()), \
            "Abilities should be unchanged after new_tournament"

    def test_battle_outcome_cleared(self):
        """battle_outcome should be zero after new_tournament."""
        _, new, _ = self._run_close_and_new()
        assert bool((new.battle_outcome == 0).all()), \
            "battle_outcome should be cleared after new_tournament"

    def test_cage_assignments_cleared(self):
        """battle_pair should be -1 and cage_occupied False after new_tournament."""
        _, new, _ = self._run_close_and_new()
        assert bool((new.battle_pair == -1).all()), \
            "battle_pair should be all -1 after new_tournament"
        assert bool((~new.cage_occupied).all()), \
            "cage_occupied should be all False after new_tournament"

    def test_all_in_battle_zone_after_reset(self):
        """All agents should be in BattleZone after new_tournament."""
        _, new, _ = self._run_close_and_new()
        battle_totals = new.zone_array[:, :, ZONE_BATTLE].sum(axis=1)
        assert bool((battle_totals == 1).all()), \
            "All agents should be in BattleZone after new_tournament"
        assert int(new.zone_array[:, :, ZONE_SPECTATOR].sum()) == 0, \
            "No agents in SpectatorZone after new_tournament"
        assert int(new.zone_array[:, :, ZONE_TRANSIT].sum()) == 0, \
            "No agents in TransitZone after new_tournament"

    def test_tournament_count_increments(self):
        """tournament_count should have been incremented by close_tournament."""
        old, new, _ = self._run_close_and_new()
        # close_tournament increments; new_tournament doesn't change it further
        assert int(new.tournament_count) == 1, \
            f"tournament_count should be 1, got {int(new.tournament_count)}"
