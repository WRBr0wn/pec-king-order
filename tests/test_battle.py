"""
tests/test_battle.py — Battle Outcome Correctness.

Verifies spec requirements for the dominance battle system:
  - Higher ability always wins (100 battles with known unequal abilities = 100% win)
  - Equal ability produces ~50% win rate (1000 battles, +-5% tolerance)
  - Battle outcome correctly populates BattleOutcome[i,j,k] and BattleOutcome[j,i,k]
  - Ability scores sampled from truncated Poisson (10000 samples, tolerance check)

Uses fixed PRNGKey seeds for reproducibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.state import EnvParams, empty_env_state
from envs.spaces import ZONE_BATTLE
from envs.pec_king_order import PecKingOrder
from events.dominance_battle import battle_outcome, dominance_battle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL = EnvParams(M=8, N_coop=2, k=4, T=5)


def _make_state(params, *, abilities=None, zone_array=None, battle_pair=None, cage_occupied=None):
    """Build an EnvState with optional overrides."""
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
        battle_pair=battle_pair if battle_pair is not None else base.battle_pair,
        cage_occupied=cage_occupied if cage_occupied is not None else base.cage_occupied,
    )


def _all_in_battle_zone(M, N):
    z = jnp.zeros((M, N, 3), dtype=jnp.int32)
    coops = jnp.arange(M) % N
    return z.at[jnp.arange(M), coops, ZONE_BATTLE].set(1)


# ---------------------------------------------------------------------------
# Higher ability always wins (deterministic cases)
# ---------------------------------------------------------------------------

class TestDeterministicWins:
    """Higher ability must always win. Test 100 battles with known unequal abilities."""

    def test_higher_always_wins_100(self):
        """100 battles where ability_L=3 > ability_R=1: left should win every time."""
        wins = 0
        for seed in range(100):
            key = jax.random.PRNGKey(seed)
            result = int(battle_outcome(jnp.int32(3), jnp.int32(1), key))
            wins += (result == 1)
        assert wins == 100, \
            f"Higher ability should win 100/100 times, but won {wins}/100"

    def test_lower_always_loses_100(self):
        """100 battles where ability_L=1 < ability_R=4: left should lose every time."""
        losses = 0
        for seed in range(100):
            key = jax.random.PRNGKey(seed + 1000)
            result = int(battle_outcome(jnp.int32(1), jnp.int32(4), key))
            losses += (result == -1)
        assert losses == 100, \
            f"Lower ability should lose 100/100 times, but lost {losses}/100"

    def test_various_ability_gaps(self):
        """Test multiple ability gaps: (2,1), (3,2), (4,1), (4,3)."""
        pairs = [(2, 1), (3, 2), (4, 1), (4, 3)]
        for a_L, a_R in pairs:
            for seed in range(20):
                key = jax.random.PRNGKey(seed + a_L * 100 + a_R * 10)
                result = int(battle_outcome(jnp.int32(a_L), jnp.int32(a_R), key))
                assert result == 1, \
                    f"ability {a_L} vs {a_R}: expected +1 (left wins), got {result}"


# ---------------------------------------------------------------------------
# Equal ability ~ 50% win rate (stochastic)
# ---------------------------------------------------------------------------

class TestTieRandomness:
    """Equal ability should produce ~50% win rate over many trials."""

    def test_equal_ability_near_50_percent(self):
        """1000 tied battles should give win rate within [0.45, 0.55]."""
        wins = 0
        n = 1000
        for seed in range(n):
            key = jax.random.PRNGKey(seed)
            result = int(battle_outcome(jnp.int32(2), jnp.int32(2), key))
            assert result in (1, -1), f"Tie battle returned {result}, expected +1 or -1"
            wins += (result == 1)
        ratio = wins / n
        assert 0.45 <= ratio <= 0.55, \
            f"Tie win ratio {ratio:.3f} outside [0.45, 0.55] tolerance"

    def test_equal_ability_all_levels(self):
        """Test ties at each ability level 1..4."""
        for ability_val in range(1, 5):
            wins = 0
            n = 200
            for seed in range(n):
                key = jax.random.PRNGKey(seed + ability_val * 1000)
                result = int(battle_outcome(jnp.int32(ability_val), jnp.int32(ability_val), key))
                wins += (result == 1)
            ratio = wins / n
            assert 0.38 <= ratio <= 0.62, \
                f"Tie at ability={ability_val}: ratio {ratio:.3f} outside tolerance"


# ---------------------------------------------------------------------------
# BattleOutcome array population (antisymmetric entry writing)
# ---------------------------------------------------------------------------

class TestBattleOutcomePopulation:
    """Battle outcome correctly populates [i,j,k] and [j,i,k] in the state array."""

    def test_outcome_written_correctly(self):
        """After dominance_battle, outcome[i,j,c] and outcome[j,i,c] are populated."""
        M, N = SMALL.M, SMALL.N_coop
        abilities = jnp.ones((M, N), dtype=jnp.int32)
        abilities = abilities.at[0, 0].set(3)  # agent 0 ability=3 in coop 0
        abilities = abilities.at[1, 0].set(1)  # agent 1 ability=1 in coop 0

        zone = _all_in_battle_zone(M, N)
        pair = jnp.full((M // 2, N, 2), -1, dtype=jnp.int32)
        pair = pair.at[0, 0, :].set(jnp.array([0, 1]))
        occ = jnp.zeros((M // 2, N), dtype=jnp.bool_)
        occ = occ.at[0, 0].set(True)

        state = _make_state(SMALL, abilities=abilities, zone_array=zone,
                            battle_pair=pair, cage_occupied=occ)
        key = jax.random.PRNGKey(42)
        new = dominance_battle(key, state, SMALL)

        # Agent 0 (ability=3) beats agent 1 (ability=1) in coop 0
        assert int(new.battle_outcome[0, 1, 0]) == 1, \
            "outcome[0,1,0] should be +1 (agent 0 wins)"
        assert int(new.battle_outcome[1, 0, 0]) == -1, \
            "outcome[1,0,0] should be -1 (antisymmetric)"

    def test_outcome_zero_for_non_battlers(self):
        """Agents not in a cage should have zero outcome."""
        M, N = SMALL.M, SMALL.N_coop
        zone = _all_in_battle_zone(M, N)
        # No cages occupied
        state = _make_state(SMALL, zone_array=zone)
        key = jax.random.PRNGKey(42)
        new = dominance_battle(key, state, SMALL)

        assert bool((new.battle_outcome == 0).all()), \
            "All outcomes should be zero when no cages are occupied"


# ---------------------------------------------------------------------------
# Truncated Poisson distribution check
# ---------------------------------------------------------------------------

class TestAbilityDistribution:
    """Ability scores should follow the truncated Poisson from the spec."""

    def test_truncated_poisson_distribution(self):
        """Sample 10000 ability scores and check they match the PMF within tolerance."""
        N = 4
        params = EnvParams(M=100, N_coop=N, k=4, T=5)  # M=100 for more samples

        # Collect samples across many resets (100 agents * 100 resets * 4 coops = 40000)
        env = PecKingOrder()
        all_abilities = []
        for seed in range(100):
            key = jax.random.PRNGKey(seed)
            _, _pec = env.reset_env(key, params)
            state = _pec.env
            all_abilities.append(np.array(state.abilities).ravel())  # flatten all M*N

        samples = np.concatenate(all_abilities)  # ~40000 samples
        total = len(samples)

        # Expected PMF
        expected_pmf = np.array(params.ability_pmf)

        # Observed frequencies
        for v in range(1, N + 1):
            observed_freq = (samples == v).sum() / total
            expected_freq = expected_pmf[v - 1]
            assert abs(observed_freq - expected_freq) < 0.03, \
                f"Ability={v}: observed {observed_freq:.4f}, expected {expected_freq:.4f} " \
                f"(diff={abs(observed_freq - expected_freq):.4f} > 0.03 tolerance)"

    def test_all_scores_in_valid_range(self):
        """All sampled scores must be in [1, N_coop]."""
        env = PecKingOrder()
        for seed in range(10):
            key = jax.random.PRNGKey(seed + 5000)
            _, _pec = env.reset_env(key, SMALL)
            state = _pec.env
            assert bool((state.abilities >= 1).all()), f"Ability below 1 at seed {seed}"
            assert bool((state.abilities <= SMALL.N_coop).all()), \
                f"Ability above N_coop at seed {seed}"
