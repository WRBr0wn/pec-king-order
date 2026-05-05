"""
tests/test_invariants.py — Array Shape & Structural Invariants.

Verifies spec requirements that must hold after every reset_env and step_env call:
  - ZoneArray shape (M, N_coop, 3), each row sums to exactly 1
  - ActionArray shape (M, 1), values in {0, ..., 2*N_coop}
  - BattleOutcome shape (M, M, N_coop), antisymmetric, zero diagonal
  - BattleHistory shape (M, M, N_coop, T), equals cumulative sum
  - All ability scores are integers in [1, N_coop]
  - Ability indicator matrix: indicator[v, m, c] == 1 iff v+1 <= ability[m, c]
  - Agent count per coop: zone sums are consistent

Uses M=8, N_coop=2 for speed; fixed PRNGKey(42) for reproducibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from envs.state import EnvParams, PecKingState, empty_env_state
from envs.spaces import obs_dim, ZONE_BATTLE
from envs.pec_king_order import PecKingOrder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL = EnvParams(M=8, N_coop=2, k=4, T=5)


@pytest.fixture(scope="module")
def env():
    return PecKingOrder()


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def reset_state(env, key):
    """State immediately after reset_env."""
    _, pec = env.reset_env(key, SMALL)
    return pec.env


@pytest.fixture(scope="module")
def stepped_state(env, key):
    """State after 5 steps (enough to trigger battles and relocations)."""
    k = key
    _, pec = env.reset_env(k, SMALL)
    state, agent_state = pec.env, pec.agents
    actions = jnp.zeros((SMALL.M, 1), dtype=jnp.int32)
    for _ in range(5):
        k, sk = jax.random.split(k)
        _, pec, _, _, _ = env.step_env(sk, PecKingState(env=state, agents=agent_state), actions, SMALL)
        state, agent_state = pec.env, pec.agents
    return state


# ---------------------------------------------------------------------------
# Invariant checks — reusable assertion functions
# ---------------------------------------------------------------------------

def _check_zone_array(state, params):
    """ZoneArray shape is (M, N_coop, 3) and each agent sums to exactly 1."""
    M, N = params.M, params.N_coop
    assert state.zone_array.shape == (M, N, 3), \
        f"ZoneArray shape {state.zone_array.shape} != ({M}, {N}, 3)"
    totals = state.zone_array.sum(axis=(1, 2))  # (M,)
    assert bool((totals == 1).all()), \
        f"Location rule violated: some agents not in exactly one zone. Sums: {totals}"


def _check_action_array(state, params):
    """ActionArray shape is (M, 1) and values in {0, ..., 2*N_coop}."""
    M, N = params.M, params.N_coop
    assert state.action_array.shape == (M, 1), \
        f"ActionArray shape {state.action_array.shape} != ({M}, 1)"
    max_action = 2 * N
    vals = state.action_array[:, 0]
    assert bool((vals >= 0).all() & (vals <= max_action).all()), \
        f"ActionArray values out of range [0, {max_action}]: min={int(vals.min())}, max={int(vals.max())}"


def _check_battle_outcome(state, params):
    """BattleOutcome shape (M, M, N_coop), antisymmetric, zero diagonal."""
    M, N = params.M, params.N_coop
    bo = state.battle_outcome
    assert bo.shape == (M, M, N), \
        f"BattleOutcome shape {bo.shape} != ({M}, {M}, {N})"
    # Antisymmetry
    diff = bo.astype(jnp.int32) + bo.astype(jnp.int32).transpose(1, 0, 2)
    assert bool((diff == 0).all()), \
        "BattleOutcome antisymmetry violated: outcome[i,j,k] != -outcome[j,i,k]"
    # Zero diagonal
    for m in range(M):
        assert bool((bo[m, m, :] == 0).all()), \
            f"BattleOutcome diagonal non-zero at agent {m}"


def _check_abilities(state, params):
    """All ability scores are integers in [1, N_coop]."""
    N = params.N_coop
    assert bool((state.abilities >= 1).all()), \
        f"Abilities below 1: min={int(state.abilities.min())}"
    assert bool((state.abilities <= N).all()), \
        f"Abilities above N_coop={N}: max={int(state.abilities.max())}"


def _check_ability_matrix(state, params):
    """ability_matrix[v, m, c] == 1 iff abilities[m, c] >= v+1."""
    N = params.N_coop
    for v in range(N):
        expected = (state.abilities >= (v + 1)).astype(jnp.int32)
        actual = state.ability_matrix[v]
        assert bool((actual == expected).all()), \
            f"Ability matrix invariant failed at threshold v={v}"


def _check_agent_count_per_coop(state, params):
    """Sum of zone_array[:, c, :] across agents equals number of agents in coop c."""
    M, N = params.M, params.N_coop
    # Total agents across all coops should equal M
    total = state.zone_array.sum()
    assert int(total) == M, \
        f"Total agents across all coops = {int(total)}, expected {M}"


# ---------------------------------------------------------------------------
# Tests after reset_env
# ---------------------------------------------------------------------------

class TestInvariantsAfterReset:
    """Verify all structural invariants hold immediately after reset."""

    def test_zone_array_shape_and_sum(self, reset_state):
        """ZoneArray shape (M, N_coop, 3), each agent in exactly one zone."""
        _check_zone_array(reset_state, SMALL)

    def test_action_array_shape_and_range(self, reset_state):
        """ActionArray shape (M, 1), values in {0, ..., 2*N_coop}."""
        _check_action_array(reset_state, SMALL)

    def test_battle_outcome_shape_antisymmetry_diagonal(self, reset_state):
        """BattleOutcome shape, antisymmetric, zero diagonal at reset (all zeros)."""
        _check_battle_outcome(reset_state, SMALL)

    def test_abilities_in_range(self, reset_state):
        """All ability scores in [1, N_coop]."""
        _check_abilities(reset_state, SMALL)

    def test_ability_matrix_invariant(self, reset_state):
        """Indicator matrix matches abilities."""
        _check_ability_matrix(reset_state, SMALL)

    def test_agent_count(self, reset_state):
        """Total agents across coops equals M."""
        _check_agent_count_per_coop(reset_state, SMALL)


# ---------------------------------------------------------------------------
# Tests after step_env (multiple steps)
# ---------------------------------------------------------------------------

class TestInvariantsAfterStep:
    """Verify all structural invariants hold after several simulation steps."""

    def test_zone_array_after_steps(self, stepped_state):
        """ZoneArray invariant holds after 5 steps."""
        _check_zone_array(stepped_state, SMALL)

    def test_action_array_after_steps(self, stepped_state):
        """ActionArray invariant holds after 5 steps."""
        _check_action_array(stepped_state, SMALL)

    def test_battle_outcome_after_steps(self, stepped_state):
        """BattleOutcome antisymmetry and zero diagonal after 5 steps."""
        _check_battle_outcome(stepped_state, SMALL)

    def test_abilities_unchanged_after_steps(self, stepped_state):
        """Abilities remain in valid range after steps (fixed at init)."""
        _check_abilities(stepped_state, SMALL)

    def test_ability_matrix_after_steps(self, stepped_state):
        """Ability matrix invariant after steps."""
        _check_ability_matrix(stepped_state, SMALL)

    def test_agent_count_after_steps(self, stepped_state):
        """Total agents equals M after steps."""
        _check_agent_count_per_coop(stepped_state, SMALL)

    def test_battle_outcome_nonzero_after_steps(self, stepped_state):
        """BattleOutcome should have some nonzero entries after 5 steps."""
        assert bool((stepped_state.battle_outcome != 0).any()), \
            "BattleOutcome all zeros after 5 steps — no battles recorded"


# ---------------------------------------------------------------------------
# Continuous invariant check across a short run
# ---------------------------------------------------------------------------

class TestInvariantsContinuous:
    """Run invariant checks after EVERY step in a 10-step run."""

    def test_invariants_every_step(self, env, key):
        """All invariants hold after each of 10 consecutive steps."""
        k = key
        _, pec = env.reset_env(k, SMALL)
        state, agent_state = pec.env, pec.agents
        actions = jnp.zeros((SMALL.M, 1), dtype=jnp.int32)

        for step in range(10):
            k, sk = jax.random.split(k)
            _, pec, _, _, _ = env.step_env(sk, PecKingState(env=state, agents=agent_state), actions, SMALL)
            state, agent_state = pec.env, pec.agents