"""
Tests for HW5 sack mechanics and social actions.

Tests EnvState.sack / sack_owners initialization, crown accumulation,
and all four social action functions from events/social.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from envs.pec_king_order import PecKingOrder
from envs.state import EnvParams, PecKingState
from events.social import (
    share_observations,
    transfer_crowns,
    extend_sack_ownership,
    compute_final_scores,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SMALL = EnvParams(M=8, N_coop=4, k=4, T=10)
KEY   = jax.random.PRNGKey(99)


@pytest.fixture(scope="module")
def env():
    return PecKingOrder()


@pytest.fixture(scope="module")
def reset_state(env):
    _, pec = env.reset_env(KEY, SMALL)
    return pec.env


@pytest.fixture(scope="module")
def post_tournament_state(env):
    """Run enough steps that at least one tournament closes."""
    key = jax.random.PRNGKey(7)
    _, pec = env.reset_env(key, SMALL)
    state, agent_state = pec.env, pec.agents
    step_jit = jax.jit(env.step_env, static_argnums=(3,))
    M = SMALL.M

    for _ in range(500):
        key, sk = jax.random.split(key)
        actions = jnp.ones((M, 1), dtype=jnp.int32)
        _, pec, _, _, info = step_jit(sk, PecKingState(env=state, agents=agent_state), actions, SMALL)
        state, agent_state = pec.env, pec.agents
        if int(state.tournament_count) >= 1:
            break

    return state


# ---------------------------------------------------------------------------
# TestSackState
# ---------------------------------------------------------------------------

class TestSackState:

    def test_sack_initialized_to_zero(self, reset_state):
        """After reset_env, state.sack is all zeros with shape (M,)."""
        sack = np.array(reset_state.sack)
        assert sack.shape == (SMALL.M,), f"Expected shape ({SMALL.M},), got {sack.shape}"
        assert (sack == 0).all(), f"Expected all zeros, got {sack}"

    def test_sack_owners_initialized_as_identity(self, reset_state):
        """After reset_env, sack_owners is the identity matrix with shape (M, M)."""
        owners = np.array(reset_state.sack_owners)
        M = SMALL.M
        assert owners.shape == (M, M), f"Expected shape ({M}, {M}), got {owners.shape}"
        expected = np.eye(M, dtype=bool)
        assert (owners == expected).all(), "sack_owners is not identity at reset"

    def test_sack_accumulates_after_tournament(self, post_tournament_state):
        """After at least one tournament closes, some crowns must be in sacks."""
        sack = np.array(post_tournament_state.sack)
        assert sack.sum() > 0, f"Expected crowns in sacks after tournament, got {sack}"

    def test_sack_never_negative(self, post_tournament_state):
        """Sack values must always be >= 0."""
        sack = np.array(post_tournament_state.sack)
        assert (sack >= 0).all(), f"Negative sack values found: {sack}"


# ---------------------------------------------------------------------------
# TestSocialActions
# ---------------------------------------------------------------------------

class TestSocialActions:

    def test_transfer_crowns_basic(self):
        """Transfer 5 crowns from chicken 0 (sack=10) to chicken 1 (sack=0)."""
        sack = np.array([10, 0, 0], dtype=np.int32)
        transfer_crowns(0, 1, 5, sack)
        assert sack[0] == 5, f"Expected sack[0]=5, got {sack[0]}"
        assert sack[1] == 5, f"Expected sack[1]=5, got {sack[1]}"

    def test_transfer_crowns_clamp(self):
        """Transfer clamped to available balance: can't send more than sack holds."""
        sack = np.array([3, 0, 0], dtype=np.int32)
        transfer_crowns(0, 1, 100, sack)
        assert sack[0] == 0, f"Expected sack[0]=0, got {sack[0]}"
        assert sack[1] == 3, f"Expected sack[1]=3, got {sack[1]}"

    def test_transfer_crowns_to_self_is_noop(self):
        """Transferring crowns from a chicken to itself leaves sack unchanged."""
        sack = np.array([10, 5, 0], dtype=np.int32)
        transfer_crowns(0, 0, 5, sack)
        assert sack[0] == 10, f"Expected sack[0]=10 (unchanged), got {sack[0]}"

    def test_extend_sack_ownership(self):
        """extend_sack_ownership(0, 1) sets sack_owners[0, 1] = True."""
        M = 4
        owners = np.eye(M, dtype=bool)
        extend_sack_ownership(0, 1, owners)
        assert owners[0, 1] == True, "sack_owners[0,1] should be True after extend"

    def test_extend_sack_ownership_does_not_add_reverse(self):
        """Extending ownership is asymmetric: sack_owners[1, 0] stays False."""
        M = 4
        owners = np.eye(M, dtype=bool)
        extend_sack_ownership(0, 1, owners)
        assert owners[1, 0] == False, "sack_owners[1,0] should remain False (asymmetric)"

    def test_share_observations_copies_nonzero(self):
        """Non-zero entries from src are copied into tgt where tgt has no data."""
        M, T = 4, 10
        obs = np.zeros((M, M, M, T), dtype=np.int16)
        obs[0, 1, 2, 3] = 1   # src (chicken 0) observed outcome between 1 and 2 at t=3
        share_observations(0, 2, obs)
        assert obs[2, 1, 2, 3] == 1, "tgt should have received src's non-zero observation"

    def test_share_observations_does_not_overwrite(self):
        """Existing non-zero entries in tgt are not overwritten by share."""
        M, T = 4, 10
        obs = np.zeros((M, M, M, T), dtype=np.int16)
        obs[0, 1, 2, 3] = 1    # src has +1 for (1,2) at t=3
        obs[2, 1, 2, 3] = -1   # tgt already has -1 for same slot
        share_observations(0, 2, obs)
        assert obs[2, 1, 2, 3] == -1, "tgt's existing observation should not be overwritten"


# ---------------------------------------------------------------------------
# TestFinalScores
# ---------------------------------------------------------------------------

class TestFinalScores:

    def test_solo_owner_gets_full_sack(self):
        """With identity ownership each chicken's score equals its sack."""
        sack = np.array([10, 20], dtype=np.int32)
        owners = np.eye(2, dtype=bool)
        scores = compute_final_scores(sack, owners)
        assert abs(scores[0] - 10.0) < 1e-9, f"Expected 10.0, got {scores[0]}"
        assert abs(scores[1] - 20.0) < 1e-9, f"Expected 20.0, got {scores[1]}"

    def test_shared_sack_splits_equally(self):
        """Chicken 0 sack=120 shared among owners 0,1,2 gives 40 each."""
        M = 3
        sack = np.array([120, 0, 0], dtype=np.int32)
        owners = np.eye(M, dtype=bool)
        owners[0, 1] = True
        owners[0, 2] = True
        scores = compute_final_scores(sack, owners)
        for j in range(M):
            assert abs(scores[j] - 40.0) < 1e-9, f"Expected 40.0 for chicken {j}, got {scores[j]}"

    def test_example_from_spec(self):
        """Replicate the spec example exactly.

        Chicken0 sack=120, owners=[0,1,2]  → contributes 120/3 = 40 to each
        Chicken1 sack=60,  owners=[0,1]    → contributes 60/2  = 30 to each
        Chicken2 sack=40,  owners=[0,2]    → contributes 40/2  = 20 to each

        Final scores:
          C0: 120/3 + 60/2 + 40/2 = 40 + 30 + 20 = 90
          C1: 120/3 + 60/2        = 40 + 30       = 70
          C2: 120/3 + 40/2        = 40 + 20       = 60
        """
        M = 3
        sack = np.array([120, 60, 40], dtype=np.int32)
        owners = np.eye(M, dtype=bool)
        # Chicken0's sack shared with C1 and C2
        owners[0, 1] = True
        owners[0, 2] = True
        # Chicken1's sack shared with C0
        owners[1, 0] = True
        # Chicken2's sack shared with C0
        owners[2, 0] = True

        scores = compute_final_scores(sack, owners)
        assert abs(scores[0] - 90.0) < 1e-9, f"C0: expected 90.0, got {scores[0]}"
        assert abs(scores[1] - 70.0) < 1e-9, f"C1: expected 70.0, got {scores[1]}"
        assert abs(scores[2] - 60.0) < 1e-9, f"C2: expected 60.0, got {scores[2]}"
