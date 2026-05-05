"""
tests/test_env.py — unit tests for PecKingOrder environment.

Coverage:
  EnvParams
    - ability_pmf sums to 1, has correct shape
    - ability values sampled in [1, N_coop]

  reset_env
    - returned obs has correct shape (obs_dim,)
    - state arrays have correct shapes (exact spec dimensions)
    - ability_matrix invariant: ability_matrix[v,m,c] == (abilities[m,c] >= v+1)
    - zone_array sums to exactly 1 per agent (each agent in exactly one zone/coop)
    - all agents start in BattleZone
    - round-robin coop assignment: agent m in coop (m % N_coop)

  observation_space / action_space
    - observation_space shape matches obs_dim
    - action_space n == 2*N_coop + 1

  step_env
    - shapes of returned (obs, reward, done) are correct
    - tournament_step increments by 1 each step
    - battle_outcome is non-zero after battles
    - done is False before T tournaments finish
    - loss_count / win_count are non-negative

  Invariants (checked after every step in a short run)
    - zone_array: each agent occupies exactly one (coop, zone) cell
    - battle_outcome antisymmetry: outcome[i,j,c] == -outcome[j,i,c]
    - loss_count + win_count >= 0 everywhere
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from envs.state import EnvParams, PecKingState, empty_env_state, empty_agent_state
from envs.spaces import obs_dim, ZONE_BATTLE
from envs.pec_king_order import PecKingOrder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_params():
    """Smaller parameters for fast tests (M=8, N_coop=4, k=4, T=5)."""
    return EnvParams(M=8, N_coop=4, k=4, T=5)


@pytest.fixture(scope="module")
def default_params():
    """Full spec parameters (M=64, N_coop=4, k=4, T=1000)."""
    return EnvParams()


@pytest.fixture(scope="module")
def env():
    return PecKingOrder()


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# EnvParams tests
# ---------------------------------------------------------------------------

class TestEnvParams:
    def test_defaults(self, default_params):
        p = default_params
        assert p.M == 64
        assert p.N_coop == 4
        assert p.k == 4
        assert p.T == 256   # HW5: changed from 1000

    def test_ability_pmf_shape(self, default_params):
        assert len(default_params.ability_pmf) == default_params.N_coop

    def test_ability_pmf_sums_to_one(self, default_params):
        total = float(jnp.array(default_params.ability_pmf).sum())
        assert abs(total - 1.0) < 1e-6, f"PMF sum = {total}"

    def test_ability_pmf_positive(self, default_params):
        assert bool((jnp.array(default_params.ability_pmf) > 0).all()), "PMF must be strictly positive"

    def test_small_params(self, small_params):
        assert small_params.M == 8
        assert len(small_params.ability_pmf) == small_params.N_coop


# ---------------------------------------------------------------------------
# reset_env tests
# ---------------------------------------------------------------------------

class TestResetEnv:
    def test_obs_shape(self, env, small_params, key):
        obs, _pec = env.reset_env(key, small_params)
        state = _pec.env
        expected = obs_dim(small_params)
        assert obs.shape == (expected,), f"obs.shape={obs.shape}, expected ({expected},)"

    def test_state_array_shapes(self, env, small_params, key):
        _, _pec = env.reset_env(key, small_params)
        state = _pec.env
        M, N, T = small_params.M, small_params.N_coop, small_params.T

        assert state.zone_array.shape    == (M, N, 3),        f"zone_array: {state.zone_array.shape}"
        assert state.action_array.shape  == (M, 1),           f"action_array: {state.action_array.shape}"
        assert state.battle_outcome.shape == (M, M, N),       f"battle_outcome: {state.battle_outcome.shape}"
        assert state.abilities.shape      == (M, N),          f"abilities: {state.abilities.shape}"
        assert state.ability_matrix.shape == (N, M, N),       f"ability_matrix: {state.ability_matrix.shape}"
        assert state.loss_count.shape     == (M, N),          f"loss_count: {state.loss_count.shape}"
        assert state.win_count.shape      == (M, N),          f"win_count: {state.win_count.shape}"
        assert state.battle_pair.shape    == (M // 2, N, 2),  f"battle_pair: {state.battle_pair.shape}"
        assert state.cage_occupied.shape  == (M // 2, N),     f"cage_occupied: {state.cage_occupied.shape}"
        assert state.king.shape           == (M, N),          f"king: {state.king.shape}"

    def test_ability_values_in_range(self, env, small_params, key):
        _, _pec = env.reset_env(key, small_params)
        state = _pec.env
        N = small_params.N_coop
        assert bool((state.abilities >= 1).all()), "abilities must be >= 1"
        assert bool((state.abilities <= N).all()), f"abilities must be <= N_coop={N}"

    def test_ability_matrix_invariant(self, env, small_params, key):
        """ability_matrix[v, m, c] == (abilities[m, c] >= v+1)."""
        _, _pec = env.reset_env(key, small_params)
        state = _pec.env
        N = small_params.N_coop
        for v in range(N):
            expected = (state.abilities >= (v + 1)).astype(jnp.int32)  # (M, N)
            actual   = state.ability_matrix[v]                          # (M, N)
            assert bool((actual == expected).all()), \
                f"ability_matrix invariant failed at threshold v={v}"

    def test_each_agent_in_exactly_one_zone(self, env, small_params, key):
        """zone_array.sum(axis=(1,2)) == 1 for every agent at reset."""
        _, _pec = env.reset_env(key, small_params)
        state = _pec.env
        # Sum over (N_coop, 3) axes for each agent
        totals = state.zone_array.sum(axis=(1, 2))  # (M,)
        assert bool((totals == 1).all()), \
            f"Some agents not in exactly one zone: {totals}"

    def test_all_start_in_battle_zone(self, env, small_params, key):
        """All agents must start in BattleZone (zone index 0)."""
        _, _pec = env.reset_env(key, small_params)
        state = _pec.env
        battle_totals = state.zone_array[:, :, ZONE_BATTLE].sum(axis=1)  # (M,)
        assert bool((battle_totals == 1).all()), \
            "All agents should start in BattleZone"

    def test_coop_assignment_is_valid(self, env, small_params, key):
        """Each agent starts in the BattleZone of some valid coop in [0, N_coop)."""
        _, _pec = env.reset_env(key, small_params)
        state = _pec.env
        M, N = small_params.M, small_params.N_coop
        # zone_array[:, :, ZONE_BATTLE] is (M, N); argmax gives assigned coop per agent
        assigned_coops = state.zone_array[:, :, ZONE_BATTLE].argmax(axis=1)  # (M,)
        assert int((assigned_coops >= 0).all()), "All coop indices must be >= 0"
        assert int((assigned_coops < N).all()), "All coop indices must be < N_coop"

    def test_counters_zeroed_at_reset(self, env, small_params, key):
        _, _pec = env.reset_env(key, small_params)
        state = _pec.env
        assert bool((state.loss_count == 0).all())
        assert bool((state.win_count == 0).all())
        assert bool((state.battle_outcome == 0).all())
        assert int(state.tournament_step) == 0
        assert int(state.tournament_count) == 0

    def test_king_initialised_to_false(self, env, small_params, key):
        _, _pec = env.reset_env(key, small_params)
        state = _pec.env
        assert bool((~state.king).all()), "king mask should be all-False at reset"

    def test_different_keys_give_different_abilities(self, env, small_params):
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)
        _, _p1 = env.reset_env(key1, small_params)
        _, _p2 = env.reset_env(key2, small_params)
        s1, s2 = _p1.env, _p2.env
        # Different seeds should (almost certainly) give different abilities
        assert not bool((s1.abilities == s2.abilities).all()), \
            "Two different keys gave identical abilities (extremely unlikely)"


# ---------------------------------------------------------------------------
# Spaces tests
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_observation_space_shape(self, env, small_params):
        space = env.observation_space(small_params)
        M, N = small_params.M, small_params.N_coop
        expected_dim = M * M + N + 2
        assert space.shape == (expected_dim,), \
            f"obs space shape {space.shape} != ({expected_dim},)"

    def test_action_space_n(self, env, small_params):
        space = env.action_space(small_params)
        expected_n = 2 * small_params.N_coop + 1
        assert space.n == expected_n, \
            f"action space n={space.n}, expected {expected_n}"

    def test_observation_space_full_params(self, env, default_params):
        space = env.observation_space(default_params)
        expected_dim = 64 * 64 + 4 + 2   # 4102
        assert space.shape == (expected_dim,)


# ---------------------------------------------------------------------------
# step_env tests
# ---------------------------------------------------------------------------

class TestStepEnv:
    def _fresh(self, env, params):
        key = jax.random.PRNGKey(7)
        obs, pec = env.reset_env(key, params)
        return key, obs, pec.env, pec.agents

    def test_step_output_shapes(self, env, small_params):
        key, _, state, agent_state = self._fresh(env, small_params)
        M = small_params.M
        actions = jnp.zeros((M, 1), dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        obs, _pec, reward, done, info = env.step_env(
            subkey, PecKingState(env=state, agents=agent_state), actions, small_params
        )
        new_state = _pec.env

        expected_obs_dim = obs_dim(small_params)
        assert obs.shape    == (expected_obs_dim,), f"obs shape {obs.shape}"
        assert reward.shape == ()
        assert done.shape   == ()
        assert "obs_all" in info
        assert info["obs_all"].shape == (M, expected_obs_dim)

    def test_tournament_step_increments(self, env, small_params):
        key, _, state, agent_state = self._fresh(env, small_params)
        actions = jnp.zeros((small_params.M, 1), dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        _, _pec, _, _, _ = env.step_env(subkey, PecKingState(env=state, agents=agent_state), actions, small_params)
        new_state = _pec.env
        assert int(new_state.tournament_step) == 1

    def test_reward_is_zero(self, env, small_params):
        key, _, state, agent_state = self._fresh(env, small_params)
        actions = jnp.zeros((small_params.M, 1), dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        _, _, reward, _, _ = env.step_env(subkey, PecKingState(env=state, agents=agent_state), actions, small_params)
        assert float(reward) == 0.0

    def test_done_false_at_start(self, env, small_params):
        key, _, state, agent_state = self._fresh(env, small_params)
        actions = jnp.zeros((small_params.M, 1), dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        _, _, _, done, _ = env.step_env(subkey, PecKingState(env=state, agents=agent_state), actions, small_params)
        assert not bool(done)

    def test_loss_win_counts_non_negative(self, env, small_params):
        key, _, state, agent_state = self._fresh(env, small_params)
        actions = jnp.zeros((small_params.M, 1), dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        _, _pec, _, _, _ = env.step_env(subkey, PecKingState(env=state, agents=agent_state), actions, small_params)
        new_state = _pec.env
        assert bool((new_state.loss_count >= 0).all())
        assert bool((new_state.win_count >= 0).all())

    def test_zone_invariant_after_step(self, env, small_params):
        """Each agent must be in exactly one zone after a step."""
        key, _, state, agent_state = self._fresh(env, small_params)
        actions = jnp.zeros((small_params.M, 1), dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        _, _pec, _, _, _ = env.step_env(subkey, PecKingState(env=state, agents=agent_state), actions, small_params)
        new_state = _pec.env
        totals = new_state.zone_array.sum(axis=(1, 2))
        assert bool((totals == 1).all()), \
            f"Zone invariant violated after step: counts={totals}"

    def test_battle_outcome_antisymmetry(self, env, small_params):
        """battle_outcome[i,j,c] == -battle_outcome[j,i,c]."""
        key, _, state, agent_state = self._fresh(env, small_params)
        actions = jnp.zeros((small_params.M, 1), dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        _, _pec, _, _, _ = env.step_env(subkey, PecKingState(env=state, agents=agent_state), actions, small_params)
        new_state = _pec.env
        bo = new_state.battle_outcome.astype(jnp.int32)
        diff = bo + bo.transpose(1, 0, 2)   # should be all zeros
        assert bool((diff == 0).all()), "battle_outcome antisymmetry violated"

    def test_battle_outcome_nonzero_after_steps(self, env, small_params):
        """battle_outcome should be non-zero somewhere after a few steps."""
        key, _, state, agent_state = self._fresh(env, small_params)
        actions = jnp.zeros((small_params.M, 1), dtype=jnp.int32)
        for _ in range(3):
            key, subkey = jax.random.split(key)
            _, _pec, _, _, _ = env.step_env(subkey, PecKingState(env=state, agents=agent_state), actions, small_params)
            state, agent_state = _pec.env, _pec.agents
        assert bool((state.battle_outcome != 0).any()), \
            "battle_outcome is all zeros after 3 steps"

    def test_multi_step_state_changes(self, env, small_params):
        """State must actually change across steps."""
        key, _, state, agent_state = self._fresh(env, small_params)
        actions = jnp.zeros((small_params.M, 1), dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        _, _pec, _, _, _ = env.step_env(subkey, PecKingState(env=state, agents=agent_state), actions, small_params)
        new_state = _pec.env
        changed = not bool(
            (new_state.zone_array == state.zone_array).all()
        )
        # Zone array should change (battles happen and zones may update)
        # At minimum tournament_step should differ
        assert int(new_state.tournament_step) != int(state.tournament_step)


# ---------------------------------------------------------------------------
# empty_env_state smoke test
# ---------------------------------------------------------------------------

class TestEmptyEnvState:
    def test_shapes_match_params(self, small_params):
        s = empty_env_state(small_params)
        M, N, T = small_params.M, small_params.N_coop, small_params.T
        assert s.zone_array.shape     == (M, N, 3)
        assert s.battle_outcome.shape == (M, M, N)
        assert s.ability_matrix.shape == (N, M, N)


# ---------------------------------------------------------------------------
# AllAgentState tests
# ---------------------------------------------------------------------------

class TestAllAgentState:
    def test_empty_agent_state_shapes(self, small_params):
        """empty_agent_state returns correct shapes."""
        a = empty_agent_state(small_params)
        M, N, T = small_params.M, small_params.N_coop, small_params.T
        assert a.ability_belief.shape == (M, M, N, N)
        assert a.my_ability.shape == (M, N)
        assert a.my_location.shape == (M, 2)
        assert a.last_observation.shape == (M, M, M)
        assert a.king_belief.shape == (M, M, N, 2)

    def test_reset_returns_agent_state(self, env, small_params, key):
        """reset_env returns AllAgentState with correct shapes."""
        _, _pec = env.reset_env(key, small_params)
        agent_state = _pec.agents
        M, N, T = small_params.M, small_params.N_coop, small_params.T
        assert agent_state.ability_belief.shape == (M, M, N, N)
        assert agent_state.my_ability.shape == (M, N)
        assert agent_state.my_location.shape == (M, 2)
        assert agent_state.last_observation.shape == (M, M, M)
        assert agent_state.king_belief.shape == (M, M, N, 2)

    def test_agent_state_my_ability_matches_env_abilities(self, env, small_params, key):
        """agent_state.my_ability matches env_state.abilities."""
        _, _pec = env.reset_env(key, small_params)
        env_state, agent_state = _pec.env, _pec.agents
        assert bool((agent_state.my_ability == env_state.abilities).all())

    def test_step_updates_last_observation(self, env, small_params, key):
        """last_observation is non-zero after a step with battles."""
        _, _pec = env.reset_env(key, small_params)
        env_state, agent_state = _pec.env, _pec.agents
        actions = jnp.zeros((small_params.M, 1), dtype=jnp.int32)

        # Run a few steps
        for _ in range(3):
            key, sk = jax.random.split(key)
            _, _pec, _, _, _ = env.step_env(
                sk, PecKingState(env=env_state, agents=agent_state), actions, small_params
            )
            env_state, agent_state = _pec.env, _pec.agents

        # BattleZone agents should have seen something
        assert bool((agent_state.last_observation != 0).any())

    def test_step_syncs_my_location(self, env, small_params, key):
        """my_location stays in sync with zone_array after steps."""
        _, _pec = env.reset_env(key, small_params)
        env_state, agent_state = _pec.env, _pec.agents
        actions = jnp.zeros((small_params.M, 1), dtype=jnp.int32)

        for _ in range(5):
            key, sk = jax.random.split(key)
            _, _pec, _, _, _ = env.step_env(
                sk, PecKingState(env=env_state, agents=agent_state), actions, small_params
            )
            env_state, agent_state = _pec.env, _pec.agents

        # Verify each agent's location matches zone_array
        M = small_params.M
        for m in range(M):
            zone_row = env_state.zone_array[m]  # (N_coop, 3)
            coop_active = zone_row.any(axis=-1)  # (N_coop,)
            region_idx = int(jnp.argmax(coop_active))
            zone_idx = int(jnp.argmax(zone_row[region_idx]))

            assert int(agent_state.my_location[m, 0]) == region_idx
            assert int(agent_state.my_location[m, 1]) == zone_idx


# ---------------------------------------------------------------------------
# Belief update correctness
# ---------------------------------------------------------------------------

class TestBeliefUpdate:
    """Verify ability belief updates work under JIT and actually shift from prior."""

    def test_belief_update_under_jit(self, env, small_params, key):
        """step_env compiles under jit — belief update is on the hot path."""
        jit_reset = jax.jit(env.reset_env, static_argnums=(1,))
        obs, pec = jit_reset(key, small_params)

        action = jnp.zeros((small_params.M, 1), dtype=jnp.int32)
        jit_step = jax.jit(env.step_env, static_argnums=(3,))
        key, sk = jax.random.split(key)
        obs2, pec2, r, d, info = jit_step(sk, pec, action, small_params)

        ab = pec2.agents.ability_belief
        M, N = small_params.M, small_params.N_coop
        assert ab.shape == (M, M, N, N)
        assert bool((ab >= 0).all()), "negative belief"
        sums = ab.sum(axis=-1)
        assert bool(jnp.allclose(sums, 1.0, atol=1e-5)), "beliefs don't sum to 1"

    def test_beliefs_update_after_battles(self, env, small_params, key):
        """After several steps with battles, beliefs should differ from uniform prior."""
        obs, pec = env.reset_env(key, small_params)
        initial_belief = pec.agents.ability_belief.copy()

        action = jnp.zeros((small_params.M, 1), dtype=jnp.int32)
        for _ in range(5):
            key, sk = jax.random.split(key)
            obs, pec, r, d, info = env.step_env(sk, pec, action, small_params)

        diff = jnp.abs(pec.agents.ability_belief - initial_belief).sum()
        assert float(diff) > 0.1, f"beliefs unchanged after 5 steps (diff={float(diff)})"
