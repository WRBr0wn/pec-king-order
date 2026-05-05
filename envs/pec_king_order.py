"""
PecKingOrder — gymnax Environment class for the Pec-King Order domain.

Required interface (per spec / gymnax convention):
  step_env(key, state, action, params)  -> (obs, state, reward, done, info)
  reset_env(key, params)                -> (obs, state)
  observation_space(params)             -> spaces.Box
  action_space(params)                  -> spaces.Discrete

Simulation loop (per spec):
  env   = PecKingOrder()
  params = EnvParams(M=64, N_coop=4, k=4)
  key, state, obs = env.reset(key, params)

  for tournament in range(1000):
      while not tournament_done:
          actions = action_array          # M × 1 int32
          obs, state, reward, done, info = env.step(key, state, actions, params)
      # close_tournament + new_tournament handled automatically inside step

Naming conventions (per spec):
  CamelCase for all classes; snake_case for all functions inside a class.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import chex
from gymnax.environments import environment, spaces

from envs.state import (
    EnvState,
    EnvParams,
    AllAgentState,
    PecKingState,
    empty_env_state,
    empty_agent_state,
    default_ability_pmf,
)
from envs.spaces import (
    observation_space  as _observation_space,
    action_space       as _action_space,
    make_observation,
    ZONE_BATTLE,
    ZONE_SPECTATOR,
)
from events.assign_agents    import assign_agents_to_cage, relocate_zone
from events.dominance_battle import dominance_battle
from events.region_view      import region_view
from events.relocate         import relocate_region, arrive_from_transit
from events.tournament       import (
    tournament_is_done,
    close_tournament,
    new_tournament,
)
from inference.domain_inference import (
    incremental_ability_update,
    derive_king_belief_from_ability,
)


class PecKingOrder(environment.Environment):
    """Pec-King Order multi-agent dominance tournament environment.

    Implements the gymnax Environment interface. All state transitions are
    pure functions; state is never mutated in place.

    Step order inside step_env:
      0. Check tournament_is_done (spec: "assessed at beginning of a round")
           If done: close_tournament -> new_tournament
      1. Write action_array into state
      2. Assign BattleZone eligible chickens to cages
      3. Relocate unpaired BattleZone chickens -> SpectatorZone
      4. SpectatorZone a_move -> TransitZone (runs after relocate_zone
         so newly-unpaired chickens can depart in the same round)
      5. Concurrent:
           a. dominance_battle  — all cage fights
           b. region_view       — compute per-chicken observations
           c. record observed_coop per agent
      6. entity_update:
           a. arrive_from_transit  — TransitZone -> BattleZone/SpectatorZone
           b. increment tournament_step
      7. Build per-agent observation vectors
      8. Return (obs, state, reward, done, info)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # gymnax interface
    # ------------------------------------------------------------------

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters (required by gymnax Environment ABC)."""
        return EnvParams()

    # ------------------------------------------------------------------
    # reset_env
    # ------------------------------------------------------------------

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> tuple[chex.Array, PecKingState]:
        """Sample abilities, assign chickens uniformly at random, return (obs, PecKingState)."""
        M, N_coop, T = params.M, params.N_coop, params.T

        # --- Ensure ability_pmf matches N_coop ---
        ability_pmf = jnp.array(params.ability_pmf)
        if len(params.ability_pmf) != N_coop:
            ability_pmf = default_ability_pmf(N_coop)

        # --- Sample ability scores ---
        key, subkey = jax.random.split(key)
        # Sample M × N_coop independent draws from the truncated-Poisson PMF.
        # ability values are in {1, ..., N_coop} → add 1 after sampling index.
        ability_idx = jax.random.choice(
            subkey,
            a=N_coop,
            shape=(M, N_coop),
            replace=True,
            p=ability_pmf,
        )  # int32, shape (M, N_coop), values in [0, N_coop)
        abilities = (ability_idx + 1).astype(jnp.int32)  # values in [1, N_coop]

        # --- Build indicator ability_matrix ---
        # ability_matrix[v, m, c] = 1  iff  abilities[m, c] >= (v+1)
        # v in {0, ..., N_coop-1}  (threshold = v+1)
        thresholds    = jnp.arange(1, N_coop + 1, dtype=jnp.int32)  # (N_coop,)
        # Broadcast: (N_coop, 1, 1) vs (1, M, N_coop)
        ability_matrix = (
            abilities[None, :, :] >= thresholds[:, None, None]
        ).astype(jnp.int32)  # (N_coop, M, N_coop)

        # --- Initial coop assignments: uniform random ---
        key, subkey = jax.random.split(key)
        initial_coops = jax.random.randint(subkey, shape=(M,), minval=0, maxval=N_coop)  # (M,)

        # --- Build zone_array: all in BattleZone of their assigned coop ---
        zone_array = jnp.zeros((M, N_coop, 3), dtype=jnp.int32)
        zone_array = zone_array.at[
            jnp.arange(M),
            initial_coops,
            ZONE_BATTLE,
        ].set(jnp.int32(1))

        # --- Assemble EnvState ---
        env_state = EnvState(
            zone_array=zone_array,
            action_array=jnp.zeros((M, 1), dtype=jnp.int32),
            battle_outcome=jnp.zeros((M, M, N_coop), dtype=jnp.int8),
            abilities=abilities,
            ability_matrix=ability_matrix,
            loss_count=jnp.zeros((M, N_coop), dtype=jnp.int32),
            win_count=jnp.zeros((M, N_coop), dtype=jnp.int32),
            cumulative_battles=jnp.zeros((M, N_coop), dtype=jnp.int32),
            battle_pair=jnp.full((M // 2, N_coop, 2), -1, dtype=jnp.int32),
            cage_occupied=jnp.zeros((M // 2, N_coop), dtype=jnp.bool_),
            has_battled=jnp.zeros((M, M, N_coop), dtype=jnp.bool_),
            king=jnp.zeros((M, N_coop), dtype=jnp.bool_),
            sack=jnp.zeros((M,), dtype=jnp.int32),
            sack_owners=jnp.eye(M, dtype=jnp.bool_),
            tournament_step=jnp.int32(0),
            tournament_count=jnp.int32(0),
            done=jnp.bool_(False),
        )

        # --- Assemble AllAgentState with uniform priors ---
        # my_location: [region_idx, zone_idx] — all start in BattleZone (zone=0)
        my_location = jnp.stack([initial_coops, jnp.zeros(M, dtype=jnp.int32)], axis=1)

        agent_state = AllAgentState(
            ability_belief=jnp.ones((M, M, N_coop, N_coop), dtype=jnp.float32) / N_coop,
            my_ability=abilities,  # each agent knows its own abilities
            my_location=my_location,
            last_observation=jnp.zeros((M, M, M), dtype=jnp.int8),
            king_belief=jnp.ones((M, M, N_coop, 2), dtype=jnp.float32) * 0.5,
        )

        # --- Build observation for agent 0 (gymnax convention) ---
        obs = self._make_agent_obs(agent_idx=0, state=env_state, params=params)

        return obs, PecKingState(env=env_state, agents=agent_state)

    # ------------------------------------------------------------------
    # step_env
    # ------------------------------------------------------------------

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PecKingState,
        action: chex.Array,   # int32, shape (M, 1)
        params: EnvParams,
    ) -> tuple[chex.Array, PecKingState, chex.Array, chex.Array, dict]:
        """Advance one round: pair, battle, observe, update beliefs, maybe close tournament.

        Returns (obs, state, reward=0.0, done, info). info has obs_all, king_masks, etc.
        """
        env_state   = state.env
        agent_state = state.agents
        M, N_coop = params.M, params.N_coop
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        # 0. Check if tournament is over (spec: "assessed at beginning of a round")
        t_done = tournament_is_done(env_state, params)
        info: dict[str, Any] = {"tournament_done": t_done}
        env_state, agent_state, info = self._maybe_close_and_reopen(
            k3, env_state, agent_state, params, t_done, info
        )

        # 1. Write actions into state
        env_state = env_state.replace(action_array=action.reshape(M, 1))

        # 2. Assign eligible BattleZone chickens to cages (spec step 1)
        env_state = assign_agents_to_cage(k1, env_state, params)

        # 3. Unpaired BattleZone -> SpectatorZone (spec step 1 cont.)
        env_state = relocate_zone(env_state, params)

        # 4. SpectatorZone a_move -> TransitZone (spec step 2-3)
        env_state = relocate_region(env_state, params)

        # 5a. Execute all cage battles (spec step 3 — concurrent)
        env_state = dominance_battle(k2, env_state, params)

        # 5b. Compute per-chicken observations from this round's outcomes
        obs_received = region_view(env_state, params)  # (M, M, M) int8

        # 5c. Record which coop each agent observed this round
        observed_coop = self._compute_observed_coop(env_state, params)  # (M,) int32

        # 5d. Update agent_state with new observations
        agent_state = self._update_agent_observations(agent_state, obs_received)

        # 5e. Update ability beliefs incrementally from this round's battles
        agent_state = self._update_beliefs_from_observation(
            agent_state, env_state, observed_coop, params
        )

        # 6a. Arrive from TransitZone -> BattleZone or SpectatorZone
        env_state = arrive_from_transit(env_state, params)

        # 6b. Increment round counter
        env_state = env_state.replace(
            tournament_step=env_state.tournament_step + jnp.int32(1)
        )

        # 6c. Sync agent_state.my_location with zone_array
        agent_state = self._sync_agent_locations(agent_state, env_state, params)

        # 7. Build per-agent observation vectors
        obs_all = self._make_all_obs(obs_received, env_state, params)  # (M, obs_dim)
        info["obs_all"] = obs_all
        info["obs_received"] = obs_received
        info["observed_coop"] = observed_coop

        obs = obs_all[0]
        reward = jnp.float32(0.0)

        return obs, PecKingState(env=env_state, agents=agent_state), reward, env_state.done, info

    # ------------------------------------------------------------------
    # Space descriptors
    # ------------------------------------------------------------------

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Fixed-size per-agent observation Box.

        Shape: (M*M + N_coop + 2,)  =  (4102,) for default params.
        """
        return _observation_space(params)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Discrete action space: 0=no-op, 1..N=a_watch, N+1..2N=a_move.

        n = 2*N_coop + 1  =  9 for default params.
        """
        return _action_space(params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_agent_obs(
        self,
        agent_idx: int | chex.Array,
        state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Build the flat observation vector for a single agent.

        Uses zeros for last_battle_obs (called at reset; no battles yet).
        """
        M, N_coop = params.M, params.N_coop
        last_obs    = jnp.zeros((M, M), dtype=jnp.int8)
        my_ability  = state.abilities[agent_idx]              # (N_coop,)
        # Determine current location from zone_array
        zone_row    = state.zone_array[agent_idx]             # (N_coop, 3)
        # region_idx: which coop has any zone bit set
        coop_active = zone_row.any(axis=-1)                   # (N_coop,) bool
        region_idx  = jnp.argmax(coop_active).astype(jnp.int32)
        # zone_idx: which zone within that coop
        zone_idx    = jnp.argmax(zone_row[region_idx]).astype(jnp.int32)
        my_location = jnp.stack([region_idx, zone_idx])       # (2,)

        return make_observation(last_obs, my_ability, my_location)

    def _make_all_obs(
        self,
        obs_received: chex.Array,   # int8, (M, M, M)
        state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Build flat observation vectors for all M agents.

        Vectorized: computes all M observation vectors with batched array ops
        instead of a per-agent Python loop.

        Returns float32, shape (M, obs_dim).
        """
        M, N_coop = params.M, params.N_coop

        # Flatten per-agent battle observations: (M, M, M) -> (M, M*M)
        flat_obs = obs_received.reshape(M, M * M).astype(jnp.float32)

        # Abilities: (M, N_coop) -> float32
        abilities = state.abilities.astype(jnp.float32)

        # Location from zone_array for all agents at once
        zone = state.zone_array                                      # (M, N_coop, 3)
        coop_active = zone.any(axis=-1)                              # (M, N_coop) bool
        region_idx = jnp.argmax(coop_active, axis=1)                 # (M,)
        zone_per_agent = zone[jnp.arange(M), region_idx]             # (M, 3)
        zone_idx = jnp.argmax(zone_per_agent, axis=1)                # (M,)
        locations = jnp.stack([region_idx, zone_idx], axis=1).astype(jnp.float32)  # (M, 2)

        return jnp.concatenate([flat_obs, abilities, locations], axis=1)  # (M, obs_dim)

    def _compute_observed_coop(
        self,
        state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Return which coop each agent observed this round.

        Must be called AFTER region_view and BEFORE arrive_from_transit,
        when zone_array still reflects the positions during observation.

        Returns int32, shape (M,).  -1 means no observation.
        """
        M, N_coop = params.M, params.N_coop
        action_flat = state.action_array[:, 0]  # (M,) int32

        # BattleZone agents observe their own coop unconditionally
        in_bz = state.zone_array[:, :, ZONE_BATTLE].astype(jnp.bool_)  # (M, N_coop)
        has_bz = in_bz.any(axis=1)  # (M,)
        bz_coop = jnp.argmax(in_bz.astype(jnp.int32), axis=1)  # (M,)

        # SpectatorZone agents with a_watch observe their chosen coop
        in_sz = state.zone_array[:, :, ZONE_SPECTATOR].astype(jnp.bool_)  # (M, N_coop)
        has_sz = in_sz.any(axis=1)  # (M,)
        is_watch = (action_flat >= 1) & (action_flat <= N_coop)
        watch_target = action_flat - 1  # coop 0..N_coop-1 (valid only when is_watch)
        sz_watches = has_sz & is_watch  # (M,)

        # BattleZone takes priority; then SpectatorZone a_watch; else -1
        return jnp.where(
            has_bz, bz_coop,
            jnp.where(sz_watches, watch_target, jnp.int32(-1))
        )

    def _update_agent_observations(
        self,
        agent_state: AllAgentState,
        obs_received: chex.Array,   # int8, (M, M, M)
    ) -> AllAgentState:
        """Store this round's observations as last_observation for all agents.

        ObservationMemory (M, M, M, T) is accumulated outside the JIT loop
        in run.py to avoid carrying 500 MB of dead-write state through step_env.
        """
        return agent_state.replace(last_observation=obs_received)

    def _sync_agent_locations(
        self,
        agent_state: AllAgentState,
        env_state: EnvState,
        params: EnvParams,
    ) -> AllAgentState:
        """Sync agent_state.my_location with env_state.zone_array.

        Called AFTER arrive_from_transit so locations reflect final positions.
        """
        M = params.M
        zone = env_state.zone_array  # (M, N_coop, 3)

        # For each agent, find which (coop, zone) they occupy
        coop_active = zone.any(axis=-1)  # (M, N_coop) bool
        region_idx = jnp.argmax(coop_active, axis=1)  # (M,)
        zone_per_agent = zone[jnp.arange(M), region_idx]  # (M, 3)
        zone_idx = jnp.argmax(zone_per_agent, axis=1)  # (M,)

        new_location = jnp.stack([region_idx, zone_idx], axis=1).astype(jnp.int32)

        return agent_state.replace(my_location=new_location)

    def _update_beliefs_from_observation(
        self,
        agent_state: AllAgentState,
        env_state: EnvState,
        observed_coop: chex.Array,  # int32 (M,) — which coop each agent observed (-1 if none)
        params: EnvParams,
    ) -> AllAgentState:
        """Update ability beliefs incrementally from this round's battles.

        Iterates over actual cage battles (at most M/2 * N_coop) instead of
        scanning all M^2 pairs. Vectorizes Bayes updates across all observers.
        """
        M, N_coop = params.M, params.N_coop

        new_ability_belief = incremental_ability_update(
            agent_state.ability_belief,
            env_state.battle_pair,
            env_state.cage_occupied,
            env_state.battle_outcome,
            observed_coop,
            N_coop,
            M,
        )

        return agent_state.replace(ability_belief=new_ability_belief)

    def _maybe_close_and_reopen(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        agent_state: AllAgentState,
        params: EnvParams,
        t_done: chex.Array,   # bool scalar (traced under jit)
        info: dict,
    ) -> tuple[EnvState, AllAgentState, dict]:
        """If the tournament just ended, run close + new tournament.

        Uses jax.lax.cond so this is safe inside jit. Both branches return
        the same pytree structure: (EnvState, AllAgentState, king_masks, king_repr).
        """
        M, N_coop = params.M, params.N_coop

        def _do_close(args):
            key, state, agent_state = args
            new_king_belief = derive_king_belief_from_ability(
                agent_state.ability_belief, N_coop, M
            )
            agent_state = agent_state.replace(king_belief=new_king_belief)
            state, close_info = close_tournament(state, params)
            state = new_tournament(key, state, params, close_info["king_masks"])
            return state, agent_state, close_info["king_masks"], close_info["king_repr"]

        def _no_op(args):
            _, state, agent_state = args
            return (
                state,
                agent_state,
                jnp.zeros((N_coop, M), dtype=jnp.bool_),
                jnp.full((N_coop,), -1, dtype=jnp.int32),
            )

        state, agent_state, king_masks, king_repr = jax.lax.cond(
            t_done, _do_close, _no_op, (key, state, agent_state)
        )
        info["king_masks"] = king_masks
        info["king_repr"]  = king_repr
        return state, agent_state, info

    # ------------------------------------------------------------------
    # Ability sampling utility (standalone, for testing / inference)
    # ------------------------------------------------------------------

    @staticmethod
    def sample_abilities(
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> chex.Array:
        """Sample M × N_coop ability scores from the truncated-Poisson PMF.

        Returns int32, shape (M, N_coop), values in [1, N_coop].
        """
        M, N_coop = params.M, params.N_coop
        ability_pmf = jnp.array(params.ability_pmf)
        if len(params.ability_pmf) != N_coop:
            ability_pmf = default_ability_pmf(N_coop)
        idx = jax.random.choice(
            key,
            a=N_coop,
            shape=(M, N_coop),
            replace=True,
            p=ability_pmf,
        )
        return (idx + 1).astype(jnp.int32)

