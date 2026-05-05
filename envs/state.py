"""
EnvParams and EnvState definitions for Pec-King Order.

Array shapes (EXACT from spec):
  ZoneArray     M x N_coop x 3       — agent zone occupancy per coop (BZ/SZ/TZ)
  ActionArray   M x 1                — current action per agent (0=no action)
  BattleOutcome M x M x N_coop       — per-round battle results (+1/-1/0)

Note: BattleHistory (M x M x N_coop x T) and ObservationMemory (M x M x M x T)
are accumulated outside the JIT loop in numpy to avoid carrying 500+ MB of dead
write state through every step_env call.  See run.py for the accumulators.
"""

from __future__ import annotations

from typing import Optional

import chex
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# EnvParams — static configuration (does not change after initialization)
# ---------------------------------------------------------------------------

@struct.dataclass
class EnvParams:
    """Static environment configuration.

    Attributes:
        M:           Number of agents (default 64).
        N_coop:      Number of coops / regions (default 4).
        k:           Max losses before a chicken is excluded from a coop (default 4).
        T:           Number of tournaments to simulate (default 256, per HW5 spec).
        ability_pmf: Precomputed truncated-Poisson PMF table, shape (N_coop,).
                     p(x_i) = exp(-x_i / (N_coop/3)) / Z  for x_i in [1, N_coop].
                     Stored as a length-N_coop float array (values sum to 1).
    """

    M: int = 64
    N_coop: int = 4
    k: int = 4
    T: int = 256
    # Truncated-Poisson probability table, length N_coop.
    # Stored as a plain tuple so EnvParams is hashable (needed for static_argnums).
    # Index i corresponds to ability value (i+1).
    ability_pmf: tuple = struct.field(
        default_factory=lambda: tuple(map(float, default_ability_pmf(N_coop=4).tolist()))
    )


def default_ability_pmf(N_coop: int = 4) -> chex.Array:
    """Compute truncated Poisson PMF for ability values in [1, N_coop].

    p(x) = exp(-x / (N_coop / 3)) / Z,   x in {1, ..., N_coop}
    """
    xs = jnp.arange(1, N_coop + 1, dtype=jnp.float32)
    unnorm = jnp.exp(-xs / (N_coop / 3.0))
    return unnorm / unnorm.sum()


# ---------------------------------------------------------------------------
# EnvState — mutable simulation state (JAX pytree via chex.dataclass)
# ---------------------------------------------------------------------------

@chex.dataclass
class EnvState:
    """Full (objective) state of the Pec-King Order environment.

    All fields are JAX arrays so the struct is a valid pytree and can be
    passed through jit / lax.scan without retracing.

    Array shapes use the variable names from the spec:
        M      = number of agents
        N      = N_coop = number of coops
        T      = number of tournaments

    Fields
    ------
    zone_array : int32, shape (M, N, 3)
        Occupancy indicator.  axis-2 indices:
          0 = BattleZone, 1 = SpectatorZone, 2 = TransitZone
        Value is 1 if agent m occupies that zone in coop c, else 0.
        Each agent must occupy exactly one (coop, zone) at all times.

    action_array : int32, shape (M, 1)
        Current action for each agent.
          0  = no action
          1..N_coop          = a_watch for each coop   (stay + observe)
          N_coop+1..2*N_coop = a_move  for each coop   (move + no observation)
        Converts to a 2*N_coop one-hot vector for downstream processing.

    battle_outcome : int8, shape (M, M, N)
        Per-round battle results.
          (i, j, k) = +1  chicken i beat  chicken j in coop k this round
          (i, j, k) = -1  chicken i lost to chicken j in coop k this round
          (i, j, k) =  0  no battle between i and j in coop k this round
        Antisymmetric: outcome[i,j,k] == -outcome[j,i,k].

    abilities : int32, shape (M, N)
        Ability score of each chicken per coop challenge type.
        Values in [1, N_coop].  Sampled once at reset and never changed.

    ability_matrix : int32, shape (N, M, N)
        Indicator matrix representation of abilities.
        ability_matrix[v, m, c] = 1  iff  abilities[m, c] >= (v+1).
        Encodes cumulative dominance; dot products reveal shared levels.
        First axis ranges over v in {0, ..., N_coop-1} (threshold = v+1).

    loss_count : int32, shape (M, N)
        Number of losses per chicken per coop in the current tournament.
        Reset to zero at the start of each new tournament.

    win_count : int32, shape (M, N)
        Number of wins per chicken per coop in the current tournament.
        Reset to zero at the start of each new tournament.

    battle_pair : int32, shape (M // 2, N, 2)
        Cage assignments for the current round.
        battle_pair[cage_idx, coop_idx, :] = [chicken_i, chicken_j] or [-1,-1].

    cage_occupied : bool_, shape (M // 2, N)
        True if cage is occupied this round.

    king : bool_, shape (M, N)
        Crown-holder mask. king[m, c] = True if chicken m holds the crown
        for coop c. All False before a tournament closes.

    tournament_step : int32, scalar
        Current step index within the active tournament.

    tournament_count : int32, scalar
        Number of completed tournaments so far.

    done : bool_, scalar
        True when all T tournaments have been completed.
    """

    # --- core zone / action state ---
    zone_array: chex.Array       # int32  (M, N, 3)
    action_array: chex.Array     # int32  (M, 1)

    # --- battle records ---
    battle_outcome: chex.Array   # int8   (M, M, N)

    # --- agent attributes (fixed after reset) ---
    abilities: chex.Array        # int32  (M, N)
    ability_matrix: chex.Array   # int32  (N, M, N)   threshold x agent x coop

    # --- per-tournament counters ---
    loss_count: chex.Array       # int32  (M, N)
    win_count: chex.Array        # int32  (M, N)

    # --- cumulative battle counter (never reset) ---
    cumulative_battles: chex.Array  # int32  (M, N)  total battles across all tournaments

    # --- cage assignments for current round ---
    battle_pair: chex.Array      # int32  (M//2, N, 2)
    cage_occupied: chex.Array    # bool_  (M//2, N)

    # --- per-tournament "have they fought" tracker ---
    has_battled: chex.Array      # bool_  (M, M, N)  True if i and j battled in coop c

    # --- tournament metadata ---
    king: chex.Array             # bool_  (M, N)

    # --- sack mechanics ---
    sack: chex.Array             # int32  (M,)
    sack_owners: chex.Array      # bool_  (M, M)

    # --- step / episode counters (scalars stored as 0-d arrays) ---
    tournament_step: chex.Array  # int32  scalar
    tournament_count: chex.Array # int32  scalar
    done: chex.Array             # bool_  scalar


# ---------------------------------------------------------------------------
# Factory — build a zeroed EnvState (populated properly in reset_env)
# ---------------------------------------------------------------------------

def empty_env_state(params: EnvParams) -> EnvState:
    """Return an EnvState filled with zeros / False of the correct shapes.

    This is intentionally *not* a valid initial state — call reset_env to
    obtain a properly initialized state.  The factory exists so that type
    checkers and tests can verify array shapes without running JAX tracing.
    """
    M, N, T = params.M, params.N_coop, params.T

    return EnvState(
        zone_array=jnp.zeros((M, N, 3), dtype=jnp.int32),
        action_array=jnp.zeros((M, 1), dtype=jnp.int32),
        battle_outcome=jnp.zeros((M, M, N), dtype=jnp.int8),
        abilities=jnp.zeros((M, N), dtype=jnp.int32),
        ability_matrix=jnp.zeros((N, M, N), dtype=jnp.int32),
        loss_count=jnp.zeros((M, N), dtype=jnp.int32),
        win_count=jnp.zeros((M, N), dtype=jnp.int32),
        cumulative_battles=jnp.zeros((M, N), dtype=jnp.int32),
        battle_pair=jnp.full((M // 2, N, 2), fill_value=-1, dtype=jnp.int32),
        cage_occupied=jnp.zeros((M // 2, N), dtype=jnp.bool_),
        has_battled=jnp.zeros((M, M, N), dtype=jnp.bool_),
        king=jnp.zeros((M, N), dtype=jnp.bool_),
        sack=jnp.zeros((M,), dtype=jnp.int32),
        sack_owners=jnp.eye(M, dtype=jnp.bool_),
        tournament_step=jnp.zeros((), dtype=jnp.int32),
        tournament_count=jnp.zeros((), dtype=jnp.int32),
        done=jnp.zeros((), dtype=jnp.bool_),
    )


# ---------------------------------------------------------------------------
# AllAgentState — stacked per-agent components (JAX pytree via chex.dataclass)
# ---------------------------------------------------------------------------

@chex.dataclass
class AllAgentState:
    """Stacked per-agent components for all M agents.

    For any array with leading dimension M, index [m, ...] gives agent m's data.
    This replaces the parallel flock list and ad-hoc observation arrays.

    Shape reference (per-agent shape -> stacked shape):
      ability_belief      (M, N_coop, N_coop) -> (M, M, N_coop, N_coop)
      my_ability          (N_coop,)         -> (M, N_coop)
      my_location         (2,)              -> (M, 2)
      last_observation    (M, M)            -> (M, M, M)
      king_belief         (M, N_coop, 2)    -> (M, M, N_coop, 2)

    ObservationMemory (M, M, M, T) is accumulated outside JIT in run.py.

    The 7th component (action_policy) is a callable and stays Python-side.
    """
    # Component 1: AbilityBelief — agent m's belief about everyone's abilities
    # ability_belief[m, j, c, v] = P(agent j has ability v+1 in coop c | m's obs)
    ability_belief: chex.Array       # float32 (M, M, N_coop, N_coop)

    # Component 3: MyAbility — each agent's own ability scores (known to self)
    my_ability: chex.Array           # int32  (M, N_coop)

    # Component 4: MyLocation — [region_idx, zone_idx] per agent
    my_location: chex.Array          # int32  (M, 2)

    # Component 5: LastObservation — most recent round's observations per agent
    # last_observation[m, i, j] = outcome m saw this round between i and j
    last_observation: chex.Array     # int8   (M, M, M)

    # Component 6: KingBelief — agent m's belief about who is king
    # king_belief[m, j, c, 0/1] = P(j is not king / is king in coop c | m's obs)
    king_belief: chex.Array          # float32 (M, M, N_coop, 2)


@chex.dataclass
class PecKingState:
    """Compound state: env bookkeeping + per-agent beliefs.

    Packs EnvState and AllAgentState into the single 'state' slot that
    gymnax's reset/step interface expects, so step_env/reset_env match
    gymnax's 4-arg signature without custom overrides.
    """
    env:    EnvState
    agents: AllAgentState


def empty_agent_state(params: EnvParams) -> AllAgentState:
    """Create AllAgentState with uniform priors and zeroed observations.

    Initial states:
      - ability_belief: uniform prior (1/N_coop for each ability value)
      - my_ability: zeros (filled properly in reset_env)
      - my_location: zeros (filled properly in reset_env)
      - last_observation: zeros (no observations yet)
      - king_belief: uniform prior (0.5 for king/not-king per coop)
    """
    M, N_coop, T = params.M, params.N_coop, params.T

    return AllAgentState(
        # AbilityBelief: uniform prior 1/N_coop for each (agent, target, coop, value)
        ability_belief=jnp.full((M, M, N_coop, N_coop), 1.0 / N_coop, dtype=jnp.float32),

        # MyAbility: zeros — populated in reset_env from sampled abilities
        my_ability=jnp.zeros((M, N_coop), dtype=jnp.int32),

        # MyLocation: zeros — populated in reset_env
        my_location=jnp.zeros((M, 2), dtype=jnp.int32),

        # LastObservation: zeros — no battles seen yet
        last_observation=jnp.zeros((M, M, M), dtype=jnp.int8),

        # KingBelief: uniform prior 0.5 for king/not-king per (observer, subject, coop)
        king_belief=jnp.full((M, M, N_coop, 2), 0.5, dtype=jnp.float32),
    )