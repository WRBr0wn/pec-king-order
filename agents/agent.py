"""
Agent classes for HW5: NPC, AgentSetA (PPO), AgentSetB (MCTS/UCT).

Action encoding (from envs/spaces.py):
  0                    — noop
  1 .. N_coop          — a_watch coop (c+1 for coop c)
  N_coop+1 .. 2*N_coop — a_move  coop (N_coop+1+c for coop c)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import distrax
from flax.training.train_state import TrainState
from typing import NamedTuple
from inference.domain_inference import p_i_beats_j

# ---------------------------------------------------------------------------
# Module-level PPO constants
# ---------------------------------------------------------------------------

OBS_DIM        = 14   # 4 abilities + 4 king_belief + 1 sack_self + 1 tourn_frac + 4 teammate_sacks
ACTION_DIM     = 9    # 2 * N_coop + 1 = 9

LR             = 2.5e-4
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.5
ENT_COEF       = 0.01
UPDATE_EPOCHS  = 4
NUM_MINIBATCHES = 4


# ---------------------------------------------------------------------------
# ActorCritic — shared by all AgentSetA instances
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Actor branch
        actor = nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                         bias_init=nn.initializers.constant(0.0))(x)
        actor = nn.tanh(actor)
        actor = nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                         bias_init=nn.initializers.constant(0.0))(actor)
        actor = nn.tanh(actor)
        actor = nn.Dense(self.action_dim,
                         kernel_init=nn.initializers.orthogonal(0.01),
                         bias_init=nn.initializers.constant(0.0))(actor)
        policy = distrax.Categorical(logits=actor)
        # Critic branch
        critic = nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                          bias_init=nn.initializers.constant(0.0))(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                          bias_init=nn.initializers.constant(0.0))(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0),
                          bias_init=nn.initializers.constant(0.0))(critic)
        return policy, jnp.squeeze(critic, axis=-1)


# ---------------------------------------------------------------------------
# Trajectory buffer element
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs:      np.ndarray   # float32 (OBS_DIM,)
    action:   int
    log_prob: float
    value:    float
    reward:   float
    done:     bool


# ---------------------------------------------------------------------------
# GAE computation (pure NumPy, backward scan)
# ---------------------------------------------------------------------------

def _compute_gae(
    rewards: np.ndarray,   # float32 (T,)
    values:  np.ndarray,   # float32 (T,)
    dones:   np.ndarray,   # bool    (T,)
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (advantages, returns), both float32 (T,)."""
    T   = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nv    = 0.0 if (dones[t] or t == T - 1) else float(values[t + 1])
        delta = float(rewards[t]) + GAMMA * nv * (1.0 - float(dones[t])) - float(values[t])
        last  = delta + GAMMA * GAE_LAMBDA * (1.0 - float(dones[t])) * last
        adv[t] = last
    return adv, adv + values.astype(np.float32)


# ---------------------------------------------------------------------------
# JIT-cached PPO step (one cached function per (network_id, minibatch_size))
# ---------------------------------------------------------------------------

_ppo_step_cache: dict = {}


def _get_ppo_step(network: ActorCritic, mb_size: int):
    """Return a JIT-compiled PPO update step, cached by (network identity, mb_size).

    Caching prevents JAX from re-tracing whenever the trajectory length changes,
    which would be expensive.  We pad trajectories to fixed multiples of
    NUM_MINIBATCHES, so mb_size stabilises after the first few tournaments.
    """
    cache_key = (id(network), mb_size)
    if cache_key in _ppo_step_cache:
        return _ppo_step_cache[cache_key]

    @jax.jit
    def ppo_step(ts, obs_mb, acts_mb, lp_mb, adv_mb, ret_mb):
        def loss_fn(params):
            pi, vals = jax.vmap(lambda x: network.apply(params, x))(obs_mb)
            lp_new  = pi.log_prob(acts_mb)
            entropy = pi.entropy().mean()
            ratio   = jnp.exp(lp_new - lp_mb)
            # PPO clipped surrogate loss (sign convention: maximise, so negate)
            al = jnp.maximum(
                -adv_mb * ratio,
                -adv_mb * jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS),
            ).mean()
            vl = ((vals - ret_mb) ** 2).mean()
            return al + VF_COEF * vl - ENT_COEF * entropy, (al, vl, entropy)

        (loss, _aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
        return ts.apply_gradients(grads=grads), loss

    _ppo_step_cache[cache_key] = ppo_step
    return ppo_step


# ---------------------------------------------------------------------------
# NPC — simple greedy strategy (unchanged from HW3)
# ---------------------------------------------------------------------------

def _choose_action_npc(
    agent_idx: int,
    abilities: np.ndarray,   # (N_coop,) own ability scores
    N_coop: int,
    zone_array: np.ndarray,  # (M, N_coop, 3)
) -> int:
    """Shared action logic: move to best-ability coop, or watch if already there."""
    best_coop    = int(np.argmax(abilities))
    current_coop = int(np.argmax(zone_array[agent_idx, :, 0]))
    if current_coop == best_coop:
        return best_coop + 1           # a_watch
    return N_coop + 1 + best_coop     # a_move


class NPC:
    def __init__(
        self,
        agent_idx: int,
        abilities: np.ndarray,  # int32 (N_coop,)
        N_coop: int,
        teammates: list[int],
    ) -> None:
        self.agent_idx = agent_idx
        self.abilities = abilities
        self.N_coop    = N_coop
        self.teammates = teammates

    def choose_action(
        self,
        zone_array: np.ndarray,    # (M, N_coop, 3)
        king_belief: np.ndarray,   # (N_coop,)
        abilities: np.ndarray,     # (N_coop,) own abilities
        N_coop: int,
        key=None,
    ) -> int:
        return _choose_action_npc(self.agent_idx, abilities, N_coop, zone_array)

    def choose_social_actions(
        self,
        sack: np.ndarray,               # int32 (M,)
        sack_owners: np.ndarray,        # bool  (M, M)
        observation_memory: np.ndarray, # int16 (M, M, M, T)
        tournament_count: int,
    ) -> list[dict]:
        return []


# ---------------------------------------------------------------------------
# AgentSetA — Tournament-level PPO (Flax ActorCritic + distrax + optax)
# ---------------------------------------------------------------------------

class AgentSetA:
    """Strategy A — PPO with tournament-sparse rewards.

    Trajectory is collected within each tournament (one Transition per env step).
    PPO update runs once at tournament close using GAE advantages.
    Network: two-headed MLP (actor + critic), 2x64 tanh layers each.
    """

    def __init__(
        self,
        agent_idx: int,
        abilities: np.ndarray,  # int32 (N_coop,)
        N_coop: int,
        teammates: list[int],
    ) -> None:
        self.agent_idx = agent_idx
        self.abilities = abilities
        self.N_coop    = N_coop
        self.teammates = teammates
        self.T         = 256  # total tournaments (for tournament_frac normalisation)

        # Build network and optimizer
        self._network = ActorCritic(action_dim=ACTION_DIM)
        init_key      = jax.random.PRNGKey(agent_idx)
        init_obs      = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
        params        = self._network.init(init_key, init_obs)
        optimizer     = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(LR, eps=1e-5),
        )
        self._ts = TrainState.create(
            apply_fn=self._network.apply, params=params, tx=optimizer
        )

        # Trajectory buffer (reset each tournament)
        self._traj: list[Transition] = []
        self._prev_sack              = 0
        self._prev_win_rate          = 0.0
        self._last_tournament_count  = 0
        # Sack snapshot — updated each choose_social_actions call, used in next obs build
        self._last_sack: np.ndarray | None = None
        # Internal PRNG key (used when run.py does not pass one)
        self._key = jax.random.PRNGKey(agent_idx + 1000)

    # ------------------------------------------------------------------
    def choose_action(
        self,
        zone_array: np.ndarray,    # (M, N_coop, 3)
        king_belief: np.ndarray,   # (N_coop,) P(self is king per coop)
        abilities: np.ndarray,     # (N_coop,) own abilities
        N_coop: int,
        key=None,
    ) -> int:
        tourn_frac = self._last_tournament_count / float(self.T)
        # Sack values from last social call (one step stale — best available without
        # changing choose_action's signature). Zero on first step before any social call.
        if self._last_sack is not None:
            sack_self     = float(self._last_sack[self.agent_idx]) / 1024.0
            teammate_idxs = [t for t in self.teammates if t != self.agent_idx]
            sack_teammates = np.array(
                [float(self._last_sack[t]) / 1024.0 for t in teammate_idxs],
                dtype=np.float32,
            )
            # Pad to 4 values in case team size varies
            if len(sack_teammates) < 4:
                sack_teammates = np.pad(sack_teammates, (0, 4 - len(sack_teammates)))
        else:
            sack_self      = 0.0
            sack_teammates = np.zeros(4, dtype=np.float32)
        obs_vec = np.concatenate([
            abilities.astype(np.float32) / N_coop,  # (4,) normalised abilities
            king_belief.astype(np.float32),          # (4,) P(king) per coop
            [sack_self],                             # own sack / 1024
            [tourn_frac],                            # tournament progress signal
            sack_teammates,                          # teammate sacks / 1024 (4,)
        ]).astype(np.float32)  # total: 14

        obs_jax = jnp.array(obs_vec)
        if key is None:
            self._key, key = jax.random.split(self._key)

        pi, value = self._ts.apply_fn(self._ts.params, obs_jax)
        action    = int(pi.sample(seed=key))
        log_prob  = float(pi.log_prob(action))

        self._traj.append(Transition(
            obs=obs_vec, action=action, log_prob=log_prob,
            value=float(value), reward=0.0, done=False,
        ))
        return action

    # ------------------------------------------------------------------
    def choose_social_actions(
        self,
        sack: np.ndarray,               # int32 (M,)
        sack_owners: np.ndarray,        # bool  (M, M)
        observation_memory: np.ndarray, # int16 (M, M, M, T)
        tournament_count: int,
        win_count: int = 0,
        loss_count: int = 0,
    ) -> list[dict]:
        social: list[dict] = []

        # Cache sack snapshot for use in next choose_action observation build
        self._last_sack = sack

        if tournament_count > self._last_tournament_count:
            crowns_earned = int(sack[self.agent_idx]) - self._prev_sack
            win_rate = float(win_count) / max(float(win_count + loss_count), 1.0)
            win_rate_delta = win_rate - self._prev_win_rate
            reward = float(crowns_earned) + 0.3 * win_rate_delta
            self._prev_win_rate = win_rate
            # Run PPO update if we have enough steps
            if len(self._traj) >= NUM_MINIBATCHES:
                self._traj[-1] = self._traj[-1]._replace(
                    reward=reward, done=True
                )
                self._ppo_update()
            self._traj = []
            self._prev_sack             = int(sack[self.agent_idx])
            self._last_tournament_count = tournament_count

            # On first tournament close: share ownership with all teammates
            if tournament_count == 1:
                for t in self.teammates:
                    if t != self.agent_idx:
                        social.append({"type": "extend_sack_ownership", "target": t})

            # After tournament 10: concentrate crowns into the highest-earning teammate.
            # All 4 already co-own each other's sacks from tournament 1, so crowns in
            # the high-earner's sack benefit the whole team at scoring time.
            # Only non-high-earners transfer; high-earner keeps its crowns in place.
            if tournament_count >= 10:
                team = [self.agent_idx] + [t for t in self.teammates if t != self.agent_idx]
                team_sacks = np.array([int(sack[i]) for i in team])
                best_idx   = int(np.argmax(team_sacks))
                best_val   = team_sacks[best_idx]
                sorted_vals = np.sort(team_sacks)[::-1]
                second_val  = sorted_vals[1] if len(sorted_vals) > 1 else 0
                # Only transfer when one teammate is clearly ahead and has meaningful crowns
                clear_leader = best_val > 5 and best_val > 1.5 * max(second_val, 1)
                if clear_leader:
                    high_earner = team[best_idx]
                    if self.agent_idx != high_earner:
                        own_crowns = int(sack[self.agent_idx])
                        if own_crowns > 0:
                            social.append({
                                "type": "transfer_crowns",
                                "target": high_earner,
                                "count": own_crowns,
                            })

        # Every step: share observations with teammates
        for t in self.teammates:
            if t != self.agent_idx:
                social.append({"type": "share_observations", "target": t})

        return social

    # ------------------------------------------------------------------
    def _ppo_update(self) -> None:
        """Run PPO update over the completed tournament's trajectory."""
        traj = self._traj
        n    = len(traj)

        obs_arr  = np.stack([t.obs      for t in traj])            # (n, OBS_DIM)
        acts_arr = np.array([t.action   for t in traj], dtype=np.int32)
        lp_arr   = np.array([t.log_prob for t in traj], dtype=np.float32)
        val_arr  = np.array([t.value    for t in traj], dtype=np.float32)
        rew_arr  = np.array([t.reward   for t in traj], dtype=np.float32)
        don_arr  = np.array([t.done     for t in traj], dtype=bool)

        adv_arr, ret_arr = _compute_gae(rew_arr, val_arr, don_arr)
        # Normalise advantages
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # Pad to multiple of NUM_MINIBATCHES so minibatch size is fixed for JIT caching
        remainder = n % NUM_MINIBATCHES
        if remainder != 0:
            pad      = NUM_MINIBATCHES - remainder
            obs_arr  = np.concatenate([obs_arr,  np.zeros((pad, OBS_DIM), dtype=np.float32)])
            acts_arr = np.concatenate([acts_arr, np.zeros(pad, dtype=np.int32)])
            lp_arr   = np.concatenate([lp_arr,  np.zeros(pad, dtype=np.float32)])
            adv_arr  = np.concatenate([adv_arr, np.zeros(pad, dtype=np.float32)])
            ret_arr  = np.concatenate([ret_arr, np.zeros(pad, dtype=np.float32)])

        n_padded    = len(obs_arr)
        mb_size     = n_padded // NUM_MINIBATCHES
        update_step = _get_ppo_step(self._network, mb_size)

        for _epoch in range(UPDATE_EPOCHS):
            perm = np.random.permutation(n_padded)
            for mb_indices in np.array_split(perm, NUM_MINIBATCHES):
                self._ts, _loss = update_step(
                    self._ts,
                    jnp.array(obs_arr[mb_indices]),
                    jnp.array(acts_arr[mb_indices]),
                    jnp.array(lp_arr[mb_indices]),
                    jnp.array(adv_arr[mb_indices]),
                    jnp.array(ret_arr[mb_indices]),
                )


# ---------------------------------------------------------------------------
# AgentSetB — Monte Carlo Tree Search (MCTS/UCT) for coop selection
# ---------------------------------------------------------------------------

class AgentSetB:
    """Strategy B — UCT-guided MCTS over coops at each tournament close.

    At each tournament close, runs n_iter=20 UCT iterations. Each iteration
    picks the coop with the highest UCT score, runs K rollouts against
    opponents sampled from the last observed zone_array, and updates Q/N.
    Win probabilities come from p_i_beats_j over stored ability beliefs.
    Fallback to prob=0.5 when no belief is available yet.

    Social: symmetric — share observations with all teammates every step;
    extend sack ownership to all teammates on tournament 1; concentrate
    crowns into the highest-earning teammate after tournament 10.
    """

    def __init__(
        self,
        agent_idx: int,
        abilities: np.ndarray,  # int32 (N_coop,)
        N_coop: int,
        teammates: list[int],
        k_rollouts: int = 10,
    ) -> None:
        self.agent_idx = agent_idx
        self.abilities = abilities
        self.N_coop    = N_coop
        self.teammates = teammates

        self._current_coop = int(np.argmax(abilities))
        self._uct_Q        = np.zeros(N_coop, dtype=np.float64)
        self._uct_N        = np.zeros(N_coop, dtype=np.int64)
        self._K            = k_rollouts
        self._n_iter       = 20

        self._last_ability_belief: np.ndarray | None = None
        self._last_zone_array:     np.ndarray | None = None
        self._last_tournament_count = 0
        self._prev_sack             = 0

    # ------------------------------------------------------------------
    def choose_action(
        self,
        zone_array: np.ndarray,    # (M, N_coop, 3)
        king_belief: np.ndarray,   # (N_coop,)
        abilities: np.ndarray,     # (N_coop,) own abilities
        N_coop: int,
        key=None,
    ) -> int:
        self._last_zone_array = zone_array
        current_coop = int(np.argmax(zone_array[self.agent_idx].any(axis=-1)))
        if current_coop == self._current_coop:
            return self._current_coop + 1       # a_watch
        return N_coop + 1 + self._current_coop  # a_move toward target

    # ------------------------------------------------------------------
    def _single_rollout(self, coop: int) -> int:
        """Simulate one tournament in coop c; return 1 if crowned, 0 otherwise."""
        M = self._last_zone_array.shape[0] if self._last_zone_array is not None else 64
        if self._last_zone_array is not None:
            occ = self._last_zone_array[:, coop, :].any(axis=-1)
            opponents = [j for j in np.where(occ)[0] if j != self.agent_idx]
        else:
            opponents = [j for j in range(M) if j != self.agent_idx]

        if len(opponents) == 0:
            return 1  # uncontested

        k_limit = 4  # matches env K
        wins = 0
        losses = 0
        opp_pool = list(opponents)
        np.random.shuffle(opp_pool)
        for j in opp_pool:
            if self._last_ability_belief is not None:
                prob = float(p_i_beats_j(
                    self.agent_idx, j, coop,
                    self._last_ability_belief, self.N_coop,
                ))
            else:
                prob = 0.5
            if np.random.random() < prob:
                wins += 1
            else:
                losses += 1
            if losses >= k_limit:
                break

        return 1 if losses == 0 else 0

    # ------------------------------------------------------------------
    def _run_uct(self) -> None:
        """Run n_iter UCT iterations; commit to argmax(Q) as next coop."""
        C_uct = 1.0
        for _ in range(self._n_iter):
            total_visits = int(self._uct_N.sum())
            scores = (
                self._uct_Q
                + C_uct * np.sqrt(np.log(total_visits + 1) / (self._uct_N + 1))
            )
            c = int(np.argmax(scores))
            new_crowns = sum(self._single_rollout(c) for _ in range(self._K))
            old_total = self._uct_Q[c] * self._uct_N[c]
            self._uct_N[c] += self._K
            self._uct_Q[c] = (old_total + new_crowns) / self._uct_N[c]
        self._current_coop = int(np.argmax(self._uct_Q))

    # ------------------------------------------------------------------
    def choose_social_actions(
        self,
        sack: np.ndarray,               # int32 (M,)
        sack_owners: np.ndarray,        # bool  (M, M)
        observation_memory: np.ndarray, # int16 (M, M, M, T)
        tournament_count: int,
        ability_belief: np.ndarray | None = None,
    ) -> list[dict]:
        social: list[dict] = []

        if tournament_count > self._last_tournament_count:
            if ability_belief is not None:
                self._last_ability_belief = ability_belief
            self._run_uct()
            self._prev_sack             = int(sack[self.agent_idx])
            self._last_tournament_count = tournament_count

            if tournament_count == 1:
                for t in self.teammates:
                    if t != self.agent_idx:
                        social.append({"type": "extend_sack_ownership", "target": t})

            # After tournament 10: concentrate crowns into the highest-earning teammate.
            # All 4 co-own each other's sacks from tournament 1, so crowns held by the
            # high-earner benefit the whole team at final scoring.
            if tournament_count >= 10:
                team = [self.agent_idx] + [t for t in self.teammates if t != self.agent_idx]
                team_sacks = np.array([int(sack[i]) for i in team])
                best_idx   = int(np.argmax(team_sacks))
                best_val   = team_sacks[best_idx]
                sorted_vals = np.sort(team_sacks)[::-1]
                second_val  = sorted_vals[1] if len(sorted_vals) > 1 else 0
                clear_leader = best_val > 5 and best_val > 1.5 * max(second_val, 1)
                if clear_leader:
                    high_earner = team[best_idx]
                    if self.agent_idx != high_earner:
                        own_crowns = int(sack[self.agent_idx])
                        if own_crowns > 0:
                            social.append({
                                "type": "transfer_crowns",
                                "target": high_earner,
                                "count": own_crowns,
                            })

        for t in self.teammates:
            if t != self.agent_idx:
                social.append({"type": "share_observations", "target": t})

        return social
