# Pec-King Order

A multi-agent dominance tournament simulation built on the [gymnax](https://github.com/RobertTLange/gymnax) (JAX) framework. 64 chickens compete across 4 coops over 256 tournaments, accumulating crowns through battles and social coordination. This repository contains the full domain specification, a reference implementation, and two strategic agent implementations — one using tournament-level PPO and one using MCTS/UCT — evaluated against a population of 56 greedy NPCs.

## Design Motivation

Pec-King Order is built around a tension that most multi-agent environments avoid: ability is fixed, private, and drawn from a distribution skewed heavily toward low values. An agent cannot improve its fighting strength through training or experience. What it can do is learn where its strength is, and coordinate with teammates to make that strength count at scoring time.

The sack mechanic is the core design lever. Crowns accumulate in individual wallets, but ownership can be shared, and a shared sack benefits all co-owners equally regardless of who earned the crowns. This creates a cooperative incentive that is structurally unusual: the rational move is often to concentrate crowns into the teammate with the best ability draw rather than accumulate individually. A chicken with ability 4 in a coop with two NPC rivals is a better crown engine than four chickens with ability 2 competing against twelve. If the team identifies that agent early and routes resources toward it, the whole set benefits at scoring time.

The three social actions exist to operationalize this. `ExtendSackOwnership` is the foundational move: without it, no coordination is possible. `ShareObservations` lets agents pool their partial views of a 64-chicken world into a shared belief about who is strong where, sharpening the inference that both PPO and MCTS depend on. `TransferCrowns` is the endgame action: once a team leader is identified, non-leaders can concentrate the pool rather than leaving crowns stranded in low-yield sacks.

An interesting direction this opens up is emergent exchange between agents that are not pre-specified as teammates. In the current implementation, social actions flow within fixed teams of four. But the action space does not enforce this: any agent can extend sack ownership or transfer crowns to any other. A self-interested agent with weak ability draws but rich observation history could, in principle, offer `ShareObservations` to a strong fighter in exchange for a cut of that fighter's sack via `ExtendSackOwnership` or a direct `TransferCrowns` payment. This would require agents to reason about the value of information to others, negotiate implicitly through sequential action, and decide whether a proposed exchange is worth the cost. None of that is implemented here, but the domain supports it natively, and it represents a natural extension toward fully decentralized multi-agent learning without pre-defined team structure.

---

## Table of Contents

- [Setup](#setup)
- [Running the Simulation](#running-the-simulation)
- [Domain Specification](#domain-specification)
  - [Parameters](#parameters)
  - [Ability Prior](#ability-prior)
  - [Zone System](#zone-system)
  - [Action Encoding](#action-encoding)
  - [Tournament Structure](#tournament-structure)
  - [Crown Assignment](#crown-assignment)
  - [Sack Mechanics and Final Scoring](#sack-mechanics-and-final-scoring)
  - [Social Actions](#social-actions)
  - [Observation Model](#observation-model)
  - [Win Condition](#win-condition)
- [Inference](#inference)
- [Reference Implementation](#reference-implementation)
  - [NPC Strategy](#npc-strategy)
  - [Set-A: Tournament-Level PPO](#set-a-tournament-level-ppo)
  - [Set-B: MCTS/UCT Coop Selection](#set-b-mctsuct-coop-selection)
- [Experiments](#experiments)
- [Tests](#tests)
- [Architecture](#architecture)
- [State Reference](#state-reference)

---

## Setup

```bash
pip install jax jaxlib gymnax chex flax optax distrax pytest
```

## Running the Simulation

```bash
python run.py
```

### Configuration

Edit the flags at the top of `run.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `M` | `64` | Number of agents |
| `N_COOP` | `4` | Number of coops |
| `K` | `4` | Max losses before elimination from a coop |
| `T` | `256` | Number of tournaments |
| `MCTS_K` | `10` | MCTS rollouts per coop per UCT iteration (Set-B only) |
| `SEED` | `None` | RNG seed — `None` = random each run; `int` = reproducible |
| `INTERACTIVE` | `False` | Launch belief query REPL after simulation |
| `VERBOSE` | `False` | Print sampled abilities for first few agents at startup |
| `TRACK_CHICKEN` | `True` | Enable per-chicken step-by-step tracing |
| `TRACK_IDX` | `None` | Which NPC to trace (`None` = randomly chosen) |
| `SELECT_IDX` | `False` | Prompt at startup to pick the traced chicken |
| `TRACK_FILE` | `"tracing/chicken_trace.txt"` | Path for NPC trace output |

Strategic agent traces are written to `tracing/agent_setA_*.txt` and `tracing/agent_setB_*.txt` on every run. The seed used is printed in the run header so any run can be exactly reproduced.

Two complete runs are archived in `example_runs/` — one with `SEED=42` (the K-sweep reference seed) and one unseeded run — so you can inspect outputs without running the simulation yourself. Each folder contains `chicken_abilities.txt`, `chicken_trace.txt`, `agent_setA_*.txt`, `agent_setB_*.txt`, and a `run_summary.md` with the terminal output and final scores.

Simulation time scales with `MCTS_K`: approximately 13 min at K=1, 28 min at K=10, 2h at K=50 (CPU, M=64, T=256).

### Interactive Mode

When `INTERACTIVE = True`, a belief query REPL launches after the simulation:

| Command | Description |
|---------|-------------|
| `chicken <idx> [from <obs>]` | Overview of beliefs about a chicken |
| `beats <i> <j> [coop <c>] [from <obs>]` | P(i beats j \| observations) |
| `king <idx> [from <obs>]` | P(king) per coop for a chicken |
| `profile <idx> [from <obs>]` | Ability posterior (MAP vs true abilities) |
| `abilities [idx]` | Ground-truth ability scores |
| `help` | List available commands |
| `quit` | Exit |

All commands accept `from <observer_idx>` to query beliefs from a specific chicken's perspective.

---

## Domain Specification

### Parameters

| Parameter | Value | Description |
|---|---|---|
| `M` | 64 | Total agents |
| `N_coop` | 4 | Independent coops |
| `k` | 4 | Max losses per agent per tournament |
| `T` | 256 | Total tournaments |

### Ability Prior

Each chicken has a fixed ability score per coop, sampled once at the start of the simulation and never changed. Scores are drawn i.i.d. from a truncated Poisson prior with PMF:

$$P(\text{ability} = v) \propto \exp\!\left(\frac{-v}{N_{\text{coop}}/3}\right), \quad v \in \{1, 2, 3, 4\}$$

For `N_coop=4`, this normalizes to:

```
P(ability = 1) = 0.5553
P(ability = 2) = 0.2623
P(ability = 3) = 0.1239
P(ability = 4) = 0.0585
```

Abilities are **private**: each agent knows only its own scores. Opponents' abilities must be inferred from observed battles. With 64 agents and 4 coops, the expected number of ability-4 agents per coop is `64 × 0.0585 / 4 ≈ 0.94`, making high-ability draws rare and structurally valuable.

### Zone System

Each chicken occupies exactly one zone at any time. Zones are tracked in `zone_array` of shape `(M, N_coop, 3)` — a one-hot indicator across the three zones for each (agent, coop) pair.

| Zone | Index | Name | Description |
|---|---|---|---|
| 0 | `ZONE_BATTLE` | BattleZone | Eligible for cage assignment; observes own coop unconditionally |
| 1 | `ZONE_SPECTATOR` | SpectatorZone | Watches a chosen coop or moves to another; cannot battle |
| 2 | `ZONE_TRANSIT` | TransitZone | In transit between coops; no observations; arrives next round |

At reset, all agents are placed uniformly at random into BattleZone, one coop each.

### Action Encoding

The action space is `Discrete(2 * N_coop + 1)` — 9 actions for `N_coop=4`:

| Action value | Meaning |
|---|---|
| `0` | No-op (do nothing) |
| `1` to `N_coop` | `a_watch(c)` — watch coop `c-1` from SpectatorZone |
| `N_coop+1` to `2*N_coop` | `a_move(c)` — move to coop `c - N_coop - 1` via TransitZone |

`a_watch` is only effective from SpectatorZone. BattleZone agents observe their own coop regardless of action. `a_move` sends the agent into TransitZone; they arrive in the target coop's BattleZone the following round.

### Tournament Structure

Each tournament runs in discrete rounds until no coop can produce a new eligible pair. The step order within each round is:

**1. Tournament close check** (assessed at the start of each round)

`tournament_is_done` returns True if no coop contains two eligible BattleZone agents that have not yet fought each other this tournament. When True, `close_tournament` and `new_tournament` run before proceeding.

**2. Cage assignment** (`assign_agents_to_cage`)

For each coop independently, eligible BattleZone agents are paired into cages. Eligibility requires:
- Currently in BattleZone for that coop
- `loss_count[agent, coop] < k` (under the loss limit for this tournament)
- The candidate pair has **not already fought each other** in this coop this tournament (tracked in `has_battled[i, j, coop]`)

Pairing order: fewest losses first, with uniform random tiebreak within the same loss count. Pairing is greedy — for each agent in sorted order, find the first eligible opponent (by sorted position) they haven't yet fought. At most `M // 2` cages per coop.

**3. Unpaired relocation** (`relocate_zone`)

Any BattleZone agent that was not assigned to a cage moves to SpectatorZone for this round.

**4. Move actions** (`relocate_region`)

SpectatorZone agents that issued `a_move` enter TransitZone. They observe nothing this round and arrive in the target coop's BattleZone next round. This runs after step 3, so newly-relocated agents can choose to depart in the same round.

**5. Concurrent events**

- **DominanceBattle** (`dominance_battle`): All cage pairs fight simultaneously. Higher ability wins; equal ability resolves as a fair coin flip (50/50). The loser's `loss_count[coop]` increments by 1. `has_battled[i, j, coop]` and `has_battled[j, i, coop]` are set to True. `battle_outcome[i, j, coop]` is set to +1 (i wins) or −1 (i loses); antisymmetric by construction.

- **RegionView** (`region_view`): SpectatorZone agents using `a_watch(c)` observe all battles that occurred in coop `c` this round. BattleZone agents observe all battles in their own coop. TransitZone agents observe nothing. Observations are stored as `last_observation[observer, i, j]` = outcome of the (i,j) battle this round, from observer's perspective.

**6. Transit arrival** (`arrive_from_transit`)

TransitZone agents arrive in the target coop, entering BattleZone.

**7. Belief update**

Ability beliefs are updated incrementally from the round's battles using Bayesian variable elimination (see [Inference](#inference)).

### Crown Assignment

At `close_tournament`, crowns are assigned **per coop** among agents that fought at least once in that coop that tournament:

- If any agents have `loss_count[coop] == 0` (zero losses, non-dominated): **all** such agents receive one crown.
- Otherwise: **all** agents tied at the maximum `win_count[coop]` receive one crown.

Multiple simultaneous crowns per coop are possible when abilities are equal. Each crown is immediately deposited into the winner's sack (`sack[agent] += crowns_earned`).

After crown assignment, `new_tournament` resets `loss_count`, `win_count`, `has_battled`, and `battle_outcome` to zero. `cumulative_battles` (total battles across all tournaments) is preserved and used to determine placement in the new tournament: agents who have fought more are placed in their best coop, others placed uniformly at random.

### Sack Mechanics and Final Scoring

Each agent owns a **sack** — a persistent crown wallet that accumulates across all 256 tournaments and is never reset mid-game. Sacks can have **co-owners**: any agent listed as a co-owner receives a share of that sack's value at the end.

Final score for agent j:

$$\text{score}[j] = \sum_i \frac{\text{sack}[i]}{|\text{owners}(i)|} \quad \text{for all } i \text{ where } \text{sack\_owners}[i,j] = \text{True}$$

**Example:** Agent A has sack=120, shared with B and C. Agent B has sack=60, shared with A. Agent C has sack=40, shared with A.

- A scores: 120/3 + 60/2 + 40/2 = 40 + 30 + 20 = **90**
- B scores: 120/3 + 60/2 = 40 + 30 = **70**
- C scores: 120/3 + 40/2 = 40 + 20 = **60**

A wins solo despite sharing both its own sack and receiving shares of others'. If all three had extended ownership to each other, all would share equally and win together.

### Social Actions

Three social actions are available to strategic agents after each environment step. There is no limit on how many social actions an agent can issue per round. NPCs never use social actions.

**`ShareObservations(target)`**

Copies the caller's full observation history into target's `observation_memory`. For all positions `(i, j)` where the target has no data (value 0), the caller's observation is written in. This improves the target's ability belief inference without requiring the target to have been present for those battles.

**`TransferCrowns(target, count)`**

Moves `min(count, sack[caller])` crowns from the caller's sack to the target's sack. Clamped to the caller's current balance — cannot go negative.

**`ExtendSackOwnership(target)`**

Adds target as a co-owner of the caller's sack (`sack_owners[caller, target] = True`). Only the original owner of a sack can extend ownership. Co-ownership is asymmetric and unilateral — the target does not need to consent, and this does not make the caller a co-owner of the target's sack. Co-owners share the sack's value equally at final scoring. Once extended, co-ownership cannot be revoked.

Social actions execute outside the JAX JIT loop in NumPy and are synced back into JAX state after each step.

### Observation Model

Observations are stored in `observation_memory` of shape `(M, M, M, T)`, accumulated in NumPy outside the JIT loop:

```
observation_memory[observer, i, j, tournament] = outcome of (i,j) battle in that tournament
  +1 : observer saw i beat j
  -1 : observer saw i lose to j
   0 : observer did not see this battle (or it didn't occur)
```

An observer records a battle if and only if:
- They were in BattleZone of that coop during that round (observers see their own coop unconditionally), or
- They were in SpectatorZone and issued `a_watch(c)` targeting that coop

TransitZone agents record nothing. Observation memory is observer-specific — two agents in the same coop will have the same observations, but an agent that was elsewhere will have zeros for that battle.

`last_observation` (shape `(M, M, M)`) stores only the most recent round's outcomes and is the value written into the cumulative `observation_memory` buffer each step.

### Win Condition

The simulation ends after T=256 tournaments. Final scores are computed from the sack state using the formula above. **If any or all agents within a set of 4 are among the highest-scoring chickens, the set wins.** The winning condition is collective — a single well-placed agent benefits the whole team if sack co-ownership has been established.

---

## Inference

The inference module (`inference/domain_inference.py`) implements three Bayesian belief computations over the truncated-Poisson ability prior. All three are available to strategic agents via the `ability_belief` and `king_belief` arrays passed through `AllAgentState`.

### Ability Belief

Each agent maintains an ability belief over every other agent, per coop, updated incrementally each round:

```
ability_belief[observer, subject, coop, v] = P(ability[subject, coop] == v+1 | observations)
```

Shape: `(M, M, N_coop, N_coop)`, initialized to a uniform prior `1/N_coop`.

Updates use Bayesian variable elimination over the v-structure Bayes net:

```
Ability_i → BattleOutcome(i,j) ← Ability_j
```

For each observed battle (i,j) in coop c with outcome o, the observer updates beliefs about both i and j:

$$P(a_i | \text{obs}) \propto P(a_i) \cdot \sum_{a_j} P(o | a_i, a_j) \cdot P(a_j | \text{obs})$$

where:
- `P(i wins | a_i, a_j) = 1.0` if `a_i > a_j`
- `P(i wins | a_i, a_j) = 0.0` if `a_i < a_j`
- `P(i wins | a_i, a_j) = 0.5` if `a_i == a_j`

A small Laplace-smoothing floor (`1e-8`) is added before normalizing to prevent belief collapse when conflicting evidence would otherwise produce a zero posterior.

### P(i beats j)

```python
from inference.domain_inference import p_i_beats_j

prob = p_i_beats_j(i, j, coop, ability_belief[observer], N_coop)
# Returns scalar float: P(i beats j in coop | observer's beliefs)
```

Computed by marginalizing over the joint ability distribution:

$$P(i \text{ beats } j) = \sum_{a_i, a_j} P(i \text{ beats } j \mid a_i, a_j) \cdot P(a_i) \cdot P(a_j)$$

O(N_coop²) per query.

### King Belief

At each tournament close, king beliefs are recomputed from ability beliefs via expected pairwise wins:

```
king_belief[observer, subject, coop, 1] = P(subject is king in coop | observer's beliefs)
```

Shape: `(M, M, N_coop, 2)`. The score for each chicken is its expected number of wins against all other chickens (normalized to a simplex over M chickens per coop):

$$\text{score}[m, c] = \sum_{j \neq m} P(m \text{ beats } j \mid \text{obs})$$

### Ability Profile

```python
from inference.domain_inference import p_ability_profile

profile = p_ability_profile(j, ability_belief[observer], N_coop)
# Returns (N_coop, N_coop): profile[c, v] = P(ability[j, c] == v+1)
```

Under the independence assumption across coops, this is the full marginal profile distribution for agent j.

---

## Reference Implementation

### NPC Strategy

56 NPCs (indices 8–63) use a greedy strategy: always move to the coop where their personal ability is highest. If already there, watch. No social actions are ever used.

```python
class NPC:
    def choose_action(self, zone_array, king_belief, abilities, N_coop, key=None):
        best_coop    = int(np.argmax(abilities))
        current_coop = int(np.argmax(zone_array[self.agent_idx].any(axis=-1)))
        if current_coop == best_coop:
            return best_coop + 1           # a_watch
        return N_coop + 1 + best_coop     # a_move

    def choose_social_actions(self, sack, sack_owners, observation_memory, tournament_count):
        return []
```

Because abilities are fixed and known to each NPC exactly, `argmax(abilities)` never changes. NPCs converge immediately to their optimal coop and stay there for all 256 tournaments. This sets a high performance baseline: strategic agents need a structurally favorable ability draw to place in the top 10 against 56 near-optimal competitors.

### Set-A: Tournament-Level PPO

Indices 0–3. Proximal Policy Optimization using a Flax `ActorCritic` network: two separate branches (actor and critic), each with two 64-unit tanh hidden layers with orthogonal initialization. The actor outputs a 9-way Categorical distribution over the discrete action space. The critic outputs a scalar value estimate.

**Observation vector (14 dimensions):**

| Indices | Content |
|---|---|
| 0:4 | Own abilities / N_coop (normalized to [0, 1]) |
| 4:8 | P(self is king) per coop, from `king_belief` |
| 8 | Own sack / 1024.0 |
| 9 | `tournament_count / T` (game progress signal) |
| 10:14 | Teammate sack values / 1024.0 |

**Training loop:** One trajectory is collected per tournament (~17 transitions on average). At tournament close, the last transition receives a sparse shaped reward:

$$r = \Delta\text{crowns} + 0.3 \times (w_t - w_{t-1})$$

where $\Delta\text{crowns} = \text{sack[agent]} - \text{prev\_sack}$ and $w_t$ is the win rate `win_count / (win_count + loss_count)` in the agent's best coop at tournament close $t$. The win-rate delta term reduces variance on the sparse crown signal. All within-tournament transitions have reward 0.0. GAE advantages (γ=0.99, λ=0.95) propagate the signal backwards through the trajectory.

**PPO update** (at each tournament close): 4 epochs × 4 minibatches, clip ε=0.2, value coef=0.5, entropy coef=0.01, Adam lr=2.5e-4 with global norm clipping at 0.5. Trajectories are padded to a fixed multiple of `NUM_MINIBATCHES` to avoid JAX retracing on variable-length rollouts.

**Social strategy:**
- Tournament 1 only: extend sack ownership to all 3 teammates (`ExtendSackOwnership`)
- Every step: share observations with all 3 teammates (`ShareObservations`)
- After tournament 10: if a clear team leader emerges (>5 crowns and >1.5× the second-highest teammate), non-leaders transfer their full sack balance to the leader (`TransferCrowns`)

**Agent interface:**

```python
class AgentSetA:
    def __init__(self, agent_idx, abilities, N_coop, teammates):
        ...

    def choose_action(self, zone_array, king_belief, abilities, N_coop, key=None) -> int:
        # Builds 14-dim obs, runs ActorCritic forward pass, samples action
        # Appends Transition to trajectory buffer
        ...

    def choose_social_actions(self, sack, sack_owners, observation_memory,
                               tournament_count, win_count=0, loss_count=0) -> list[dict]:
        # At tournament close: assigns reward, runs PPO update, clears buffer
        # Returns list of social action dicts
        ...
```

### Set-B: MCTS/UCT Coop Selection

Indices 4–7. Frames coop selection as a planning problem. No neural network. Architecturally distinct from Set-A: model-based planning via forward simulation rather than model-free policy gradient.

**UCT selection:** At each tournament close, runs `n_iter=20` UCT iterations. Each iteration selects the coop with the highest UCT score, where $n$ is the total visit count across all coops and $N[c]$ is visits to coop $c$:

$$\text{UCT}(c) = Q[c] + C \cdot \sqrt{\frac{\log(n + 1)}{N[c] + 1}}, \quad C = 1.0$$

then simulates `MCTS_K` rollouts in that coop using the ability belief forward model. After all iterations, commits to `argmax(Q)` as the target coop for the next tournament.

**Single rollout for coop c:**
1. Identify opponents present in coop c from the last observed `zone_array` (fallback: sample from full population)
2. For each simulated battle step, sample an opponent j and resolve win/loss using `p_i_beats_j(self_idx, j, c, ability_belief, N_coop)`
3. Track losses; stop if losses ≥ k
4. Apply crown rule: crown if 0 losses (non-dominated), otherwise if tied at max wins among simulated participants
5. Return 1 if crowned, 0 otherwise

Q[c] and N[c] are updated with the mean rollout return after each iteration. As the ability belief posterior sharpens with more observed battles, the forward model becomes more accurate and UCT can increasingly exploit real differences between coops.

**State tracked per agent:**

```python
self._uct_Q: np.ndarray   # float64 (N_coop,) — mean simulated crowns per coop
self._uct_N: np.ndarray   # int64   (N_coop,) — total rollout visits per coop
self._K: int               # rollouts per coop per UCT iteration
self._n_iter: int = 20     # UCT iterations per tournament close
self._last_ability_belief  # (M, N_coop, N_coop) — stored from social call
self._last_zone_array      # (M, N_coop, 3) — stored from action call
```

**Social strategy:** Symmetric with Set-A — extend sack ownership to all 3 teammates at tournament 1; share observations with all 3 teammates every step; transfer crowns to team leader after tournament 10.

**Agent interface:**

```python
class AgentSetB:
    def __init__(self, agent_idx, abilities, N_coop, teammates, k_rollouts=10):
        ...

    def choose_action(self, zone_array, king_belief, abilities, N_coop, key=None) -> int:
        # Moves toward self._current_coop; watches if already there
        # Stores zone_array for rollout opponent estimation
        ...

    def choose_social_actions(self, sack, sack_owners, observation_memory,
                               tournament_count, ability_belief=None) -> list[dict]:
        # At tournament close: runs UCT, updates self._current_coop
        # Returns list of social action dicts
        ...
```

---

## Experiments

### Ability Draw Structure

The PMF `[0.5553, 0.2623, 0.1239, 0.0585]` means that most agents draw ability 1 or 2. With 64 agents and 4 coops, a strategic agent with ability 2 competing against 12 NPCs at the same ability level wins approximately `1/13 ≈ 7.7%` of tournaments in that coop regardless of strategy. **The ability draw is the first-order driver of outcomes at T=256.** A strategic agent needs a favorable draw — high ability in a coop with few NPC rivals — to accumulate crowns systematically.

### K-Sweep (SEED=42, K=1,5,10,20,50)

SEED=42 gives Set-B a structurally hard draw — all four agents have abilities 2–3 competing against 10–12 NPC rivals at the same level. Set-A drew ability 4 in coops 2 and 3 with only 2–3 NPC rivals.

```
chicken_4  Set-B  abilities [2,2,2,2]  best coop 0  12 NPC rivals at ability 2
chicken_5  Set-B  abilities [2,2,2,2]  best coop 0  12 NPC rivals at ability 2
chicken_6  Set-B  abilities [1,3,2,2]  best coop 1  11 NPC rivals at ability 3
chicken_7  Set-B  abilities [3,2,1,1]  best coop 0  10 NPC rivals at ability 3
```

| K | Set-A mean score | Set-B mean score | Set-B best | Set-B top 10 | Sim time |
|---|---|---|---|---|---|
| 1  | 41.00 | 0.00 | 0.00 | No | 13m 13s |
| 5  | 54.00 | 1.25 | 1.25 | No | 19m 36s |
| 10 | 51.50 | 0.00 | 0.00 | No | 27m 57s |
| 20 | 46.50 | 0.25 | 0.25 | No | 1h 4m 15s |
| 50 | 38.25 | 0.25 | 0.25 | No | 2h 12m 34s |

Varying K from 1 to 50 produces no systematic improvement. This is the correct output of a planning algorithm on a flat reward landscape — with 12 rivals at equal ability, all coops yield the same expected reward, so UCT correctly converges to uniform visits:

```
K=10 UCT visit counts per coop (chicken_4, SEED=42):
  [1280, 1280, 1280, 1280]  — uniform, confirming no exploitable structure
```

Set-B fractional scores at K=5 (1.25) and K=20/50 (0.25) reflect co-ownership pooling from a single lucky crown — not a strategy effect. Simulation time scales approximately linearly with K because the per-tournament MCTS cost dominates: each close runs `n_iter × K = 20K` rollout calls in a Python loop.

Full terminal output for each K is in `tracing/k_sweep/`.

### Variance Run (SEED=None, K=10)

An unseeded run with a structurally better Set-B draw:

```
chicken_4  Set-B  abilities [1,1,2,3]  best coop 3   7 NPC rivals at ability 3
chicken_5  Set-B  abilities [3,1,3,1]  best coop 0  10 NPC rivals at ability 3
chicken_6  Set-B  abilities [3,2,1,1]  best coop 0  10 NPC rivals at ability 3
chicken_7  Set-B  abilities [1,1,1,1]  best coop 0  32 NPC rivals at ability 1
```

```
Set-A scores: all 32.00  (128 crowns pooled, 4-way co-ownership)
Set-B scores: all 7.50   (30 crowns pooled, 4-way co-ownership)
Set-A in top 10: Yes (ranks 9 and 10)
Set-B in top 10: No
Sim time: 30m 31s
```

With a better draw, UCT correctly identifies and exploits structure: chicken_4 accumulated 17,530 visits on coop 3 vs. ~11,180 on each other coop (1.57× preference ratio) and earned 3 crowns there. The forward model is working — the bottleneck is whether the ability draw creates a detectable signal between coops.

### Key Findings

**Ability draw dominates strategy.** Across K=1 to K=50 with SEED=42, Set-B scores stay near zero because no coop is differentiable — all are equally competitive for these agents. This is not an algorithm failure; it is the correct behavior on an undifferentiated reward landscape.

**MCTS works when structure exists.** In the unseeded run, UCT correctly concentrates visits on the better coop when the ability draw creates a detectable difference. The `p_i_beats_j` forward model over belief-propagation posteriors is accurate enough to guide planning.

**Symmetric co-ownership is robust.** All 4 agents in each set share sack ownership from tournament 1, so a single well-placed agent lifts the whole team's final score. The crown-concentration transfer rule (tournament 10+) amplifies this further.

**PPO convergence is limited by reward sparsity.** With ~17 transitions per tournament and 256 total updates, the policy gradient signal is too weak to reliably learn coop selection within a single run. Set-A's crown accumulation is partly explained by the tournament crown-placement mechanic — a winner is moved to the coop where it has won most, pulling it toward its strongest coop independently of the policy.

**K=10 selected as submission default.** K=10 sits in the middle of the sweep range, has two documented runs providing the most comparative evidence, and completes in ~28 minutes. K=5 produced the best Set-B score in the sweep but via co-ownership pooling from a single lucky crown, not a strategy effect.

---

## Tests

```bash
pytest tests/ -v
```

| File | What it covers |
|------|----------------|
| `test_env.py` | `EnvParams` defaults, ability PMF, `reset_env` shapes + invariants, `step_env` outputs, belief update correctness under JIT |
| `test_battle.py` | Deterministic win (higher ability), tie randomness (~50%), truncated Poisson distribution, outcome matrix population |
| `test_events.py` | Each event module in isolation: cage assignment, battle resolution, region view, relocation, tournament lifecycle |
| `test_invariants.py` | Zone-occupancy invariant, array shapes, outcome antisymmetry, ability matrix — checked after reset and multi-step sequences |
| `test_preconditions.py` | No duplicate battles per tournament, loss-limit exclusion, fewest-losses pairing order, observation gating, transit arrival eligibility |
| `test_tournament.py` | Crown rules, reset state, uniform placement, sack initialization |
| `test_sack.py` | Sack deposit, co-ownership mechanics, final score formula |
| `test_agents.py` | NPC action/social, AgentSetA PPO trajectory + shaped reward + co-ownership, AgentSetB UCT convergence + visit counts + symmetric social |

---

## Architecture

```
pec-king-order/
├── run.py                        # Entry point: config, simulation loop, final scoring
├── requirements.txt
├── conftest.py                   # pytest path resolution
├── example_runs/                 # Pre-generated outputs — inspect without running
│   ├── Seed=42/                  # K=10, SEED=42 (K-sweep reference run)
│   │   ├── chicken_abilities.txt
│   │   ├── chicken_trace.txt
│   │   ├── agent_setA_{0..3}.txt
│   │   ├── agent_setB_{4..7}.txt
│   │   └── run_summary.md        # Terminal output + final scores
│   └── Seed=Random/              # K=10, unseeded (variance reference run)
│       ├── chicken_abilities.txt
│       ├── chicken_trace.txt
│       ├── agent_setA_{0..3}.txt
│       ├── agent_setB_{4..7}.txt
│       └── run_summary.md
├── agents/
│   └── agent.py                  # NPC, AgentSetA (PPO), AgentSetB (MCTS/UCT)
├── envs/
│   ├── pec_king_order.py         # PecKingOrder env (reset_env, step_env)
│   ├── state.py                  # EnvState, AllAgentState, PecKingState, EnvParams
│   └── spaces.py                 # Observation / action space definitions
├── events/
│   ├── assign_agents.py          # AssignAgentsToCage + unpaired relocation
│   ├── dominance_battle.py       # Battle resolution, outcome + history updates
│   ├── region_view.py            # Observation distribution to all watchers
│   ├── relocate.py               # RelocateRegion (move) + arrive_from_transit
│   ├── tournament.py             # CloseTournament (crowns → sack) + NewTournament (reset)
│   └── social.py                 # ShareObservations, TransferCrowns, ExtendSackOwnership, compute_final_scores
├── inference/
│   ├── domain_inference.py       # p_i_beats_j, p_ability_profile, incremental belief update, king belief
│   ├── enumeration.py            # Variable elimination utilities
│   └── belief_propagation.py    # Belief propagation (batch update from full history)
├── tracing/                      # Runtime output — generated each run
│   ├── chicken_abilities.txt
│   ├── chicken_trace.txt
│   ├── agent_set{A,B}_*.txt
│   └── k_sweep/                  # Captured terminal output from K-sweep experiments
│       ├── run_k1_seed42.txt
│       ├── run_k5_seed42.txt
│       ├── run_k10_seed42.txt
│       ├── run_k20_seed42.txt
│       ├── run_k50_seed42.txt
│       └── run_k10_seedNone.txt
└── tests/
    ├── test_env.py
    ├── test_battle.py
    ├── test_events.py
    ├── test_invariants.py
    ├── test_preconditions.py
    ├── test_tournament.py
    ├── test_sack.py
    └── test_agents.py
```

---

## State Reference

**`EnvState`** — objective simulation state (all JAX arrays, pytree via `@chex.dataclass`):

| Field | Shape | dtype | Notes |
|-------|-------|-------|-------|
| `zone_array` | (M, N, 3) | int32 | One-hot: [agent, coop, zone]; zones 0=BZ, 1=SZ, 2=TZ |
| `action_array` | (M, 1) | int32 | 0=noop, 1–N=watch coop N−1, N+1–2N=move to coop N−N_coop |
| `battle_outcome` | (M, M, N) | int8 | Current round: +1/−1/0; antisymmetric |
| `battle_history` | (M, M, N, T) | int32 | Cumulative outcomes across all tournaments |
| `has_battled` | (M, M, N) | bool | True if pair fought in this coop this tournament; reset each close |
| `abilities` | (M, N) | int32 | Fixed ability scores per agent per coop; values in {1,…,N_coop} |
| `ability_matrix` | (N, M, N) | int32 | `[v,m,c]=1` iff `abilities[m,c] ≥ v+1` |
| `loss_count` | (M, N) | int32 | Losses this tournament per agent per coop; reset each close |
| `win_count` | (M, N) | int32 | Wins this tournament per agent per coop; reset each close |
| `cumulative_battles` | (M, N) | int32 | Total battles across all tournaments; used for new-tournament placement |
| `sack` | (M,) | int32 | Crown wallet per agent; accumulates across all tournaments; never reset |
| `sack_owners` | (M, M) | bool | `[i,j]=True` iff agent j co-owns agent i's sack; initialized as identity |
| `battle_pair` | (M//2, N, 2) | int32 | Current cage assignments per coop; −1 if empty |
| `cage_occupied` | (M//2, N) | bool | True if cage assigned this round |
| `king` | (M, N) | bool | True for each agent that won a crown in each coop this tournament |
| `tournament_step` | scalar | int32 | Rounds elapsed in current tournament |
| `tournament_count` | scalar | int32 | Completed tournaments |
| `done` | scalar | bool | True after T tournaments |

**`AllAgentState`** — per-agent belief state (M agents stacked, pytree via `@chex.dataclass`):

| Field | Shape | dtype | Notes |
|-------|-------|-------|-------|
| `ability_belief` | (M, M, N, N) | float32 | `[obs, subj, coop, v]` = P(ability[subj, coop] == v+1 \| obs's observations) |
| `king_belief` | (M, M, N, 2) | float32 | `[obs, subj, coop, 1]` = P(subj is king in coop \| obs's beliefs); recomputed at tournament close |
| `my_ability` | (M, N) | int32 | Each agent's own known ability scores |
| `my_location` | (M, 2) | int32 | [coop_idx, zone_idx] per agent |
| `last_observation` | (M, M, M) | int8 | Most recent round's battle outcomes, per observer |

`observation_memory` (shape `(M, M, M, T)`, int16) is maintained in NumPy outside JAX state to avoid carrying ~500 MB through every JIT call. It is accumulated in `run.py` and passed directly to agents' `choose_social_actions`.

### JAX / gymnax Notes

- `step_env` is JIT-compiled via `jax.jit(env.step_env, static_argnums=(3,))`. `params` must be static because Python loop bounds and array shape expressions are resolved at trace time.
- All conditionals on traced values use `jax.lax.cond`; all loops over dynamic bounds use `jax.lax.fori_loop`. Static `N_coop` loops are unrolled at trace time.
- `@chex.dataclass` makes all state structs JAX pytrees — they pass through `jit` without retracing.
- Sack state (`sack`, `sack_owners`), social action dispatch, and `observation_memory` all live outside the JIT loop in NumPy. Sack state is synced back into JAX state after each step via `state.replace(sack=..., sack_owners=...)`.
- The tournament close/open check uses `jax.lax.cond` inside `step_env` so it is safe under JIT. Both branches return the same pytree structure.
