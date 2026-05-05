"""
run.py — Pec-King Order simulation runner (HW5)

Runs T=256 tournaments with M=64 agents:
  - 4  Set-A agents  (indices 0-3)   — Strategy A (PPO: tournament-level policy gradient)
  - 4  Set-B agents  (indices 4-7)   — Strategy B (MCTS/UCT: Monte Carlo coop selection)
  - 56 NPC agents    (indices 8-63)  — always move to best-ability coop

After simulation, computes final sack scores and reports the winner.

Per-agent partial observability: each agent maintains its own AbilityBelief
and KingBelief, updated incrementally after each observation.  Social actions
(ShareObservations, TransferCrowns, ExtendSackOwnership) are dispatched
outside the JIT loop after each step.

Usage:
    python run.py
    Set SEED to an int for reproducibility, or None for random.
    Set INTERACTIVE = True to enter belief query mode after simulation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import jax
import jax.numpy as jnp

from envs.pec_king_order import PecKingOrder
from envs.state import EnvParams, PecKingState
from inference.domain_inference import (
    p_i_beats_j,
    p_ability_profile,
)
from agents.agent import NPC, AgentSetA, AgentSetB
from events.social import apply_social_action, compute_final_scores

# ==================================================================
# Configuration — edit these flags to control the simulation
# ==================================================================
M      = 64       # Number of agents (spec: 64)
N_COOP = 4        # Number of coops/regions (spec: 4)
K      = 4        # Max losses before exclusion (spec: 4)
T      = 256      # Number of tournaments (HW5 spec: 256)
MCTS_K = 10       # MCTS rollouts per coop per UCT iteration — sweep for experiments
SEED        = None     # None -> random each run; int -> reproducible
INTERACTIVE = False    # True -> belief query mode after simulation
VERBOSE     = False    # True -> print sampled abilities for first few agents at startup

# Agent index assignments (fixed for entire run)
AGENT_SET_A_INDICES = list(range(0, 4))    # chickens 0-3
AGENT_SET_B_INDICES = list(range(4, 8))    # chickens 4-7
NPC_INDICES         = list(range(8, 64))   # chickens 8-63

# Chicken tracing — log one chicken's actions/battles to a file
TRACK_CHICKEN = True       # Enable per-chicken trace logging
TRACK_IDX     = None       # None -> random; int -> specific chicken index
SELECT_IDX    = False      # True -> prompt user to pick chicken at startup
TRACK_FILE    = "tracing/chicken_trace.txt"
# ==================================================================


# ------------------------------------------------------------------
# Vectorized action policy — kept for reference (HW3 style)
# HW5 uses per-agent dispatch via agent.choose_action() instead.
# ------------------------------------------------------------------

def batch_action_policy(zone_array, king_probs, abilities, N_coop, M):
    """Compute actions for ALL M agents in one vectorized call (HW3 style).

    Strategy:
      - Go to the coop where the agent infers it is most likely to be King.
      - Fallback to own ability argmax when king_belief is still uniform.
      - If already in best coop: a_watch (stay + observe).
      - Otherwise: a_move (move to best coop).

    Parameters
    ----------
    zone_array : int32, (M, N_coop, 3) — current zone occupancy
    king_probs : float32, (M, N_coop) or None — per-agent self-king probability
    abilities  : int32, (M, N_coop) — ability scores
    N_coop     : int — number of coops
    M          : int — number of agents

    Returns
    -------
    int32, (M, 1) — action per agent
    """
    # Current coop per agent: which coop has any zone bit set
    coop_active = zone_array.any(axis=-1)          # (M, N_coop) bool
    current_coop = jnp.argmax(coop_active, axis=1)  # (M,)

    if king_probs is None:
        best_coop = jnp.argmax(abilities, axis=1).astype(jnp.int32)
    else:
        is_uniform = jnp.all(
            jnp.abs(king_probs - king_probs[:, :1]) < 1e-6, axis=1
        )  # (M,) bool
        ability_best = jnp.argmax(abilities, axis=1).astype(jnp.int32)
        king_best    = jnp.argmax(king_probs, axis=1).astype(jnp.int32)
        best_coop    = jnp.where(is_uniform, ability_best, king_best)

    stay = (best_coop == current_coop)
    watch_action = current_coop + jnp.int32(1)
    move_action  = jnp.int32(N_coop + 1) + best_coop

    return jnp.where(stay, watch_action, move_action).astype(jnp.int32).reshape(M, 1)


def _agent_type_label(idx: int) -> str:
    """Return 'SetA', 'SetB', or 'NPC' for a given chicken index."""
    if idx in AGENT_SET_A_INDICES:
        return "SetA"
    if idx in AGENT_SET_B_INDICES:
        return "SetB"
    return "NPC"


if __name__ == "__main__":
    params = EnvParams(M=M, N_coop=N_COOP, k=K, T=T)
    env    = PecKingOrder()

    # JIT-compile step_env with params as static
    step_jit = jax.jit(env.step_env, static_argnums=(3,))

    if SEED is not None:
        seed = SEED
    else:
        seed = int.from_bytes(os.urandom(4), "big")
    key = jax.random.PRNGKey(seed)

    # Reset environment
    key, reset_key = jax.random.split(key)
    obs, pec_state = env.reset_env(reset_key, params)
    state, agent_state = pec_state.env, pec_state.agents

    # ------------------------------------------------------------------
    # Build agent objects (one per chicken, sorted by index)
    # ------------------------------------------------------------------
    abilities_np = np.array(state.abilities)  # (M, N_coop)
    agents = []
    for i in AGENT_SET_A_INDICES:
        agents.append(AgentSetA(i, abilities_np[i], N_COOP, AGENT_SET_A_INDICES))
    for i in AGENT_SET_B_INDICES:
        agents.append(AgentSetB(i, abilities_np[i], N_COOP, AGENT_SET_B_INDICES, k_rollouts=MCTS_K))
    for i in NPC_INDICES:
        agents.append(NPC(i, abilities_np[i], N_COOP, []))
    agents.sort(key=lambda a: a.agent_idx)

    # Numpy mirrors of JAX sack state (kept in sync after every step)
    sack_np        = np.zeros(M, dtype=np.int32)
    sack_owners_np = np.eye(M, dtype=bool)

    print("=" * 62)
    print("  Pec-King Order Simulation -- CSCI 5512 HW5")
    print("=" * 62)
    print(f"  M={M} agents  |  N_coop={N_COOP}  |  k={K}  |  T={T} tournaments  |  seed={seed}")
    print(f"  Agent types: {len(AGENT_SET_A_INDICES)} Set-A  |  {len(AGENT_SET_B_INDICES)} Set-B  |  {len(NPC_INDICES)} NPCs")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Ability PMF (truncated Poisson): {np.array(params.ability_pmf).round(4)}")
    if VERBOSE:
        print(f"\nSampled abilities (first 4 agents):\n{np.array(state.abilities[:4])}")
        print(f"  (rows=agents, cols=coop ability scores, values in [1,{N_COOP}])")

    # ------------------------------------------------------------------
    # Ensure tracing directory exists and write chicken abilities file
    # ------------------------------------------------------------------
    os.makedirs("tracing", exist_ok=True)

    _abilities_np = np.array(state.abilities)  # (M, N_coop)
    with open("tracing/chicken_abilities.txt", "w") as _af:
        _af.write(f"=== All Chicken Abilities ===\n")
        _af.write(f"M={M}, N_coop={N_COOP}\n")
        _af.write(f"{'=' * 40}\n\n")
        hdr = f"{'#':>3s}  {'Name':<16s}  {'Type':<5s}" + "".join(f"  C{c}" for c in range(N_COOP))
        _af.write(hdr + "\n")
        for i in range(M):
            scores = "".join(f"  {int(_abilities_np[i, c]):2d}" for c in range(N_COOP))
            _af.write(f"{i:3d}  {'chicken_no' + str(i):<16s}  {_agent_type_label(i):<5s}{scores}\n")
        _af.write(f"\n{'=' * 40}\n")
        _af.write(f"Distribution (count per ability value per coop):\n")
        _af.write(f"{'Value':>5s}" + "".join(f"   C{c}" for c in range(N_COOP)) + "\n")
        for v in range(1, N_COOP + 1):
            counts = "".join(f"  {int((_abilities_np[:, c] == v).sum()):3d}" for c in range(N_COOP))
            _af.write(f"  {v:>3d}{counts}\n")
    print(f"  Abilities written to tracing/chicken_abilities.txt")

    # ------------------------------------------------------------------
    # Interactive chicken index selection
    # ------------------------------------------------------------------
    if SELECT_IDX and TRACK_CHICKEN:
        _user_input = input(f"  Select NPC index to track [{NPC_INDICES[0]}-{NPC_INDICES[-1]}] (anything else = random): ").strip()
        try:
            _selected = int(_user_input)
            if _selected in NPC_INDICES:
                TRACK_IDX = _selected
            else:
                TRACK_IDX = None
        except ValueError:
            TRACK_IDX = None

    # ------------------------------------------------------------------
    # Trace file setup — 8 strategic agents (always) + 1 NPC (if TRACK_CHICKEN)
    # ------------------------------------------------------------------
    ZONE_NAMES = {0: "BattleZone", 1: "SpectatorZone", 2: "TransitZone"}

    _trace_hdr = (
        f"{'Tourn':>5s} {'Step':>4s} | {'Pre-Location':>25s} -> {'Post-Location':<25s} | "
        f"{'Action':<34s} | {'W/L':<5s} | {'share_obs':<22s} | {'transfer':<12s} | extend_own"
    )

    # Open trace files for all 8 strategic agents (always, regardless of TRACK_CHICKEN)
    trace_files: dict = {}
    for _si in AGENT_SET_A_INDICES:
        _sf = open(f"tracing/agent_setA_{_si}.txt", "w")
        _sf.write(f"=== Set-A Agent Trace: chicken_{_si} ===\n")
        _sf.write(f"Abilities: {np.array(state.abilities[_si]).tolist()}\n")
        _sf.write(f"Simulation: M={M}, N_coop={N_COOP}, k={K}, T={T}\n")
        _sf.write(f"{'='*60}\n\n")
        _sf.write(_trace_hdr + "\n")
        _sf.write("-" * len(_trace_hdr) + "\n")
        trace_files[_si] = _sf
    for _si in AGENT_SET_B_INDICES:
        _sf = open(f"tracing/agent_setB_{_si}.txt", "w")
        _sf.write(f"=== Set-B Agent Trace: chicken_{_si} ===\n")
        _sf.write(f"Abilities: {np.array(state.abilities[_si]).tolist()}\n")
        _sf.write(f"Simulation: M={M}, N_coop={N_COOP}, k={K}, T={T}\n")
        _sf.write(f"{'='*60}\n\n")
        _sf.write(_trace_hdr + "\n")
        _sf.write("-" * len(_trace_hdr) + "\n")
        trace_files[_si] = _sf

    # Open NPC trace file if TRACK_CHICKEN (NPC only — not strategic agents)
    _track_idx = None
    if TRACK_CHICKEN:
        import random as _pyrandom
        if TRACK_IDX is not None:
            _track_idx = TRACK_IDX
        else:
            _track_idx = _pyrandom.choice(NPC_INDICES)
        _track_abilities = np.array(state.abilities[_track_idx])
        _npc_tf = open(TRACK_FILE, "w")
        _npc_tf.write(f"=== NPC Trace: chicken_{_track_idx} ===\n")
        _npc_tf.write(f"Abilities: {_track_abilities.tolist()}\n")
        _npc_tf.write(f"Simulation: M={M}, N_coop={N_COOP}, k={K}, T={T}\n")
        _npc_tf.write(f"{'='*60}\n\n")
        _npc_tf.write(_trace_hdr + "\n")
        _npc_tf.write("-" * len(_trace_hdr) + "\n")
        trace_files[_track_idx] = _npc_tf
        print(f"\n  [TRACK] NPC trace: chicken_{_track_idx} -> {TRACK_FILE}")
        print(f"  [TRACK] Abilities: {_track_abilities.tolist()}")
    print(f"  [TRACK] Strategic traces -> tracing/agent_setA_*.txt  tracing/agent_setB_*.txt")
    _all_tracked = sorted(trace_files.keys())

    # ------------------------------------------------------------------
    # JIT warm-up
    # ------------------------------------------------------------------
    print("  Compiling step_env...", end=" ", flush=True)
    _wup_key = jax.random.PRNGKey(0)
    _wup_act  = jnp.zeros((M, 1), dtype=jnp.int32)
    _wup_out  = step_jit(_wup_key, pec_state, _wup_act, params)
    _wup_out[1].env.done.block_until_ready()
    print("done", flush=True)
    print(f"  Starting simulation: {M} agents, {N_COOP} coops, {T} tournaments", flush=True)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    step_count = 0
    tournament_steps = []
    t_start = time.time()
    last_kings = None
    crown_counts = np.zeros((M, N_COOP), dtype=np.int32)

    # Per-tournament sack delta tracking (for trace close lines + Set-A analysis)
    _sack_at_last_close = np.zeros(M, dtype=np.int32)
    _setA_crown_history: dict = {idx: [] for idx in AGENT_SET_A_INDICES}  # list of (t, delta)

    battle_history_np     = np.zeros((M, M, N_COOP, T), dtype=np.int32)
    observation_memory_np = np.zeros((M, M, M, T),      dtype=np.int16)

    while not bool(state.done):
        splits     = jax.random.split(key, M + 2)
        key        = splits[0]
        step_key   = splits[1]
        agent_keys = splits[2:]   # (M,) — one per chicken index

        # ---- Per-agent action dispatch (HW5) ----
        zone_array_np    = np.array(state.zone_array)    # (M, N_coop, 3)
        abilities_np_cur = np.array(state.abilities)     # (M, N_coop)
        _agent_idx_diag  = jnp.arange(M)
        king_probs_np = np.array(
            agent_state.king_belief[_agent_idx_diag, _agent_idx_diag, :, 1]
        )  # (M, N_coop)

        actions = np.zeros(M, dtype=np.int32)
        for agent in agents:
            i = agent.agent_idx
            actions[i] = agent.choose_action(
                zone_array_np,
                king_probs_np[i],
                abilities_np_cur[i],
                N_COOP,
                key=agent_keys[i],
            )
        action_arr = jnp.array(actions).reshape(M, 1)

        # Capture pre-step location for all tracked agents
        _pre_info: dict = {}
        for _ti in _all_tracked:
            _z = zone_array_np[_ti]                          # (N_coop, 3)
            _c = int(np.argmax(_z.any(axis=-1)))
            _pre_info[_ti] = (_c, int(np.argmax(_z[_c])))   # (coop, zone_id)

        obs, pec_state, reward, done, info = step_jit(
            step_key, PecKingState(env=state, agents=agent_state), action_arr, params
        )
        state, agent_state = pec_state.env, pec_state.agents
        step_count += 1

        # Accumulate battle_history and observation_memory outside JIT
        _t = int(state.tournament_count)
        _t_clamped = min(_t, T - 1)
        battle_history_np[:, :, :, _t_clamped] += np.array(state.battle_outcome, dtype=np.int32)
        observation_memory_np[:, :, :, _t_clamped] += np.array(
            info["obs_received"], dtype=np.int16
        )

        # ---- Sync sack from JAX state ----
        sack_np = np.array(state.sack)

        # ---- Social actions (outside JIT) ----
        t_count = int(state.tournament_count)
        _step_social: dict = {}   # {agent_idx: list[dict]} — captured for trace logging
        for agent in agents:
            if isinstance(agent, AgentSetB):
                ab = np.array(agent_state.ability_belief[agent.agent_idx])
                social_actions = agent.choose_social_actions(
                    sack_np, sack_owners_np, observation_memory_np, t_count,
                    ability_belief=ab
                )
            elif isinstance(agent, AgentSetA):
                i = agent.agent_idx
                best_c = int(np.argmax(abilities_np_cur[i]))
                wc = int(np.array(state.win_count)[i, best_c])
                lc = int(np.array(state.loss_count)[i, best_c])
                social_actions = agent.choose_social_actions(
                    sack_np, sack_owners_np, observation_memory_np, t_count,
                    win_count=wc, loss_count=lc
                )
            else:
                social_actions = agent.choose_social_actions(
                    sack_np, sack_owners_np, observation_memory_np, t_count
                )
            _step_social[agent.agent_idx] = social_actions
            for sa in social_actions:
                apply_social_action(
                    sa, agent.agent_idx, sack_np, sack_owners_np, observation_memory_np
                )

        # ---- Push sack/sack_owners back into JAX state ----
        state = state.replace(
            sack=jnp.array(sack_np),
            sack_owners=jnp.array(sack_owners_np),
        )
        pec_state = PecKingState(env=state, agents=agent_state)

        # Write trace rows for all tracked agents
        _t_num    = int(state.tournament_count)
        _t_step   = int(state.tournament_step)
        _zone_post = np.array(state.zone_array)    # (M, N_coop, 3) — post-step
        _bat_post  = np.array(state.battle_outcome) # (M, M, N_coop)
        for _ti in _all_tracked:
            _pre_c, _pre_zid = _pre_info[_ti]
            # Post-step zone info
            _pz = _zone_post[_ti]
            _post_c   = int(np.argmax(_pz.any(axis=-1)))
            _post_zid = int(np.argmax(_pz[_post_c]))
            # Policy action
            _act = int(actions[_ti])
            if _act == 0:
                _pol = "no-op"
            elif 1 <= _act <= N_COOP:
                _pol = f"a_watch->coop{_act - 1}"
            else:
                _pol = f"a_move->coop{_act - N_COOP - 1}"
            # Effective action (depends on zone and whether battle happened)
            _bout_row = _bat_post[_ti]   # (M, N_coop)
            _had_bat  = np.any(_bout_row != 0)
            if _pre_zid == 0:
                _act_str = f"battle(policy:{_pol})" if _had_bat else f"unpaired(policy:{_pol})"
            elif _pre_zid == 2:
                _act_str = f"arriving(policy:{_pol})"
            else:
                _act_str = _pol
            # W/L counters at this agent's pre-step coop
            _wins   = int(state.win_count[_ti,  _pre_c])
            _losses = int(state.loss_count[_ti, _pre_c])
            # Social action columns
            _sa_list = _step_social.get(_ti, [])
            _so = ','.join(str(sa['target']) for sa in _sa_list if sa['type'] == 'share_observations')
            _tr = ','.join(f"{sa['target']}:{sa['count']}" for sa in _sa_list if sa['type'] == 'transfer_crowns')
            _eo = ','.join(str(sa['target']) for sa in _sa_list if sa['type'] == 'extend_sack_ownership')
            trace_files[_ti].write(
                f"T{_t_num:>4d} step{_t_step:>3d} | "
                f"{ZONE_NAMES[_pre_zid]:>20s}@coop{_pre_c} -> "
                f"{ZONE_NAMES[_post_zid]:<20s}@coop{_post_c} | "
                f"{_act_str:<34s} | W{_wins:>2d}/L{_losses:>2d} | "
                f"{_so:<22s} | {_tr:<12s} | {_eo}\n"
            )

        if info.get('tournament_done', False):
            tournament_steps.append(step_count)
            t = int(state.tournament_count)
            last_kings = np.array(info['king_repr'])
            crown_counts += np.array(info['king_masks'], dtype=np.int32).T
            # Inject tournament close summary line into every open trace file
            for _ti in _all_tracked:
                _delta = int(sack_np[_ti]) - int(_sack_at_last_close[_ti])
                trace_files[_ti].write(
                    f"  --- Tournament {t} closed | sack: {int(sack_np[_ti])} (+{_delta}) ---\n\n"
                )
                trace_files[_ti].flush()
            # Record per-tournament crown delta for Set-A analysis
            for _ai in AGENT_SET_A_INDICES:
                _delta = int(sack_np[_ai]) - int(_sack_at_last_close[_ai])
                _setA_crown_history[_ai].append((t, _delta))
            _sack_at_last_close = np.array(sack_np)
            if t % max(1, T // 10) == 0 or t == T:
                print(f"  Tournament {t:4d}/{T} done  (step {step_count})", flush=True)

    elapsed = time.time() - t_start
    completed = int(state.tournament_count)

    h  = int(elapsed) // 3600
    m  = (int(elapsed) % 3600) // 60
    s  = elapsed % 60
    duration_str = (f"{h}h {m}m {s:.1f}s" if h else f"{m}m {s:.1f}s" if m else f"{s:.2f}s")

    print(f"\n{'=' * 62}")
    print(f"  Simulation complete")
    print(f"{'=' * 62}")
    print(f"  {step_count} total steps across {completed} tournaments")
    print(f"  Total time:  {duration_str}")
    print(f"  Per step:    {elapsed/step_count*1000:.1f} ms")
    print(f"  Per tournament: {elapsed/completed:.2f}s avg")
    print(f"  BattleHistory nonzero entries: {int((battle_history_np != 0).sum())}")

    # ------------------------------------------------------------------
    # Final Sack Scores (HW5)
    # ------------------------------------------------------------------
    final_scores = compute_final_scores(sack_np, sack_owners_np)
    top10_idx    = np.argsort(final_scores)[::-1][:10]
    total_crowns = int(sack_np.sum())

    print(f"\n{'=' * 62}")
    print(f"  Final Sack Scores (HW5)")
    print(f"{'=' * 62}")
    print(f"  Total crowns in circulation: {total_crowns}")
    print(f"\n  Top 10 scorers:")
    print(f"  {'Rank':>4s}  {'Chicken':>8s}  {'Type':<5s}  {'Score':>8s}")
    print(f"  {'-'*35}")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"  {rank:>4d}  chicken_{idx:<4d}   {_agent_type_label(idx):<5s}  {final_scores[idx]:>8.2f}")

    top10_set = set(int(i) for i in top10_idx)
    set_a_wins = any(i in top10_set for i in AGENT_SET_A_INDICES)
    set_b_wins = any(i in top10_set for i in AGENT_SET_B_INDICES)
    print(f"\n  Set-A wins (any in top 10): {set_a_wins}")
    print(f"  Set-B wins (any in top 10): {set_b_wins}")

    # ------------------------------------------------------------------
    # Crown frequency summary
    # ------------------------------------------------------------------
    _abilities_np = np.array(state.abilities)  # (M, N_coop)

    print(f"\n  {'-' * 62}")
    print(f"  Crown Frequency Summary  ({completed} tournaments)")
    print(f"  {'-' * 62}")
    print(f"  {'Coop':>4s}  {'Sim King':>8s}  {'Crowns':>6s}  {'Rate':>6s}  {'Match':>5s}  Top-5 (chicken: crowns)")
    print(f"  {'-' * 62}")

    for _c in range(N_COOP):
        _col = crown_counts[:, _c]
        _order = np.argsort(_col)[::-1]
        _sim_king   = int(_order[0])
        _sim_crowns = int(_col[_sim_king])
        _rate = _sim_crowns / completed if completed > 0 else 0.0

        _max_ability = int(_abilities_np[:, _c].max())
        _true_kings  = [int(m) for m in np.where(_abilities_np[:, _c] == _max_ability)[0]]
        _match = "yes" if _sim_king in _true_kings else "no"
        _true_str = f"[{', '.join(str(m) for m in _true_kings)}]  (ability={_max_ability})"

        _top5 = "  ".join(
            f"c{int(_order[i])}:{int(_col[_order[i]])}"
            for i in range(min(5, M))
            if _col[_order[i]] > 0
        )
        print(f"  {_c:>4d}  {_sim_king:>8d}  {_sim_crowns:>6d}  {_rate:>5.1%}  {_match:>5s}  {_top5}")
        print(f"        Possible true kings: {_true_str}")

    print(f"  {'-' * 62}")

    # ------------------------------------------------------------------
    # HW5 Strategy Analysis
    # ------------------------------------------------------------------
    print(f"\n{'=' * 62}")
    print(f"  HW5 Strategy Analysis")
    print(f"{'=' * 62}")

    # 1. Score distribution by agent type
    print(f"\n  1. Score distribution by agent type")
    print(f"  {'Agent type':<10s} | {'Count':>5s} | {'Min':>8s} | {'Mean':>8s} | {'Max':>8s} | Top scorer")
    print(f"  {'-' * 72}")
    for _lbl, _idxs in [("NPC", NPC_INDICES), ("Set-A", AGENT_SET_A_INDICES), ("Set-B", AGENT_SET_B_INDICES)]:
        _sc = final_scores[list(_idxs)]
        _top = _idxs[int(np.argmax(_sc))]
        print(f"  {_lbl:<10s} | {len(_idxs):>5d} | {_sc.min():>8.2f} | {_sc.mean():>8.2f} | {_sc.max():>8.2f} | chicken_{_top} ({final_scores[_top]:.2f})")

    # 2. Set-B coop convergence
    print(f"\n  2. Set-B coop convergence")
    print(f"  {'Agent':<12s} | {'Final coop':>10s} | {'n_visits per coop':<28s} | {'True best':>9s} | Match")
    print(f"  {'-' * 72}")
    for _bi in AGENT_SET_B_INDICES:
        _ba = agents[_bi]
        _fc = _ba._current_coop
        _nv = _ba._uct_N.tolist()
        _tb = int(np.argmax(np.array(state.abilities[_bi])))
        _mt = "yes" if _fc == _tb else "no"
        print(f"  chicken_{_bi:<4d} | coop {_fc:>5d} | {str(_nv):<28s} | coop {_tb:>4d} | {_mt}")

    # 3. Set-A reward history
    print(f"\n  3. Set-A reward history")
    print(f"  {'Agent':<12s} | {'Total crowns':>12s} | {'Tourns':>6s} | {'Avg/tourn':>9s} | Best tourn")
    print(f"  {'-' * 65}")
    for _ai in AGENT_SET_A_INDICES:
        _hist = _setA_crown_history[_ai]
        _tot  = sum(d for _, d in _hist)
        _nt   = len(_hist)
        _avg  = _tot / _nt if _nt > 0 else 0.0
        _best = (f"t={max(_hist, key=lambda x: x[1])[0]} (+{max(_hist, key=lambda x: x[1])[1]})"
                 if _hist else "none")
        print(f"  chicken_{_ai:<4d} | {_tot:>12d} | {_nt:>6d} | {_avg:>9.2f} | {_best}")

    # 4. Sack co-ownership map
    print(f"\n  4. Sack co-ownership map")
    _any_coown = False
    for _i in range(M):
        _co = [_j for _j in range(M) if _j != _i and sack_owners_np[_i, _j]]
        if _co:
            _any_coown = True
            _nc    = 1 + len(_co)
            _split = int(sack_np[_i]) / _nc if _nc > 0 else 0.0
            print(f"  chicken_{_i:<4d} | co-owners {_co} | sack {int(sack_np[_i]):>6d} | {_split:.2f} per owner")
    if not _any_coown:
        print(f"  No sack co-ownership established.")

    # 5. Win determination
    print(f"\n  5. Win determination")
    _sa_sc = [round(float(final_scores[i]), 2) for i in AGENT_SET_A_INDICES]
    _sb_sc = [round(float(final_scores[i]), 2) for i in AGENT_SET_B_INDICES]
    print(f"  Set-A final scores: {_sa_sc}")
    print(f"  Set-B final scores: {_sb_sc}")
    _gmax = float(final_scores.max())
    _winners = []
    if any(abs(float(final_scores[i]) - _gmax) < 1e-9 for i in AGENT_SET_A_INDICES):
        _winners.append("Set-A")
    if any(abs(float(final_scores[i]) - _gmax) < 1e-9 for i in AGENT_SET_B_INDICES):
        _winners.append("Set-B")
    if any(abs(float(final_scores[i]) - _gmax) < 1e-9 for i in NPC_INDICES):
        _winners.append("NPC field")
    print(f"  Winner: {' / '.join(_winners) if _winners else 'none'}")

    # 6. Strategic agent ability context
    print(f"\n  6. Strategic agent ability context")
    print(f"  {'Agent':<12s} {'Type':<8s} {'Abilities':<22s} {'Best':>4s} {'Val':>4s} {'Rivals':>6s} {'Crowns':>7s}")
    print(f"  {'-'*68}")
    for _ai in AGENT_SET_A_INDICES + AGENT_SET_B_INDICES:
        _atype = "Set-A" if _ai in AGENT_SET_A_INDICES else "Set-B"
        _ab = abilities_np[_ai]
        _best_coop = int(np.argmax(_ab))
        _best_val  = int(_ab[_best_coop])
        _rivals = sum(1 for _ni in NPC_INDICES if int(abilities_np[_ni][_best_coop]) == _best_val)
        _crowns = int(sack_np[_ai])
        _ab_str = str(_ab.tolist())
        print(f"  chicken_{_ai:<4d} {_atype:<8s} {_ab_str:<22s} {_best_coop:>4d} {_best_val:>4d} {_rivals:>6d} {_crowns:>7d}")
    print()
    for _ai in AGENT_SET_A_INDICES + AGENT_SET_B_INDICES:
        _ab = abilities_np[_ai]
        _best_coop = int(np.argmax(_ab))
        _best_val  = int(_ab[_best_coop])
        _rivals = sum(1 for _ni in NPC_INDICES if int(abilities_np[_ni][_best_coop]) == _best_val)
        _difficulty = "hard" if _rivals >= 5 else ("moderate" if _rivals >= 2 else "easy")
        print(f"  chicken_{_ai}: ability={_best_val} in coop{_best_coop}, competing against {_rivals} NPCs with the same ability")
        print(f"    -> {_difficulty} field")

    print(f"\n  {'-' * 62}")

    # Close all trace files
    for _ti, _tf in trace_files.items():
        _tf.write(f"\n{'='*60}\n")
        _tf.write(f"Simulation complete: {step_count} steps, {completed} tournaments\n")
        _tf.close()
    _n_strat_files = len(AGENT_SET_A_INDICES) + len(AGENT_SET_B_INDICES)
    print(f"  [TRACK] Closed {_n_strat_files} strategic trace files"
          + (f" + NPC trace ({TRACK_FILE})" if TRACK_CHICKEN else ""))

    # ------------------------------------------------------------------
    # Interactive Belief Query Mode
    # ------------------------------------------------------------------

    def _interactive_loop():
        """Interactive prompt for querying per-agent beliefs after simulation."""

        ab_all = np.array(agent_state.ability_belief)
        kb_all = np.array(agent_state.king_belief)
        abilities = np.array(state.abilities)
        bh = battle_history_np

        def _validate_idx(val, label="chicken"):
            try:
                idx = int(val)
            except (ValueError, TypeError):
                print(f"  Invalid {label} index: {val}")
                return None
            if not (0 <= idx < M):
                print(f"  {label} index must be 0-{M-1}, got {idx}")
                return None
            return idx

        def _parse_from(tokens):
            if "from" in tokens:
                fi = tokens.index("from")
                if fi + 1 < len(tokens):
                    obs = _validate_idx(tokens[fi + 1], "observer")
                    remaining = tokens[:fi] + tokens[fi + 2:]
                    return remaining, obs
                else:
                    print("  'from' requires an observer index")
                    return tokens, -1
            return tokens, None

        def _parse_coop(tokens):
            if "coop" in tokens:
                ci = tokens.index("coop")
                if ci + 1 < len(tokens):
                    try:
                        c = int(tokens[ci + 1])
                    except ValueError:
                        print(f"  Invalid coop: {tokens[ci + 1]}")
                        return tokens, -1
                    if not (0 <= c < N_COOP):
                        print(f"  Coop must be 0-{N_COOP-1}, got {c}")
                        return tokens, -1
                    remaining = tokens[:ci] + tokens[ci + 2:]
                    return remaining, c
                else:
                    print("  'coop' requires a coop index")
                    return tokens, -1
            return tokens, None

        def _expected_p(a_i, a_j):
            if a_i > a_j: return 1.0
            elif a_i < a_j: return 0.0
            else: return 0.5

        def cmd_chicken(tokens):
            tokens, obs = _parse_from(tokens)
            if obs == -1: return
            if len(tokens) < 1:
                print("  Usage: chicken <idx> [from <observer>]")
                return
            idx = _validate_idx(tokens[0])
            if idx is None: return
            observer = obs if obs is not None else idx
            ab = ab_all[observer, :, :, :]
            kb = kb_all[observer, :, :, :]
            true_ab = abilities[idx]
            map_est = np.argmax(ab[idx], axis=1) + 1
            king_p = kb[idx, :, 1]
            best_coop = int(np.argmax(king_p))
            obs_label = f"chicken_{observer}" if observer != idx else "self"
            print(f"\n  chicken_{idx} ({_agent_type_label(idx)}, observed by {obs_label}):")
            print(f"    True abilities: {true_ab.tolist()}")
            print(f"    MAP estimate:   {map_est.tolist()}  "
                  f"{'(matches)' if np.array_equal(true_ab, map_est) else '(partial match)'}")
            print(f"    P(king): [{', '.join(f'{v:.4f}' for v in king_p)}]  best_coop={best_coop}")
            print(f"\n    MAP estimates of other chickens (first 16):")
            print(f"    {'#':>4s}  {'True':>12s}  {'MAP':>12s}  Match")
            for j in range(min(16, M)):
                t = abilities[j].tolist()
                m = (np.argmax(ab[j], axis=1) + 1).tolist()
                match = "yes" if t == m else "no"
                print(f"    {j:4d}  {str(t):>12s}  {str(m):>12s}  {match}")
            if M > 16:
                print(f"    ... ({M - 16} more, use 'profile <idx>' for details)")
            print()

        def cmd_beats(tokens):
            tokens, obs = _parse_from(tokens)
            if obs == -1: return
            tokens, coop_filter = _parse_coop(tokens)
            if coop_filter == -1: return
            if len(tokens) < 2:
                print("  Usage: beats <i> <j> [coop <c>] [from <observer>]")
                return
            i = _validate_idx(tokens[0], "chicken i")
            j = _validate_idx(tokens[1], "chicken j")
            if i is None or j is None: return
            if i == j:
                print("  Cannot compute self-battle probability")
                return
            observer = obs if obs is not None else i
            ab = ab_all[observer]
            obs_label = f"chicken_{observer}" if observer != i else f"chicken_{i}"
            coops = [coop_filter] if coop_filter is not None else range(N_COOP)
            print(f"\n  P(chicken_{i} beats chicken_{j} | obs of {obs_label}):")
            for c in coops:
                p = float(p_i_beats_j(i, j, c, ab, N_COOP))
                a_i, a_j = int(abilities[i, c]), int(abilities[j, c])
                exp = _expected_p(a_i, a_j)
                print(f"    coop {c}: P = {p:.4f}  (a_{i}={a_i} vs a_{j}={a_j}, expected: {exp:.1f})")
            print()

        def cmd_king(tokens):
            tokens, obs = _parse_from(tokens)
            if obs == -1: return
            if len(tokens) < 1:
                print("  Usage: king <idx> [from <observer>]")
                return
            idx = _validate_idx(tokens[0])
            if idx is None: return
            observer = obs if obs is not None else idx
            ab = ab_all[observer].astype(np.float64)
            obs_label = f"chicken_{observer}" if observer != idx else "self"

            king_p = np.zeros(N_COOP)
            for c in range(N_COOP):
                ab_c  = ab[:, c, :]
                cdf_c = np.cumsum(ab_c, axis=1)
                scores = np.zeros(M)
                for v in range(N_COOP):
                    log_cdf      = np.log(np.clip(cdf_c[:, v], 1e-30, None))
                    prod_others  = np.exp(log_cdf.sum() - log_cdf)
                    scores      += ab_c[:, v] * prod_others
                s = scores.sum()
                king_p[c] = scores[idx] / s if s > 1e-30 else 0.0

            best_coop = int(np.argmax(king_p))
            best_ability_coop = int(np.argmax(abilities[idx]))
            print(f"\n  chicken_{idx}'s king beliefs (observed by {obs_label}):")
            for c in range(N_COOP):
                print(f"    coop {c}: P(king) = {float(king_p[c]):.4f}")
            print(f"    Best coop by P(king): {best_coop}  "
                  f"(true strongest: coop {best_ability_coop}, "
                  f"ability={int(abilities[idx, best_ability_coop])})")
            print()

        def cmd_profile(tokens):
            tokens, obs = _parse_from(tokens)
            if obs == -1: return
            if len(tokens) < 1:
                print("  Usage: profile <idx> [from <observer>]")
                return
            idx = _validate_idx(tokens[0])
            if idx is None: return
            observer = obs if obs is not None else idx
            ab = ab_all[observer, idx, :, :]
            true_ab = abilities[idx]
            map_est = np.argmax(ab, axis=1) + 1
            obs_label = f"chicken_{observer}" if observer != idx else "self"
            print(f"\n  Ability belief for chicken_{idx} (observed by {obs_label}):")
            print(f"    True abilities: {true_ab.tolist()}")
            print(f"    MAP estimate:   {map_est.tolist()}  "
                  f"{'(matches)' if np.array_equal(true_ab, map_est) else '(partial match)'}")
            print(f"    Posterior P(ability = v+1 | obs):")
            for c in range(N_COOP):
                probs = [f"{float(ab[c, v]):.4f}" for v in range(N_COOP)]
                print(f"      coop {c}: [{', '.join(probs)}]")
            print()

        def cmd_abilities(tokens):
            if len(tokens) >= 1:
                idx = _validate_idx(tokens[0])
                if idx is None: return
                print(f"\n  chicken_{idx} abilities: {abilities[idx].tolist()}")
                print()
            else:
                print(f"\n  {'#':>4s}  {'Name':<16s}  {'Type':<5s}" + "".join(f"  C{c}" for c in range(N_COOP)))
                for i in range(M):
                    scores = "".join(f"  {int(abilities[i, c]):2d}" for c in range(N_COOP))
                    print(f"  {i:4d}  chicken_{i:<8d}  {_agent_type_label(i):<5s}{scores}")
                print()

        def cmd_help(_tokens):
            print("""
  Commands (all belief commands accept optional 'from <observer>'):
    chicken <idx> [from <obs>]              Overview of beliefs about a chicken
    beats <i> <j> [coop <c>] [from <obs>]  P(i beats j | observations)
    king <idx> [from <obs>]                 P(king) per coop for a chicken
    profile <idx> [from <obs>]              Ability posterior (MAP vs true)
    abilities [idx]                         True abilities (ground truth)
    help                                    Show this message
    quit / q                                Exit
            """)

        dispatch = {
            "chicken": cmd_chicken,
            "beats": cmd_beats,
            "king": cmd_king,
            "profile": cmd_profile,
            "abilities": cmd_abilities,
            "help": cmd_help,
        }

        print(f"\n{'=' * 62}")
        print(f"  Belief Query Mode")
        print(f"{'=' * 62}")
        print(f"  AbilityBelief shape = {ab_all.shape}  (observer, subject, coop, value)")
        print(f"  KingBelief shape    = {kb_all.shape}  (observer, subject, coop, 0/1)")
        print(f"  Type 'help' for commands, 'quit' to exit.\n")

        while True:
            try:
                raw = input("pec-king> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting.")
                break
            if not raw:
                continue
            tokens = raw.split()
            cmd = tokens[0].lower()
            if cmd in ("quit", "q", "exit"):
                print("  Exiting.")
                break
            handler = dispatch.get(cmd)
            if handler:
                handler(tokens[1:])
            else:
                print(f"  Unknown command: '{cmd}'. Type 'help' for a list of commands.")

    if INTERACTIVE:
        _interactive_loop()
