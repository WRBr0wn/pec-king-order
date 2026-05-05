"""
Fast unit tests for NPC, AgentSetA (PPO), AgentSetB (MCTS).
M=8, N_coop=4. No full simulation runs.
"""

import numpy as np
import pytest

from agents.agent import NPC, AgentSetA, AgentSetB, NUM_MINIBATCHES

M      = 8
N_COOP = 4


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _zone(agent_idx, current_coop):
    """Place each agent in its home coop (i % N_COOP), override agent_idx."""
    z = np.zeros((M, N_COOP, 3), dtype=np.int32)
    for i in range(M):
        z[i, i % N_COOP, 0] = 1
    z[agent_idx, :, :] = 0
    z[agent_idx, current_coop, 0] = 1
    return z


def _abilities(best_coop, best_val=4):
    a = np.ones(N_COOP, dtype=np.int32)
    a[best_coop] = best_val
    return a


def _sack(idx=0, val=0):
    s = np.zeros(M, dtype=np.int32)
    s[idx] = val
    return s


def _obs_mem():
    return np.zeros((M, M, M, 8), dtype=np.int16)


def _sack_owners():
    return np.eye(M, dtype=bool)


def _belief_dominant(agent_idx, coop):
    """ability_belief where agent_idx has max ability in coop; opponents have min."""
    b = np.ones((M, N_COOP, N_COOP), dtype=np.float32) / N_COOP
    b[agent_idx, coop, :] = 0.0
    b[agent_idx, coop, N_COOP - 1] = 1.0   # prob 1 of max ability
    for j in range(M):
        if j != agent_idx:
            b[j, coop, :] = 0.0
            b[j, coop, 0] = 1.0             # prob 1 of min ability
    return b


# ---------------------------------------------------------------------------
# 1 & 2. NPC
# ---------------------------------------------------------------------------

class TestNPC:
    def test_moves_to_best_coop(self):
        agent = NPC(0, _abilities(best_coop=2), N_COOP, [])
        action = agent.choose_action(_zone(0, current_coop=0),
                                     np.ones(N_COOP) / N_COOP, _abilities(2), N_COOP)
        assert action == N_COOP + 1 + 2   # a_move to coop 2

    def test_watches_when_at_best_coop(self):
        agent = NPC(0, _abilities(best_coop=2), N_COOP, [])
        action = agent.choose_action(_zone(0, current_coop=2),
                                     np.ones(N_COOP) / N_COOP, _abilities(2), N_COOP)
        assert action == 2 + 1            # a_watch coop 2

    def test_social_always_empty(self):
        agent = NPC(0, _abilities(0), N_COOP, [])
        for t in range(5):
            assert agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), t) == []


# ---------------------------------------------------------------------------
# 3 & 4 & 5. AgentSetA
# ---------------------------------------------------------------------------

class TestAgentSetA:
    TEAMMATES = [0, 1, 2, 3]

    def _make(self, idx=0):
        return AgentSetA(idx, _abilities(best_coop=1), N_COOP, self.TEAMMATES)

    def test_choose_action_in_valid_range(self):
        agent = self._make()
        for _ in range(6):
            a = agent.choose_action(_zone(0, 1), np.ones(N_COOP) / N_COOP,
                                    _abilities(1), N_COOP)
            assert 0 <= a <= 2 * N_COOP

    def test_trajectory_grows_each_step(self):
        agent = self._make()
        for i in range(1, 6):
            agent.choose_action(_zone(0, 1), np.ones(N_COOP) / N_COOP,
                                _abilities(1), N_COOP)
            assert len(agent._traj) == i

    def test_trajectory_clears_at_tournament_close(self):
        agent = self._make()
        for _ in range(NUM_MINIBATCHES):
            agent.choose_action(_zone(0, 1), np.ones(N_COOP) / N_COOP,
                                _abilities(1), N_COOP)
        agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 1)
        assert len(agent._traj) == 0

    def test_shaped_reward_prev_win_rate_starts_zero(self):
        assert self._make()._prev_win_rate == pytest.approx(0.0)

    def test_shaped_reward_prev_win_rate_updates(self):
        agent = self._make()
        for _ in range(NUM_MINIBATCHES):
            agent.choose_action(_zone(0, 1), np.ones(N_COOP) / N_COOP,
                                _abilities(1), N_COOP)
        agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(),
                                    1, win_count=8, loss_count=2)
        # win_rate = 8/10 = 0.8
        assert agent._prev_win_rate == pytest.approx(0.8)

    def test_shaped_reward_win_rate_delta_applied_second_tournament(self):
        agent = self._make()
        for _ in range(NUM_MINIBATCHES):
            agent.choose_action(_zone(0, 1), np.ones(N_COOP) / N_COOP,
                                _abilities(1), N_COOP)
        agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(),
                                    1, win_count=8, loss_count=2)
        for _ in range(NUM_MINIBATCHES):
            agent.choose_action(_zone(0, 1), np.ones(N_COOP) / N_COOP,
                                _abilities(1), N_COOP)
        agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(),
                                    2, win_count=4, loss_count=6)
        # win_rate = 4/10 = 0.4; delta = 0.4 - 0.8 = -0.4 was applied
        assert agent._prev_win_rate == pytest.approx(0.4)

    def test_tournament1_extends_ownership_to_3_teammates(self):
        agent = self._make(idx=0)
        actions = agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 1)
        extend = [a for a in actions if a["type"] == "extend_sack_ownership"]
        assert len(extend) == 3
        assert {a["target"] for a in extend} == {1, 2, 3}

    def test_tournament2_no_extend_ownership(self):
        agent = self._make(idx=0)
        agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 1)
        actions2 = agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 2)
        assert not any(a["type"] == "extend_sack_ownership" for a in actions2)

    def test_every_step_shares_with_3_teammates(self):
        agent = self._make(idx=0)
        for t in [0, 1, 2, 5]:
            actions = agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), t)
            share = [a for a in actions if a["type"] == "share_observations"]
            assert len(share) == 3
            assert {a["target"] for a in share} == {1, 2, 3}

    def test_self_not_in_social_targets(self):
        agent = self._make(idx=0)
        actions = agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 1)
        for a in actions:
            assert a["target"] != 0


# ---------------------------------------------------------------------------
# 6 & 7. AgentSetB
# ---------------------------------------------------------------------------

class TestAgentSetB:
    TEAMMATES = [4, 5, 6, 7]

    def _make(self, idx=4, best_coop=0):
        return AgentSetB(idx, _abilities(best_coop), N_COOP, self.TEAMMATES)

    def test_uct_selects_dominant_coop(self):
        """When agent has max ability in coop 0 per belief, UCT should pick coop 0."""
        agent = self._make(idx=4, best_coop=0)
        agent.choose_action(_zone(4, 0), np.ones(N_COOP) / N_COOP, _abilities(0), N_COOP)
        belief = _belief_dominant(4, coop=0)
        agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(),
                                    1, ability_belief=belief)
        assert agent._current_coop == 0

    def test_uct_n_accumulates(self):
        """After one tournament close, total UCT visits == n_iter * K."""
        agent = self._make(idx=4)
        agent.choose_action(_zone(4, 0), np.ones(N_COOP) / N_COOP, _abilities(0), N_COOP)
        agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 1)
        assert int(agent._uct_N.sum()) == agent._n_iter * agent._K

    def test_tournament1_extends_ownership_to_3_teammates(self):
        agent = self._make(idx=4)
        agent.choose_action(_zone(4, 0), np.ones(N_COOP) / N_COOP, _abilities(0), N_COOP)
        actions = agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 1)
        extend = [a for a in actions if a["type"] == "extend_sack_ownership"]
        assert len(extend) == 3
        assert {a["target"] for a in extend} == {5, 6, 7}

    def test_tournament2_no_extend_ownership(self):
        agent = self._make(idx=4)
        agent.choose_action(_zone(4, 0), np.ones(N_COOP) / N_COOP, _abilities(0), N_COOP)
        agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 1)
        agent.choose_action(_zone(4, 0), np.ones(N_COOP) / N_COOP, _abilities(0), N_COOP)
        actions2 = agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 2)
        assert not any(a["type"] == "extend_sack_ownership" for a in actions2)

    def test_every_step_shares_with_3_teammates(self):
        agent = self._make(idx=4)
        agent.choose_action(_zone(4, 0), np.ones(N_COOP) / N_COOP, _abilities(0), N_COOP)
        for t in [0, 1, 2]:
            actions = agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), t)
            share = [a for a in actions if a["type"] == "share_observations"]
            assert len(share) == 3
            assert {a["target"] for a in share} == {5, 6, 7}

    def test_symmetric_all_agents_share_all_teammates(self):
        """Every Set-B agent shares with all 3 teammates, not just a sub-role."""
        for idx in [4, 5, 6, 7]:
            agent = self._make(idx=idx)
            agent.choose_action(_zone(idx, 0), np.ones(N_COOP) / N_COOP,
                                _abilities(0), N_COOP)
            actions = agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 0)
            share_targets = {a["target"] for a in actions
                             if a["type"] == "share_observations"}
            expected = set(self.TEAMMATES) - {idx}
            assert share_targets == expected, f"agent {idx}: got {share_targets}"

    def test_self_not_in_social_targets(self):
        agent = self._make(idx=4)
        agent.choose_action(_zone(4, 0), np.ones(N_COOP) / N_COOP, _abilities(0), N_COOP)
        actions = agent.choose_social_actions(_sack(), _sack_owners(), _obs_mem(), 1)
        for a in actions:
            assert a["target"] != 4
