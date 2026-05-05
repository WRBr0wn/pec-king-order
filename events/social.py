"""
Social actions for HW5: pure Python/NumPy, executed outside the JIT loop.

Functions
---------
share_observations     — merge src's observation history into tgt's
transfer_crowns        — move crowns between sacks (clamped to balance)
extend_sack_ownership  — add a co-owner to a sack
compute_final_scores   — compute end-of-game scores from sack/ownership state
apply_social_action    — dispatcher for action dicts produced by agents
"""

from __future__ import annotations

import numpy as np


def share_observations(
    src: int,
    tgt: int,
    obs_memory: np.ndarray,  # int16 (M, M, M, T)
) -> None:
    """Copy src's non-zero observations into tgt wherever tgt has no data."""
    obs_memory[tgt] = np.where(obs_memory[tgt] == 0, obs_memory[src], obs_memory[tgt])


def transfer_crowns(
    src: int,
    tgt: int,
    count: int,
    sack: np.ndarray,  # int32 (M,)
) -> None:
    """Transfer min(count, sack[src]) crowns from src to tgt."""
    actual = min(count, int(sack[src]))
    sack[src] -= actual
    sack[tgt] += actual


def extend_sack_ownership(
    owner: int,
    new_co_owner: int,
    sack_owners: np.ndarray,  # bool (M, M)
) -> None:
    """Add new_co_owner as a co-owner of owner's sack."""
    sack_owners[owner, new_co_owner] = True


def compute_final_scores(
    sack: np.ndarray,        # int32 (M,)
    sack_owners: np.ndarray, # bool  (M, M)
) -> np.ndarray:
    """Compute each chicken's final crown score.

    score[j] = sum over all i where sack_owners[i, j] of sack[i] / |owners of i|
    """
    M = len(sack)
    scores = np.zeros(M, dtype=np.float64)
    for i in range(M):
        owners_of_i = np.where(sack_owners[i])[0]
        if len(owners_of_i) == 0:
            continue
        share = sack[i] / len(owners_of_i)
        for j in owners_of_i:
            scores[j] += share
    return scores


def apply_social_action(
    action: dict,
    src: int,
    sack: np.ndarray,        # int32 (M,)
    sack_owners: np.ndarray, # bool  (M, M)
    obs_memory: np.ndarray,  # int16 (M, M, M, T)
) -> None:
    """Dispatch a social action dict to the appropriate function."""
    action_type = action.get("type")
    if action_type == "share_observations":
        share_observations(src, action["target"], obs_memory)
    elif action_type == "transfer_crowns":
        transfer_crowns(src, action["target"], action["count"], sack)
    elif action_type == "extend_sack_ownership":
        extend_sack_ownership(src, action["target"], sack_owners)
