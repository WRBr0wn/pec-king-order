"""
Inference layer for Pec-King Order.

Three required computations (Problem 1 spec):

  1. P(chicken_i beats chicken_j | ObservationHistory)
       → domain_inference.py :: p_i_beats_j

  2. P(chicken_i will be King in Coop_k | ObservationHistory)
       → domain_inference.py :: derive_king_belief_from_ability
       → stored in KingBelief  (M × N_coop × 2)

  3. P(ability profile of chicken_j | ObservationHistory)
       → domain_inference.py :: p_ability_profile
       → stored in AbilityBelief  (M × N_coop × N_coop)

Problems 2-4 (written + coding):

  enumeration.py       — Problem 3: Batcave P(S=5 | A=T) by enumeration
  belief_propagation.py — Problem 4: Wet-grass Pearl BP
"""

from inference.domain_inference import (
    p_i_beats_j,
    p_ability_profile,
    incremental_ability_update,
    derive_king_belief_from_ability,
    p_king_in_coop,
    update_king_belief,
    update_king_belief_vectorized,
    update_king_beliefs_all_agents,
    update_ability_belief_from_history,
    update_ability_beliefs_all_agents,
)
from inference.enumeration        import p_s5_given_alarm, full_posterior_s_given_alarm
from inference.belief_propagation import (
    bel_w_no_obs,
    bel_h_no_obs,
    bel_w_given_h,
    bel_s_given_h_and_w,
)

__all__ = [
    # Problem 1 — domain inference
    "p_i_beats_j",
    "p_ability_profile",
    "incremental_ability_update",
    "derive_king_belief_from_ability",
    "p_king_in_coop",
    "update_king_belief",
    "update_king_belief_vectorized",
    "update_king_beliefs_all_agents",
    "update_ability_belief_from_history",
    "update_ability_beliefs_all_agents",
    # Problem 3 — enumeration
    "p_s5_given_alarm",
    "full_posterior_s_given_alarm",
    # Problem 4 — belief propagation
    "bel_w_no_obs",
    "bel_h_no_obs",
    "bel_w_given_h",
    "bel_s_given_h_and_w",
]