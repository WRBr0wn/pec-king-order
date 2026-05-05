==============================================================
  Pec-King Order Simulation -- CSCI 5512 HW5
==============================================================
  M=64 agents  |  N_coop=4  |  k=4  |  T=256 tournaments  |  seed=50398775
  Agent types: 4 Set-A  |  4 Set-B  |  56 NPCs
  Observation shape: (4102,)
  Ability PMF (truncated Poisson): [0.5553 0.2623 0.1239 0.0585]
  Abilities written to tracing/chicken_abilities.txt

  [TRACK] NPC trace: chicken_31 -> tracing/chicken_trace.txt
  [TRACK] Abilities: [3, 2, 1, 2]
  [TRACK] Strategic traces -> tracing/agent_setA_*.txt  tracing/agent_setB_*.txt
  Compiling step_env... done
  Starting simulation: 64 agents, 4 coops, 256 tournaments
  Tournament   25/256 done  (step 418)
  Tournament   50/256 done  (step 841)
  Tournament   75/256 done  (step 1265)
  Tournament  100/256 done  (step 1680)
  Tournament  125/256 done  (step 2103)
  Tournament  150/256 done  (step 2524)
  Tournament  175/256 done  (step 2935)
  Tournament  200/256 done  (step 3344)
  Tournament  225/256 done  (step 3752)
  Tournament  250/256 done  (step 4170)
  Tournament  256/256 done  (step 4265)

==============================================================
  Simulation complete
==============================================================
  4265 total steps across 256 tournaments
  Total time:  35m 50.0s
  Per step:    504.1 ms
  Per tournament: 8.40s avg
  BattleHistory nonzero entries: 174028

==============================================================
  Final Sack Scores (HW5)
==============================================================
  Total crowns in circulation: 1142

  Top 10 scorers:
  Rank   Chicken  Type      Score
  -----------------------------------
     1  chicken_39     NPC      249.00
     2  chicken_13     NPC      132.00
     3  chicken_11     NPC      124.00
     4  chicken_45     NPC      103.00
     5  chicken_16     NPC       97.00
     6  chicken_44     NPC       93.00
     7  chicken_29     NPC       73.00
     8  chicken_58     NPC       56.00
     9  chicken_53     NPC       56.00
    10  chicken_2      SetA      31.25

  Set-A wins (any in top 10): True
  Set-B wins (any in top 10): False

  --------------------------------------------------------------
  Crown Frequency Summary  (256 tournaments)
  --------------------------------------------------------------
  Coop  Sim King  Crowns    Rate  Match  Top-5 (chicken: crowns)
  --------------------------------------------------------------
     0        45     103  40.2%    yes  c45:103  c44:93  c29:72  c2:1  c1:1
        Possible true kings: [29, 44, 45]  (ability=4)
     1        16      81  31.6%    yes  c16:81  c58:56  c53:56  c3:53  c0:47
        Possible true kings: [0, 3, 16, 53, 58]  (ability=4)
     2        13     131  51.2%    yes  c13:131  c11:124  c16:16  c0:5  c3:1
        Possible true kings: [11, 13, 16]  (ability=4)
     3        39     249  97.3%    yes  c39:249  c50:9  c2:9  c41:7  c40:7
        Possible true kings: [39]  (ability=4)
  --------------------------------------------------------------

==============================================================
  HW5 Strategy Analysis
==============================================================

  1. Score distribution by agent type
  Agent type | Count |      Min |     Mean |      Max | Top scorer
  ------------------------------------------------------------------------
  NPC        |    56 |     0.00 |    18.14 |   249.00 | chicken_39 (249.00)
  Set-A      |     4 |    31.25 |    31.25 |    31.25 | chicken_0 (31.25)
  Set-B      |     4 |     0.25 |     0.25 |     0.25 | chicken_4 (0.25)

  2. Set-B coop convergence
  Agent        | Final coop | n_visits per coop            | True best | Match
  ------------------------------------------------------------------------
  chicken_4    | coop     3 | [12790, 12780, 12780, 12850] | coop    0 | no
  chicken_5    | coop     0 | [12800, 12800, 12800, 12800] | coop    1 | no
  chicken_6    | coop     3 | [12790, 12780, 12780, 12850] | coop    1 | no
  chicken_7    | coop     3 | [12790, 12780, 12780, 12850] | coop    1 | no

  3. Set-A reward history
  Agent        | Total crowns | Tourns | Avg/tourn | Best tourn
  -----------------------------------------------------------------
  chicken_0    |           58 |    256 |      0.23 | t=74 (+2)
  chicken_1    |            2 |    256 |      0.01 | t=30 (+1)
  chicken_2    |           10 |    256 |      0.04 | t=7 (+1)
  chicken_3    |           55 |    256 |      0.21 | t=116 (+2)

  4. Sack co-ownership map
  chicken_0    | co-owners [1, 2, 3] | sack     58 | 14.50 per owner
  chicken_1    | co-owners [0, 2, 3] | sack      2 | 0.50 per owner
  chicken_2    | co-owners [0, 1, 3] | sack     10 | 2.50 per owner
  chicken_3    | co-owners [0, 1, 2] | sack     55 | 13.75 per owner
  chicken_4    | co-owners [5, 6, 7] | sack      0 | 0.00 per owner
  chicken_5    | co-owners [4, 6, 7] | sack      0 | 0.00 per owner
  chicken_6    | co-owners [4, 5, 7] | sack      1 | 0.25 per owner
  chicken_7    | co-owners [4, 5, 6] | sack      0 | 0.00 per owner

  5. Win determination
  Set-A final scores: [31.25, 31.25, 31.25, 31.25]
  Set-B final scores: [0.25, 0.25, 0.25, 0.25]
  Winner: NPC field

  6. Strategic agent ability context
  Agent        Type     Abilities              Best  Val Rivals  Crowns
  --------------------------------------------------------------------
  chicken_0    Set-A    [1, 4, 2, 2]              1    4      3      58
  chicken_1    Set-A    [3, 1, 1, 1]              0    3     11       2
  chicken_2    Set-A    [1, 1, 1, 2]              3    2     14      10
  chicken_3    Set-A    [2, 4, 1, 1]              1    4      3      55
  chicken_4    Set-B    [3, 2, 2, 1]              0    3     11       0
  chicken_5    Set-B    [1, 3, 1, 1]              1    3      9       0
  chicken_6    Set-B    [1, 2, 1, 2]              1    2     18       1
  chicken_7    Set-B    [1, 2, 1, 1]              1    2     18       0

  chicken_0: ability=4 in coop1, competing against 3 NPCs with the same ability
    -> moderate field
  chicken_1: ability=3 in coop0, competing against 11 NPCs with the same ability
    -> hard field
  chicken_2: ability=2 in coop3, competing against 14 NPCs with the same ability
    -> hard field
  chicken_3: ability=4 in coop1, competing against 3 NPCs with the same ability
    -> moderate field
  chicken_4: ability=3 in coop0, competing against 11 NPCs with the same ability
    -> hard field
  chicken_5: ability=3 in coop1, competing against 9 NPCs with the same ability
    -> hard field
  chicken_6: ability=2 in coop1, competing against 18 NPCs with the same ability
    -> hard field
  chicken_7: ability=2 in coop1, competing against 18 NPCs with the same ability
    -> hard field

  --------------------------------------------------------------
  [TRACK] Closed 8 strategic trace files + NPC trace (tracing/chicken_trace.txt)