==============================================================
  Pec-King Order Simulation -- CSCI 5512 HW5
==============================================================
  M=64 agents  |  N_coop=4  |  k=4  |  T=256 tournaments  |  seed=42
  Agent types: 4 Set-A  |  4 Set-B  |  56 NPCs
  Observation shape: (4102,)
  Ability PMF (truncated Poisson): [0.5553 0.2623 0.1239 0.0585]
  Abilities written to tracing/chicken_abilities.txt

  [TRACK] NPC trace: chicken_60 -> tracing/chicken_trace.txt
  [TRACK] Abilities: [1, 1, 1, 3]
  [TRACK] Strategic traces -> tracing/agent_setA_*.txt  tracing/agent_setB_*.txt
  Compiling step_env... done
  Starting simulation: 64 agents, 4 coops, 256 tournaments
  Tournament   25/256 done  (step 417)
  Tournament   50/256 done  (step 842)
  Tournament   75/256 done  (step 1250)
  Tournament  100/256 done  (step 1668)
  Tournament  125/256 done  (step 2086)
  Tournament  150/256 done  (step 2512)
  Tournament  175/256 done  (step 2933)
  Tournament  200/256 done  (step 3343)
  Tournament  225/256 done  (step 3757)
  Tournament  250/256 done  (step 4164)
  Tournament  256/256 done  (step 4263)

==============================================================
  Simulation complete
==============================================================
  4263 total steps across 256 tournaments
  Total time:  39m 39.5s
  Per step:    558.2 ms
  Per tournament: 9.29s avg
  BattleHistory nonzero entries: 173572

==============================================================
  Final Sack Scores (HW5)
==============================================================
  Total crowns in circulation: 1150

  Top 10 scorers:
  Rank   Chicken  Type      Score
  -----------------------------------
     1  chicken_51     NPC      145.00
     2  chicken_8      NPC      112.00
     3  chicken_52     NPC      106.00
     4  chicken_16     NPC       89.00
     5  chicken_57     NPC       88.00
     6  chicken_19     NPC       87.00
     7  chicken_18     NPC       84.00
     8  chicken_17     NPC       80.00
     9  chicken_62     NPC       73.00
    10  chicken_20     NPC       72.00

  Set-A wins (any in top 10): False
  Set-B wins (any in top 10): False

  --------------------------------------------------------------
  Crown Frequency Summary  (256 tournaments)
  --------------------------------------------------------------
  Coop  Sim King  Crowns    Rate  Match  Top-5 (chicken: crowns)
  --------------------------------------------------------------
     0        51     145  56.6%    yes  c51:145  c8:112  c1:5  c0:5  c3:1
        Possible true kings: [8, 51]  (ability=4)
     1        16      89  34.8%    yes  c16:89  c57:88  c18:84  c0:5  c1:3
        Possible true kings: [16, 18, 57]  (ability=4)
     2        52     106  41.4%    yes  c52:106  c19:87  c0:55  c2:48  c37:3
        Possible true kings: [0, 2, 19, 52]  (ability=4)
     3        17      80  31.2%    yes  c17:80  c62:73  c20:72  c3:69  c2:2
        Possible true kings: [3, 17, 20, 62]  (ability=4)
  --------------------------------------------------------------

==============================================================
  HW5 Strategy Analysis
==============================================================

  1. Score distribution by agent type
  Agent type | Count |      Min |     Mean |      Max | Top scorer
  ------------------------------------------------------------------------
  NPC        |    56 |     0.00 |    17.02 |   145.00 | chicken_51 (145.00)
  Set-A      |     4 |    49.25 |    49.25 |    49.25 | chicken_0 (49.25)
  Set-B      |     4 |     0.00 |     0.00 |     0.00 | chicken_4 (0.00)

  2. Set-B coop convergence
  Agent        | Final coop | n_visits per coop            | True best | Match
  ------------------------------------------------------------------------
  chicken_4    | coop     3 | [12790, 12780, 12780, 12850] | coop    0 | no
  chicken_5    | coop     0 | [12800, 12800, 12800, 12800] | coop    0 | yes
  chicken_6    | coop     0 | [12800, 12800, 12800, 12800] | coop    1 | no
  chicken_7    | coop     0 | [12800, 12800, 12800, 12800] | coop    0 | yes

  3. Set-A reward history
  Agent        | Total crowns | Tourns | Avg/tourn | Best tourn
  -----------------------------------------------------------------
  chicken_0    |          197 |    256 |      0.77 | t=10 (+6)
  chicken_1    |            0 |    256 |      0.00 | t=1 (+0)
  chicken_2    |            0 |    256 |      0.00 | t=4 (+1)
  chicken_3    |            0 |    256 |      0.00 | t=1 (+1)

  4. Sack co-ownership map
  chicken_0    | co-owners [1, 2, 3] | sack    197 | 49.25 per owner
  chicken_1    | co-owners [0, 2, 3] | sack      0 | 0.00 per owner
  chicken_2    | co-owners [0, 1, 3] | sack      0 | 0.00 per owner
  chicken_3    | co-owners [0, 1, 2] | sack      0 | 0.00 per owner
  chicken_4    | co-owners [5, 6, 7] | sack      0 | 0.00 per owner
  chicken_5    | co-owners [4, 6, 7] | sack      0 | 0.00 per owner
  chicken_6    | co-owners [4, 5, 7] | sack      0 | 0.00 per owner
  chicken_7    | co-owners [4, 5, 6] | sack      0 | 0.00 per owner

  5. Win determination
  Set-A final scores: [49.25, 49.25, 49.25, 49.25]
  Set-B final scores: [0.0, 0.0, 0.0, 0.0]
  Winner: NPC field

  6. Strategic agent ability context
  Agent        Type     Abilities              Best  Val Rivals  Crowns
  --------------------------------------------------------------------
  chicken_0    Set-A    [3, 1, 4, 1]              2    4      2     197
  chicken_1    Set-A    [3, 2, 1, 2]              0    3     10       0
  chicken_2    Set-A    [1, 1, 4, 2]              2    4      2       0
  chicken_3    Set-A    [2, 1, 1, 4]              3    4      3       0
  chicken_4    Set-B    [2, 2, 2, 2]              0    2     12       0
  chicken_5    Set-B    [2, 2, 2, 2]              0    2     12       0
  chicken_6    Set-B    [1, 3, 2, 2]              1    3     11       0
  chicken_7    Set-B    [3, 2, 1, 1]              0    3     10       0

  chicken_0: ability=4 in coop2, competing against 2 NPCs with the same ability
    -> moderate field
  chicken_1: ability=3 in coop0, competing against 10 NPCs with the same ability
    -> hard field
  chicken_2: ability=4 in coop2, competing against 2 NPCs with the same ability
    -> moderate field
  chicken_3: ability=4 in coop3, competing against 3 NPCs with the same ability
    -> moderate field
  chicken_4: ability=2 in coop0, competing against 12 NPCs with the same ability
    -> hard field
  chicken_5: ability=2 in coop0, competing against 12 NPCs with the same ability
    -> hard field
  chicken_6: ability=3 in coop1, competing against 11 NPCs with the same ability
    -> hard field
  chicken_7: ability=3 in coop0, competing against 10 NPCs with the same ability
    -> hard field

  --------------------------------------------------------------
  [TRACK] Closed 8 strategic trace files + NPC trace (tracing/chicken_trace.txt)