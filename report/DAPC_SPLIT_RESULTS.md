# DAPC v2: Budget Split Between DP Class-Conditional Marginals and DP Flow Matching

**Date:** 2026-09-26.
**Code:**
- `acfm_net/postprocessing/conditional_marginals.py` (Stage 7c)
- `AdaptiveBudgetAllocator.marginal_noise_multiplier` (split rule)
- `ContinuousOTCFM.compute_loss(noise_multiplicity=K)`

**Run:** `python lumisyn/backend/dapc_split_eval.py --dataset adult|cardio|pima [--eps E] [--configs ...]`. Raw data in `runs/dapc_split/results.jsonl`, 3 seeds per cell. Every run passes the accountant check (`verified=True`).

## Mechanism
- **Stage 7c** measures one noisy (label × feature) table per feature on public bins, using the Gaussian mechanism: L2 sensitivity √M, noise multiplier z. It is composed with DP-SGD in the **same PRV accountant**, and the flow's σ is re-calibrated so that the total stays exactly at DAPC's ε\*.
- **Recalibration** (post-processing): class sizes come from the noisy tables. Within each class, every column is quantile-mapped to its noisy conditional marginal while keeping the flow's within-class ranks (Sklar: the flow supplies the copula).
- **Split rules:**
  - fixed share ρ: the marginals alone cost ρ·ε\*;
  - adaptive: z = r·n/(K√M) with r = 0.25, capped at a 0.6 share.
- Design follows MST/AIM (noisy low-order marginals on public domains) and DPDM (noise multiplicity).

## Results (target ε = 1 → DAPC-selected ε\* shown)
| dataset (ε\*) | config | TSTR AUC | 1-way TVD | 2-way TVD (all pairs) |
|---|---|---|---|---|
| Adult (0.741), TRTR 0.917 | flow only | 0.492 ± 0.054 | 0.510 | 0.719 |
| | flow + 1-way (old 7b) | 0.497 ± 0.066 | 0.135 | 0.271 |
| | share 0.2 / 0.4 / 0.6 | 0.879 / 0.897 / 0.901 | 0.105 / 0.058 / 0.041 | 0.207 / 0.138 / 0.114 |
| | adaptive | 0.904 ± 0.002 | 0.042 | 0.116 |
| | **share 0.9** | **0.906 ± 0.002** | **0.029** | **0.097** |
| | share 0.9, marginals only (no flow) | 0.905 ± 0.003 | 0.027 | 0.095 |
| | *AIM (same ε)* | *0.790 ± 0.011* | *0.018* | *0.112* |
| | *MST (same ε)* | *0.713 ± 0.031* | *0.014* | *0.128* |
| Cardio 20k (0.849), TRTR 0.779 | flow only | 0.732 ± 0.017 | 0.213 | 0.365 |
| | adaptive | 0.780 ± 0.005 | 0.027 | 0.070 |
| | share 0.6 | 0.780 ± 0.003 | 0.012 | 0.049 |
| | **share 0.9** | **0.778 ± 0.004** | **0.008** | **0.044** |
| | share 0.9, marginals only | 0.779 ± 0.003 | 0.008 | 0.045 |
| Pima 614 (0.753), TRTR 0.812 | flow only | 0.514 ± 0.085 | 0.540 | 0.795 |
| | adaptive | 0.729 ± 0.069 | 0.344 | 0.601 |
| | share 0.6 | 0.768 ± 0.027 | 0.331 | 0.590 |
| | adaptive, marginals only | 0.700 ± 0.042 | 0.326 | 0.579 |

**Higher ε (Adult).** At ε\* = 2.45 and 4.72:
- adaptive: 0.905 / 0.906;
- marginals only: 0.906 / 0.907;
- flow only: 0.526 / 0.581.

**Noise multiplicity (K=4, hidden 64).** No measurable change on any dataset (Adult 0.900, Cardio 0.778, Pima 0.748).

## Honest reading
1. **DP class-conditional marginals are the utility fix.**
   - Adult rises from 0.49 to 0.906 TSTR AUC at ε = 0.741, beating AIM (0.790) and MST (0.713). Its 2-way TVD over all pairs (0.097) is also better than AIM's (0.112).
   - Cardio reaches the real-data ceiling (0.778 vs TRTR 0.779).
   - Caveat: the method is **target-aware**, while AIM and MST optimise a general workload. A fair comparison states this, or gives AIM the label × feature workload too.
2. **The flow contributes nothing measurable at ε ≤ 4.7.**
   - Marginals-only equals flow + marginals on every dataset and ε, within one standard deviation.
   - The best split sends ≥ 90% of the budget to the marginals.
   - The DP-SGD-trained flow does not learn a useful dependence structure in the budgets tested.
3. **The adaptive rule is no better than a fixed 0.9 share.**
   - Adaptive: Adult 0.904 vs 0.906; Cardio TVD 0.070 vs 0.044.
   - The data says "spend almost everything on marginals". An adaptive allocator therefore has nothing to adapt between unless the second mechanism earns budget.

## Consequence for the paper's claim ("adaptive allocator + flow matching")
The experiments do not support "flow matching improves DP tabular synthesis" or "the adaptive split beats a fixed split". They do support:
- (a) the OT-coupling privacy flaw and its certified audit;
- (b) a sound DP pipeline whose task-aware marginal stage gives strong TSTR utility;
- (c) the negative result that DP-SGD flow matching adds no measurable value below ε ≈ 5 on these datasets.

## Update 2026-09-26: blockers closed
- **10-seed equivalence** (`flow_ablation_stats.py` → `runs/dapc_split/flow_ablation_stats.json`). Share 0.9 with the flow copula vs marginals only, paired by seed:
  - Adult AUC 0.9060 vs 0.9061 (diff CI [−0.0023, +0.0021]).
  - Cardio AUC 0.7785 vs 0.7789 (CI [−0.0038, +0.0030]).
  - TOST p ≤ 0.0001 for every metric at margins ±0.01 AUC and ±0.005 TVD.
  - The flow's only detectable effect: Cardio 2-way TVD improves by 0.0011 (Wilcoxon p = 0.01), well inside the margin.
- **Fair baseline.** AIM restricted to the label × feature workload (`baselines_gen.py aim_target`) scores TSTR 0.872 ± 0.002 with 2-way TVD 0.267, vs ours 0.906 / 0.096.
- **Second audit dataset (Cardio).** OT exceeds the bound in 69% of batches (max 1.484×). The white-box audit gives μ ≥ 1.427 vs claimed 0.966, so the minimum valid ε is 1.474 where 0.839 was reported. I-CFM's max is 0.975× and it passes. Raw data: `runs/privacy_audit.json` key `stronger_adversary_cardio`.
