# Pre-registration: adaptive budget split with a warm-started DP flow (written before the reduced grid, 2026-09-27)

**Pilot data already seen:**
- Adult at target ε = 1 (all configs);
- Adult at target ε = 8 (warm_0.2, warm_0.5, marginals_only_0.9, warm-only ablations).

**Grid (reduced, as agreed):**
- Datasets: Adult, Cardio.
- Target ε: 1, 4, 8.
- Marginal share ρ: 0.2, 0.5, 0.9 with warm start and DP fine-tuning.
- Baselines: marginals-only at 0.9, and a warm-only ablation at 0.2.
- 3 seeds per cell.

**Objective, fixed now.** Within each (dataset, ε) cell, each configuration gets a composite utility: the mean of four min–max-normalised metrics across the configurations of that cell, with higher always better:
1. TSTR AUC (label);
2. all-column accuracy (general-purpose utility);
3. −(2-way TVD);
4. −(3-way label TVD).

**Adaptive rule.** The split ρ is chosen from public quantities only (the released ε*, and n and d if needed).
- The rule is fitted on **Adult only**: for each ε, take the ρ with the highest composite, then fit the simplest monotone rule of ε* that reproduces those choices.
- It is evaluated on **Cardio**, which is held out.

**Success criterion for the claim.** On the held-out dataset, averaged over the three ε levels, the adaptive rule's composite must beat every fixed ρ ∈ {0.2, 0.5, 0.9}. In addition, at the ε where the rule gives the flow a real share, DP fine-tuning must beat the warm-only ablation on all-column accuracy (flow learns from private data). If either condition fails, the claim is not supported and will not be made.

## Addendum A: 10-seed confirmation (written 2026-09-27, after seeing the 3-seed grid above)

The 3-seed grid showed a pattern that we did not predict: the warm-started flow at ρ = 0.9 (`warm_0.9`) looked at least as good as marginals-only on all four metrics. Because we saw this pattern before writing this addendum, it is a confirmatory test of an exploratory observation, and we report it that way.

**Grid:**
- Datasets: Adult, Cardio.
- Target ε: 1, 4, 8.
- Configs: `marginals_only_0.9`, `warm_0.9`, `warm_0.9_pretrain_only`.
- Seeds: 0–9. Seeds 0–2 are reused where they exist.

**Hypotheses.** Both are one-sided, tested per cell with Welch's t-test on all-column accuracy, and Holm-corrected over the 6 cells at α = 0.05:
- **H1:** `warm_0.9` has higher all-column accuracy than `marginals_only_0.9`.
- **H2:** `warm_0.9` has higher all-column accuracy than `warm_0.9_pretrain_only`. This isolates what the DP fine-tuning on private records adds.

**Secondary:** TSTR AUC for `warm_0.9` is equivalent to `marginals_only_0.9`, tested with TOST at a ±0.01 margin.

**Decision rule.**
- H1 and H2 hold in at least 5 of 6 cells: the paper states that DP flow matching adds cross-column dependence beyond DP marginals.
- Otherwise: the paper states the pilot result as not confirmed.
