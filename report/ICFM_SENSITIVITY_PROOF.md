# Mini-Batch Optimal-Transport Coupling Breaks DP-SGD's Sensitivity Bound; Independent Coupling (I-CFM) Restores It

**Status:** core technical claim for submission. Revised 2026-09-26: adds the stronger adversary and white-box audit (§4), post-leak-fix utility (§5), matched-ε baselines (§6b); §7 now retracts three earlier claims. End-to-end guarantee and threat model: `PRIVACY_PROOF.md`.
**Data:** UCI Adult, N=15,000 sample of `lumisyn/data/kaggle_datasets/adult.csv`, encoded with the declared public schema `lumisyn/data/schemas/adult_schema.yaml` (encoded dimension d=78).
**Reproduce:** `python lumisyn/backend/verify_icfm_sensitivity.py` (Parts 1–2; `--worst-case-only` for 1b, `--utility-only` for 2), `python lumisyn/backend/audit_privacy.py` (1c and audits), `python lumisyn/backend/baselines_eval.py` (§6b). Raw outputs in `runs/`: `icfm_sensitivity_verification.json`, `icfm_worst_case_sensitivity.json`, `privacy_audit.json`, `baselines/results.json`.

---

## 1. Setting

Conditional flow matching trains $v_\theta(z_t,t)$ to regress $u = z_1 - z_0$ along $z_t=(1-t)z_0+t z_1$, with $z_0\sim\mathcal N(0,I)$ and $z_1$ a private record. Within a mini-batch of size $B$ the pairing of $z_{0,i}$ with a record is a design choice:

- **Batch-OT (OT-CFM):** $\pi^\star=\arg\min_\pi\sum_i\|z_{0,i}-z_{1,\pi(i)}\|^2$ (Hungarian algorithm over the whole batch).
- **I-CFM:** $z_{0,i}$ paired with $z_{1,i}$; no batch-level solve.

DP-SGD (Abadi et al. 2016) clips each per-example gradient to norm $C$, sums, and adds $\mathcal N(0,\sigma^2C^2I)$. Its accountant is valid only if the **sum of clipped gradients** $S(B)=\sum_i \bar g_i$ has L2 sensitivity $\Delta\le C$ (add/remove) or $\Delta\le 2C$ (replace-one) between neighbouring batches.

## 2. Proposition 1 (I-CFM satisfies the DP-SGD sensitivity bound)

Let $B,B'$ differ only in record $j$. Under I-CFM, $\ell_i$ depends only on $(z_{0,i},z_{1,i},t_i,\theta)$, so $\bar g_i(B)=\bar g_i(B')$ for all $i\ne j$ and
$$\|S(B)-S(B')\|_2=\|\bar g_j(B)-\bar g_j(B')\|_2\le 2C.$$
The standard DP-SGD accountant therefore applies unchanged. $\blacksquare$

## 3. Theorem 1 (Batch-OT violates it)

Under Batch-OT, changing $z_{1,j}$ changes column $j$ of the cost matrix, which can change $\pi^\star(i)$ for $k$ other records. Then
$$\|S(B)-S(B')\|_2=\Big\|\sum_{i\in\{j\}\cup R}\big(\bar g_i(B)-\bar g_i(B')\big)\Big\|_2\le 2C(k+1),$$
where $R$ is the set of reassigned records, $|R|=k\le B-1$. The upper bound is attained only if the changes align; the claim that matters for DP is that **the bound $2C$ is not guaranteed**, which a single counterexample establishes (DP is a worst-case property). §4 exhibits such counterexamples on real data. The accountant's reported $\varepsilon$ is then not a valid upper bound, even though it is numerically identical for both couplings (it reads only $\sigma,q,T$). $\blacksquare$

## 4. Empirical verification on real data

**Part 1b — worst case (the evidence that matters).** 10 seeds × 10 trials, $B=64$, $C$ set so every per-example gradient is clipped (matching the 100% clip fraction observed in every training run of this framework). Adversary: for each trial, screen 200 real replacement records + 63 records placed on other examples' prior draws + 50 domain corners by how many *other* records they reassign under OT, then compute exact clipped-sum sensitivity for the top 5.

| | Batch-OT | I-CFM |
|---|---|---|
| other records reassigned by one replacement | mean 13.9, max 26 | 0 |
| $\Delta/2C$, mean ± std | 0.85 ± 0.14 | 0.44 ± 0.12 |
| $\Delta/2C$, max | **1.147** | 0.648 |
| batches exceeding the DP-SGD bound | **15%** | 0% (impossible, Prop. 1) |

**Part 1a — random neighbour, default $C=1.25$** (for context; not a worst-case test): OT $\Delta$ = 1.29 ± 0.45 (max 2.47), I-CFM 0.63 ± 0.23; neither exceeded $2C=2.5$ in 100 trials. OT roughly doubles typical sensitivity even when it does not breach the bound.

**Part 1c — stronger adversary** (`audit_privacy.py`, same 100 trials). Start from the 3 best screened candidates, now restricted to *valid* encoded records (one-hot blocks one-hot, flags ±1, numerics inside public bounds), then 150 steps of hill climbing over valid records on the exact objective $\Delta/2C$.

| | Batch-OT | I-CFM |
|---|---|---|
| $\Delta/2C$, mean ± std | 0.89 ± 0.11 | 0.49 ± 0.12 |
| $\Delta/2C$, max | **1.182** | 0.681 |
| batches exceeding the bound | **17%** | 0% |

**White-box audit of the step (Nasr et al. 2021; 2023).** For the worst pair from 1c, run the real Gaussian mechanism $S(B)+\mathcal N(0,\sigma^2C^2I)$ with the training $\sigma=2.617$, 10⁶ times under each hypothesis (full-dimensional noise on GPU). Apply the Neyman–Pearson test (projection on $S(B')-S(B)$, threshold $\Delta/2$), and bound FPR and FNR with Clopper–Pearson (Bonferroni, 95% overall). Then $\mu_{lb}=\Phi^{-1}(1-\overline{FPR})-\Phi^{-1}(\overline{FNR})$ lower-bounds the step's Gaussian-DP parameter (Dong, Roth & Su 2022). DP-SGD's analysis claims $\mu\le 2/\sigma=0.764$ for a replace-one step.

| | Batch-OT | I-CFM |
|---|---|---|
| empirical FPR / FNR | 0.326 / 0.327 | 0.397 / 0.397 |
| $\mu$ lower bound (95%) | **0.896** | 0.518 |
| claimed $\mu$ | 0.764 | 0.764 |
| violation certified | **yes** | no |
| smallest valid full-run certificate ($q=512/12000$, $T=120$, $\delta=8\times10^{-6}$) | **ε ≥ 0.898** | 0.735 (= reported) |

The reported ε for Batch-OT (0.735) is therefore below what any valid Gaussian-mechanism certificate could state (0.898), with statistical confidence. This is a white-box, chosen-batch adversary, the strongest threat model DP is meant to cover.

**Second dataset (Cardio, 20k rows, d=12; same adversary and audit; `audit_privacy.py --dataset cardio`).**

| | Batch-OT | I-CFM |
|---|---|---|
| Δ/2C max | **1.484** | 0.975 |
| batches exceeding the bound | **69%** | 0% |
| μ lower bound (95%) vs claimed | **1.427** vs 0.966 | 0.936 vs 0.966 |
| minimum valid full-run ε vs reported | **1.474** vs 0.839 | 0.839 (= reported) |

**Black-box end-to-end canary audit** (Steinke, Nasr & Jagielski 2023, one run). 1000 random valid-domain canaries, each included w.p. ½; auditor sees only the synthetic data; score $=-$ distance to the nearest synthetic record; $k=100$ guesses per side fixed a priori.
- Every run gives $\varepsilon_{lb}=0$: 3 I-CFM seeds, 3 OT seeds, **and** a near-non-private control ($\varepsilon=28.6$, 30 epochs).
- So this black-box audit has **no power** against this generator: the synthetic data does not reproduce individual canaries even with almost no noise.
- We report it for completeness and rest the empirical claim on the white-box audit.

**Consequence for $\varepsilon$ (illustrative).** For the Adult configuration ($q=512/12000$, $T=120$, $\delta=8\times10^{-6}$), σ is calibrated to $\varepsilon=0.741$. If every step's sensitivity were $1.147\times2C$, the same σ gives $\varepsilon=0.881$ (19% under-reported); at $1.5\times$ it is 1.27, at $2\times$ 1.99. The observed excess is modest; a stronger adversary (optimising the replacement record directly) may find more.

## 5. Utility cost of the fix (Part 2)

Full pipeline, identity latent mode, 5 epochs, batch 512, DAPC-selected $\varepsilon=0.741$, Stage 7b off. 10 seeds per coupling, same seeds, paired Wilcoxon signed-rank. Re-run 2026-09-26 **after** the Stage 3b / 7b leak fixes (`PRIVACY_PROOF.md` §5). The pre-fix numbers are kept in `runs/icfm_sensitivity_verification.pre_leakfix.json`.

| metric | I-CFM | Batch-OT | $p$ |
|---|---|---|---|
| TSTR AUC (RF, synthetic→real holdout) | 0.533 ± 0.049 | 0.526 ± 0.091 | 0.49 |
| correlation Frobenius error / $d^2$ | 0.01086 ± 0.0004 | 0.01089 ± 0.0004 | 0.006 |
| mean DCR (z-scored) | 5.20 ± 0.17 | 5.16 ± 0.17 | 0.002 |
| TRTR AUC (real→real, reference) | 0.917 | 0.917 | — |

Reading:
- The fix has **no significant utility cost** (p = 0.49).
- The pre-fix run showed a borderline 0.05 AUC gap (p = 0.049), which did not survive the re-run (§7). The correlation and DCR differences are statistically significant but practically negligible (<1%).
- Both couplings are near chance at this ε, so this table establishes only the *relative* cost of the fix (§6).

## 6. Where the utility goes: DP noise vs model capacity vs training budget

Adult, I-CFM, TSTR AUC (TRTR = 0.917). Pre-leak-fix runs: the removed Stage-3b adjacency is exactly the one this table shows has no effect. Indicative only. Logs: `runs/_probe_ablation.log`, `runs/_probe_ceiling.log`; scripts `runs/_probe_ablation.py`, `runs/_probe_utility.py`.

| config | ε (reported) | σ | TSTR AUC |
|---|---|---|---|
| 5 ep, batch 512 (10 seeds) | 0.741 | 2.60 | 0.497 ± 0.071 |
| 5 ep, no Stage-3b adjacency (3 seeds) | 0.741 | 2.60 | 0.506 |
| 30 ep, batch 2048 (2 seeds) | 0.741 | — | 0.530 |
| 30 ep, batch 2048, Stage 7b on (1 seed) | 0.741 | — | 0.552 |
| 60 ep, batch 512 (1 seed) | 0.741 | 8.22 | 0.556 |
| 5 ep, near non-private (3 seeds) | 28.6 | 0.42 | 0.564 |
| **60 ep, near non-private (1 seed)** | 28.6 | 0.67 | **0.861** |

Reading: (i) the flow generator can learn Adult (0.861 vs 0.917 real) given enough optimizer steps; (ii) at ε≈0.74, adding steps does not help because the accountant raises σ in step (2.6 → 8.2), so utility stays near chance at every budget tried; (iii) the Stage-3b adjacency in the velocity network has no measurable effect; (iv) Stage 7b recalibration does not rescue label–feature structure (it fixes 1-way marginals only). The synthesizer's ε≈1 utility on Adult is therefore **not competitive** with published DP tabular synthesizers (marginal-based methods such as AIM/MST are known to be strong on Adult at this ε), and the paper should not claim otherwise. Single-seed rows are indicative only.

## 6b. Matched-ε external baselines (Adult, ε = 0.741, δ = 1e-5, 3 seeds)

All methods get the same public schema; the smartnoise baselines discretise numerics on public bounds (32 bins) and spend no ε on preprocessing. Script: `baselines_gen.py` / `baselines_eval.py`. TVD = mean total-variation distance over all 1-way / 2-way marginals on public bins (the AIM/MST workload error).

| method | TSTR AUC | 1-way TVD | 2-way TVD |
|---|---|---|---|
| AIM (McKenna et al. 2022) | **0.790 ± 0.011** | 0.018 | **0.112** |
| MST (McKenna et al. 2021) | 0.713 ± 0.031 | **0.014** | 0.128 |
| ours, I-CFM + Stage 7b | 0.534 ± 0.061 | 0.140 | 0.273 |
| PATE-CTGAN | 0.534 ± 0.043 | 0.506 | 0.705 |
| ours, I-CFM | 0.492 ± 0.054 | 0.510 | 0.719 |
| DP-CTGAN | 0.484 ± 0.036 | 0.298 | 0.468 |

Reading:
- Our generator is on par with the deep DP generators (DP-CTGAN, PATE-CTGAN) and far behind the marginal-based methods. This matches published DP-synthesis benchmarks.
- The paper's contribution is the sensitivity pitfall and its certified audit, **not** synthesis quality.

## 7. Retractions relative to earlier versions of this file

1. *"20% of seeds exceed the clip bound C=1.25."* That measured one **unclipped** per-example gradient difference against $C$ — not the quantity DP-SGD's guarantee depends on. Replaced by §4 (clipped-sum sensitivity vs $2C$, adversarial neighbour).
2. *"I-CFM matches or exceeds Batch-OT utility; the fix costs nothing."* From 5 seeds under an older DAPC.
3. *"The valid mechanism costs ≈0.05 AUC (p = 0.049)."* (2026-09-24 version.) That run still had the Stage-3b leak. After the fix, with the same 10 seeds, there is no significant difference (p = 0.49; §5). Claims 2 and 3 were each the result of one small, borderline experiment. The paper should state only "no significant utility difference detected at 10 seeds".

## 8. Status of pre-submission items

- Threat model and end-to-end composition proof: **done**, `PRIVACY_PROOF.md`.
- Empirical privacy audit: **done**, §4. The white-box audit certifies the violation; the black-box audit has no power and is reported as such.
- External baselines at matched ε: **done** (AIM, MST, DP-CTGAN, PATE-CTGAN), §6b. PrivBayes is not in smartnoise-synth, and AIM subsumes it on Adult in published comparisons.
- Remaining limitations to state in the paper:
  - Non-cryptographic PRNG (Opacus `secure_mode` is the fix).
  - Public-N assumption (`PRIVACY_PROOF.md` A2).
