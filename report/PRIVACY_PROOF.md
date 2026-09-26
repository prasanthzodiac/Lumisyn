# Threat Model and End-to-End Privacy Proof (identity latent mode, I-CFM)

**Scope.** `ACFMNetV5Pipeline(strict_release_privacy=True, latent_mode="identity", use_ot_coupling=False, use_public_graph_only=True)`, optionally with Stage 7b (`enable_dp_recalibration=True`). This is the configuration every result in the paper uses. HVAE mode (`latent_mode="vae"`), OT coupling, and the noised-dependency-graph ablation are **not** covered (see §6).
**Revision 2026-09-26.** Writing this proof line by line against the code found two further leaks, both fixed in this revision (§5). All experiments in the paper were re-run after the fixes.

---

## 1. Threat model

**Adversary.** Knows the code, every hyper-parameter, the public schema, and all records except the target's (the standard DP adversary). Sees the released outputs (§1.3). May choose the neighbouring dataset adversarially.

**Neighbouring relation.** Add/remove one record: $D \simeq D'$ if $|D \,\triangle\, D'| = 1$. This is the relation Opacus' accountants assume; replace-one guarantees follow by group privacy (factor 2 in $\varepsilon$).

**Released outputs** (`report["privacy_contract"]["dp_covered_outputs"]`):
1. the synthetic dataset;
2. the released velocity-network weights (EMA of the DP-SGD iterates);
3. the privacy summary ($\varepsilon$, $\delta$, $\sigma$, $q$, $T$).

Everything else in the run report — training loss, per-example gradient norms, clip fractions, row and missing-cell counts, the dataset-hash `audit_trail_handle`, flow diagnostics — is computed from private data **without noise**. It is operator telemetry and must not be released. The guarantee below says nothing about it.

**Assumptions.**
- **A1 (public schema).** Each column's type, `kind`, bounds (numeric) or category list (categorical) is declared independently of $D$. Under strict mode the pipeline warns about any column where the schema is incomplete (`privacy_contract.schema_incomplete_columns`), because the encoder then falls back to private statistics. All five schemas used in the paper are complete (checked).
- **A2 (public cohort size).** $N$ is treated as public. It sets the sampling rate $q=B/N$ (as in every DP-SGD deployment we know of) and the DAPC input $\mathrm{round}_{50}(N)$. Strictly, under add/remove adjacency $N$ is private. The guarantee is therefore for the mechanism family indexed by a publicly known $N$.
- **A3 (public names).** Column names and the target-column name are public.
- **A4 (randomness).** Noise is drawn from unpredictable generators: no fixed seeds anywhere on a noise path (fixed in the previous revision). The PRNGs are PyTorch/NumPy, not cryptographic; floating-point attacks on noise sampling (Mironov 2012) are out of scope. Opacus `secure_mode` is the production remedy.

---

## 2. Data flow, stage by stage

| Stage | What touches private data | Classification |
|---|---|---|
| 0 CDGL | drops/hashes columns chosen by **name**; computes a dataset hash for the audit handle | column selection = public function; hash = telemetry (not released) |
| 1 UDIE | sensitivity vector from **column-name** rules (`use_data_dependent_risk=False`); missingness, outlier, dependency and fingerprint analyses disabled; dependency matrix $=I$ | public |
| 2 ADPE | imputation with midpoints of public bounds; missing-indicator columns from the schema; one-hot / ordinal encoding with public categories; affine scaling to $[-1,1]$ from public bounds; class balancing disabled | **row-local map** $\phi$ (Lemma 1) |
| 3 DCGB | graph from column names (ontology and name-family edges), sensitivity vector, $W$ from dependency $=I$ | public |
| 3b refine | strict mode: **no data** (§5, fix 1) | public |
| 5 DAPC | $\varepsilon^\star, \sigma$ from $(\mathrm{round}_{50}N, d, \text{name-sensitivity}, \text{target-name flag}, q, T)$ | public (A2, A3) |
| 6/8 DP-SGD | I-CFM loss, per-example clip $C$, Gaussian noise $\sigma C$, Poisson sampling; EMA of iterates | $(\varepsilon_{\text{sgd}}, \delta_{\text{sgd}})$-DP (Lemma 2) |
| inference, 7 | prior samples → RK4 through the released net → clamp → public decode → public domain constraints | post-processing |
| 7b | $M$ one-way histograms on **public** bins / categories + Laplace$(M/\varepsilon_m)$; recalibration of the synthetic data to them | $\varepsilon_m$-DP (Lemma 3), then post-processing |

---

## 3. Lemmas

**Lemma 1 (row-local preprocessing).** Under A1, Stage 2 is a fixed map $\phi:\mathcal X\to[-1,1]^d$ applied to each record independently: every constant it uses (fill values, categories, bounds) comes from the schema. Hence $D\simeq D' \Rightarrow \phi(D)\simeq\phi(D')$. Any mechanism that is DP on the encoded data is therefore DP on the raw data.

*Proof.* Checked in `preparation/imputation.py` (fill $=(lo+hi)/2$ from the schema), `encoding.py` (categories and bounds from the schema; kind from the schema), and `scaling.py` (privacy-mode affine map from public bounds). No cross-row statistic is computed when A1 holds. $\square$

**Lemma 2 (DP-SGD on I-CFM).** Let $x_i=\phi(r_i)$. At each step the loader includes each record independently with probability $q$ (Opacus `DPDataLoader`), and for each included record draws $z_{0,i}\sim\mathcal N(0,I)$ and $t_i\sim U[0,1]$ independently of all other records. The per-example loss $\ell_i(\theta)=\|v_\theta((1-t_i)z_{0,i}+t_ix_i,t_i)-(x_i-z_{0,i})\|^2$ depends on no other record (I-CFM, Proposition 1 of `ICFM_SENSITIVITY_PROOF.md`). After clipping to norm $C$, the sum $S=\sum_i\bar g_i$ therefore has add/remove $\ell_2$-sensitivity $\le C$. Adding $\mathcal N(0,\sigma^2C^2I)$ makes each step a Poisson-subsampled Gaussian mechanism. $T$ adaptive steps compose to $(\varepsilon_{\text{sgd}},\delta_{\text{sgd}})$ as computed by the PRV accountant (Gopi et al. 2021; RDP of Mironov 2017 reported alongside as a second formal bound). The EMA weights and the synthetic samples are post-processing of the iterates.

Two implementation details:
- The accountant uses $q=B/N$. Opacus' loader samples at $1/\lceil N/B\rceil\le B/N$, and $\varepsilon$ is increasing in $q$, so the accounting is conservative.
- Empty Poisson batches are still stepped (pure noise) and counted.

The run asserts $\varepsilon_{\text{acct}}\le\varepsilon_{\text{sgd}}^{\text{target}}$ before release. $\square$

**Lemma 3 (Stage 7b).** With public bins and categories, one record changes at most one count in each of the $M$ histograms, so the concatenated count vector has add/remove $\ell_1$-sensitivity $M$. Laplace noise of scale $M/\varepsilon_m$ gives $\varepsilon_m$-DP (Dwork et al. 2006). Clamping negative counts to 0 and normalising are post-processing, as is the recalibration map $\text{recal}(\tilde D, \tilde H)$, which reads only the synthetic data and the noisy histograms. Values outside the public bounds or categories are simply not counted, which does not increase sensitivity. $\square$

**Lemma 3b (Stage 7c, class-conditional marginals).** There are $M$ tables of shape (label levels × feature levels), all on public bins and categories. One record changes one cell of each table by 1, so the concatenated count vector has add/remove $\ell_2$-sensitivity $\sqrt M$. Adding $\mathcal N(0,(z\sqrt M)^2)$ per cell is a Gaussian mechanism with noise multiplier $z$. The pipeline records it in the **same** PRV accountant as the DP-SGD steps (history entry $(z,1,1)$), and the flow's $\sigma$ is calibrated so that the joint composition meets $\varepsilon^\star$. Stage 7b is then disabled and not charged. Recalibration reads only the synthetic data and the noisy tables, so it is post-processing. $z$ is chosen from public $(n,K,M)$ (Lemma 4). $\square$

**Lemma 4 (DAPC consumes no budget).** $\varepsilon^\star$ and $\sigma$ are deterministic functions of public quantities (A2, A3) and are fixed before any mechanism touches private data. They are therefore constants of the mechanism, not adaptively chosen outputs. $\square$

---

## 4. Theorem (end-to-end guarantee)

Under A1–A4, the released outputs of the scoped pipeline satisfy $(\varepsilon_{\text{sgd}}+\varepsilon_m,\ \delta_{\text{sgd}})$-DP with respect to add/remove adjacency on the raw records. Here $\varepsilon_m=0$ without Stage 7b.

The pipeline sets $\varepsilon_{\text{sgd}}^{\text{target}}=\varepsilon^\star-\varepsilon_m$ and $\delta_{\text{sgd}}=0.8\,\delta$, and asserts the accountant stays within target. So the release is $(\varepsilon^\star,\delta)$-DP, which is the value `privacy_summary.prv_upper_with_laplace` reports.

*Proof.* By Lemma 1 we may work on encoded records. Stages 0, 1, 3, 3b and 5 are public (Lemma 4 and §2). Stage 8 is $(\varepsilon_{\text{sgd}},\delta_{\text{sgd}})$-DP (Lemma 2). Stage 7b is $\varepsilon_m$-DP (Lemma 3) and uses fresh noise independent of Stage 8. Basic sequential composition gives the sum. Synthetic sampling, repair and recalibration are post-processing of the two DP outputs. $\square$

**Corollary (OT coupling is not covered).** With batch-Hungarian OT the per-example loss depends on the whole batch. Theorem 1 of `ICFM_SENSITIVITY_PROOF.md` shows the replace-one sensitivity can exceed $2C$, and the audit exhibits 1.18× on real data. By the triangle inequality $\|S(B)-S(B')\|\le\|S(B)-S(B\setminus j)\|+\|S(B\setminus j)-S(B')\|$, so a replace-one violation implies an add/remove violation of $C$ at one of the two intermediate neighbours. Lemma 2 therefore fails under the relation Opacus assumes, and the reported $\varepsilon$ is not a valid bound.

---

## 5. Leaks found and fixed while writing this proof

1. **Stage 3b fed an un-noised private statistic into the released model.** `PostLatentGraphRefiner.refine_graph(G_0, W_0, X_bal)` computed $|\cos|$ similarity between feature columns over **all private rows**. The result became the flow network's `adj_matrix` buffer, so it was part of the released weights and shaped every synthetic sample.
   - Fix: strict mode passes `None`, so refinement uses only the public $W_0$.
   - Impact on results: the Stage-3b ablation had already shown the adjacency has no measurable effect on utility (`ICFM_SENSITIVITY_PROOF.md` §6). All paper experiments were re-run after the fix.
2. **Stage 7b bin edges used a private minimum.** For zero-inflated columns the lower log-bin edge was `min(positive private values)`, which revealed the smallest positive value (e.g. of `capital-gain`).
   - Fix: the edge is now `max(1, public lower bound)`.

Earlier revisions had already fixed: fixed noise seeds (Stage 7b, dependency-matrix sanitiser, DAPC), DAPC's private inputs and noise clipping, and the dependency-matrix sensitivity claim (now warned as uncertified).

---

## 6. Not covered

- **OT coupling.** It is provably not covered (Corollary). `use_ot_coupling=True` requires `allow_nonprivate_ot=True`.
- **HVAE mode.** Encoder and decoder DP-SGD, followed by DP-SGD of the flow on the encoded latents, would compose. However, the mode is experimental: it collapses posteriors under DP, and its diagnostics are telemetry. It is not claimed in the paper.
- **`use_public_graph_only=False`.** The Laplace dependency matrix's per-entry sensitivity $2/N$ is not established for Spearman / ANOVA / Cramér's V. It is uncertified and a runtime warning says so.
- **Incomplete schemas (A1 fails).** The encoder reads private min/max/kind/categories, and a runtime warning lists the affected columns.
- **Operator telemetry** (§1).
