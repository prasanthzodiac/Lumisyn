// Supplementary Material. Per-seed tables are read directly from the committed result files.
const fs = require("fs");
const path = require("path");
const { REFS } = require("./refs.js");
const RUNS = path.resolve(__dirname, "..", "..", "runs");
const J = (f) => JSON.parse(fs.readFileSync(path.join(RUNS, f), "utf8"));
const stats = J("dapc_split/flow_ablation_stats.json");
const util = J("icfm_sensitivity_verification.json").utility;
const dapc = J("dapc_sensitivity.json");
const leaks = J("leak_demos.json");
const fva = J("flow_value/analysis.json");
const fvc = fva.confirmation.cells;
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const sci = (x) => { const [m, e] = x.toExponential(1).split("e"); return `${m} × 10${[...String(Number(e))].map((c) => SUP[c]).join("")}`; };
const f3 = (x) => x.toFixed(3), f4 = (x) => x.toFixed(4);

const B = [];
const p = (t) => B.push({ type: "p", text: t });
const h1 = (t) => B.push({ type: "h1", text: t });
const h2 = (t) => B.push({ type: "h2", text: t });

B.push({ type: "title", text: `Supplementary Material for: ${process.env.MAIN_TITLE || "Minibatch Optimal Transport Invalidates Standard DP-SGD Accounting in Private Flow Matching for Tabular Data"}` });
B.push({ type: "authors", text: "[Author Name]¹, [Author Name]², [Author Name]³, [Author Name]¹*" });
B.push({ type: "affil", lines: ["[Affiliations as in the main text]"] });

// ------------------------------------------------------------------ S1
h1("S1 Dataset-Adaptive Privacy Budget Controller (DAPC)");
h2("S1.1 Complete Specification");
p("DAPC maps a requested budget ε_max to the released budget");
B.push({ type: "eq", omml: "dapc", n: "S1" });
p("where R is a public risk score. DAPC reads only public inputs: column names, the target-column name, the cohort size N (assumption A2) and the encoded dimension d, which is fixed by the declared schema. Table S1 lists every rule and constant. Column sensitivity is assigned by case-insensitive substring matching of the column name against three keyword lists, checked in the order shown; the first list that matches determines the score.");
B.push({ type: "table", caption: "**Table S1.** Complete DAPC specification. All inputs are public.", size: 17,
  header: ["Quantity", "Definition", "Value or rule"], widths: [1700, 3300, 4000], rows: [
    ["Column score s_i (high)", "name contains any of", "ssn, social_security, name, mrn, address, zip, phone, email, ip_address, biometric → s_i = 0.95"],
    ["Column score s_i (medium)", "else, name contains any of", "age, gender, sex, race, ethnicity, dob, birth, income, marital, insurance → s_i = 0.65"],
    ["Column score s_i (clinical)", "else, name contains any of", "diagnosis, disease, icd, cancer, hiv, mental_health, substance, mortality → s_i = 0.50"],
    ["Column score s_i (default)", "otherwise", "s_i = 0.10"],
    ["r_sens", "mean of s_i² over all columns", "in [0.01, 0.9025]"],
    ["r_scale", "exp(−n/600), n = N rounded to the nearest 50", "in (0, 1]"],
    ["r_dim", "1 − exp(−d/100), d = encoded dimension", "in [0, 1)"],
    ["r_task", "0.85 if the target name contains mortality, death, cancer or icu; otherwise 0.30", "public flag"],
    ["Aggregation", "R = max(r_sens, 1 − Π_k (1 − w_k r_k)^{γ_k})", "w = (0.35, 0.30, 0.20, 0.15), γ = (1.5, 1, 1, 1)"],
    ["Budget map", "Equation S1", "ε_min = 0.4, β = 2.5, ε_max = requested budget (1.0; 4 or 8 in the sweep)"],
    ["Abstention", "raw N < 100", "no release"],
    ["Rejection", "d > 500", "no release"],
    ["Stage 7c split (fixed)", "z such that the marginal release alone costs ρ ε*", "ρ ∈ {0.2, 0.4, 0.6, 0.9}"],
    ["Stage 7c split (adaptive)", "z = max(0.25 n / (K √M), z_cap)", "K = largest table size, M = number of tables, z_cap gives a 0.6 share"],
    ["Flow noise σ", "bisection on the PRV accountant", "joint composition with Stage 7c equals ε* at δ_SGD = 8 × 10⁻⁶"],
  ] });
h2("S1.2 Robustness to the Hand-Set Constants");
p(`Across ten datasets, DAPC selected ε* between 0.655 and 0.905 and abstained below 100 records. We perturbed all constants jointly over 2000 draws: weights from Dirichlet(50 w), exponents scaled by U(0.8, 1.2), and β from U(2, 3). The 5–95% range of ε* per dataset had a width between 0.06 and 0.15. The dataset ranking was stable, with a median Kendall τ of ${dapc.kendall_tau_median.toFixed(2)} and a 5th percentile of ${dapc.kendall_tau_p05.toFixed(2)}. A 10% change to any single weight moved ε* by at most ${Math.max(...Object.values(dapc.max_eps_change_for_10pct_weight_bump)).toFixed(3)}. The adaptive split rule gave TSTR AUC / 2-way TVD of 0.904 / 0.116 on Adult, 0.780 / 0.070 on Cardio and 0.729 / 0.601 on Pima (3 seeds). It matched the best fixed split on Adult (0.906 / 0.096 at a 0.9 share) but not on Cardio, where the 0.9 share reached a 2-way TVD of 0.044. It is therefore not used in the main results.`);
h2("S1.3 Pre-Registered Test of an ε-Adaptive Split");
p(`Because the ad hoc rule above was fitted by hand, we also ran a pre-registered test (runs/flow_value/PREREGISTRATION.md, written before the grid was run). Configurations used the warm-started flow of main-text Section 2.6 with marginal shares ρ ∈ {0.2, 0.5, 0.9}, on Adult and Cardio at requested budgets of 1, 4 and 8, with 3 seeds each. Each configuration received a composite score, the mean of the min–max-normalised TSTR AUC, all-column accuracy, −(2-way TVD) and −(3-way label TVD) within its (dataset, ε) cell. The rule was fitted on Adult only, as the simplest monotone function of the released ε* that reproduces the best ρ at each budget, and was evaluated on Cardio. The claim required the rule to beat every fixed ρ on Cardio when averaged over the three budgets. On Adult, ρ = 0.9 was best at every budget, so the fitted rule is the constant ρ = 0.9 and adapts to nothing. On Cardio, its mean composite was ${fva.cardio_mean_composite.adaptive.toFixed(3)}, against ${fva.cardio_mean_composite["warm_0.5"].toFixed(3)} for a fixed ρ = 0.5 (Table S1b). The pre-registered criterion therefore failed, and we make no claim that an ε-adaptive budget split improves utility.`);
B.push({ type: "table", caption: "**Table S1b.** Pre-registered adaptive-split test: composite score per (dataset, requested ε) cell, 3 seeds (higher is better; the scores are normalised within each cell).", size: 17,
  header: ["Cell (ε*)", "ρ = 0.2", "ρ = 0.5", "ρ = 0.9", "Marginals only (ρ = 0.9)", "Rule choice"], widths: [1900, 1300, 1300, 1300, 1900, 1300],
  rows: Object.entries(fva.cells).map(([k, c]) => [`${k.split("@")[0] === "adult" ? "Adult" : "Cardio"}, ${Number(k.split("@")[1])} (${c.eps_star.toFixed(2)})`,
    ...["warm_0.2", "warm_0.5", "warm_0.9", "marginals_only_0.9"].map((g) => c.composite[g].toFixed(3)),
    k.startsWith("adult") ? "(fitted)" : "ρ = 0.9"]) });
B.push({ type: "fig", file: "fig7_dapc.png", width: 5.8, caption: "**Figure S1.** Budget selected by DAPC for ten datasets, with 5–95% ranges under joint perturbation of its constants (2000 draws). n: rounded cohort size; d: encoded dimension." });

// ------------------------------------------------------------------ S2
h1("S2 Formal Mechanism Specification and Proof of Theorem 2");
h2("S2.1 Setting");
p("**Adjacency.** D ≃ D′ if one is obtained from the other by adding or removing one record (add/remove). Replace-one guarantees follow from Equation 1 of the main text.");
p("**Released outputs.** (i) the synthetic table; (ii) the EMA weights of the velocity network; (iii) the privacy summary (ε*, δ_SGD, σ, z, q, T). Everything else computed by the pipeline, including loss curves, gradient norms, clip fractions, row and missing-cell counts and a dataset hash used as an internal audit handle, is operator telemetry that is computed without noise and is never released.");
p("**Assumptions.** A1 public schema (types, bounds, category lists, kinds); A2 public cohort size N; A3 public column and target names; A4 unpredictable noise. The pipeline warns at run time and lists every column for which A1 is not met, because the encoder would otherwise fall back to private statistics.");
h2("S2.2 Mechanisms");
p("**M₁ (DP-SGD on I-CFM).** For t = 1, …, T with T = epochs × ⌈N/B⌉ fixed in advance: sample B_t by including each encoded record independently with probability 1/⌈N/B⌉; for each included record draw z₀ ~ N(0, I) and t ~ U(0, 1) independently; compute the per-example loss and gradient, clip to norm C = 1.25, sum, add N(0, σ²C²I), divide by the public constant qN = B, and apply AdamW; update the EMA copy. Initialisation uses PyTorch defaults with a zero-initialised output layer and does not depend on data; with the warm start, it is instead the result of non-private pretraining on records sampled from the output of M₂ (Lemma S5). Empty batches are processed with the noise term only and counted.");
p("**M₂ (Stage 7c).** For each non-target column c, compute the count table n_c(y, b) over public target levels y and public bins or categories b, and release n_c + N(0, z²M) independently per cell, where M is the number of tables.");
p("**Accounting.** The PRV accountant receives the history {(σ, q = B/N) × T, (z, 1) × 1} and σ is found by bisection so that the composed ε at δ_SGD = 8 × 10⁻⁶ equals ε*. The run asserts the accountant value before any output is released. RDP is reported alongside as a second bound.");
h2("S2.3 Lemmas");
p("**Lemma S1 (row-local preprocessing).** Under A1, imputation uses the midpoint of public bounds, encoding uses public categories and bounds, and scaling is an affine map fixed by public bounds. Hence the encoded versions of D ≃ D′ are again add/remove neighbours.");
p("**Lemma S2 (DP-SGD).** By Proposition 1 of the main text, the clipped sum of each step has add/remove L2 sensitivity at most C. Each step is therefore a Poisson-subsampled Gaussian mechanism with noise multiplier σ. The accountant uses q = B/N ≥ 1/⌈N/B⌉, and since ε is non-decreasing in q, the accounting is conservative. Division by the public constant B and the AdamW and EMA updates are post-processing of the noisy sums under A2.");
p("**Lemma S3 (Stage 7c).** One record contributes exactly one count to each of the M tables, so the concatenated vector has L2 sensitivity √M, and adding N(0, z²M) per cell is a Gaussian mechanism with noise multiplier z. Its noise is drawn from an unseeded generator, independently of M₁.");
p("**Lemma S4 (public controller).** ε*, z, σ and T are deterministic functions of public inputs (Table S1) and are fixed before M₁ or M₂ is run, so they are constants of the composed mechanism.");
p("**Lemma S5 (warm start).** When the warm start is enabled, M₂ is run first. N records are drawn from its noisy tables, independently within each class, and encoded with the public schema. The velocity network is then trained on them without noise, starting from the data-independent initialisation. The only inputs of this step are the output of M₂, the public schema, the public N (A2) and fresh randomness, so the starting point of M₁ is a post-processing of M₂. M₁ is thus chosen adaptively after M₂. PRV composition bounds adaptive compositions of Gaussian mechanisms exactly as it bounds non-adaptive ones, and σ, q and T are still fixed in advance (Lemma S4), so the accounting history is unchanged. In the ablation without DP fine-tuning, every step of M₁ is still executed and accounted, with the learning rate set to zero.");
p("**Proof of Theorem 2.** By Lemma S1 we may work on encoded records. By Lemma S4 the mechanism parameters are constants, and by Lemma S5 a warm start makes M₁ an adaptive choice after M₂. M₁ and M₂ are Gaussian mechanisms with the stated sensitivities (Lemmas S2 and S3), and their PRV composition at δ_SGD yields ε*. Sampling from the released EMA model with fresh prior noise, clamping, decoding, and the within-class quantile map read only the released model, fresh randomness and the noisy tables, and are therefore post-processing. The released outputs are thus (ε*, δ_SGD)-DP, and δ_SGD ≤ δ. ∎");
h2("S2.4 Scope of the White-Box Audits");
p("The audits in the main text condition on a fixed pair of batches with fixed prior draws and times. They certify, with 95% confidence, a lower bound on the sensitivity of that conditional Gaussian step, and hence that the sensitivity assumption of the standard accountant fails. They do not include the Poisson sampling of the trained mechanism, which amplifies privacy, so they are not lower bounds on its privacy loss. The accounting-consistent ε is the smallest ε that the standard subsampled-Gaussian accountant can certify with a sensitivity bound large enough to cover the audited pair: since that pair has replace-one sensitivity at least μ_lb σC, one of its add/remove neighbours has sensitivity at least μ_lb σC/2, and the accountant with noise σC and this bound is equivalent to noise multiplier 2/μ_lb. A lower bound on the privacy loss of the full mechanism would require auditing the sampling event, for example with many training runs or a one-run audit of the subsampled step, which we leave to future work.");
B.push({ type: "table", caption: "**Table S2.** Every code path that reads private data, with its classification.", size: 17,
  header: ["Stage", "What reads private data", "Classification"], widths: [1700, 4500, 2800], rows: [
    ["0 De-identification", "Column selection by name; dataset hash for an internal audit handle", "Public function; hash is telemetry, not released"],
    ["1 Dataset intelligence", "Column-name sensitivity rules; missingness, outlier and dependency analyses disabled in strict mode", "Public"],
    ["2 Preprocessing", "Imputation, encoding and scaling from the public schema", "Row-local map (Lemma S1)"],
    ["3 Constraint graph", "Graph from column names and ontology; dependency matrix set to the identity", "Public"],
    ["3b Graph refinement", "Strict mode passes no data (leak fixed, Section S3)", "Public"],
    ["5 DAPC", "Rounded N, d, names", "Public (Lemma S4)"],
    ["6 Warm start (optional)", "Reads only the noisy Stage 7c tables, the public schema and N", "Post-processing of M₂ (Lemma S5)"],
    ["6/8 Flow training", "Algorithm 1 with I-CFM", "M₁ (Lemma S2)"],
    ["7c Marginals", "Label × feature count tables", "M₂ (Lemma S3)"],
    ["7 Repair and decoding", "Released model, fresh prior noise, public domain", "Post-processing"],
  ] });

// ------------------------------------------------------------------ S3
h1("S3 Leak Reproductions");
p("The reproductions run on UCI Adult with the pre-fix behaviour (script leak_demos.py, output leak_demos.json in the artifact). Where the fixed code no longer exposes the old path, the pre-fix line is re-implemented verbatim in the script.");
B.push({ type: "algo", caption: "**Listing S1.** Minimal pre-fix code patterns of the five leak classes (simplified).", lines: [
  "# 1 fixed seed on a noise path",
  "rng = np.random.default_rng(42); noisy = counts + rng.laplace(0, M / eps, K)",
  "# 2 clipping the noise instead of the query",
  "release = f(D) + np.clip(rng.laplace(0, b), -c, c)",
  "# 3 un-noised statistic stored in a model buffer",
  "cos = abs(cosine_between_columns(X_private)); model.register_buffer('adj_matrix', normalise(a*W0 + (1-a)*cos))",
  "# 4 data-dependent bin edge",
  "lo = max(1.0, X_private[X_private > 0].min()); bins = np.geomspace(lo, hi, K)",
  "# 5 silent schema fallback",
  "lo, hi = schema.get('bounds', [col.min(), col.max()])      # private min/max when bounds are missing",
] });
B.push({ type: "table", caption: "**Table S3.** Reproduction results (from leak_demos.json).", size: 17,
  header: ["Leak class", "Reproduction", "Result"], widths: [2300, 4300, 2400], rows: [
    ["Fixed seed", "Laplace histogram of race at ε = 0.1 with seed 42; regenerate and subtract the noise", `${leaks.fixed_seed.cells_recovered_exactly} of ${leaks.fixed_seed.cells} counts recovered exactly`],
    ["Clipped noise", `Noise clipped to ±${leaks.clipped_noise.noise_clip}, query gap ${leaks.clipped_noise.query_gap}`, "Output ranges disjoint, ε unbounded"],
    ["Model buffer", "Read adj_matrix from the state_dict; deg_{i} = 1/N_ii; A = N ⊙ √(deg degᵀ) − I; |cos| = (A − αW₀)/(1 − α)", `${leaks.model_buffer.matrix_size} × ${leaks.model_buffer.matrix_size} matrix, max error ${sci(leaks.model_buffer.max_abs_error_recovering_private_cosine)}`],
    ["Bin edge", "Pre-fix lower log-bin edge for capital-gain", `Published ${leaks.bin_edge.published_edge} = private minimum ${leaks.bin_edge.smallest_private_positive_value}`],
    ["Schema fallback", "Encode age without declared bounds", `Private min ${leaks.schema_fallback.private_min} and max ${leaks.schema_fallback.private_max} stored by the encoder`],
  ] });
p("Regression tests in the artifact: test_leak_regressions.py (graph refinement ignores data in strict mode; histogram bin edges do not depend on private values), test_conditional_marginals.py (Stage 7c noise differs across releases; joint calibration never exceeds the budget), test_phase1_privacy.py (DAPC is invariant to shuffling private values; I-CFM per-example gradient independence), and a run-time warning that lists columns not covered by the schema.");

// ------------------------------------------------------------------ S4
h1("S4 Per-Seed Results and Statistical Tests");
const seedRows = (ds, metric, fmt) => Object.entries(stats[ds][metric].per_seed).map(([s, [a, b]]) => [s, fmt(a), fmt(b), (a - b >= 0 ? "+" : "") + fmt(a - b)]);
B.push({ type: "table", caption: "**Table S4a.** Adult, share 0.9: full pipeline vs marginals only, per seed.", header: ["Seed", "AUC full", "AUC marginals only", "Difference"], widths: [1500, 2500, 2500, 2500], rows: seedRows("adult", "tstr_auc", f4) });
B.push({ type: "table", caption: "**Table S4b.** Cardio, share 0.9: full pipeline vs marginals only, per seed.", header: ["Seed", "AUC full", "AUC marginals only", "Difference"], widths: [1500, 2500, 2500, 2500], rows: seedRows("cardio", "tstr_auc", f4) });
B.push({ type: "table", caption: "**Table S4c.** 2-way TVD per seed: full pipeline vs marginals only.", header: ["Seed", "Adult full", "Adult marginals only", "Cardio full", "Cardio marginals only"], widths: [1000, 2000, 2000, 2000, 2000],
  rows: Object.keys(stats.adult.tvd_2way_all.per_seed).map((s) => [s, ...stats.adult.tvd_2way_all.per_seed[s].map(f4), ...stats.cardio.tvd_2way_all.per_seed[s].map(f4)]) });
B.push({ type: "table", caption: "**Table S4d.** I-CFM vs batch-OT (flow only), Adult, per seed (TSTR AUC).", header: ["Seed", "I-CFM", "Batch-OT", "Difference"], widths: [1500, 2500, 2500, 2500],
  rows: util.i_cfm.map((r, i) => [String(r.seed), f3(r.tstr_auc), f3(util.batch_ot[i].tstr_auc), ((r.tstr_auc - util.batch_ot[i].tstr_auc) >= 0 ? "+" : "") + f3(r.tstr_auc - util.batch_ot[i].tstr_auc)]) });
const pm = (a) => `${a[0].toFixed(4)} ± ${a[1].toFixed(4)}`;
B.push({ type: "table", caption: "**Table S4e.** Pre-registered 10-seed confirmation (Addendum A), all-column accuracy, mean ± sd. H1: warm start + DP fine-tuning > marginals only; H2: warm start + DP fine-tuning > warm start without fine-tuning. One-sided Welch tests, Holm-corrected over the six cells.", size: 16,
  header: ["Cell (ε*)", "Marginals only", "Warm, no fine-tuning", "Warm + DP fine-tuning", "H1 diff (p Holm)", "H2 diff (p Holm)"], widths: [1500, 1500, 1500, 1600, 1450, 1450],
  rows: Object.entries(fvc).map(([k, c]) => [`${k.startsWith("adult") ? "Adult" : "Cardio"} (${c.eps_star.toFixed(2)})`, pm(c.allcol_acc_marg), pm(c.allcol_acc_warm_only), pm(c.allcol_acc_warm),
    `+${c.H1_diff.toFixed(4)} (${sci(c.H1_p_holm)})`, `${c.H2_diff >= 0 ? "+" : ""}${c.H2_diff.toFixed(4)} (${sci(c.H2_p_holm)})`]) });
B.push({ type: "table", caption: "**Table S4f.** The same runs, other metrics (mean over 10 seeds). AUC equivalence: TOST at ±0.01 (Welch).", size: 16,
  header: ["Cell", "AUC marginals only", "AUC warm + DP", "TOST p", "3-way TVD marginals only", "3-way TVD warm + DP"], widths: [1300, 1500, 1500, 1200, 1800, 1700],
  rows: Object.entries(fvc).map(([k, c]) => [`${k.startsWith("adult") ? "Adult" : "Cardio"} ${Number(k.split("@")[1])}`, c.tstr_auc_marg[0].toFixed(4), c.tstr_auc_warm[0].toFixed(4), sci(c.auc_tost_p),
    c.tvd_3way_label_marg[0].toFixed(4), c.tvd_3way_label_warm[0].toFixed(4)]) });
p("Equivalence was tested with paired two one-sided t-tests at margins of ±0.01 AUC and ±0.005 TVD, fixed before the 10-seed confirmation runs. For each metric, the lower statistic is t_L = (d̄ + m)/(s_d/√n) and the upper is t_U = (d̄ − m)/(s_d/√n), with n − 1 = 9 degrees of freedom; equivalence is declared when both one-sided p-values are below 0.05. The resulting statistics are reported in Table 6 of the main text.");

// ------------------------------------------------------------------ S5
h1("S5 Full Configuration and Software Environment");
B.push({ type: "table", caption: "**Table S5.** Complete configuration.", size: 17, header: ["Component", "Setting"], widths: [2500, 6500], rows: [
  ["Main environment", "Python 3.12.10; PyTorch 2.6.0+cu124 (CUDA 12.4, cuDNN 9.1); Opacus 1.6.0; NumPy 2.5.1; SciPy 1.18.0; scikit-learn 1.9.0; pandas 2.3.3; exact versions pinned in requirements-lock.txt"],
  ["Baseline environment", "Separate virtual environment: smartnoise-synth 1.0.8, mbi 1.1.0, PyTorch 2.14, Opacus 0.14; pinned in requirements-baselines-lock.txt"],
  ["Hardware", "NVIDIA GeForce RTX 4060 Ti (8 GB), driver 610.88; FP32 with TF32 matrix multiplication"],
  ["Encoding", "Public-schema min–max scaling to [−1, 1]; one-hot for categories (≤ 20 levels), ordinal otherwise; missing indicators from the schema"],
  ["Velocity network", "Time embedding: sinusoidal (64) → Linear(64) → SiLU → Linear(64). Input: learnable elementwise affine plus 0.1 × public-graph propagation, concatenated with the time embedding → Linear(128) → LayerNorm → SiLU. Two residual blocks h + [Linear–LayerNorm–SiLU]×2. Output Linear, zero-initialised"],
  ["Training", "AdamW (lr 10⁻³, weight decay 10⁻⁵), cosine annealing to 10⁻⁵ stepped per epoch; per-example loss = MSE averaged over d coordinates; t ~ U(0, 1); z₀ ~ N(0, I); no gradient accumulation; no early stopping"],
  ["DP-SGD", "Microbatched per-example autograd (chunks of 128); clip C = 1.25; noise N(0, σ²C²I) on the clipped sum; divide by B = 512; Poisson sampling via Opacus DPDataLoader; empty batches stepped with noise"],
  ["Accounting", "Opacus PRV accountant at δ_SGD = 8 × 10⁻⁶ (release δ = 10⁻⁵), RDP reported alongside; σ by 48-step bisection"],
  ["Stage 7c", "32 public bins (log-spaced when the upper-to-lower bound ratio exceeds 1000), public categories; Gaussian noise z√M per cell; unseeded generator"],
  ["Warm start", "Stage 7c released first; N records drawn from the noisy tables with independent within-class ranks and encoded with the public schema; 20 epochs of AdamW (lr 10⁻³, weight decay 10⁻⁵, batch 512) without noise; I-CFM loss; the EMA starts after the warm start. Ablation: DP-SGD steps run and accounted with learning rate 0"],
  ["Sampling", "Fourth-order Runge–Kutta, 30 steps; clamp to [−1, 1]; argmax decoding of one-hot blocks"],
  ["Sensitivity search", "Batch 64; 10 initialisations × 10 batches; C = 0.5 × minimum per-example gradient norm (all clipped); screened: 200 real + 63 prior-anchored + 50 random valid candidates, exact sensitivity for the top 5; hill-climb: 150 steps from the top 3, 1–3 block mutations per step"],
  ["White-box audit", "10⁶ trials per hypothesis in chunks of 4000 on GPU; threshold Δ/2 on the projection; Clopper–Pearson at 97.5% per rate (Bonferroni); implied ε from the PRV accountant at q = 512/N, T = 5⌈N/512⌉"],
  ["Canary audit", "1000 canaries, inclusion probability 1/2, k = 100 guesses per side fixed in advance, 95% confidence (pure-DP form of the one-run bound)"],
  ["Baselines", "Numeric columns discretised into 32 public bins; preprocessor ε = 0; DP-CTGAN 50 epochs; δ = 10⁻⁵; 3 seeds"],
  ["Data provenance", "adult.csv: 48,842 records, SHA-256 d273d956c28eba72241f69dcbb78e37e93862248503a1873fb44b9fc2a6ae00b; cardio_train.csv: 70,000 records, SHA-256 2423cc5534fdc70887471778fc92b673ebb53d0d5188537b9029643b8a10e275; pima_diabetes.csv: 768 records, SHA-256 123ce738f394095a201da0f65c5e174cdcb19cdf5ec1535aee65e2b0e1e2aae2. Samples: uniform without replacement, pandas sample(n, random_state=42); splits: scikit-learn train_test_split(test_size=0.2, random_state=42), unstratified"],
  ["Loader and accountant rates", "Opacus DPDataLoader.from_data_loader samples at q_L = 1/⌈N/B⌉ (Adult 0.0417, Cardio 0.0313); the accountant uses q = B/N (0.0427, 0.0320) ≥ q_L; regression test test_loader_sampling_rate_never_exceeds_accountant_rate"],
  ["Seeds", "PyTorch and NumPy generators seeded with the run index for experiments (fixes DP-SGD noise for reproducibility only); Stage 7c noise unseeded; data splits use seed 42"],
] });

// ------------------------------------------------------------------ S6
h1("S6 Artifact Manifest");
B.push({ type: "table", caption: "**Table S6.** Where each result comes from. Paths are relative to the repository root.", size: 17, header: ["Result", "Script", "Raw output"], widths: [2600, 3600, 2800], rows: [
  ["Table 2 (random, screened)", "lumisyn/backend/verify_icfm_sensitivity.py --worst-case-only", "runs/icfm_worst_case_sensitivity.json"],
  ["Table 2 (hill-climbed), Table 3, Fig. 4", "lumisyn/backend/audit_privacy.py [--dataset cardio --part A]", "runs/privacy_audit.json"],
  ["Section 3.4, Table S4d", "lumisyn/backend/verify_icfm_sensitivity.py --utility-only", "runs/icfm_sensitivity_verification.json"],
  ["Table 4, Table S3", "lumisyn/backend/leak_demos.py", "runs/leak_demos.json"],
  ["Tables 5–6, Figs. 5–6, Tables S4a–c", "lumisyn/backend/dapc_split_eval.py; flow_ablation_stats.py", "runs/dapc_split/results.jsonl; flow_ablation_stats.json"],
  ["Table 7, Fig. 7", "baselines_eval.py --prepare; baselines_gen.py (baseline environment); baselines_eval.py", "runs/baselines/results.json"],
  ["Section 3.8", "lumisyn/backend/run_hard_stress_test.py --seeds 5", "runs/hard_stress_test_results.json"],
  ["Section 3.7, Table 8, Fig. 8, Tables S1b, S4e–f", "lumisyn/backend/flow_value_eval.py; flow_value_analysis.py", "runs/flow_value/results.jsonl; analysis.json; PREREGISTRATION.md"],
  ["Section S1, Fig. S1", "lumisyn/backend/dapc_sensitivity.py", "runs/dapc_sensitivity.json"],
  ["All figures", "paper/docx_build/make_figures.py", "paper/docx_build/fig/"],
  ["Long-form proof", "–", "report/PRIVACY_PROOF.md"],
  ["Tests (90)", "python -m pytest lumisyn/backend/tests", "–"],
] });

module.exports = { B, REFS };
