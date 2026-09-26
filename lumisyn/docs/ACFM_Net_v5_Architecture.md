# ACFM-Net v5 — Expert Redesign
### 30-Year Research-Grade Architecture

---

## Critical Issues Fixed from v4 → v5

| Issue | v4 Flaw | v5 Fix |
|---|---|---|
| **DP Violation** | ε = f_θ(I,S,Z) — learned budget breaks formal DP guarantees | Fixed ε*, PRV Accountant, learn per-feature allocation only |
| **KS in training loss** | KS test is non-differentiable — cannot backprop | Replaced with Sliced Wasserstein Distance |
| **Chicken-and-egg** | Graph built before latent Z, but Z needs graph | Stage 3b: graph refinement AFTER Stage 4 representation |
| **VAE posterior collapse** | Vanilla VAE fails on mixed-type data | Hierarchical VAE with disentangled subspaces |
| **Gradient conflict** | 6 loss terms fight each other — instability | PCGrad + MGDA + GradNorm adaptive weighting |
| **Linear fusion** | F = αM+βG+γP+δZ — no nonlinear interaction | Gated Multi-Source Cross-Attention |
| **Undefined Repair** | Repair branch in Stage 7 never defined | Explicit constrained gradient descent repair loop |
| **Fixed graph** | Graph update in Stage 8 optimization — feedback instability | Graph FROZEN after Stage 3b; only edge weights updated |
| **No governance** | No IRB/consent/audit mechanism | Stage 0: Clinical Data Governance Layer |
| **No curriculum** | All losses from epoch 0 — unstable | 3-phase curriculum training in Stage 8 |
| **No DP warm-up** | Privacy injected from step 1 — kills early learning | Privacy warm-up: anneal ε from ε_warm → ε* |
| **Unnormalized convergence** | C = ω₁ΔL+ω₂ΔU+ω₃ΔP+ω₄ΔF — different scales dominate | Normalized: C = Σ ωᵢ × (ΔXᵢ / σ(ΔXᵢ)) |
| **No discrete flow** | Single CFM for all types — wrong for categorical | Separate: CFM (continuous) + Discrete Flow (categorical) |
| **No model selection** | No checkpoint selection criteria | Best model = highest utility subject to privacy constraint |

---

```
                         ┌──────────────────────────────────────────────────────┐
                         │         Any Structured Healthcare Dataset            │
                         │──────────────────────────────────────────────────────│
                         │ NHANES │ MIMIC-IV │ eICU │ Heart Disease │ Stroke   │
                         │ Diabetes │ Breast Cancer │ ICU │ EHR │ Genomics     │
                         └──────────────────────────────────────────────────────┘
                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 0 : CLINICAL DATA GOVERNANCE LAYER (CDGL)                      ★ NEW IN v5 ★
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  ┌──────────────────────┐     ┌──────────────────────┐                     │
   │  │  0.1 IRB / Ethics    │     │  0.2 De-identification│                    │
   │  │  Compliance Check    │     │  Verifier             │                    │
   │  │                      │     │                       │                    │
   │  │ • IRB Protocol ID    │     │ • Safe Harbor Check   │                    │
   │  │ • HIPAA Compliance   │     │ • Expert Determination│                    │
   │  │ • GDPR Compliance    │     │ • Re-ID Risk < τ      │                    │
   │  │ • Data Use Agreement │     │ • PHI Residual Scan   │                    │
   │  └──────────┬───────────┘     └──────────┬────────────┘                    │
   │             └──────────────┬─────────────┘                                 │
   │                            ▼                                                │
   │               ┌─────────────────────────┐                                  │
   │               │  0.3 Consent Framework  │                                  │
   │               │      Validator          │                                  │
   │               │                         │                                  │
   │               │ • Consent Scope Check   │                                  │
   │               │ • Secondary Use Auth    │                                  │
   │               │ • Research Exception    │                                  │
   │               └────────────┬────────────┘                                  │
   │                            ▼                                                │
   │               ┌─────────────────────────┐                                  │
   │               │  0.4 Audit Trail        │                                  │
   │               │      Initializer        │                                  │
   │               │                         │                                  │
   │               │ • Dataset Hash (SHA-256)│                                  │
   │               │ • Access Timestamp      │                                  │
   │               │ • Governance Certificate│                                  │
   │               │ • Immutable Log Start   │                                  │
   │               └────────────┬────────────┘                                  │
   │                            │                                                │
   │                    ┌───────┴────────┐                                       │
   │                    ▼               ▼                                        │
   │                  PASS            FAIL                                       │
   │                    │               │                                        │
   │                    │            HALT — data access denied                  │
   │                    ▼                                                        │
   │            Governance Certificate                                           │
   └─────────────────────────────────────────────────────────────────────────────┘

   ✓ Governance Certificate    (required for all downstream stages)
   ✓ Audit Trail Handle        (all operations logged)

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 1 : UNIVERSAL DATASET INTELLIGENCE ENGINE (UDIE) v5
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  ┌───────────────────┐  ┌───────────────────┐  ┌────────────────────────┐ │
   │  │  1.1 Dataset      │  │  1.2 Metadata     │  │  1.3 Mixed-Type        │ │
   │  │  Validator        │  │  Extractor        │  │  Feature Profiler      │ │
   │  │                   │  │                   │  │                        │ │
   │  │ • Duplicates      │  │ • n_samples       │  │ • Numerical            │ │
   │  │ • Constant feats  │  │ • n_features      │  │ • Categorical          │ │
   │  │ • Inf / NaN       │  │ • Target var      │  │ • Ordinal              │ │
   │  │ • Invalid types   │  │ • Missing %       │  │ • Binary / Boolean     │ │
   │  │ • Corrupt records │  │ • Data types      │  │ • Datetime             │ │
   │  │                   │  │ • Units / Scales  │  │ • Clinical Code        │ │
   │  │                   │  │ • File format     │  │ • Free Text            │ │
   │  └─────────┬─────────┘  └─────────┬─────────┘  └──────────┬─────────────┘ │
   │            └─────────────────────┬┘──────────────────────┘                │
   │                                  ▼                                          │
   │  ┌────────────────────────────────────────────────────────────────────┐    │
   │  │  1.4 Multi-resolution Missing Value Analyzer                       │    │
   │  │                                                                    │    │
   │  │  Global level: overall missing %                                   │    │
   │  │  Feature level: per-column missing pattern                         │    │
   │  │  Pair level: joint missing (detect structured missingness)         │    │
   │  │  MCAR test: Little's MCAR test (χ²)                               │    │
   │  │  MAR / MNAR: logistic regression on missingness indicator          │    │
   │  └────────────────────────────────────┬───────────────────────────────┘    │
   │                                        ▼                                    │
   │  ┌────────────────────────────────────────────────────────────────────┐    │
   │  │  1.5 Multi-method Outlier Detection Engine                         │    │
   │  │                                                                    │    │
   │  │  Ensemble: Z-score ∪ IQR ∪ Isolation Forest ∪ LOF ∪ ECOD        │    │
   │  │  Final outlier score = weighted vote across methods                │    │
   │  │  Clinical outlier vs. statistical outlier — distinguished          │    │
   │  └────────────────────────────────────┬───────────────────────────────┘    │
   │                                        ▼                                    │
   │  ┌────────────────────────────────────────────────────────────────────┐    │
   │  │  1.6 Sensitive Attribute Detector                                  │    │
   │  │                                                                    │    │
   │  │  Rule-based: Age / Gender / Ethnicity / ZIP / Diagnosis / SSN     │    │
   │  │  NLP-based: Clinical NER on column names + value samples          │    │
   │  │  k-Anonymity pre-check on raw data                                │    │
   │  │  Output: Sensitivity score sᵢ ∈ [0,1] for each feature           │    │
   │  └────────────────────────────────────┬───────────────────────────────┘    │
   │                                        ▼                                    │
   │  ┌────────────────────────────────────────────────────────────────────┐    │
   │  │  1.7 Dependency Estimation Engine                                  │    │
   │  │                                                                    │    │
   │  │  Continuous-Continuous: Pearson + Spearman + Distance Correlation │    │
   │  │  Continuous-Categorical: Point-biserial + ANOVA F                 │    │
   │  │  Categorical-Categorical: Cramér's V + Mutual Information         │    │
   │  │  All pairs: Conditional Entropy + Feature Interaction Score       │    │
   │  │  Temporal: Granger Causality (if longitudinal detected)           │    │
   │  └────────────────────────────────────┬───────────────────────────────┘    │
   │                                        ▼                                    │
   │  ┌────────────────────────────────────────────────────────────────────┐    │
   │  │  1.8 Dataset Complexity Analyzer                                   │    │
   │  │                                                                    │    │
   │  │  Class imbalance: IR = n_majority / n_minority                    │    │
   │  │  Feature redundancy: pairwise mutual info matrix                  │    │
   │  │  Intrinsic dimension: TwoNN estimator (Facco et al.)              │    │
   │  │  Noise level: label consistency under k-NN                        │    │
   │  │  Sample diversity: average pairwise distance in feature space     │    │
   │  └────────────────────────────────────┬───────────────────────────────┘    │
   │                                        ▼                                    │
   │  ┌────────────────────────────────────────────────────────────────────┐    │
   │  │  1.9 Dataset Intelligence Generator            ★ v5 REDESIGNED ★  │    │
   │  │                                                                    │    │
   │  │  NOT a raw MLP (v4 flaw: no training signal for ψ)                │    │
   │  │                                                                    │    │
   │  │  Instead: Dataset2Vec-style Meta-Encoder                          │    │
   │  │  • Pre-trained on 500+ diverse tabular datasets                   │    │
   │  │  • Permutation-invariant set encoder                              │    │
   │  │  • Input: set of (feature_stats, type, role) tuples              │    │
   │  │  • Output: I ∈ ℝ²⁵⁶ — universal dataset fingerprint             │    │
   │  │                                                                    │    │
   │  │  Also computes:                                                    │    │
   │  │  • Dataset Similarity Index — nearest known dataset in I-space   │    │
   │  │  • Transfer Learning Warm-start recommendation                    │    │
   │  └────────────────────────────────────────────────────────────────────┘    │
   └─────────────────────────────────────────────────────────────────────────────┘

Outputs

   ✓ Metadata Vector              M
   ✓ Feature Type Map
   ✓ Sensitivity Score Vector     S  (sᵢ ∈ [0,1] per feature)
   ✓ Dependency Matrix            R  (type-appropriate correlation)
   ✓ Complexity Vector            C
   ✓ Dataset Intelligence Vector  I  (Meta-encoder output, ℝ²⁵⁶)
   ✓ Dataset Similarity Index     (nearest known dataset)

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 2 : ADAPTIVE DATA PREPARATION ENGINE (ADPE) v5
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  2.1 Semantic-guided Adaptive Imputation      ★ v5 REDESIGNED ★     │  │
   │  │                                                                      │  │
   │  │  Step 1: Use clinical NER output to identify semantic role          │  │
   │  │  Step 2: Select strategy by (feature_type, missing_pattern, size)   │  │
   │  │                                                                      │  │
   │  │  MCAR + Numerical:           MICE / MissForest                      │  │
   │  │  MNAR + Clinical:            Clinical-rule-guided imputation         │  │
   │  │  MAR + Categorical:          Learned Embedding Imputation            │  │
   │  │  Extreme sparsity (>60%):    Conditional generation (mini-VAE)      │  │
   │  │                                                                      │  │
   │  │  ⚠ Uncertainty propagated: imputation confidence score per cell    │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  2.2 Distribution-preserving Feature Encoding ★ v5 REDESIGNED ★    │  │
   │  │                                                                      │  │
   │  │  Numerical:  Identity (preserve raw distribution)                   │  │
   │  │  Ordinal:    Ordinal encoding (preserve rank)                       │  │
   │  │  Categorical:                                                        │  │
   │  │    Low cardinality (≤10): One-hot                                   │  │
   │  │    Medium (11–50):        Learned embedding (dim = ⌈log₂(C)⌉)     │  │
   │  │    High (>50):            Hashing + Learned embedding               │  │
   │  │  Clinical codes: Ontology-aware embedding (ICD2Vec, Med2Vec)        │  │
   │  │  Free text:  Clinical BERT embedding (frozen)                       │  │
   │  │                                                                      │  │
   │  │  Key: encoding is INVERTIBLE — needed for decoder in Stage 6       │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────┐  ┌─────────────────────────┐                     │
   │  │  2.3 Adaptive        │  │  2.4 Clinical-aware     │                     │
   │  │  Feature Scaling     │  │  Outlier Treatment      │                     │
   │  │                      │  │                         │                     │
   │  │ Auto-selects by      │  │ Distinguish:            │                     │
   │  │ skewness + kurtosis: │  │ • Clinical outlier      │                     │
   │  │ • Robust (skewed)    │  │   → KEEP (real signal)  │                     │
   │  │ • Quantile (heavy    │  │ • Statistical outlier   │                     │
   │  │   tail)              │  │   → Winsorize           │                     │
   │  │ • Standard (normal)  │  │ • Data entry error      │                     │
   │  │ • MinMax (bounded)   │  │   → Impute              │                     │
   │  └──────────┬───────────┘  └──────────┬──────────────┘                     │
   │             └──────────────┬───────────┘                                   │
   │                            ▼                                                │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  2.5 Stratified Class Balancing                                      │  │
   │  │                                                                      │  │
   │  │  IR < 3:    No balancing needed                                      │  │
   │  │  3 ≤ IR < 10: SMOTE / ADASYN (oversample minority)                 │  │
   │  │  IR ≥ 10:   Borderline-SMOTE + Random Undersample (hybrid)          │  │
   │  │  Rare disease (n < 50): Warning — generative model may overfit      │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  2.6 Feature Consistency Validator                                   │  │
   │  │                                                                      │  │
   │  │  • Range constraints (Age ∈ [0,120], Glucose ∈ [20,600])           │  │
   │  │  • Type consistency (encoded match declared type)                   │  │
   │  │  • Distribution shift check (KS test pre/post preparation)          │  │
   │  └──────────────────────────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────────────────────┘

Outputs

   ✓ Prepared Feature Matrix    X
   ✓ Imputation Confidence Map  (per-cell uncertainty)
   ✓ Encoding Map               (invertible)
   ✓ Scaling Parameters
   ✓ Consistency Report

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 3 : DOMAIN CONSTRAINT GRAPH BUILDER (DCGB) v5
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  ★ v5: Multi-relational graph with 4 edge types                           │
   │                                                                             │
   │  ┌─────────────────────────────────────────────────────────────────────┐   │
   │  │  Clinical Ontology Subgraph  (if clinical dataset detected)         │   │
   │  │                                                                     │   │
   │  │  • SNOMED CT — disease-symptom, disease-drug edges                 │   │
   │  │  • ICD-10/11 — diagnostic hierarchy edges                          │   │
   │  │  • Drug interaction edges (from DrugBank)                          │   │
   │  │  • Lab-vital sign consistency edges (Harrison's Principles)        │   │
   │  └─────────────────────────────────────────────────────────────────────┘   │
   │                                                                             │
   │  ┌─────────────────────────────────────────────────────────────────────┐   │
   │  │  Statistical Subgraph  (always computed)                            │   │
   │  │                                                                     │   │
   │  │  • Correlation edges:  |ρ| > τ_corr                               │   │
   │  │  • Mutual Info edges:  I(X;Y) > τ_mi                              │   │
   │  │  • Causal edges:       PC algorithm (FCI for hidden confounders)   │   │
   │  └─────────────────────────────────────────────────────────────────────┘   │
   │                                                                             │
   │  ┌─────────────────────────────────────────────────────────────────────┐   │
   │  │  Privacy Subgraph  (from Stage 1 sensitive attributes)             │   │
   │  │                                                                     │   │
   │  │  • Quasi-identifier combination edges                              │   │
   │  │  • Attribute linkage risk edges                                    │   │
   │  └─────────────────────────────────────────────────────────────────────┘   │
   │                                                                             │
   │  ┌─────────────────────────────────────────────────────────────────────┐   │
   │  │  Temporal Subgraph  (if longitudinal data detected)                │   │
   │  │                                                                     │   │
   │  │  • Granger causality edges                                         │   │
   │  │  • Temporal precedence edges                                       │   │
   │  └─────────────────────────────────────────────────────────────────────┘   │
   │                                                                             │
   │                               ▼                                             │
   │               ┌─────────────────────────────────────┐                      │
   │               │  Graph Fusion + Sparsification      │                      │
   │               │                                     │                      │
   │               │  G₀ = (V, E_clinical ∪ E_stat      │                      │
   │               │           ∪ E_privacy ∪ E_temporal, │                      │
   │               │        W₀)                          │                      │
   │               │                                     │                      │
   │               │  Sparsify: keep top-k edges per     │                      │
   │               │  node (k=15) to prevent density     │                      │
   │               │  explosion                          │                      │
   │               └────────────────────────────────────-┘                      │
   └─────────────────────────────────────────────────────────────────────────────┘

Outputs

   ✓ Initial Constraint Graph    G₀ = (V, E, W₀)   (frozen structure)
   ✓ Learnable Edge Weights      W  (updated in Stage 3b)
   ✓ Graph Topology Report

                       ┌──────────────────────────────────────────────┐
                       │  NOTE: Graph refinement occurs in Stage 3b   │
                       │  AFTER Stage 4 learns the latent space       │
                       │  This fixes the v4 chicken-and-egg problem   │
                       └──────────────────────────────────────────────┘

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 4 : HIERARCHICAL CONSTRAINT-AWARE REPRESENTATION LEARNING (HCARLN) v5
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  Prepared Features X                   Initial Graph G₀                   │
   │           │                                    │                            │
   │           └────────────────┬───────────────────┘                           │
   │                            ▼                                                │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  4.1 Graph Attention Encoder (GAT)                                   │  │
   │  │                                                                      │  │
   │  │  Multi-head GAT: H = ATT(X, G₀)                                    │  │
   │  │  • Multi-head: K=8 heads, dim=64 each                               │  │
   │  │  • Edge-type-aware attention (4 edge types → 4 attention heads)     │  │
   │  │  • Residual connections between GAT layers                          │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  4.2 Gated Cross-Attention                ★ v5: replaces vanilla    │  │
   │  │                                              cross-attention         │  │
   │  │  Q = Feature representation                                          │  │
   │  │  K,V = Graph node embeddings (from GAT)                             │  │
   │  │                                                                      │  │
   │  │  Gate: g = σ(W_g [Q; K])  — learns WHEN to attend to graph        │  │
   │  │  Output: Ã = g ⊙ softmax(QKᵀ/√d) V                               │  │
   │  │                                                                      │  │
   │  │  Prevents graph from dominating when graph is noisy                │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  4.3 Hierarchical VAE (HVAE)              ★ v5: replaces VAE        │  │
   │  │                                                                      │  │
   │  │  Level 1 — Global latent z_g (disease-level, d=64):                │  │
   │  │    Captures: overall health state, disease class                    │  │
   │  │                                                                      │  │
   │  │  Level 2 — Feature-group latents z_k (d=32 each):                  │  │
   │  │    z_demo: demographic subspace                                     │  │
   │  │    z_lab:  laboratory values subspace                               │  │
   │  │    z_vital: vital signs subspace                                    │  │
   │  │    z_diag: diagnosis codes subspace                                 │  │
   │  │                                                                      │  │
   │  │  Level 3 — Feature-level residuals z_f (d=16 each)                 │  │
   │  │                                                                      │  │
   │  │  Joint latent: Z = [z_g ‖ z_demo ‖ z_lab ‖ z_vital ‖ z_diag]    │  │
   │  │                                                                      │  │
   │  │  ⚠ Disentangled by design — sensitive features in z_demo ONLY     │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  4.4 Contrastive Representation Regularization  ★ NEW IN v5        │  │
   │  │                                                                      │  │
   │  │  Pull together: same diagnosis / same risk group                    │  │
   │  │  Push apart: different diagnosis / different risk profile           │  │
   │  │  Use: supervised contrastive loss (Khosla et al. 2020)             │  │
   │  │                                                                      │  │
   │  │  Benefit: richer structure in latent space → better generation     │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │                                                                             │
   │  Representation Loss:                                                       │
   │  L_rep = L_recon + β_KL(t) × L_KL + λ_graph × L_graph + λ_cl × L_cl    │
   │                                                                             │
   │  β_KL(t): KL annealing (0→1 over first 20 epochs — prevents collapse)    │
   │  λ_graph, λ_cl: ADAPTIVE (updated by Stage 8 GradNorm)                   │
   │                                                                             │
   └─────────────────────────────────────────────────────────────────────────────┘

Outputs

   ✓ Graph-aware Hierarchical Latent Representation  Z = [z_g ‖ z_k]
   ✓ Disentangled Subspace Map
   ✓ Representation Quality Score

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 3b : GRAPH REFINEMENT                                          ★ NEW IN v5 ★
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  Inputs: Latent Z (from Stage 4) + Initial Graph G₀                       │
   │                                                                             │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  Latent-space Edge Weight Refinement                                 │  │
   │  │                                                                      │  │
   │  │  Refined weight: W_ij = α × W₀_ij + (1-α) × cos(z_i, z_j)        │  │
   │  │                                                                      │  │
   │  │  α learned — balances prior graph vs. learned similarity            │  │
   │  │  Prune edges: remove if W_ij < τ (sparsification)                  │  │
   │  └──────────────────────────────────────────────────────────────────────┘  │
   │                                                                             │
   │  Result: Refined Graph  G = (V, E, W)   ← FIXED from this point on       │
   │  Graph structure frozen — only W updated via small learning rate           │
   │                                                                             │
   └─────────────────────────────────────────────────────────────────────────────┘

   ✓ Refined Constraint Graph  G = (V, E, W)   (used by all remaining stages)

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 5 : ADAPTIVE PRIVACY CONTROLLER (APC) v5   ★ FORMALLY CORRECT DP ★
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  ⚠ v4 FLAW FIXED: Learning ε itself breaks formal DP guarantees.          │
   │    v5: ε* is FIXED by policy. We learn how to ALLOCATE it.                │
   │                                                                             │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  5.1 Policy-fixed Privacy Budget                                     │  │
   │  │                                                                      │  │
   │  │  ε* = user-specified (e.g., 1.0 for HIPAA-level, 0.1 for GDPR)    │  │
   │  │  δ  = user-specified (typically 1/n²)                               │  │
   │  │                                                                      │  │
   │  │  THIS IS FIXED. NEVER LEARNED. NEVER CHANGED.                       │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  5.2 PRV Accountant (Privacy Random Variable Accountant)            │  │
   │  │                                                                      │  │
   │  │  Tracks precise DP composition across all training steps            │  │
   │  │  More accurate than moments accountant (Mironov 2017)              │  │
   │  │  Real-time budget consumed: ε_used(t) ≤ ε*                        │  │
   │  │  Alert if ε_used approaching ε* → reduce noise, stop early        │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  5.3 Adaptive Per-feature Budget Allocation   ★ v5 INNOVATION       │  │
   │  │                                                                      │  │
   │  │  Learn:  εᵢ = f_θ(sᵢ, I)    subject to  Σᵢ εᵢ ≤ ε*              │  │
   │  │                                                                      │  │
   │  │  sᵢ = sensitivity score of feature i (from Stage 1)                │  │
   │  │  I  = dataset intelligence vector                                   │  │
   │  │                                                                      │  │
   │  │  High sᵢ → small εᵢ (more noise, stronger protection)             │  │
   │  │  Low sᵢ  → large εᵢ (less noise, preserve utility)               │  │
   │  │                                                                      │  │
   │  │  Constraint enforced via softmax: εᵢ = ε* × softmax(f_θ(sᵢ, I)) │  │
   │  │  Formally correct — global budget ε* is never violated             │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  5.4 Sensitivity Estimator                                           │  │
   │  │                                                                      │  │
   │  │  Per-feature clip norm Cᵢ (learned):                                │  │
   │  │    Cᵢ = percentile_p(‖∂L/∂xᵢ‖)  p=75th → adaptive clip           │  │
   │  │                                                                      │  │
   │  │  Global sensitivity:  Δfᵢ = Cᵢ (by clipping mechanism)            │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  5.5 Calibrated Noise Injection                                      │  │
   │  │                                                                      │  │
   │  │  Noise scale (closed form, formally correct):                       │  │
   │  │    σᵢ = Cᵢ × √(2 ln(1.25/δ)) / εᵢ                                │  │
   │  │                                                                      │  │
   │  │  Mechanism: Gaussian(0, σᵢ²) per feature gradient                  │  │
   │  │                                                                      │  │
   │  │  Privacy Warm-up: anneal ε_effective: ε_warm → ε*                  │  │
   │  │  Epoch 0:   ε_eff = 10×ε*  (nearly no noise — learn structure)    │  │
   │  │  Epoch 1–T: ε_eff linearly → ε*  (gradually enforce privacy)      │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  5.6 Privacy State Generator                                         │  │
   │  │                                                                      │  │
   │  │  P = MLP([ε*, ε_used, σ_vector, C_vector, s_vector])               │  │
   │  │                                                                      │  │
   │  │  P encodes current privacy state — passed to Stage 6               │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  5.7 Privacy Quality Evaluator                                       │  │
   │  │                                                                      │  │
   │  │  • Membership inference attack (shadow model)                       │  │
   │  │  • Attribute inference risk (Yeom et al.)                          │  │
   │  │  • k-anonymity, l-diversity, t-closeness                           │  │
   │  │  • Privacy–utility Pareto point                                     │  │
   │  └──────────────────────────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────────────────────┘

Outputs

   ✓ Fixed Privacy Budget        ε*  (user-specified, never modified)
   ✓ Per-feature Budget          εᵢ  (learned allocation, Σεᵢ ≤ ε*)
   ✓ Per-feature Clip Norm       Cᵢ  (learned)
   ✓ Noise Scale Vector          σᵢ  (closed-form, formally correct)
   ✓ Privacy State Vector        P
   ✓ Budget Tracker              ε_used(t)
   ✓ Privacy Quality Score

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 6 : UNIFIED ADAPTIVE CONSTRAINT-GUIDED FLOW MATCHING GENERATOR (UACFMG) v5
                              ★ Core Scientific Contribution ★
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  Z (Latent)  G (Refined Graph)  M (Metadata)  P (Privacy)  I (Intel)      │
   │      │              │                │               │           │          │
   │      └──────────────┴────────────────┴───────────────┴───────────┘         │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  6.1 Gated Multi-Source Fusion Network   ★ v5: replaces linear F   │  │
   │  │                                                                      │  │
   │  │  Each source projected to same dim d:                               │  │
   │  │    h_M = MLP(M),  h_G = GNN(G),  h_P = MLP(P),  h_Z = MLP(Z)    │  │
   │  │                                                                      │  │
   │  │  Cross-source attention (each source attends to others):           │  │
   │  │    A = MultiHeadAttn(Q=[h_M,h_G,h_P,h_Z],                        │  │
   │  │                       K=[h_M,h_G,h_P,h_Z],                        │  │
   │  │                       V=[h_M,h_G,h_P,h_Z])                        │  │
   │  │                                                                      │  │
   │  │  Gating: F = g ⊙ A + (1-g) ⊙ h_Z     g = σ(MLP([A; h_Z]))      │  │
   │  │                                                                      │  │
   │  │  Gate ensures latent Z remains primary signal                      │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  6.2 Type-split Flow Architecture        ★ v5 MAJOR REDESIGN ★     │  │
   │  │                                                                      │  │
   │  │  ⚠ v4 flaw: single flow model for all types — wrong for discrete   │  │
   │  │                                                                      │  │
   │  │  CONTINUOUS branch (numerical + ordinal):                          │  │
   │  │  ┌─────────────────────────────────────────────────────────────┐   │  │
   │  │  │  Conditional Flow Matching (Lipman et al. 2022)             │   │  │
   │  │  │                                                             │   │  │
   │  │  │  Optimal Transport path: p_t = (1-t)p_0 + t p_data        │   │  │
   │  │  │                                                             │   │  │
   │  │  │  Velocity field:                                            │   │  │
   │  │  │  dx_c/dt = v_θ(x_c, t, F, G, P)                          │   │  │
   │  │  │                                                             │   │  │
   │  │  │  Architecture:                                              │   │  │
   │  │  │    Time Embedding (sinusoidal)                             │   │  │
   │  │  │         ▼                                                  │   │  │
   │  │  │    Condition Injection via AdaLN (F, G, P)                │   │  │
   │  │  │         ▼                                                  │   │  │
   │  │  │    U-Net + Residual Flow Blocks × L                       │   │  │
   │  │  │         ▼                                                  │   │  │
   │  │  │    Constraint Projection Layer                            │   │  │
   │  │  │    (projects velocity onto constraint manifold)           │   │  │
   │  │  │         ▼                                                  │   │  │
   │  │  │    Velocity v_θ for continuous features                   │   │  │
   │  │  └─────────────────────────────────────────────────────────────┘   │  │
   │  │                                                                      │  │
   │  │  DISCRETE branch (categorical + binary):                            │  │
   │  │  ┌─────────────────────────────────────────────────────────────┐   │  │
   │  │  │  Masked Discrete Diffusion (Austin et al. 2021 / MDLM)     │   │  │
   │  │  │                                                             │   │  │
   │  │  │  Categorical: q(x_t | x_0) = Cat(αt x_0 + (1-αt) m)     │   │  │
   │  │  │  m = mask token                                            │   │  │
   │  │  │                                                             │   │  │
   │  │  │  Denoising network:  p_θ(x_0 | x_t, t, F, G, P)         │   │  │
   │  │  │  Conditioned on same F, G, P as continuous branch         │   │  │
   │  │  └─────────────────────────────────────────────────────────────┘   │  │
   │  │                                                                      │  │
   │  │  Joint ODE Solver: Dormand-Prince (RK45) — adaptive step size      │  │
   │  │  (replaces v4's vague "Adaptive Flow Scheduler")                   │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  6.3 Adaptive Differential Privacy Layer                             │  │
   │  │                                                                      │  │
   │  │  Per-feature noise injection using σᵢ from Stage 5:                │  │
   │  │    x̃_i = x_i + N(0, σᵢ²)                                         │  │
   │  │                                                                      │  │
   │  │  Applied to GRADIENTS during training (DP-SGD framework)           │  │
   │  │  NOT to raw features (avoids utility collapse)                     │  │
   │  │                                                                      │  │
   │  │  Budget tracked by PRV Accountant in Stage 5                       │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  6.4 Mixed-type Sample Decoder                                       │  │
   │  │                                                                      │  │
   │  │  Shared Trunk (MLP)                                                 │  │
   │  │       │                                                              │  │
   │  │  ┌────┴────┬─────────┬────────────┬──────────────┐                 │  │
   │  │  ▼         ▼         ▼            ▼              ▼                  │  │
   │  │  Num      Cat        Bin          Ord            Code               │  │
   │  │  Head     Head       Head         Head           Head               │  │
   │  │  (Linear)(Softmax) (Sigmoid)  (Ordinal CDF)  (Ontology-aware)      │  │
   │  │  + temp   + temp                                                    │  │
   │  │  scale    scale                                                      │  │
   │  │                                                                      │  │
   │  │  Temperature τ annealed: high τ → diversity, low τ → fidelity     │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  6.5 Unified Distribution Alignment                                  │  │
   │  │                                                                      │  │
   │  │  Training loss (differentiable only):                               │  │
   │  │                                                                      │  │
   │  │  L_dist = λ₁ L_MMD                                                 │  │
   │  │         + λ₂ L_Wasserstein  (Sinkhorn, ε-regularized)              │  │
   │  │         + λ₃ L_SlicedWasserstein   ← replaces KS (non-diff!)      │  │
   │  │         + λ₄ L_FeatureWise                                         │  │
   │  │                                                                      │  │
   │  │  KS test: monitoring ONLY (not in training loss)                   │  │
   │  └──────────────────────────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────────────────────┘

Outputs

   ✓ Unified Representation         F
   ✓ Continuous Samples             (via CFM)
   ✓ Discrete Samples               (via MDLM)
   ✓ DP-guaranteed Samples
   ✓ Synthetic Candidate Dataset    Ŷ
   ✓ Flow Quality Score

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 7 : MULTI-CONSTRAINT VALIDATION ENGINE (MCVE) v5
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  ★ v5: Progressive validation (cheap → expensive) + explicit Repair Loop  │
   │                                                                             │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  FAST CHECKS FIRST (cheap, deterministic)                           │  │
   │  │                                                                      │  │
   │  │  7.1 Hard Constraint Validator                                      │  │
   │  │  • Negative Age / Invalid Dates / Impossible Lab Values             │  │
   │  │  • Duplicate Records / Range Violations / Type Errors              │  │
   │  │  • FAIL → immediate REJECT (no repair — deterministic violation)   │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  MEDIUM CHECKS (statistical, differentiable)                        │  │
   │  │                                                                      │  │
   │  │  7.2 Clinical Constraint Validator                                  │  │
   │  │  • Disease-drug compatibility, lab-vital consistency                │  │
   │  │  • Clinical pathway rules                                           │  │
   │  │                                                                      │  │
   │  │  7.3 Statistical Distribution Validator                             │  │
   │  │  • KS, Wasserstein, MMD, Jensen-Shannon, Correlation matrix        │  │
   │  │                                                                      │  │
   │  │  7.4 Fairness Validator                                             │  │
   │  │  • Demographic parity, equalized odds, disparate impact            │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  EXPENSIVE CHECKS (attack simulation)                               │  │
   │  │                                                                      │  │
   │  │  7.5 Privacy Constraint Validator                                   │  │
   │  │  • MIA (shadow model), attribute inference, k-anon, l-div          │  │
   │  │                                                                      │  │
   │  │  7.6 Explainability Consistency Validator                           │  │
   │  │  • SHAP similarity (real vs. synthetic), feature importance rank   │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  7.7 Confidence Scorer                     ★ NEW IN v5              │  │
   │  │                                                                      │  │
   │  │  Per-record confidence: conf(ŷ) = harmonic_mean(all_val_scores)   │  │
   │  │  Records below conf_threshold → REPAIR queue                       │  │
   │  │  Records above → ACCEPT                                            │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  7.8 Validation Decision + EXPLICIT REPAIR LOOP  ★ v5 REDESIGNED   │  │
   │  │                                                                      │  │
   │  │        ┌────────────────────────────────┐                           │  │
   │  │        │  Hard fail?     →  REJECT       │                           │  │
   │  │        │  conf(ŷ) ≥ τ   →  ACCEPT       │                           │  │
   │  │        │  conf(ŷ) < τ   →  REPAIR       │                           │  │
   │  │        └────────────────────────────────┘                           │  │
   │  │                                                                      │  │
   │  │  REPAIR (explicit — v4 never defined this):                         │  │
   │  │    Given: failed record ŷ_bad, violation set V_fail                │  │
   │  │    Decode ŷ_bad back to latent z_bad                               │  │
   │  │    Constrained optimization:                                        │  │
   │  │      z* = argmin_z ‖z - z_bad‖² + λ_c × L_constraint(z, V_fail)  │  │
   │  │    Re-decode z* → ŷ_repaired                                       │  │
   │  │    Max repair attempts = 3 (then REJECT)                           │  │
   │  │                                                                      │  │
   │  │  REPAIR FEEDBACK: constraint violation signal fed back to Stage 8  │  │
   │  └──────────────────────────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────────────────────┘

                       │                │               │
                       ▼                ▼               ▼
                    Accept            Repair          Reject
                       │                │
                       └────────────────┘
                                │
                                ▼
                      Validated Synthetic Dataset

   Feedback arrow: Repair failures → Stage 8 constraint loss signal
   ◄─────────────────────────────────────────────────────────────────────

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 8 : PARETO-OPTIMAL MULTI-OBJECTIVE OPTIMIZATION ENGINE (PMOE) v5
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  ★ v5: 3-phase curriculum + PCGrad gradient surgery + GradNorm weights    │
   │                                                                             │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  PHASE 1: Warm-up  (Epochs 0 → T₁)                                  │  │
   │  │                                                                      │  │
   │  │  Train ONLY:  L_flow + L_reconstruction                             │  │
   │  │  Privacy:     OFF (ε_eff = ∞)                                       │  │
   │  │  Constraints: OFF                                                    │  │
   │  │  Goal:        Learn basic data distribution first                   │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  PHASE 2: Curriculum Constraint Introduction  (Epochs T₁ → T₂)     │  │
   │  │                                                                      │  │
   │  │  Add constraints progressively:                                     │  │
   │  │  Week 1: + L_graph  (structural)                                    │  │
   │  │  Week 2: + L_clinical  (medical)                                    │  │
   │  │  Week 3: + L_fairness  (demographic)                                │  │
   │  │  Week 4: + L_privacy   (DP) ε_eff: ε_warm → ε*                   │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  PHASE 3: Full Joint Optimization  (Epochs T₂ → convergence)       │  │
   │  │                                                                      │  │
   │  │  Composite loss:                                                     │  │
   │  │  L_total = L_flow + λ₁L_privacy + λ₂L_clinical + λ₃L_graph       │  │
   │  │                   + λ₄L_fairness + λ₅L_distrib  + λ₆L_cl          │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  8.1 GradNorm Adaptive Weight Learning    ★ v5: replaces fixed λ   │  │
   │  │                                                                      │  │
   │  │  λᵢ(t) updated to normalize gradient magnitudes:                   │  │
   │  │    λᵢ ← λᵢ × (G_target / G_i)   where G_i = ‖∇_W Lᵢ‖           │  │
   │  │                                                                      │  │
   │  │  Prevents any single loss from dominating gradients                │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  8.2 PCGrad Gradient Surgery      ★ NEW IN v5                       │  │
   │  │                                                                      │  │
   │  │  For each pair of conflicting gradients (gᵢ · gⱼ < 0):            │  │
   │  │    Project gᵢ onto plane perpendicular to gⱼ                       │  │
   │  │    g̃ᵢ = gᵢ - (gᵢ · gⱼ / ‖gⱼ‖²) gⱼ                            │  │
   │  │                                                                      │  │
   │  │  Resolves privacy vs. utility, fairness vs. accuracy conflicts      │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  8.3 Joint Optimizer                                                 │  │
   │  │                                                                      │  │
   │  │  Algorithm: Lion (Chen et al. 2023)                                 │  │
   │  │  • Memory efficient (no second moment)                              │  │
   │  │  • Better generalization than AdamW on generative models           │  │
   │  │                                                                      │  │
   │  │  LR Schedule:                                                       │  │
   │  │  • Phase 1: Linear warmup (0 → lr_max over 1000 steps)            │  │
   │  │  • Phase 2+3: Cosine annealing (lr_max → lr_min)                  │  │
   │  │                                                                      │  │
   │  │  Components updated:                                                │  │
   │  │  ✓ HVAE Encoder/Decoder   ✓ GAT Graph Encoder                     │  │
   │  │  ✓ Continuous Flow Net    ✓ Discrete Flow Net                      │  │
   │  │  ✓ Gated Fusion Net       ✓ Mixed-type Decoder                     │  │
   │  │  ✓ Privacy Allocator      ✓ GradNorm weights                       │  │
   │  │  ✗ Graph structure G      (frozen — only W edge weights updated)   │  │
   │  └──────────────────────────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────────────────────┘

Outputs

   ✓ Optimized Network (all components)
   ✓ Adaptive Loss Weights λ (GradNorm)
   ✓ Loss History (per-component)
   ✓ Pareto Front snapshot (utility vs. privacy)

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════
STAGE 9 : PERFORMANCE & CONVERGENCE CONTROLLER (PCC) v5
══════════════════════════════════════════════════════════════════════════════════════

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
   │  │ 9.1 Training │ │ 9.2 Utility  │ │ 9.3 Privacy  │ │ 9.4 Clinical     │ │
   │  │ Monitor      │ │ Evaluator    │ │ Evaluator    │ │ Fidelity         │ │
   │  │              │ │              │ │              │ │                  │ │
   │  │ • L_train    │ │ • TSTR*      │ │ • MIA AUC   │ │ • Clinical rule  │ │
   │  │ • L_val      │ │ • TRTS*      │ │ • k-anon    │ │   satisfaction   │ │
   │  │ • Grad norm  │ │ • Classif. F1│ │ • ε_used    │ │ • Lab range %    │ │
   │  │ • GPU mem    │ │ • Regr. RMSE │ │ • Discl. risk│ │ • Diagnosis cons.│ │
   │  │ • LR(t)      │ │              │ │              │ │                  │ │
   │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘ │
   │                                                                             │
   │  * TSTR = Train on Synthetic, Test on Real                                │
   │  * TRTS = Train on Real, Test on Synthetic                                │
   │                                                                             │
   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────────────┐  │
   │  │ 9.5 Stat.    │ │ 9.6 Fairness │ │ 9.7 Convergence Controller       │  │
   │  │ Fidelity     │ │ Evaluator    │ │                       ★ v5       │  │
   │  │              │ │              │ │                                  │  │
   │  │ • αW dist.   │ │ • Dem.parity │ │  Normalized criterion:           │  │
   │  │ • KS test    │ │ • Eq. odds   │ │                                  │  │
   │  │ • Corr matrix│ │ • Disp.impact│ │  C = Σᵢ ωᵢ × ΔXᵢ/σ(ΔXᵢ)     │  │
   │  │ • FID-equiv. │ │ • Calibration│ │                                  │  │
   │  │   (tabular)  │ │              │ │  All ΔX normalized (prevents     │  │
   │  └──────────────┘ └──────────────┘ │  scale dominance — v4 flaw)     │  │
   │                                    │                                  │  │
   │                                    │  Stop when C < τ                │  │
   │                                    │  for K=10 consecutive epochs    │  │
   │                                    │  (patience mechanism)           │  │
   │                                    └──────────────┬───────────────────┘  │
   │                                                   │                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  9.8 Multi-objective Pareto Monitor   ★ NEW IN v5                   │  │
   │  │                                                                      │  │
   │  │  Tracks Pareto front: utility U vs. privacy P_score                │  │
   │  │  Best model = checkpoint maximizing U subject to P_score ≥ τ_P    │  │
   │  │  (not just lowest loss — selects on validation Pareto optimality)  │  │
   │  └──────────────────────────────────┬───────────────────────────────────┘  │
   │                                     ▼                                       │
   │  ┌──────────────────────────────────────────────────────────────────────┐  │
   │  │  9.9 Complexity Reporter                                             │  │
   │  │                                                                      │  │
   │  │  • Parameters: total + per-module                                   │  │
   │  │  • FLOPs per generation step                                        │  │
   │  │  • Inference latency (ms / sample)                                  │  │
   │  │  • Memory (GB training / GB inference)                              │  │
   │  └──────────────────────────────────────────────────────────────────────┘  │
   │                                     ▼                                       │
   │                       ┌────────────────────────┐                           │
   │                       │                        │                           │
   │                  Continue Training       Stop Training                     │
   │                       │                        │                           │
   │                       └───────────┬────────────┘                           │
   │                                   ▼                                        │
   │                      Best Pareto Checkpoint Selected                       │
   └─────────────────────────────────────────────────────────────────────────────┘

Outputs

   ✓ Convergence Status
   ✓ Best Model (Pareto-optimal checkpoint)
   ✓ Training Report
   ✓ Performance Report
   ✓ Complexity Report (params, FLOPs, latency)

                                               │
                                               ▼

══════════════════════════════════════════════════════════════════════════════════════

FINAL OUTPUT — ACFM-Net v5

   ✓ Privacy-Preserving      (Formal (ε*, δ)-DP · k-Anon · l-Div · t-Close)
   ✓ Clinically Consistent   (SNOMED CT · ICD · 4-type Constraint Graph)
   ✓ Statistically Faithful  (MMD · Wasserstein · Sliced Wasserstein · Corr)
   ✓ Fair                    (Demographic Parity · Equalized Odds · Calib.)
   ✓ Explainable             (SHAP · Integrated Gradients · Feature Rank)
   ✓ Governed                (IRB · HIPAA · GDPR · Audit Trail)
   ✓ Pareto-optimal          (Best utility at guaranteed privacy level)
   ✓ AI-Ready Synthetic Healthcare Dataset

══════════════════════════════════════════════════════════════════════════════════════
```

---

## Key Innovations — v5 vs v4

```
┌─────────────────────────────────┬────────────────────────────────────────────────┐
│ Component                       │ v5 Contribution                                │
├─────────────────────────────────┼────────────────────────────────────────────────┤
│ Stage 0 (NEW)                   │ Clinical governance gate before data access    │
│ Stage 1: Intelligence Vector    │ Dataset2Vec meta-encoder (pre-trained)        │
│ Stage 3: Graph                  │ 4 edge types + sparsification                 │
│ Stage 3b (NEW)                  │ Graph refinement after latent learning         │
│ Stage 4: Representation         │ HVAE + Gated CrossAttn + Contrastive Reg       │
│ Stage 5: Privacy                │ FORMALLY CORRECT DP (fixed ε*, PRV accountant) │
│ Stage 6: Fusion                 │ Gated Multi-Source Attention (not linear)      │
│ Stage 6: Flow                   │ CFM (continuous) + MDLM (discrete) — split    │
│ Stage 6: Loss                   │ KS removed; Sliced Wasserstein added           │
│ Stage 7: Repair                 │ Explicit constrained gradient descent loop     │
│ Stage 7: Validation             │ Progressive (cheap → expensive)               │
│ Stage 8: Optimization           │ 3-phase curriculum + PCGrad + GradNorm + Lion  │
│ Stage 9: Convergence            │ Normalized C + patience + Pareto checkpoint    │
└─────────────────────────────────┴────────────────────────────────────────────────┘
```
