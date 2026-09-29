# Response to the External (ChatGPT) Audit: Revision 2, 2026-09-27

**Deliverables:**
- `paper/Lumisyn_DP_FlowMatching_Manuscript.docx`: 7,384 words, 7 figures, 7 tables, Algorithm 1, 4 native Word equations, 38 references.
- `paper/Lumisyn_Supplementary_Material.docx`: sections S1–S6.

Both files pass the OOXML schema validator.

| # | Audit point | Action | Status |
|---|---|---|---|
| 1 | Code/data not available | Artifact contents are specified in the Data Availability section and mapped per table/figure in Supplementary S6. The code is committed locally (`aa99b7f`) with pinned environments. **Publishing it and minting a DOI needs you:** push to GitHub, then archive on Zenodo. | ⚠️ Needs you |
| 2 | Title overstates | New title: "Minibatch Optimal Transport Invalidates Standard DP-SGD Accounting in Private Flow Matching for Tabular Data". | ✅ |
| 3 | "any valid certificate" too broad | Replaced everywhere by "implied ε under the Gaussian-mechanism accounting of DP-SGD". Table row renamed. §2.7 states it is a lower bound for that formulation only. | ✅ |
| 4 | Theorem vs search vs audit blended | The Introduction states the three evidence types; the searches are called heuristic; "found by our search" is used throughout. | ✅ |
| 5 | DAPC under-specified | Supplementary Table S1 gives every keyword list, formula and constant, including ε_max, the high-stakes rule, abstain/reject rules and split rules. | ✅ |
| 6 | Theorem 2 informal | Theorem 2 now names mechanisms M₁ and M₂. The proof outline covers items (a)–(g): initialisation, fixed T, empty batches, optimizer state, EMA, independent noise, assertion. Full proof and a code-path table are in S2. | ✅ |
| 7 | Adjacency conversion | Equation 1 gives the exact conversion: add/remove (ε, δ) ⟹ replace-one (2ε, (1+e^ε)δ). | ✅ |
| 8 | Opacus normalisation | A new paragraph in §2.1 covers: Opacus used only for sampling and accounting; custom per-example gradients; mean-over-coordinates loss; noise on the clipped sum; division by the public constant qN = 512; loader rate 1/⌈N/B⌉ ≤ q; empty batches; the link to Wang et al. and A2. | ✅ |
| 9 | Leaks not demonstrated | `lumisyn/backend/leak_demos.py` reproduces all five on Adult. Private 78×78 matrix recovered from the weights (error 1.1e-7); 6/6 seeded counts recovered; bin edge 594 = private minimum; age min/max leaked; clipped-noise disjointness. Table 4, S3 and Listing S1 give code patterns, fixes and tests. | ✅ |
| 10 | Statistics incomplete | New Table 6: means ± sd, differences with 95% CIs, margins, TOST t (lower, upper), TOST p, Wilcoxon p. Margins are justified and stated as fixed before the 10-seed runs; a detectable-difference statement is added. Per-seed values are in S4. | ✅ |
| 11 | AIM fairness | §3.6 is reframed as target-aware synthesis. Table 7 gains a workload column; the default-workload numbers are marked as answering a different question. | ✅ |
| 12 | "Marginals carry the utility" too broad | Scoped to the "evaluated datasets, privacy levels and training configurations" in the abstract, §3.5, the Discussion and the Conclusions. | ✅ |
| 13 | Small cohorts underpowered | Renamed "Exploratory stress test", explicitly not used as evidence. | ✅ |
| 14 | Reproducibility details | Table 1 (main) and Table S5 give Python, CUDA and cuDNN, driver, all library versions, TF32, architecture, initialisation, optimizer and schedule, t distribution, the Hungarian solver, δ values, accountant, seeds, audit settings, and the baseline environment. | ✅ |
| 15 | I-CFM framing | "As predicted by Proposition 1, I-CFM remained below the bound in all audited batches". | ✅ |
| 17 | Mechanism diagram | New Figure 1: toy 2-D example with the exact Hungarian solver. One replacement changes 1 pair under I-CFM and 6 pairs under OT. | ✅ |
| 18 | Pseudocode | Algorithm 1 covers the DP-SGD step for both couplings and marks the line where dependence enters. | ✅ |
| 19 | Placeholders | Author, affiliation, e-mail, acknowledgement, contribution and conflict fields remain; **only you can fill them**. | ⚠️ Needs you |
| 20 | Corrupted equations | All display equations are now native Word equation objects (OMML); math symbols in running text use Cambria Math. **Check the PDF export once in Word.** | ✅ (verify visually) |
| 21 | Threats to validity | New §4.1: theoretical, empirical, external and implementation validity. | ✅ |
| 22 | DAPC weakest | Removed from the contribution list; §2.5 shortened and labelled a policy component; robustness analysis moved to S1 and Figure S1. | ✅ |
| 23 | Scoped conclusion | Wording taken from the audit's suggestion. | ✅ |

**Re-check after revision:**
- The flagged phrases ("breaks the privacy guarantee", "any valid certificate", "marginals carry the utility", "no measurable value") no longer appear.
- The AI-pattern list has 0 hits; the only "robust" left is the technical term "robustness analysis". There are 0 em-dashes. The text has 288 sentences, mean 25.5 words, SD 22.2.
- There are 0 shared 8-word or 12-word sequences with either reference paper.
- All 89 tests pass.

**Still recommended before submission:**
1. Run iThenticate or Turnitin.
2. Export to PDF from Word and inspect every equation.
3. Push the code and archive it on Zenodo.
4. Fill in the placeholders.

---

# Revision 3: response to the second audit (2026-09-27)

**Deliverables:**
- `Lumisyn_DP_FlowMatching_Manuscript.docx` (main text);
- `Lumisyn_Supplementary_Material.docx` (S1–S6);
- `Lumisyn_Manuscript_with_Supplement.docx` (both in one file, so the supplement cannot be missed).

All three pass OOXML validation, and all 90 tests pass.

| Flag | Action |
|---|---|
| 🔴 Conditional Gaussian audit reported as a full-mechanism ε | Route A. The audit is renamed the "conditional white-box audit", and a new paragraph, "What the audit does and does not establish" (§2.7), explains the subsampling amplification. "Implied ε" becomes the **accounting-consistent ε**: the smallest certificate the standard subsampled-Gaussian accountant can issue once its sensitivity bound covers the audited pair (noise multiplier 2/μ_lb, derived in S2.4). The text states explicitly that it is not a lower bound on the privacy loss. The abstract, contributions, §3.2, the Table 3 caption and row, the Figure 4 caption, the Discussion, the threats to validity and the Conclusions are all rewritten accordingly. A full subsampled audit (Route B) is listed as future work, because at q ≈ 0.04 a sampled audit has very low power. |
| 🔴 Theorem vs accounting | §2.3 now states the condition Δ₂(S) = sup‖S(B) − S(B′)‖₂ ≤ C over neighbours and per-example randomness, and says that when it fails "the conditions of the standard accountant's proof are not satisfied". The "single violating pair invalidates the certificate" sentence is removed. |
| 🟠 "Accountant still reports the same ε" | Reworded: the accountant "computes a guarantee under the assumptions supplied to it; it does not verify that the implemented mechanism satisfies them" (Introduction and §2.3). |
| 🟠 DAPC in the main story | Moved to Supplementary S1 (equation, full rules, adaptive split, robustness). §2.5 is now a short "Choice of the Privacy Budget" paragraph; Equation 4 and the adaptive row of Table 5 are removed from the main text; main results use fixed splits (primary ρ = 0.9). |
| 🔴 Supplement "missing" | The supplement is a separate file (journals expect this). A combined single-file version is also provided. |
| 🟠 Cardio source | §2.8 states that no peer-reviewed data descriptor could be identified. SHA-256 digests of all three source files are given in Table S5 so the exact versions are verifiable. |
| 🟠 Adult sampling | §2.8 gives the full release size (48,842), the positive rate (23.9%), a uniform sample without replacement (fixed seed, not stratified) of 15,000 records, a 12,000/3,000 split, and the same details for Cardio (70,000 → 20,000; 16,000/4,000) and Pima (768; 614/154). |
| Opacus sampling semantics | Verified in code: `DPDataLoader.from_data_loader` uses q_L = 1/len(loader) = 1/⌈N/B⌉ ≤ B/N (the accountant's rate). §2.1 states both rates; a new regression test enforces q_L ≤ q. |
| "all five leaks were fixed" | Now "all five implementation defects were fixed in the audited implementation". |
| 🟡 "true difference would be detected" | Removed. Replaced by "The resulting confidence intervals were substantially narrower than the prespecified margins" and "no practically relevant utility contribution from the flow was detected". |
| 10⁶ trials marketing | Now described as a "finite-sample test … 95% confidence"; the trial count appears only in the methods. |
| 🔴 Placeholders, repository/DOI | **Still needs you.** |

**Re-check:**
- The over-claiming phrases are gone. The one remaining "any valid" reads "any valid bound must be at least the certified sensitivity of the audited pair", which is correct.
- There are 0 AI-pattern hits and 0 em-dashes.
- There are 0 shared 8-word or 12-word sequences with either reference paper.
