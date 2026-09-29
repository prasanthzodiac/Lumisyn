# Revision 3 edits (Route A for the conditional-audit issue, DAPC moved out, dataset provenance, wording).
p = 'content.js'
s = open(p, encoding='utf-8').read()


def rep(a, b):
    global s
    assert s.count(a) == 1, "NOT FOUND: " + a[:90]
    s = s.replace(a, b)


def cut(start, end, new):
    global s
    i, j = s.index(start), s.index(end)
    s = s[:i] + new + s[j:]


# abstract
rep("On a 20,000-record cardiovascular cohort our search found violations in 69% of audited batches, and a white-box audit certifies, at 95% confidence, that under the Gaussian-mechanism accounting used by DP-SGD the run implies ε ≥ 1.47 where 0.84 was reported.",
    "On a 20,000-record cardiovascular cohort, our search found violations in 69% of audited batches, and a finite-sample white-box audit of the corresponding conditional Gaussian step certified, at 95% confidence, that its sensitivity exceeds the bound assumed by standard DP-SGD accounting. Within that accounting framework, no valid certificate for the run can be smaller than ε = 1.47, whereas 0.84 was reported; this constrains the accounting and is not a lower bound on the privacy loss of the subsampled mechanism.")
rep("Checking that proof against the code exposed five further leak classes, each of which we reproduce on real data.",
    "Checking that proof against the code exposed five further classes of implementation defects, each of which we reproduce on real data.")
rep("Under the evaluated datasets, privacy levels and training configurations, removing the flow left every metric statistically equivalent (10 paired seeds, two one-sided tests, p < 0.001): the marginal stage accounted for essentially all measured utility.",
    "Under the evaluated datasets, privacy levels and training configurations, no practically relevant utility contribution from the flow was detected under prespecified equivalence margins (10 paired seeds, two one-sided tests, p < 0.001): the marginal stage accounted for essentially all measured utility.")
# introduction
rep("The accountant cannot check the invariant itself. It reads only the noise multiplier, the sampling rate and the number of steps, so a training loss that silently couples examples leaves the reported ε unchanged even though the analysis behind it no longer applies.",
    "The accountant computes a guarantee from the noise multiplier, the sampling rate and the number of steps under the assumptions supplied to it; it does not verify that the implemented mechanism satisfies them. A training loss that silently couples examples therefore leaves the reported ε unchanged even though the conditions of the analysis no longer hold.")
rep("and the white-box audit certifies a violation for a specific neighbouring pair.",
    "and the conditional white-box audit certifies a sensitivity violation for a specific neighbouring pair.")
rep("**A certified audit on two datasets.** White-box audits of the Gaussian step show that, under the Gaussian-mechanism accounting of DP-SGD, the audited pairs imply budgets 22% (UCI Adult: ε ≥ 0.898 against 0.735 reported) and 76% (cardiovascular cohort: ε ≥ 1.474 against 0.839) above the reported values.",
    "**A certified audit on two datasets.** Finite-sample white-box audits of the conditional Gaussian step certify, at 95% confidence, that the audited neighbouring batches violate the sensitivity bound. Within the standard accounting framework this forces certificates at least 22% (UCI Adult: 0.898 against 0.735 reported) and 76% (cardiovascular cohort: 1.474 against 0.839) above the reported values. We do not present these as lower bounds on the privacy loss itself, which would require auditing the subsampled mechanism.")
rep("**A leak taxonomy with reproductions.** Checking an end-to-end proof against the code exposed five leak classes that do not change the reported ε.",
    "**A taxonomy of implementation defects with reproductions.** Checking an end-to-end proof against the code exposed five classes of defects that leak private information without changing the reported ε.")
rep("and a pipeline without the flow is statistically equivalent at every ε up to 4.7 that we tested.",
    "and no practically relevant contribution from the flow was detected at any ε up to 4.7 that we tested.")
# 2.1 sampling semantics
rep("The data loader includes each record with probability 1/⌈N/B⌉, which never exceeds the rate q = B/N passed to the accountant, so the accounting is conservative; the realised batch size varies around B.",
    "Batches are drawn by the Opacus DPDataLoader.from_data_loader, which includes each record independently with probability q_L = 1/⌈N/B⌉ (0.0417 for Adult and 0.0313 for Cardio), so the realised batch size varies around q_L N. The accountant is given q = B/N (0.0427 and 0.0320). Since q_L ≤ q for every N and B, and the PRV bound is non-decreasing in the sampling rate, the accounting is conservative; a regression test checks q_L ≤ q.")
# 2.3 theorem wording
rep("and the bound assumed by standard DP-SGD accounting is not guaranteed. A single violating pair of batches suffices to invalidate the standard certificate, because DP is a worst-case property. The accountant still reports the same ε, since it reads only (σ, q, T).",
    "Standard DP-SGD accounting requires Δ₂(S) = sup ‖S(B) − S(B′)‖₂ ≤ C over add/remove neighbours and over all values of the per-example randomness (z₀, t). Under OT-CFM this condition is not guaranteed. Whenever Δ₂(S) > C, the conditions of the standard accountant's proof are not satisfied, and its reported ε is no longer established by that analysis. The accountant cannot detect this, because it computes a guarantee from (σ, q, T) under the assumptions supplied to it.")
# 2.5 DAPC out of the main text
cut('h2("2.5 Dataset-Adaptive Privacy Budget Controller (DAPC)");', 'h2("2.6 Class-Conditional Marginal Stage and Joint Accounting");',
    'h2("2.5 Choice of the Privacy Budget");\n'
    'p("The released budget ε* of each dataset was fixed before any private computation by a public-metadata policy, the dataset-adaptive privacy budget controller (DAPC). For a requested budget of ε = 1, it selects ε* = 0.741 (Adult), 0.849 (Cardio) and 0.753 (Pima) from the rounded cohort size, the encoded dimension and name-based column sensitivities. Because it reads only public quantities, it consumes no privacy budget. Its complete specification, an adaptive budget-split rule and a robustness analysis are given in Supplementary Section S1. The results of the main text use fixed budget splits and do not depend on those details.");\n\n')
rep(" A fixed split spends a share ρ of ε* on the marginals, measured as the ε that the marginal release alone would cost; the adaptive DAPC rule targets a per-cell noise of 25% of the average cell count, capped at a 0.6 share.",
    " A fixed split spends a share ρ of ε* on the marginals, measured as the ε that the marginal release alone would cost; ρ = 0.9 is our primary configuration.")
rep("(b) DAPC, the split (z, σ),", "(b) The budget policy, the split (z, σ),")
# 2.7 audit
cut('p("**White-box step audit.**', 'p("**Black-box audit.**',
    'p("**Conditional white-box step audit.** For the worst pair found, we fix the two batches together with their prior draws and times, and test the conditional Gaussian mechanism S(B) + N(0, σ²C²I) against S(B′) + N(0, σ²C²I) with the training noise multiplier. The Neyman–Pearson test projects the output onto S(B′) − S(B). We estimate its false-positive and false-negative rates from 10⁶ independent noise draws per hypothesis and bound both with Clopper–Pearson intervals [@clopper] (Bonferroni-corrected, 95% overall). This finite-sample test yields a 95% lower bound μ_lb on the GDP parameter of the conditional step, and therefore on its sensitivity, Δ ≥ μ_lb σC [@nasr21,@nasr23]. Standard accounting assumes μ ≤ 2/σ for a replace-one step, so μ_lb > 2/σ certifies that the sensitivity assumption fails for this pair.");\n'
    'p("**What the audit does and does not establish.** The audit conditions on a fixed pair of batches, whereas the trained mechanism samples each record with probability q, which amplifies privacy. It is therefore not a lower bound on the privacy loss of the subsampled mechanism, and we do not report it as one. It does constrain the standard accounting framework, which certifies a Poisson-subsampled Gaussian mechanism from a single worst-case sensitivity bound: any valid bound must be at least the certified sensitivity of the audited pair. Re-running the accountant with the correspondingly larger bound, equivalently with the noise multiplier 2/μ_lb, gives the smallest ε this framework can certify for the run, which we call the *accounting-consistent ε*. Estimating the privacy loss of the full subsampled mechanism would require auditing the sampling event as well, which we leave to future work.");\n')
rep('p("**Implied ε:** the ε, at the training δ, that the Gaussian-mechanism accounting of DP-SGD yields when the per-step parameter is set to the audited lower bound μ_lb.");',
    'p("**Accounting-consistent ε:** the smallest ε, at the training δ, that the standard subsampled-Gaussian accountant can certify when its sensitivity bound covers the audited pair (noise multiplier 2/μ_lb). It is not a lower bound on the privacy loss of the trained mechanism.");')
# 2.8 datasets
rep("The sensitivity study and the main utility study used UCI Adult [@adult] (a 15,000-record sample, 12,000 for training, 15 columns encoded to 78 dimensions, income above 50K as target) and a 20,000-record sample of the Kaggle cardiovascular disease dataset [@cardio] (11 features, 12 encoded dimensions, presence of cardiovascular disease as target).",
    "The sensitivity study and the main utility study used two datasets. UCI Adult [@adult] has 48,842 records in the combined UCI release, 23.9% of them with income above 50K (the target). We drew a uniform random sample of 15,000 records without replacement with a fixed seed (not stratified) and split it into 12,000 training and 3,000 test records; its 15 columns encode to 78 dimensions. The Kaggle cardiovascular disease dataset [@cardio] has 70,000 records from medical examinations, with 11 features and a binary disease target. We drew a uniform random sample of 20,000 records in the same way (16,000 training and 4,000 test records; 12 encoded dimensions). We could not identify a peer-reviewed data descriptor for this dataset, so Supplementary Table S5 reports the SHA-256 digests of both source files to make the exact versions verifiable.")
rep("An exploratory small-cohort stress test used Pima Indians Diabetes [@pima] (614 training records)",
    "An exploratory small-cohort stress test used Pima Indians Diabetes [@pima] (768 records; 614 training and 154 test)")
rep("All splits were 80/20 with a fixed random seed.", "All train/test splits were 80/20, unstratified, with a fixed random seed.")
# 3.2
rep('h2("3.2 White-Box Audits Certify the Violation");', 'h2("3.2 Conditional Audits Certify the Sensitivity Violation");')
cut('p("Sensitivity values found by a search could be dismissed', 'B.push({ type: "table", caption: "**Table 3.**',
    'p("Sensitivity values found by a search could be attributed to numerical artefacts of that search. The conditional white-box audit removes this objection for the audited pairs, because its conclusion is a finite-sample bound that holds with 95% confidence. On Adult, the audit of the worst OT pair gave μ ≥ 0.896 against the 0.764 assumed by standard accounting (Table 3, Figure 4), so the sensitivity of that step exceeds the assumed bound. Within the standard accounting framework, a certificate that covers this pair cannot be smaller than ε = 0.898, whereas 0.735 was reported. On Cardio, the audit gave μ ≥ 1.427 against an assumed 0.966, and the accounting-consistent ε is 1.474 against a reported 0.839, 76% higher. These numbers describe what the standard accounting can certify; they are not measured privacy losses (Section 2.7). The same audit applied to the worst I-CFM pairs gave μ ≥ 0.518 on Adult and μ ≥ 0.936 on Cardio, both below the assumed values, as Proposition 1 predicts.");\n')
rep('"**Table 3.** White-box audits of one DP-SGD step for the worst pair found (10⁶ trials per hypothesis, 95% confidence)."',
    '"**Table 3.** Conditional white-box audits of one DP-SGD step for the worst pair found (finite-sample test, 95% confidence). The last row is the smallest ε that the standard accounting framework can certify; it is not a privacy-loss lower bound."')
rep('["Implied ε (Gaussian accounting)", "≥ 0.898", "0.735", "≥ 1.474", "0.839"]', '["Accounting-consistent ε", "≥ 0.898", "0.735", "≥ 1.474", "0.839"]')
rep("Hypothesis-testing view of the audits.", "Hypothesis-testing view of the conditional audits.")
rep("which is impossible if the standard certificate were valid;", "which is impossible if the assumed sensitivity bound held;")
# 3.3, 3.5
rep("All five leaks were fixed and covered by regression tests, and all pipeline experiments",
    "All five implementation defects were fixed in the audited implementation and covered by regression tests, and all pipeline experiments")
rep("With 10 paired seeds they were statistically equivalent on every metric at the pre-specified margins.",
    "With 10 paired seeds they were statistically equivalent on every metric at the prespecified margins; no practically relevant utility contribution from the flow was detected.")
rep(" The confidence intervals are three to four times narrower than the margin, so a true difference as small as 0.004 AUC would have been detected.",
    " The resulting confidence intervals were substantially narrower than the prespecified margins.")
rep('["DAPC adaptive split", "0.904 / 0.116", "0.780 / 0.070", "0.729 / 0.601"], ', '')
rep("At ε* = 2.45 and 4.72 on Adult, the full pipeline reached 0.905 and 0.906,",
    "At ε* = 2.45 and 4.72 on Adult, the full pipeline (with the adaptive split of Supplementary S1) reached 0.905 and 0.906,")
# discussion, threats, conclusions
rep("The size of the effect depended on the data: under Gaussian-mechanism accounting, the audited pairs implied a budget 22% above the reported one on Adult and 76% above it on Cardio.",
    "The size of the effect depended on the data: within the standard accounting framework, the audited pairs force certificates 22% above the reported budget on Adult and 76% above it on Cardio.")
rep(" The budget controller behaved as intended as a policy: it is deterministic, costs no privacy budget and is stable under perturbation of its constants (Supplementary Section S1). Its adaptive split rule, however, was no better than a fixed share of 0.9, which is unsurprising while one of the two mechanisms earns no budget.",
    " An adaptive version of the budget split (Supplementary Section S1) was no better than a fixed share of 0.9, which is unsurprising while one of the two mechanisms earns no budget.")
rep("they do not exclude a weaker but valid mechanism-specific analysis of OT-CFM.",
    "they do not exclude a weaker but valid mechanism-specific analysis of OT-CFM. The white-box audits condition on fixed batches and do not include privacy amplification by subsampling; they bound what the standard accounting can certify, not the privacy loss of the subsampled mechanism.")
rep("On two datasets, certified white-box audits of the worst neighbouring pairs found by search implied budgets 22% and 76% above the reported values under Gaussian-mechanism accounting.",
    "On two datasets, certified conditional audits of the worst neighbouring pairs found by search showed that certificates within the standard accounting framework must be at least 22% and 76% larger than the reported ones.")
rep("Checking that proof against the code exposed five leak classes that no privacy summary could reveal, each reproducible on real data.",
    "Checking that proof against the code exposed five classes of implementation defects that no privacy summary could reveal, each reproducible on real data.")
rep("a jointly accounted, target-aware marginal stage accounted for essentially all measured utility, and removing the flow produced statistically equivalent results.",
    "a jointly accounted, target-aware marginal stage accounted for essentially all measured utility, and no practically relevant contribution from the flow was detected under prespecified equivalence margins.")
rep("S1, complete specification and robustness analysis of DAPC;", "S1, the budget policy (DAPC): complete specification, adaptive split rule and robustness analysis;")
open(p, 'w', encoding='utf-8').write(s)
print("main edits ok")
