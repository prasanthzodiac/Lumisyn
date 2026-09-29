# Revision 3 edits to the supplement and the figure legend.
def edit(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{path}: NOT FOUND {a[:80]}"
        s = s.replace(a, b)
    open(path, "w", encoding="utf-8").write(s)


edit("content_supp.js", [
    ('["Budget map", "Equation 4 of the main text",', '["Budget map", "Equation S1",'),
    ('p("DAPC reads only public inputs:',
     'p("DAPC maps a requested budget ε_max to the released budget");\n'
     'B.push({ type: "eq", omml: "dapc", n: "S1" });\n'
     'p("where R is a public risk score. DAPC reads only public inputs:'),
    ("The adaptive split rule matched the best fixed split on Adult (0.904 against 0.906) but not on Cardio (2-way TVD 0.070 against 0.044).`);",
     "The adaptive split rule gave TSTR AUC / 2-way TVD of 0.904 / 0.116 on Adult, 0.780 / 0.070 on Cardio and 0.729 / 0.601 on Pima (3 seeds). It matched the best fixed split on Adult (0.906 / 0.096 at a 0.9 share) but not on Cardio, where the 0.9 share reached a 2-way TVD of 0.044. It is therefore not used in the main results.`);"),
    ('B.push({ type: "table", caption: "**Table S2.** Every code path',
     'h2("S2.4 Scope of the White-Box Audits");\n'
     'p("The audits in the main text condition on a fixed pair of batches with fixed prior draws and times. They certify, with 95% confidence, a lower bound on the sensitivity of that conditional Gaussian step, and hence that the sensitivity assumption of the standard accountant fails. They do not include the Poisson sampling of the trained mechanism, which amplifies privacy, so they are not lower bounds on its privacy loss. The accounting-consistent ε is the smallest ε that the standard subsampled-Gaussian accountant can certify with a sensitivity bound large enough to cover the audited pair: since that pair has replace-one sensitivity at least μ_lb σC, one of its add/remove neighbours has sensitivity at least μ_lb σC/2, and the accountant with noise σC and this bound is equivalent to noise multiplier 2/μ_lb. A lower bound on the privacy loss of the full mechanism would require auditing the sampling event, for example with many training runs or a one-run audit of the subsampled step, which we leave to future work.");\n'
     'B.push({ type: "table", caption: "**Table S2.** Every code path'),
    ('["Seeds", "PyTorch and NumPy generators seeded',
     '["Data provenance", "adult.csv: 48,842 records, SHA-256 d273d956c28eba72241f69dcbb78e37e93862248503a1873fb44b9fc2a6ae00b; cardio_train.csv: 70,000 records, SHA-256 2423cc5534fdc70887471778fc92b673ebb53d0d5188537b9029643b8a10e275; pima_diabetes.csv: 768 records, SHA-256 123ce738f394095a201da0f65c5e174cdcb19cdf5ec1535aee65e2b0e1e2aae2. Samples: uniform without replacement, pandas sample(n, random_state=42); splits: scikit-learn train_test_split(test_size=0.2, random_state=42), unstratified"],\n  ["Loader and accountant rates", "Opacus DPDataLoader.from_data_loader samples at q_L = 1/⌈N/B⌉ (Adult 0.0417, Cardio 0.0313); the accountant uses q = B/N (0.0427, 0.0320) ≥ q_L; regression test test_loader_sampling_rate_never_exceeds_accountant_rate"],\n  ["Seeds", "PyTorch and NumPy generators seeded'),
])
edit("make_figures.py", [('"flow + marginals (DAPC split)"', '"flow + marginals (adaptive split)"')])
print("supp + figure edits ok")
