"""Revision 4 of the supplement: pre-registered adaptive-split test, warm-start lemma, 10-seed confirmation tables."""
p = "content_supp.js"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


rep('const leaks = J("leak_demos.json");',
    'const leaks = J("leak_demos.json");\nconst fva = J("flow_value/analysis.json");\nconst fvc = fva.confirmation.cells;')

# S1: the adaptive rule is now judged by the pre-registered test, not the earlier ad hoc comparison.
rep("It is therefore not used in the main results.`);",
    "It is therefore not used in the main results.`);\n"
    "h2(\"S1.3 Pre-Registered Test of an ε-Adaptive Split\");\n"
    "p(`Because the ad hoc rule above was fitted by hand, we also ran a pre-registered test (runs/flow_value/PREREGISTRATION.md, written before the grid was run). "
    "Configurations used the warm-started flow of main-text Section 2.6 with marginal shares ρ ∈ {0.2, 0.5, 0.9}, on Adult and Cardio at requested budgets of 1, 4 and 8, with 3 seeds each. "
    "Each configuration received a composite score, the mean of the min–max-normalised TSTR AUC, all-column accuracy, −(2-way TVD) and −(3-way label TVD) within its (dataset, ε) cell. "
    "The rule was fitted on Adult only, as the simplest monotone function of the released ε* that reproduces the best ρ at each budget, and was evaluated on Cardio. "
    "The claim required the rule to beat every fixed ρ on Cardio when averaged over the three budgets. "
    "On Adult, ρ = 0.9 was best at every budget, so the fitted rule is the constant ρ = 0.9 and adapts to nothing. "
    "On Cardio, its mean composite was ${fva.cardio_mean_composite.adaptive.toFixed(3)}, against ${fva.cardio_mean_composite[\"warm_0.5\"].toFixed(3)} for a fixed ρ = 0.5 (Table S1b). "
    "The pre-registered criterion therefore failed, and we make no claim that an ε-adaptive budget split improves utility.`);\n"
    "B.push({ type: \"table\", caption: \"**Table S1b.** Pre-registered adaptive-split test: composite score per (dataset, requested ε) cell, 3 seeds (higher is better; the scores are normalised within each cell).\", size: 17,\n"
    "  header: [\"Cell (ε*)\", \"ρ = 0.2\", \"ρ = 0.5\", \"ρ = 0.9\", \"Marginals only (ρ = 0.9)\", \"Rule choice\"], widths: [1900, 1300, 1300, 1300, 1900, 1300],\n"
    "  rows: Object.entries(fva.cells).map(([k, c]) => [`${k.split(\"@\")[0] === \"adult\" ? \"Adult\" : \"Cardio\"}, ${Number(k.split(\"@\")[1])} (${c.eps_star.toFixed(2)})`,\n"
    "    ...[\"warm_0.2\", \"warm_0.5\", \"warm_0.9\", \"marginals_only_0.9\"].map((g) => c.composite[g].toFixed(3)),\n"
    "    k.startsWith(\"adult\") ? \"(fitted)\" : \"ρ = 0.9\"]) });")

# S2: warm start in the mechanism specification and proof.
rep("Initialisation uses PyTorch defaults with a zero-initialised output layer and does not depend on data.",
    "Initialisation uses PyTorch defaults with a zero-initialised output layer and does not depend on data; with the warm start, it is instead the result of non-private pretraining on records sampled from the output of M₂ (Lemma S5).")
rep('p("**Proof of Theorem 2.**',
    'p("**Lemma S5 (warm start).** When the warm start is enabled, M₂ is run first. N records are drawn from its noisy tables, independently within each class, and encoded with the public schema. The velocity network is then trained on them without noise, starting from the data-independent initialisation. The only inputs of this step are the output of M₂, the public schema, the public N (A2) and fresh randomness, so the starting point of M₁ is a post-processing of M₂. M₁ is thus chosen adaptively after M₂. PRV composition bounds adaptive compositions of Gaussian mechanisms exactly as it bounds non-adaptive ones, and σ, q and T are still fixed in advance (Lemma S4), so the accounting history is unchanged. In the ablation without DP fine-tuning, every step of M₁ is still executed and accounted, with the learning rate set to zero.");\n'
    'p("**Proof of Theorem 2.**')
rep("By Lemma S4 the mechanism parameters are constants. M₁ and M₂",
    "By Lemma S4 the mechanism parameters are constants, and by Lemma S5 a warm start makes M₁ an adaptive choice after M₂. M₁ and M₂")
rep('["6/8 Flow training", "Algorithm 1 with I-CFM", "M₁ (Lemma S2)"],',
    '["6 Warm start (optional)", "Reads only the noisy Stage 7c tables, the public schema and N", "Post-processing of M₂ (Lemma S5)"],\n    ["6/8 Flow training", "Algorithm 1 with I-CFM", "M₁ (Lemma S2)"],')

# S4: per-cell confirmation table.
rep('p("Equivalence was tested with paired',
    'const pm = (a) => `${a[0].toFixed(4)} ± ${a[1].toFixed(4)}`;\n'
    'B.push({ type: "table", caption: "**Table S4e.** Pre-registered 10-seed confirmation (Addendum A), all-column accuracy, mean ± sd. H1: warm start + DP fine-tuning > marginals only; H2: warm start + DP fine-tuning > warm start without fine-tuning. One-sided Welch tests, Holm-corrected over the six cells.", size: 16,\n'
    '  header: ["Cell (ε*)", "Marginals only", "Warm, no fine-tuning", "Warm + DP fine-tuning", "H1 diff (p Holm)", "H2 diff (p Holm)"], widths: [1500, 1500, 1500, 1600, 1450, 1450],\n'
    '  rows: Object.entries(fvc).map(([k, c]) => [`${k.startsWith("adult") ? "Adult" : "Cardio"} (${c.eps_star.toFixed(2)})`, pm(c.allcol_acc_marg), pm(c.allcol_acc_warm_only), pm(c.allcol_acc_warm),\n'
    '    `+${c.H1_diff.toFixed(4)} (${c.H1_p_holm.toExponential(1)})`, `${c.H2_diff >= 0 ? "+" : ""}${c.H2_diff.toFixed(4)} (${c.H2_p_holm.toExponential(1)})`]) });\n'
    'B.push({ type: "table", caption: "**Table S4f.** The same runs, other metrics (mean over 10 seeds). AUC equivalence: TOST at ±0.01 (Welch).", size: 16,\n'
    '  header: ["Cell", "AUC marginals only", "AUC warm + DP", "TOST p", "3-way TVD marginals only", "3-way TVD warm + DP"], widths: [1300, 1500, 1500, 1200, 1800, 1700],\n'
    '  rows: Object.entries(fvc).map(([k, c]) => [`${k.startsWith("adult") ? "Adult" : "Cardio"} ${Number(k.split("@")[1])}`, c.tstr_auc_marg[0].toFixed(4), c.tstr_auc_warm[0].toFixed(4), c.auc_tost_p.toExponential(1),\n'
    '    c.tvd_3way_label_marg[0].toFixed(4), c.tvd_3way_label_warm[0].toFixed(4)]) });\n'
    'p("Equivalence was tested with paired')

# S6: manifest.
rep('["Section S1, Fig. S1",',
    '["Section 3.6, Table 8, Fig. 8, Tables S1b, S4e–f", "lumisyn/backend/flow_value_eval.py; flow_value_analysis.py", "runs/flow_value/results.jsonl; analysis.json; PREREGISTRATION.md"],\n  ["Section S1, Fig. S1",')
open(p, "w", encoding="utf-8").write(s)
print("ok")
