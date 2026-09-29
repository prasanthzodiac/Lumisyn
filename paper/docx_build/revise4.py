"""Revision 4 of the main text: warm-started flow (pre-registered), adaptive split as a pre-registered negative, IEEE-length abstract."""
p = "content.js"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


def rep_between(start, end, new):
    """Replace the text from start up to (not including) end."""
    global s
    i = s.index(start)
    j = s.index(end, i)
    s = s[:i] + new + s[j:]


rep("// Main manuscript (revision 2, responding to the external audit).", "// Main manuscript (revision 4: warm-start study and pre-registered tests).")
rep('const { REFS } = require("./refs.js");',
    'const { REFS } = require("./refs.js");\n'
    'const fs = require("fs");\nconst path = require("path");\n'
    'const FV = JSON.parse(fs.readFileSync(path.resolve(__dirname, "..", "..", "runs", "flow_value", "analysis.json"), "utf8"));\n'
    'const CF = FV.confirmation.cells, CK = Object.keys(CF);\n'
    'const nm = (k) => (k.startsWith("adult") ? "Adult" : "Cardio"), es = (k) => CF[k].eps_star.toFixed(2);\n'
    'const h2pass = CK.filter((k) => CF[k].H2_p_holm < 0.05), h2fail = CK.filter((k) => CF[k].H2_p_holm >= 0.05);\n'
    'const h1pass = CK.filter((k) => CF[k].H1_p_holm < 0.05);\n'
    'const pts = (x) => (100 * x).toFixed(1);\n'
    'const maxH2 = Math.max(...CK.map((k) => CF[k].H2_diff));\n'
    'const WORDS = ["zero", "one", "two", "three", "four", "five", "six"];\n'
    'const REF_ALLCOL = (ds) => CF[CK.find((k) => k.startsWith(ds))].ref_allcol;\n'
    'const ofSix = (n) => (n === 6 ? "all six" : `${WORDS[n]} of six`);\n'
    'const wo = (k) => CF[k].allcol_acc_warm_only[0] - CF[k].allcol_acc_marg[0];\n'
    '// the warm-only sentence in Section 3.7 states: above marginals-only on Adult, level with it on Cardio\n'
    'if (!CK.every((k) => (k.startsWith("adult") ? wo(k) > 0.005 : Math.abs(wo(k)) < 0.003))) throw new Error("warm-only sentence no longer matches the data");')

# ---------------- abstract (IEEE limit: 250 words)
rep_between('B.push({ type: "abstract", text: "', '\n\n// ------------------------------------------------------------------ 1',
    'B.push({ type: "abstract", text: `Differentially private (DP) synthetic tabular data is often proposed as a safe way to share sensitive records, and flow matching is an attractive generator because it trains stably and samples quickly. We examined this route end to end. First, minibatch optimal-transport (OT) coupling, a common way to straighten flows, makes the loss of each example depend on the rest of its batch. This breaks the per-example sensitivity bound on which standard DP-SGD accounting relies, while the reported budget stays the same. We prove that independent coupling (I-CFM) keeps the bound and OT coupling does not, and we find violating neighbouring batches on two datasets. On a 20,000-record cardiovascular cohort, 69% of audited batches violated the bound, and a finite-sample white-box audit certified the violation at 95% confidence: within standard accounting, no valid certificate for that run can be below ε = 1.47, whereas 0.84 was reported. Second, we prove an end-to-end guarantee for a public-schema pipeline built on I-CFM; checking the proof against the code exposed five further leak classes, each reproduced on real data. Third, we trace where utility comes from. Jointly accounted class-conditional marginals raise the downstream AUC on UCI Adult from 0.49 to 0.906 at ε = 0.741, above target-aware AIM (0.872), and a randomly initialised flow adds nothing measurable to them. A flow warm-started on records drawn from the noisy marginals and then fine-tuned with DP-SGD does learn cross-column dependence from the private data (pre-registered, 10 seeds, significant in ${ofSix(h2pass.length)} settings). An ε-adaptive budget split, also pre-registered, gave no benefit.` });')

# ---------------- introduction
rep("and (iv) measure how much of the downstream utility is due to the flow itself.",
    "and (iv) measure how much of the downstream utility is due to the flow itself, and whether the flow can add to the marginals once it starts from what they already provide.")
rep_between('  "**A scoped negative result about utility.**', '\n] });',
    '  "**Where the utility comes from.** A jointly accounted, target-aware marginal stage lifts utility above AIM and MST, and a randomly initialised flow adds nothing measurable to it. A flow warm-started from the noisy marginals and fine-tuned with DP-SGD does learn cross-column dependence from the private records, which a pre-registered 10-seed test confirmed in every setting we evaluated. A pre-registered ε-adaptive budget split failed, and we report it as a negative result.",')

# ---------------- methods 2.6: warm start and proof step (b)
rep("which uses no information from the flow.\");",
    "which uses no information from the flow.\");\n"
    "p(\"**Warm start.** In the default configuration the flow starts from a random initialisation, so DP-SGD has to learn everything, including the label–feature structure that Stage 7c already releases. In the warm-start configuration we run Stage 7c first, draw N records from the noisy tables with independent within-class ranks, and pretrain the velocity network on them for 20 epochs without noise. DP-SGD then fine-tunes this network on the private records with the same σ, q and T. The pretraining reads only the output of the marginal mechanism and therefore costs no privacy. StraightDP [@straightdp] proposed the same two-stage pattern for image rectified flows: release class-conditional moments once, then spend the rest of the budget on DP-SGD. We apply it to tabular data with count marginals. To isolate what DP fine-tuning learns from the private records, the **warm-only** ablation runs and accounts every DP-SGD step with a learning rate of zero, so that its released weights equal the warm start and its ε is identical.\");")
rep("(b) The budget policy, the split (z, σ), the number of steps T (fixed epochs, no data-dependent stopping), the initialisation and all hyper-parameters are functions of public inputs and constants.",
    "(b) The budget policy, the split (z, σ), the number of steps T (fixed epochs, no data-dependent stopping) and all hyper-parameters are functions of public inputs and constants. The initialisation is data-independent or, with the warm start, a post-processing of the output of M₂. In the latter case M₁ is chosen adaptively after M₂, which PRV composition covers.")

# ---------------- 2.9 metrics, 2.10 statistics
rep('p("**Sensitivity ratio Δ/2C:**',
    'p(`**All-column accuracy:** for every column in turn, a random forest trained on the synthetic table, discretised on the public bins, predicts that column from all the others; its accuracy on the real test split is averaged over columns. Unlike TSTR AUC, which only concerns the label, this metric responds to dependence among the features. The same procedure applied to the real training split gives the reference, which is ${REF_ALLCOL("adult").toFixed(3)} on Adult and ${REF_ALLCOL("cardio").toFixed(3)} on Cardio.`);\n'
    'p("**3-way label TVD:** the TVD averaged over all (target, feature, feature) marginals, that is, over feature pairs within each class.");\n'
    'p("**Sensitivity ratio Δ/2C:**')
rep("and we regard smaller differences as practically irrelevant for data release.\");",
    "and we regard smaller differences as practically irrelevant for data release. The warm-start study was pre-registered: the objective, the grid, the fitting procedure of the adaptive rule and the success criterion were written down before the runs, and a confirmatory addendum was registered after the 3-seed grid and before the 10-seed runs (PREREGISTRATION.md in the artifact). Its hypotheses were tested with one-sided Welch t-tests [@welch] and Holm-corrected over the six dataset–budget cells [@holm].\");")

# ---------------- 3.5 title scope
rep('h2("3.5 In the Evaluated Setting, Utility Comes from the Marginals");', 'h2("3.5 With a Randomly Initialised Flow, Utility Comes from the Marginals");')
rep("With 10 paired seeds they were statistically equivalent on every metric at the prespecified margins; no practically relevant utility contribution from the flow was detected.",
    "With 10 paired seeds they were statistically equivalent on every metric at the prespecified margins; no practically relevant utility contribution from the randomly initialised flow was detected.")

# ---------------- new 3.7 warm start, stress test becomes 3.8
rep('h2("3.7 Exploratory Stress Test on Small Cohorts");',
    'h2("3.7 A Warm-Started Flow Learns Dependence That the Marginals Cannot Carry");\n'
    'const gap = (k) => CF[k].H1_diff / (CF[k].ref_allcol - CF[k].allcol_acc_marg[0]);\n'
    'p(`The flow in Section 3.5 had to learn from noise-dominated gradients, starting from a random network. The warm start removes that handicap by giving the flow what the marginals already say before it sees any private record. Table 8 and Figure 8 report the pre-registered 10-seed confirmation. Downstream AUC did not change: the warm-started flow was equivalent to the marginals-only ablation at the ±0.01 margin in ${ofSix(CK.filter((k) => CF[k].auc_tost_p < 0.05).length)} cells. All-column accuracy, which responds to dependence among features, was higher with the warm-started, fine-tuned flow in ${ofSix(h1pass.length)} cells (H1, Holm-corrected p < 0.05), by ${pts(Math.min(...CK.map((k) => CF[k].H1_diff)))} to ${pts(Math.max(...CK.map((k) => CF[k].H1_diff)))} percentage points. On Adult, part of that gain did not come from the private data: the warm start alone, whose weights are a post-processing of the marginals, already scored 0.8 to 1.0 points above marginals-only, presumably because the network maps nearby prior draws to nearby records and so induces some dependence of its own. On Cardio the warm start alone was level with marginals-only. The comparison that isolates learning from private records is H2, DP fine-tuning against the warm start alone. It held in ${ofSix(h2pass.length)} cells, with gains of up to ${pts(maxH2)} percentage points` +\n'
    '  (h2fail.length ? `, and failed in ${h2fail.map((k) => `${nm(k)} at ε* = ${es(k)}`).join(" and ")}, where the fine-tuning noise is largest. ` : ". ") +\n'
    '  `The pre-registered decision rule allowed the claim only if H1 and H2 both held in at least five of six cells; ${h1pass.length >= 5 && h2pass.length >= 5 ? "this criterion was met" : "this criterion was not met, and we therefore report the result as not confirmed"}. The effect is modest. Of the gap in all-column accuracy between marginals-only and real data, the warm-started, fine-tuned flow closes between ${Math.round(100 * Math.min(...CK.map(gap)))}% and ${Math.round(100 * Math.max(...CK.map(gap)))}%.`);\n'
    'B.push({ type: "table", caption: "**Table 8.** Pre-registered 10-seed confirmation at a 0.9 marginal share: all-column accuracy (mean ± sd) and the gain from DP fine-tuning over the warm start alone (H2, one-sided Welch test, Holm-corrected over six cells). The last column is TSTR AUC for marginals only and for the warm-started, fine-tuned flow.", size: 16,\n'
    '  header: ["Dataset (ε*)", "Marginals only", "Warm start only", "Warm start + DP fine-tuning", "DP fine-tuning gain [Holm p]", "TSTR AUC"], widths: [1400, 1450, 1450, 1650, 1650, 1400],\n'
    '  rows: CK.map((k) => { const c = CF[k], f = (a) => `${a[0].toFixed(4)} ± ${a[1].toFixed(4)}`;\n'
    '    return [`${nm(k)} (${es(k)})`, f(c.allcol_acc_marg), f(c.allcol_acc_warm_only), f(c.allcol_acc_warm), `${c.H2_diff >= 0 ? "+" : ""}${c.H2_diff.toFixed(4)} [${c.H2_p_holm.toExponential(1)}]`, `${c.tstr_auc_marg[0].toFixed(3)} / ${c.tstr_auc_warm[0].toFixed(3)}`]; }) });\n'
    'B.push({ type: "fig", file: "fig8_flow_value.png", width: 6.3, caption: "**Figure 8.** All-column accuracy against the released budget. The warm-started flow is pretrained on records drawn from the noisy marginals (post-processing) and then either fine-tuned with DP-SGD (blue) or left unchanged (grey). The dotted line is the same measure on real data. Means ± standard deviation over 10 seeds." });\n'
    'p(`The same study tested the original motivation of DAPC, namely that the split of ε* between the marginals and the flow should adapt to the budget. As pre-registered, we fitted the simplest monotone rule in ε* on Adult and evaluated it on Cardio (Supplementary Section S1.3). On Adult, a 0.9 share was best at every budget, so the fitted rule never adapts. On Cardio, a fixed 0.5 share scored higher than the rule (composite ${FV.cardio_mean_composite["warm_0.5"].toFixed(3)} against ${FV.cardio_mean_composite.adaptive.toFixed(3)}). The criterion failed, and we make no claim for an adaptive split.`);\n\n'
    'h2("3.8 Exploratory Stress Test on Small Cohorts");')

# ---------------- discussion
rep("The central result on utility is a negative one, and its scope needs to be stated precisely. Under the evaluated datasets, privacy levels and training configurations, the noisy marginal stage accounted for essentially all measured downstream utility, and no practically relevant utility contribution from the flow was detected under the prespecified equivalence margins.",
    "The utility results need to be stated precisely. With a randomly initialised flow, the noisy marginal stage accounted for essentially all measured downstream utility, and no practically relevant contribution from the flow was detected under the prespecified equivalence margins.")
rep("We recommend that DP generative papers report this ablation routinely.\");",
    "We recommend that DP generative papers report this ablation routinely.\");\n"
    "p(\"The warm-start result qualifies this negative finding without reversing it. The flow did not improve the label-related utility that the marginals already provide, but once it started from them it learned dependence among features from the private records, which a set of label × feature count tables cannot represent. This agrees with the design argument of StraightDP [@straightdp]: whatever a cheap one-shot measurement can carry should not be relearned through noisy gradients, and the gradient budget is better spent on what remains. The design of the ablation mattered here as well. Had we compared the fine-tuned flow only with marginals-only, we would have credited DP-SGD with a gain that the warm start obtains without seeing any private record. The gains from fine-tuning are modest and are smallest at the lowest budget, where the fine-tuning noise is largest.\");")
rep("An adaptive version of the budget split (Supplementary Section S1) was no better than a fixed share of 0.9, which is unsurprising while one of the two mechanisms earns no budget.",
    "Finally, a pre-registered test of an ε-adaptive budget split found that a fixed share was at least as good. We see no reason to adapt the split while the flow contributes only second-order structure.")
rep("We did not test cohorts larger than 20,000 records, public pretraining of the flow, or larger architectures such as transformers, which are the settings in which the flow is most likely to contribute.",
    "We did not test cohorts larger than 20,000 records, pretraining on public data from a related population, or larger architectures such as transformers; these are the settings in which the flow is most likely to contribute more.")
rep("configuration sweeps used 3 seeds, and the small-cohort study is exploratory.",
    "configuration sweeps used 3 seeds, and the small-cohort study is exploratory. The confirmatory warm-start addendum was written after the 3-seed grid had been seen, as the pre-registration states.")

# ---------------- conclusions
rep("Under the evaluated datasets, privacy levels and training configurations, a jointly accounted, target-aware marginal stage accounted for essentially all measured utility, and no practically relevant contribution from the flow was detected under prespecified equivalence margins. These findings argue",
    "A jointly accounted, target-aware marginal stage accounted for essentially all measured downstream utility, and a randomly initialised flow added nothing measurable to it. Warm-starting the flow from the noisy marginals, at no privacy cost, let DP fine-tuning add cross-column dependence learned from the private records, whereas an ε-adaptive budget split brought no benefit. These findings argue")
rep("S1, the budget policy (DAPC): complete specification, adaptive split rule and robustness analysis;",
    "S1, the budget policy (DAPC): complete specification, robustness analysis and the pre-registered test of an adaptive split;")
# ---------------- Table 1: warm-start setting
rep('["Sampling", "RK4, 30 steps; clamp to [−1, 1]; argmax on one-hot blocks"],',
    '["Warm start (Section 2.6)", "Optional: 20 epochs of non-private AdamW (learning rate 10⁻³, batch 512) on N records drawn from the Stage 7c tables; the DP-SGD schedule is unchanged"],\n    ["Sampling", "RK4, 30 steps; clamp to [−1, 1]; argmax on one-hot blocks"],')
open(p, "w", encoding="utf-8").write(s)
print("ok")
