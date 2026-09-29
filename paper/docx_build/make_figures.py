"""Figures for the DOCX manuscript, generated from the committed result files in runs/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
C_OT, C_IC, C_M, C_F, C_B = "#c0392b", "#2471a3", "#1e8449", "#7d3c98", "#7f8c8d"

split = [json.loads(l) for l in (ROOT / "runs/dapc_split/results.jsonl").read_text().splitlines()]
audit = json.loads((ROOT / "runs/privacy_audit.json").read_text())
base = json.loads((ROOT / "runs/baselines/results.json").read_text())["summary"]
dapc = json.loads((ROOT / "runs/dapc_sensitivity.json").read_text())
worst = json.loads((ROOT / "runs/icfm_worst_case_sensitivity.json").read_text())
rand = json.loads((ROOT / "runs/icfm_sensitivity_verification.json").read_text())["sensitivity"]


def agg(ds, cfg, metric, eps=1.0):
    v = [r[metric] for r in split if r["dataset"] == ds and r["config"] == cfg and r.get("target_eps", 1.0) == eps]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0) if v else (np.nan, 0.0)


# ---------------- Fig 1: pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#34495e", lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6)
    def arr(a, b, **kw):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#34495e", **kw))
    pub, dp, pp = "#eceff1", "#d6eaf8", "#fdfefe"
    box(1, 27, 15, 9, "Public schema\n(types, bounds,\ncategories)", pub)
    box(1, 12, 15, 9, "Row-local\npreprocessing $\\phi$", pub)
    box(21, 27, 21, 9, "DAPC: $\\varepsilon^\\star$, $z$, $\\sigma$\nfrom public\nmetadata", pub)
    box(21, 12, 21, 9, "DP-SGD on I-CFM\nrate $q$, clip $C$,\nnoise $\\sigma C$", dp)
    box(21, 0.5, 21, 8, "Stage 7c: noisy\nlabel$\\times$feature tables", dp)
    box(48, 12, 16, 9, "Sample ODE,\npublic-domain\nrepair", pp)
    box(48, 0.5, 16, 8, "Within-class\nquantile map", pp)
    box(71, 0.5, 12, 8, "Synthetic\ndata", pp)
    box(49, 27, 26, 9, "One PRV accountant:\nDP-SGD steps + Gaussian\nmarginal release", "#fef9e7")
    arr((8.5, 27), (8.5, 21)); arr((16, 31.5), (21, 31.5)); arr((16, 16.5), (21, 16.5))
    arr((31.5, 27), (31.5, 21)); ax.plot([8.5, 8.5], [12, 4.5], color="#34495e", lw=0.9); arr((8.5, 4.5), (21, 4.5))
    arr((42, 16.5), (48, 16.5)); arr((56, 12), (56, 8.5)); arr((42, 4.5), (48, 4.5)); arr((64, 4.5), (71, 4.5))
    arr((42, 20), (49, 28), linestyle="--")
    ax.plot([42, 45.2, 45.2], [7.5, 7.5, 31.5], ls="--", lw=0.9, color="#34495e"); arr((45.2, 31.5), (49, 31.5), linestyle="--")
    ax.text(71, 21, "blue: the only mechanisms that\n   read private data\ngrey: public information only\nwhite: post-processing\ndashed: privacy accounting",
            ha="left", va="top", fontsize=6.8, color="#34495e")
    fig.savefig(OUT / "fig1_pipeline.png"); plt.close(fig)


# ---------------- Fig 2: sensitivity
def fig_sensitivity():
    adv = audit["stronger_adversary"]; card = audit["stronger_adversary_cardio"]
    keys = ("sensitivity_over_2C_mean", "sensitivity_over_2C_std", "sensitivity_over_2C_max")
    rows = [("Adult\nrandom", rand["batch_ot"]["sensitivity_mean"] / 2.5, rand["batch_ot"]["sensitivity_std"] / 2.5, rand["batch_ot"]["ratio_to_bound_max"],
             rand["i_cfm"]["sensitivity_mean"] / 2.5, rand["i_cfm"]["sensitivity_std"] / 2.5, rand["i_cfm"]["ratio_to_bound_max"]),
            ("Adult\nscreened",) + tuple(worst[m][k] for m in ("batch_ot", "i_cfm") for k in keys),
            ("Adult\nhill-climbed",) + tuple(adv[m][k] for m in ("batch_ot", "i_cfm") for k in keys),
            ("Cardio\nhill-climbed",) + tuple(card[m][k] for m in ("batch_ot", "i_cfm") for k in keys)]
    x = np.arange(len(rows)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for off, idx, c, lab in ((-w / 2, 1, C_OT, "Batch-OT coupling"), (w / 2, 4, C_IC, "I-CFM (independent)")):
        ax.bar(x + off, [r[idx] for r in rows], w, yerr=[r[idx + 1] for r in rows], color=c, alpha=0.75, capsize=3, label=f"{lab}: mean $\\pm$ sd")
        ax.scatter(x + off, [r[idx + 2] for r in rows], marker="v", color="black", s=22, zorder=5)
    ax.axhline(1.0, ls="--", color="black", lw=1); ax.text(-0.55, 1.045, "DP-SGD bound", ha="left", fontsize=7.5)
    ax.scatter([], [], marker="v", color="black", s=22, label="worst batch (max)")
    ax.set_xticks(x, [r[0] for r in rows]); ax.set_ylabel(r"clipped-sum sensitivity  $\Delta / 2C$")
    ax.set_ylim(0, 1.65); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.savefig(OUT / "fig2_sensitivity.png"); plt.close(fig)


# ---------------- Fig 3: audit trade-off curves
def fig_audit():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    a = np.linspace(1e-4, 1, 400)
    for ax, key, title in ((axes[0], "stronger_adversary", "Adult"), (axes[1], "stronger_adversary_cardio", "Cardio")):
        wb = audit[key]["whitebox_step_audit"]
        mc, ml = wb["batch_ot"]["mu_claimed"], wb["batch_ot"]["mu_lower_bound_95"]
        ax.plot(a, norm.cdf(norm.ppf(1 - a) - mc), color="black", lw=1.2, label=f"claimed by accountant ($\\mu$={mc:.3f})")
        ax.plot(a, norm.cdf(norm.ppf(1 - a) - ml), color=C_OT, lw=1.2, ls="--", label=f"certified OT bound ($\\mu\\geq${ml:.3f})")
        ax.scatter(wb["batch_ot"]["fpr"], wb["batch_ot"]["fnr"], color=C_OT, s=28, zorder=5, label="OT audit test")
        ax.scatter(wb["i_cfm"]["fpr"], wb["i_cfm"]["fnr"], color=C_IC, s=28, marker="s", zorder=5, label="I-CFM audit test")
        ax.set_title(title, fontsize=9); ax.set_xlabel("false-positive rate"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=6.3, loc="upper right")
    axes[0].set_ylabel("false-negative rate")
    fig.savefig(OUT / "fig3_audit.png"); plt.close(fig)


# ---------------- Fig 4: budget split
def fig_split():
    shares = [(0.0, "flow_only"), (0.2, "share_0.2"), (0.4, "share_0.4"), (0.6, "share_0.6"), (0.9, "share_0.9")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"wspace": 0.38})
    for ds, c in (("adult", C_IC), ("cardio", C_M), ("pima", C_F)):
        pts = [(s, *agg(ds, cfg, "tstr_auc"), *agg(ds, cfg, "tvd_2way_all")) for s, cfg in shares if not np.isnan(agg(ds, cfg, "tstr_auc")[0])]
        s = [p[0] for p in pts]
        axes[0].errorbar(s, [p[1] for p in pts], yerr=[p[2] for p in pts], color=c, marker="o", ms=4, capsize=2, label=ds.capitalize())
        axes[1].errorbar(s, [p[3] for p in pts], yerr=[p[4] for p in pts], color=c, marker="o", ms=4, capsize=2)
        mo = agg(ds, "share_0.9_marginals_only", "tstr_auc")
        if not np.isnan(mo[0]):
            axes[0].scatter([0.93], [mo[0]], facecolor="white", edgecolor=c, marker="D", s=26, zorder=6)
            axes[1].scatter([0.93], [agg(ds, "share_0.9_marginals_only", "tvd_2way_all")[0]], facecolor="white", edgecolor=c, marker="D", s=26, zorder=6)
    axes[0].scatter([], [], facecolor="white", edgecolor="black", marker="D", s=26, label="marginals only (no flow)")
    axes[0].set_ylabel("TSTR AUC"); axes[1].set_ylabel("2-way marginal TVD (lower is better)")
    for ax in axes:
        ax.set_xlabel("share of $\\varepsilon^\\star$ spent on Stage 7c marginals")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(OUT / "fig4_split.png"); plt.close(fig)


# ---------------- Fig 5: epsilon sweep (Adult)
def fig_eps():
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for cfg, lab, c, mk in (("flow_only", "flow only", C_OT, "o"), ("adaptive", "flow + marginals (adaptive split)", C_IC, "s"),
                            ("adaptive_marginals_only", "marginals only", C_M, "D")):
        xs, ys, es = [], [], []
        for te in (1.0, 4.0, 8.0):
            rs = [r for r in split if r["dataset"] == "adult" and r["config"] == cfg and r.get("target_eps", 1.0) == te]
            if rs:
                xs.append(rs[0]["epsilon"]); ys.append(np.mean([r["tstr_auc"] for r in rs])); es.append(np.std([r["tstr_auc"] for r in rs], ddof=1))
        dash = "--" if cfg == "adaptive" else "-"
        ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, ms=5 if cfg == "adaptive" else 3.5, capsize=2, label=lab, ls=dash,
                    mfc="white" if cfg == "adaptive" else c, zorder=6 if cfg == "adaptive" else 4)
    ax.annotate("these two lines overlap:\nno AUC gain from the flow", xy=(2.45, 0.905), xytext=(0.75, 0.79), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.axhline(0.917, color="black", ls=":", lw=1); ax.text(4.7, 0.925, "real-data reference (TRTR)", ha="right", fontsize=7)
    ax.set_xlabel("released $\\varepsilon^\\star$ (Adult)"); ax.set_ylabel("TSTR AUC"); ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    fig.savefig(OUT / "fig5_eps.png"); plt.close(fig)


# ---------------- Fig 6: baselines (Adult)
def fig_baselines():
    ours = [r for r in split if r["dataset"] == "adult" and r["config"] == "share_0.9" and r.get("target_eps", 1.0) == 1.0]
    rows = [("Ours (flow + 7c, share 0.9)", np.mean([r["tstr_auc"] for r in ours]), np.std([r["tstr_auc"] for r in ours], ddof=1), np.mean([r["tvd_2way_all"] for r in ours]), C_IC)]
    names = {"aim_target": "AIM, label$\\times$feature workload", "aim": "AIM", "mst": "MST", "patectgan": "PATE-CTGAN",
             "ours_icfm": "Ours, flow only", "dpctgan": "DP-CTGAN"}
    for k, lab in names.items():
        v = base[k]; rows.append((lab, v["tstr_auc"]["mean"], v["tstr_auc"]["std"], v["tvd_2way"]["mean"], C_B if "Ours" not in lab else C_OT))
    rows.sort(key=lambda r: r[1])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    y = np.arange(len(rows))
    axes[0].barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], color=[r[4] for r in rows], capsize=2)
    axes[0].axvline(0.5, color="black", ls=":", lw=0.9); axes[0].set_xlim(0.4, 0.95); axes[0].set_xlabel("TSTR AUC (higher is better)")
    axes[1].barh(y, [r[3] for r in rows], color=[r[4] for r in rows]); axes[1].set_xlabel("2-way marginal TVD (lower is better)")
    axes[0].set_yticks(y, [r[0] for r in rows], fontsize=7.5)
    fig.savefig(OUT / "fig6_baselines.png"); plt.close(fig)


# ---------------- Fig 7: DAPC robustness
def fig_dapc():
    d = dapc["datasets"]; items = sorted(d.items(), key=lambda kv: kv[1]["eps_nominal"])
    short = lambda n: n.replace(" (Kaggle)", "").replace(" (High-Dimensional 31-D)", "").replace(" (Heterogeneous Mixed-Type)", "").replace("Kaggle ", "").replace("Synthetic ", "Syn. ")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    y = np.arange(len(items))
    lo = [v["eps_nominal"] - v["eps_p05"] for _, v in items]; hi = [v["eps_p95"] - v["eps_nominal"] for _, v in items]
    ax.errorbar([v["eps_nominal"] for _, v in items], y, xerr=[lo, hi], fmt="o", color=C_IC, ms=4, capsize=2)
    ax.set_yticks(y, [f"{short(k)}  (n={v['rounded_n']}, d={v['d_encoded']})" for k, v in items], fontsize=7)
    ax.set_xlabel("DAPC-selected $\\varepsilon^\\star$ (nominal; bars: 5-95% under constant perturbation)")
    fig.savefig(OUT / "fig7_dapc.png"); plt.close(fig)


# ---------------- Fig: where batch dependence enters (toy 2-D example, real Hungarian solver)
def fig_coupling():
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(3)
    B = 10
    z0 = rng.normal(0, 1, (B, 2))
    z1 = np.vstack([rng.normal([3, 1.5], 0.5, (5, 2)), rng.normal([3, -1.5], 0.5, (5, 2))])
    j = 0
    def ot(data):
        cost = ((z0[:, None, :] - data[None, :, :]) ** 2).sum(-1)
        _, col = linear_sum_assignment(cost)
        return col  # prior i -> data col[i]
    best = None
    for cand in rng.normal([2.6, 0.0], 1.6, (400, 2)):   # search a replacement that re-pairs many records
        z1p = z1.copy(); z1p[j] = cand
        changed = int(np.sum(ot(z1) != ot(z1p))) - 0
        if best is None or changed > best[0]:
            best = (changed, cand)
    z1p = z1.copy(); z1p[j] = best[1]
    a, ap_ = ot(z1), ot(z1p)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharex=True, sharey=True)
    for ax, title, pairs_before, pairs_after in ((axes[0], "I-CFM: record $i$ keeps prior draw $i$", np.arange(B), np.arange(B)),
                                                 (axes[1], "Batch-OT: pairing solved over the batch", a, ap_)):
        for i in range(B):
            moved = pairs_before[i] != pairs_after[i] or pairs_after[i] == j
            d_after = z1p[pairs_after[i]]
            if moved:
                ax.plot(*zip(z0[i], z1[pairs_before[i]]), color="#bdc3c7", lw=0.9, ls=":")
            ax.plot(*zip(z0[i], d_after), color=C_OT if moved else "#7f8c8d", lw=1.4 if moved else 0.8)
        ax.scatter(*z0.T, s=16, color="black", zorder=5, label="prior draws $z_0$")
        ax.scatter(*np.delete(z1p, j, 0).T, s=20, color=C_IC, zorder=5, label="records $z_1$")
        ax.scatter(*z1p[j], s=60, marker="*", color=C_OT, zorder=6, label="replaced record")
        n_changed = int(np.sum(pairs_before != pairs_after)) if title.startswith("Batch") else 1
        ax.set_title(f"{title}\n" + ("only the replaced pair changes" if n_changed == 1 else f"{n_changed} pairs change"), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.savefig(OUT / "fig_coupling.png"); plt.close(fig)
    return best[0]


# ---------------- Fig 8: warm-started flow vs marginals only (all-column accuracy)
def fig_flow_value():
    fv = [json.loads(l) for l in (ROOT / "runs/flow_value/results.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, ds, title in ((axes[0], "adult", "Adult"), (axes[1], "cardio", "Cardio")):
        for cfg, lab, c, mk in (("marginals_only_0.9", "marginals only", C_M, "D"),
                                ("warm_0.9_pretrain_only", "warm start, no DP fine-tuning", C_B, "o"),
                                ("warm_0.9", "warm start + DP fine-tuning", C_IC, "s")):
            xs, ys, es = [], [], []
            for te in (1.0, 4.0, 8.0):
                v = [r["allcol_acc"] for r in fv if r["dataset"] == ds and r["config"] == cfg and r["target_eps"] == te]
                if len(v) > 1:
                    xs.append(next(r["epsilon"] for r in fv if r["dataset"] == ds and r["target_eps"] == te))
                    ys.append(np.mean(v)); es.append(np.std(v, ddof=1))
            ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, ms=4, capsize=2, label=lab)
        ref = next(r["allcol_acc_trtr"] for r in fv if r["dataset"] == ds)
        ax.axhline(ref, color="black", ls=":", lw=1)
        lo = ax.get_ylim()[0]
        ax.set_ylim(lo, ref + 0.07 * (ref - lo))
        ax.text(ax.get_xlim()[1], ref, f"real data: {ref:.3f} ", ha="right", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=9); ax.set_xlabel("released $\\varepsilon^\\star$")
    axes[0].set_ylabel("all-column accuracy")
    axes[1].legend(frameon=False, fontsize=7, loc="center right")
    fig.savefig(OUT / "fig8_flow_value.png"); plt.close(fig)


print("coupling toy: pairs changed =", fig_coupling())
for f in (fig_pipeline, fig_sensitivity, fig_audit, fig_split, fig_eps, fig_baselines, fig_dapc, fig_flow_value):
    f()
print(sorted(p.name for p in OUT.glob("*.png")))
