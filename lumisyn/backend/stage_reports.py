"""
Per-stage PDF reports for the Lumisyn demo.

    build_stage_pdf(session, stage_id)  -> bytes   one stage: purpose, status, measured evidence, chart
    build_all_stages_pdf(session)       -> bytes   cover page + every stage of the latest run

Everything printed comes from the pipeline report of the latest run (session["pipeline_report"]) and the
released synthetic table; raw private rows are only used for the real-vs-synthetic chart of Stage 9,
which the app already shows on screen.
"""

import datetime
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")

STAGE_INFO = {
    "0": ("Governance and de-identification", "Screens column names for direct identifiers, removes or hashes them, and issues an audit handle. Keyword-based; not a legal determination."),
    "1": ("Dataset intelligence", "Infers the task type and structural profile from a bounded sample under the strict public-schema contract."),
    "2": ("Data preparation", "Imputes, encodes and scales every column with the declared public schema only (row-local map), so no private statistic enters the encoder."),
    "3": ("Domain constraint graph", "Builds the clinical dependency graph from column names and a public ontology; the private dependency matrix is disabled in strict mode."),
    "3b": ("Post-latent graph refinement", "Refines the graph topology; in strict mode it reads no data."),
    "5": ("Dataset-adaptive privacy controller (DAPC)", "Chooses the released budget from public metadata only (column names, rounded cohort size, encoded dimension) and calibrates the DP-SGD noise with the PRV accountant."),
    "4": ("Representation learning", "Optional hierarchical VAE. In identity mode the flow trains directly on the scaled records and this stage is skipped."),
    "6": ("Flow matching generator", "Initialises the velocity network for independent-coupling conditional flow matching (I-CFM), the coupling that keeps per-example sensitivity bounded. With the warm start, the network is first pretrained on records drawn from the noisy Stage 7c tables."),
    "8": ("DP flow training", "Trains the velocity network with per-example clipping and Gaussian noise (DP-SGD); the PRV accountant tracks the privacy loss after every epoch. When Stage 7c is enabled, its marginal release is charged first, so the accumulated ε starts above zero."),
    "7": ("Constrained repair", "Enforces declared bounds, integer and category types, and public clinical rules such as systolic >= diastolic + 10 on the decoded rows (post-processing)."),
    "7c": ("DP class-conditional marginals", "Releases noisy label × feature count tables (Gaussian mechanism, jointly accounted with DP-SGD) and maps each class of the synthetic table onto them."),
    "7b": ("DP marginal recalibration", "Releases noisy one-way histograms (Laplace) and recalibrates each synthetic column to them."),
    "9": ("Synthetic data output", "Hands the released synthetic table to evaluation."),
}


def _pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DejaVu", "", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVu", "I", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf"))
    return pdf


def _header(pdf, title, subtitle):
    pdf.set_fill_color(79, 70, 229)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(12, 7)
    pdf.set_font("DejaVu", "B", 15)
    pdf.cell(0, 8, title)
    pdf.set_xy(12, 16)
    pdf.set_font("DejaVu", "", 9)
    pdf.cell(0, 6, subtitle)
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(12, 34)


def _section(pdf, title):
    pdf.ln(2)
    pdf.set_font("DejaVu", "B", 11)
    pdf.set_text_color(55, 48, 163)
    pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(199, 210, 254)
    pdf.line(pdf.l_margin, pdf.get_y(), 210 - pdf.r_margin, pdf.get_y())
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _para(pdf, text, size=9):
    pdf.set_font("DejaVu", "", size)
    pdf.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")


def _fmt(v):
    if v is None:
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        s = ", ".join(_fmt(x) for x in v[:12])
        return s + (" …" if len(v) > 12 else "")
    return str(v)


def _flatten(d, prefix=""):
    rows = []
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            rows += _flatten(v, key + " › ")
        else:
            rows.append((key.replace("_", " "), _fmt(v)))
    return rows[:120]


def _table(pdf, rows, widths=(78, 108), header=("Measure", "Value")):
    pdf.set_font("DejaVu", "B", 8.5)
    pdf.set_fill_color(238, 242, 255)
    for w, h in zip(widths, header):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()
    pdf.set_font("DejaVu", "", 8)
    for i, row in enumerate(rows):
        lines = max(len(pdf.multi_cell(w, 5, str(t), dry_run=True, output="LINES")) for w, t in zip(widths, row))
        h = 5 * max(1, lines)
        if pdf.get_y() + h > 280:
            pdf.add_page()
        x0, y0 = pdf.l_margin, pdf.get_y()
        pdf.set_fill_color(249, 250, 251) if i % 2 else pdf.set_fill_color(255, 255, 255)
        x = x0
        for w, t in zip(widths, row):
            pdf.rect(x, y0, w, h, style="DF")
            pdf.set_xy(x, y0)
            pdf.multi_cell(w, 5, str(t))
            x += w
        pdf.set_xy(x0, y0 + h)
    pdf.ln(2)


def _image(pdf, fig, width=186):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if pdf.get_y() > 190:
        pdf.add_page()
    pdf.image(path, w=width)
    os.unlink(path)
    pdf.ln(2)


def _chart_for(stage_id, session):
    """Stage-specific figure from measured values, or None."""
    rep = session.get("pipeline_report") or {}
    if stage_id == "8" and rep.get("flow_stability_log"):
        log = rep["flow_stability_log"]
        ep = [r["epoch"] for r in log]
        fig, ax = plt.subplots(1, 2, figsize=(8, 2.6))
        ax[0].plot(ep, [r["loss"] for r in log], "o-", color="#4f46e5")
        ax[0].axhline(2.0, ls="--", color="grey", lw=0.8)
        ax[0].set(title="Flow loss per epoch (dashed: zero predictor)", xlabel="epoch")
        ax[1].plot(ep, [r["accumulated_epsilon"] for r in log], "o-", color="#059669")
        ax[1].set(title="Accumulated ε (PRV accountant)", xlabel="epoch")
        return fig
    if stage_id in {"9", "7", "7c"} and session.get("df_syn") is not None:
        real, syn = session["df_real"], session["df_syn"]
        num = [c for c in syn.columns if pd.api.types.is_numeric_dtype(syn[c]) and c in real.columns and syn[c].nunique() > 2][:4]
        if not num:
            return None
        fig, axes = plt.subplots(1, len(num), figsize=(2.3 * len(num), 2.3))
        for ax, c in zip(np.atleast_1d(axes), num):
            r, s = pd.to_numeric(real[c], errors="coerce").dropna(), pd.to_numeric(syn[c], errors="coerce").dropna()
            bins = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 20)
            ax.hist(r, bins=bins, alpha=0.55, density=True, label="real", color="#94a3b8")
            ax.hist(s, bins=bins, alpha=0.55, density=True, label="synthetic", color="#4f46e5")
            ax.set_title(c, fontsize=8)
            ax.tick_params(labelsize=6)
        np.atleast_1d(axes)[0].legend(fontsize=6)
        fig.suptitle("Real vs synthetic distributions (released table)", fontsize=9, y=1.06)
        return fig
    return None


def _meta_rows(session):
    rep = session.get("pipeline_report") or {}
    pt = rep.get("privacy_training") or {}
    return [
        ("Dataset", session.get("filename")),
        ("Rows × columns", f"{len(session['df_real'])} × {len(session['df_real'].columns)}"),
        ("Target column", session.get("target_col")),
        ("Schema source", session.get("schema_source", "–")),
        ("Pipeline status", rep.get("pipeline_status")),
        ("Privacy status", rep.get("privacy_status")),
        ("Accounted ε", _fmt(pt.get("accounted_epsilon", pt.get("epsilon")))),
        ("Generation runtime (s)", _fmt(rep.get("generation_runtime_seconds"))),
        ("Report generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]


def _stage_block(pdf, session, rec):
    sid = str(rec["stage"])
    title, purpose = STAGE_INFO.get(sid, (rec.get("name", sid), ""))
    _section(pdf, f"Stage {sid}: {title}")
    status = rec.get("status", "–")
    colour = {"PASSED": (5, 150, 105), "READY": (5, 150, 105), "SKIPPED": (100, 116, 139)}.get(status, (185, 28, 28))
    pdf.set_font("DejaVu", "B", 9)
    pdf.set_text_color(*colour)
    pdf.cell(0, 6, f"Status: {status}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    if purpose:
        _para(pdf, purpose)
        pdf.ln(1)
    rows = _flatten(rec.get("evidence") or {})
    if rows:
        _table(pdf, rows)
    fig = _chart_for(sid, session)
    if fig is not None:
        _image(pdf, fig)


def _records(session):
    return (session.get("pipeline_report") or {}).get("stage_reports") or []


def build_stage_pdf(session, stage_id: str) -> bytes:
    rec = next((r for r in _records(session) if str(r["stage"]) == str(stage_id)), None)
    if rec is None:
        raise KeyError(stage_id)
    pdf = _pdf()
    pdf.add_page()
    title = STAGE_INFO.get(str(stage_id), (rec.get("name"), ""))[0]
    _header(pdf, f"Lumisyn · Stage {stage_id} Report", f"{title}  ·  {session.get('filename')}")
    _section(pdf, "Run")
    _table(pdf, [(k, _fmt(v)) for k, v in _meta_rows(session)])
    _stage_block(pdf, session, rec)
    return bytes(pdf.output())


def build_all_stages_pdf(session) -> bytes:
    recs = _records(session)
    if not recs:
        raise KeyError("no run")
    pdf = _pdf()
    pdf.add_page()
    _header(pdf, "Lumisyn · Pipeline Stage Reports", f"{session.get('filename')}  ·  {len(recs)} measured stages")
    _section(pdf, "Run")
    _table(pdf, [(k, _fmt(v)) for k, v in _meta_rows(session)])
    _section(pdf, "Stage summary")
    _table(pdf, [(f"Stage {r['stage']}: {STAGE_INFO.get(str(r['stage']), (r.get('name'), ''))[0]}", r.get("status")) for r in recs],
           header=("Stage", "Status"))
    for rec in recs:
        pdf.add_page()
        _stage_block(pdf, session, rec)
    return bytes(pdf.output())


if __name__ == "__main__":  # self-check on a tiny fake session
    df = pd.DataFrame({"systolic_bp": np.random.normal(120, 10, 200), "diastolic_bp": np.random.normal(75, 8, 200), "y": np.random.randint(0, 2, 200)})
    fake = {"filename": "demo.csv", "df_real": df, "df_syn": df.sample(frac=1.0), "target_col": "y",
            "pipeline_report": {"pipeline_status": "SUCCESS", "privacy_training": {"accounted_epsilon": 0.78},
                                "flow_stability_log": [{"epoch": e, "loss": 1.2 - 0.01 * e, "accumulated_epsilon": 0.1 * e} for e in range(1, 6)],
                                "stage_reports": [{"stage": "8", "name": "DP flow training", "status": "PASSED", "evidence": {"epochs": 5, "nested": {"a": 1.23456, "b": [1, 2]}, "ok": np.bool_(True)}},
                                                  {"stage": "9", "name": "out", "status": "READY", "evidence": {"generated_rows": 200}}]}}
    assert build_stage_pdf(fake, "8")[:4] == b"%PDF"
    assert build_all_stages_pdf(fake)[:4] == b"%PDF"
    print("stage_reports self-check ok")
