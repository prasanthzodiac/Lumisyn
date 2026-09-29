"""
PDF report generator for ACFM-Net v5 Evaluation.
Uses fpdf2 (pip install fpdf2). Falls back to JSON if fpdf2 is unavailable.
"""
from __future__ import annotations
import io
import json
import datetime
from typing import Dict, Any, Optional
import numpy as np

try:
    from fpdf import FPDF
    FPDF2_AVAILABLE = True
except ImportError:
    FPDF2_AVAILABLE = False


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


class EvaluationReportBuilder:
    """
    Builds a structured PDF (or JSON fallback) report from the evaluation results dict.

    Usage:
        builder = EvaluationReportBuilder()
        pdf_bytes = builder.build_pdf(eval_results, dataset_name="pima_diabetes")
    """

    def build_pdf(
        self,
        eval_results: Dict[str, Any],
        dataset_name: str = "Dataset",
        n_real: int = 0,
        n_syn: int = 0,
        eps_used: float = 0.0,
        target_epsilon: float = 1.0,
        privacy_verified: bool = False,
    ) -> bytes:
        """
        Generate a PDF report and return as bytes.
        Falls back to a JSON-wrapped bytes payload if fpdf2 is not installed.
        """
        if not FPDF2_AVAILABLE:
            def _json_default(obj):
                if isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                if isinstance(obj, (np.integer, int)):
                    return int(obj)
                if isinstance(obj, (np.floating, float)):
                    return float(obj)
                return str(obj)
            return json.dumps(eval_results, indent=2, default=_json_default).encode("utf-8")

        pdf = FPDF()
        import os
        import matplotlib
        font_dir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        # Unicode font: core Helvetica cannot encode '—' or 'ε' and made this endpoint return 500.
        pdf.add_font("DejaVu", "", os.path.join(font_dir, "DejaVuSans.ttf"))
        pdf.add_font("DejaVu", "B", os.path.join(font_dir, "DejaVuSans-Bold.ttf"))
        pdf.add_font("DejaVu", "I", os.path.join(font_dir, "DejaVuSans-Oblique.ttf"))
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # --- Title ---
        pdf.set_font("DejaVu", "B", 18)
        pdf.cell(0, 12, "ACFM-Net v5 — Synthetic Data Evaluation Report", ln=True, align="C")

        pdf.set_font("DejaVu", "", 10)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 8, f"Generated: {ts}   |   Dataset: {dataset_name}", ln=True, align="C")
        pdf.ln(4)

        # --- Summary banner ---
        self._section(pdf, "Dataset Overview")
        self._row(pdf, "Real samples", str(n_real))
        self._row(pdf, "Synthetic samples", str(n_syn))
        self._row(pdf, "Differential Privacy eps used", _fmt(eps_used))
        dp_status = "VERIFIED" if privacy_verified and eps_used <= target_epsilon else "UNVERIFIED"
        self._row(pdf, "DP Budget status", dp_status)
        pdf.ln(4)

        # --- Statistical ---
        self._section(pdf, "Statistical Fidelity")
        ks = eval_results.get("statistical", {}).get("ks_test", {})
        ks_summary = ks.get("_summary", {})
        self._row(pdf, "Mean KS Statistic",   _fmt(ks_summary.get("mean_ks_stat")))
        self._row(pdf, "KS Cols Passed p>0.05", str(ks_summary.get("columns_passed_005", "N/A")))
        ws = eval_results.get("statistical", {}).get("wasserstein", {}).get("_summary", {})
        self._row(pdf, "Mean Wasserstein (normalised)", _fmt(ws.get("mean_wasserstein")))
        corr = eval_results.get("statistical", {}).get("correlation", {})
        self._row(pdf, "Correlation Frobenius Norm",  _fmt(corr.get("frobenius_norm")))
        self._row(pdf, "Correlation Similarity Score",_fmt(corr.get("frobenius_similarity")))
        jsd = eval_results.get("statistical", {}).get("jsd", {}).get("_summary", {})
        self._row(pdf, "Mean Jensen-Shannon Div.",    _fmt(jsd.get("mean_jsd")))
        pdf.ln(4)

        # --- Utility ---
        self._section(pdf, "Utility Evaluation (TSTR / TRTS)")
        tstr = eval_results.get("utility", {}).get("tstr", {})
        trts = eval_results.get("utility", {}).get("trts", {})
        self._row(pdf, "TSTR Accuracy",    _fmt(tstr.get("accuracy")))
        self._row(pdf, "TSTR F1 Weighted", _fmt(tstr.get("f1_weighted")))
        self._row(pdf, "TRTS Accuracy",    _fmt(trts.get("accuracy")))
        self._row(pdf, "TRTS F1 Weighted", _fmt(trts.get("f1_weighted")))
        pdf.ln(4)

        # --- Privacy ---
        self._section(pdf, "Privacy Evaluation")
        mia = eval_results.get("privacy", {}).get("mia", {})
        self._row(pdf, "C2ST AUC",         _fmt(mia.get("c2st_auc", mia.get("mia_auc"))))
        self._row(pdf, "Indistinguishability",  str(mia.get("indistinguishability", mia.get("privacy_risk_level", "N/A"))))
        attr = eval_results.get("privacy", {}).get("attribute_ia", {})
        self._row(pdf, "Attr-IA Attack Acc",  _fmt(attr.get("attack_accuracy")))
        self._row(pdf, "Attr-IA Risk Level",  str(attr.get("risk_level", "N/A")))
        pdf.ln(4)

        # --- Clinical ---
        self._section(pdf, "Clinical Constraint Audit")
        clin_summary = eval_results.get("clinical", {}).get("summary", {})
        self._row(pdf, "Constraints Checked",     str(clin_summary.get("total_constraints_checked", "N/A")))
        self._row(pdf, "Constraints Passed",      str(clin_summary.get("total_passed", "N/A")))
        self._row(pdf, "Satisfaction Rate",       _fmt(clin_summary.get("constraint_satisfaction_rate")))
        self._row(pdf, "Fully Compliant",         str(clin_summary.get("fully_compliant", "N/A")))
        pdf.ln(4)

        # --- Fairness ---
        self._section(pdf, "Fairness Evaluation")
        fair = eval_results.get("fairness", {})
        if "sensitive_attributes" in fair:
            for s_col, s_data in fair["sensitive_attributes"].items():
                syn_gap  = s_data.get("synthetic", {}).get("demographic_parity_gap")
                real_gap = s_data.get("real", {}).get("demographic_parity_gap")
                self._row(pdf, f"{s_col} DP gap (syn)",  _fmt(syn_gap))
                self._row(pdf, f"{s_col} DP gap (real)", _fmt(real_gap))
        else:
            self._row(pdf, "Status", fair.get("message", "N/A"))
        pdf.ln(4)

        # --- Footer ---
        pdf.set_font("DejaVu", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6,
                 "ACFM-Net v5 | Adaptive Constraint-Guided Flow Matching | Confidential Research Output",
                 ln=True, align="C")

        return bytes(pdf.output())

    # --- helpers ---
    def _section(self, pdf: "FPDF", title: str):
        pdf.set_font("DejaVu", "B", 12)
        pdf.set_fill_color(30, 80, 160)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, f"  {title}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("DejaVu", "", 10)

    def _row(self, pdf: "FPDF", label: str, value: str):
        pdf.set_font("DejaVu", "B", 9)
        pdf.cell(80, 7, f"  {label}:", border=0)
        pdf.set_font("DejaVu", "", 9)
        pdf.cell(0,  7, value, border=0, ln=True)
