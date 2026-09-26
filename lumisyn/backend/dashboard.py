"""
L-OTFM Terminal Dashboard — run and inspect the ACFM-Net/L-OTFM privacy-preserving
synthetic data framework entirely from the terminal, no frontend/backend API required.

Usage:
    python dashboard.py                       interactive menu
    python dashboard.py --dataset pima         run one known dataset non-interactively
    python dashboard.py --csv path/to.csv --target col --task classification
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt
from rich import box

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "framework"))
sys.path.insert(0, str(REPO_ROOT / "lumisyn" / "backend"))

from acfm_net.pipeline import ACFMNetV5Pipeline
from schema_registry import KNOWN_DATASETS, infer_schema_for_dataframe
from eval_utils import evaluate_tstr, compute_anti_memorization_dcr, correlation_frobenius_error

# legacy_windows=False: bypass the WinAPI character-by-character renderer, which
# can't encode Rich's default unicode glyphs on a cp1252 console — write plain
# ANSI-escaped UTF-8 instead (works with the utf-8 stdout reconfigured above).
console = Console(legacy_windows=False)


# --------------------------------------------------------------------------------
# Data loading & schema resolution
# --------------------------------------------------------------------------------

def load_known_dataset(key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ds = next((d for d in KNOWN_DATASETS if d["key"] == key), None)
    if ds is None:
        raise ValueError(f"Unknown dataset key: {key}")
    df = pd.read_csv(ds["file"])
    df = df.drop(columns=[c for c in ds["drop_cols"] if c in df.columns])
    if ds["sample_n"] is not None and len(df) > ds["sample_n"]:
        df = df.sample(n=ds["sample_n"], random_state=42).reset_index(drop=True)
    schema = ds["schema_fn"](df)
    return df, {"name": ds["name"], "target": ds["target"], "task": ds["task"], "schema": schema}


def infer_task(df: pd.DataFrame, target_col: str) -> str:
    target = df[target_col].dropna()
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    if target.nunique() <= 20 and (target.values == target.values.astype(int)).all():
        return "classification"
    return "regression"


def show_schema_table(schema: Dict[str, Any], notes: Optional[list] = None) -> None:
    table = Table(title="Declared Column Schema", box=box.SIMPLE_HEAVY)
    table.add_column("Column", style="cyan")
    table.add_column("Type")
    table.add_column("Bounds / Categories")
    table.add_column("Kind")
    for col, spec in schema.items():
        if spec["type"] == "numeric":
            table.add_row(col, "numeric", f"[{spec['bounds'][0]:g}, {spec['bounds'][1]:g}]", spec.get("kind", "continuous"))
        else:
            cats = spec.get("categories", [])
            table.add_row(col, "categorical", ", ".join(cats[:5]) + (" ..." if len(cats) > 5 else ""), "-")
    console.print(table)
    if notes:
        console.print(Panel(
            "\n".join(notes),
            title="[yellow]UNVERIFIED bounds — auto-inferred from this sample, review before trusting DP/utility numbers[/yellow]",
            border_style="yellow",
        ))


# --------------------------------------------------------------------------------
# Pipeline run + report rendering
# --------------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    schema: Dict[str, Any],
    epsilon: float,
    epochs: int,
    enable_recalibration: bool,
    protocol_id: str,
) -> Dict[str, Any]:
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    pipeline = ACFMNetV5Pipeline(
        target_epsilon=epsilon,
        target_delta=1e-5,
        latent_mode="identity",
        enable_dp_recalibration=enable_recalibration,
        recalibration_epsilon=0.10,
        public_schema=schema,
    )

    console.print(Rule("[bold]PIPELINE EXECUTION LOG[/bold]", characters="-"))
    t0 = time.time()
    df_synthetic, report = pipeline.fit_generate(
        df_raw=df_train,
        num_samples=len(df_train),
        target_col=target_col,
        epochs=epochs,
        protocol_id=protocol_id,
    )
    elapsed = time.time() - t0
    console.print(Rule(characters="-"))

    mean_dcr, exact_match_pct = compute_anti_memorization_dcr(df_train, df_synthetic)
    utility = evaluate_tstr(df_train, df_test, df_synthetic, target_col, task)
    corr_err = correlation_frobenius_error(df_train, df_synthetic)

    return {
        "df_synthetic": df_synthetic,
        "report": report,
        "elapsed": elapsed,
        "mean_dcr": mean_dcr,
        "exact_match_pct": exact_match_pct,
        "utility": utility,
        "corr_err": corr_err,
    }


def render_stage_table(report: Dict[str, Any]) -> None:
    table = Table(title="Stage Verification", box=box.SIMPLE_HEAVY)
    table.add_column("Stage")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Key Evidence")
    status_style = {"PASSED": "green", "SKIPPED": "yellow", "READY": "green", "INITIALIZED": "cyan", "UNVERIFIED": "red"}
    for stage in report.get("stage_reports", []):
        status = stage.get("status", "?")
        style = status_style.get(status, "white")
        evidence = stage.get("evidence", {})
        scalar_bits = [(k, v) for k, v in evidence.items() if isinstance(v, (str, int, float, bool)) or v is None]
        evidence_str = ", ".join(f"{k}={v}" for k, v in scalar_bits[:3])
        table.add_row(stage.get("stage", "?"), stage.get("name", "?"), f"[{style}]{status}[/{style}]", evidence_str)
    console.print(table)


def render_privacy_panel(report: Dict[str, Any]) -> None:
    summary = report.get("privacy_summary", {})
    verified = report.get("release_privacy_verified", False)
    verified_str = "[green]VERIFIED[/green]" if verified else "[red]UNVERIFIED[/red]"
    body = (
        f"Headline (Opacus PRV, with Laplace charges): [bold]{summary.get('prv_upper_with_laplace', 'n/a')}[/bold]\n"
        f"Formal RDP (with Laplace charges):            {summary.get('rdp_with_laplace', 'n/a')}\n"
        f"Target epsilon:                                {report.get('target_epsilon_adaptive', 'n/a')}\n"
        f"Release privacy status:                        {verified_str}"
    )
    console.print(Panel(body, title="Differential Privacy Summary", border_style="blue"))


def render_results(result: Dict[str, Any]) -> None:
    report = result["report"]
    render_stage_table(report)
    render_privacy_panel(report)

    utility = result["utility"]
    metrics = Table(title="Downstream Utility & Fidelity", box=box.SIMPLE_HEAVY)
    metrics.add_column("Metric")
    metrics.add_column("Value")
    metrics.add_row("TSTR (trained on synthetic, tested on real)", utility["tstr_metric"])
    metrics.add_row("TRTR (trained on real, tested on real)", utility["trtr_metric"])
    metrics.add_row("Utility Retention", utility["retention"])
    metrics.add_row("Correlation Frobenius Error", f"{result['corr_err']:.4f}")
    metrics.add_row("Mean DCR (distance to closest real record)", f"{result['mean_dcr']:.3f}")
    metrics.add_row("Exact Copy Rate", f"{result['exact_match_pct']:.2f}%")
    metrics.add_row("Wall-clock time", f"{result['elapsed']:.1f}s")
    console.print(metrics)

    df_synthetic = result["df_synthetic"]
    preview = Table(title=f"Synthetic Data Preview (first 8 of {len(df_synthetic)} rows)", box=box.SIMPLE_HEAVY)
    for col in df_synthetic.columns:
        preview.add_column(str(col))
    for _, row in df_synthetic.head(8).iterrows():
        preview.add_row(*[f"{v:.3g}" if isinstance(v, float) else str(v) for v in row])
    console.print(preview)


# --------------------------------------------------------------------------------
# Interactive flow
# --------------------------------------------------------------------------------

def interactive_menu() -> None:
    console.print(Panel.fit(
        "[bold cyan]L-OTFM Terminal Dashboard[/bold cyan]\n"
        "Differentially-private synthetic tabular data — framework-only, no frontend needed",
        border_style="cyan",
    ))

    while True:
        console.print("\n[bold]Choose a dataset:[/bold]")
        for i, ds in enumerate(KNOWN_DATASETS, 1):
            console.print(f"  {i}. {ds['name']}")
        console.print(f"  {len(KNOWN_DATASETS) + 1}. Load a custom CSV")
        console.print("  q. Quit")

        choice = Prompt.ask("Selection", default="q")
        if choice.lower() == "q":
            break

        try:
            idx = int(choice)
        except ValueError:
            console.print("[red]Invalid choice.[/red]")
            continue

        if 1 <= idx <= len(KNOWN_DATASETS):
            ds = KNOWN_DATASETS[idx - 1]
            df, meta = load_known_dataset(ds["key"])
            target_col, task, schema = meta["target"], meta["task"], meta["schema"]
            notes = None
            protocol_prefix = ds["key"]
        elif idx == len(KNOWN_DATASETS) + 1:
            path = Prompt.ask("Path to CSV file")
            if not Path(path).exists():
                console.print(f"[red]File not found: {path}[/red]")
                continue
            df = pd.read_csv(path)
            console.print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns. Columns: {', '.join(df.columns)}")
            target_col = Prompt.ask("Target column", default=df.columns[-1])
            if target_col not in df.columns:
                console.print(f"[red]Column not found: {target_col}[/red]")
                continue
            task = Prompt.ask("Task", choices=["classification", "regression"], default=infer_task(df, target_col))
            schema, notes = infer_schema_for_dataframe(df)
            protocol_prefix = "custom"
        else:
            console.print("[red]Invalid choice.[/red]")
            continue

        show_schema_table(schema, notes)
        if notes and not Confirm.ask("Proceed with these auto-inferred bounds?", default=True):
            continue

        epsilon = FloatPrompt.ask("Target epsilon", default=1.0)
        epochs = IntPrompt.ask("Training epochs", default=10)
        recalibrate = Confirm.ask("Enable Stage 7b DP marginal recalibration?", default=True)

        result = run_pipeline(
            df=df,
            target_col=target_col,
            task=task,
            schema=schema,
            epsilon=epsilon,
            epochs=epochs,
            enable_recalibration=recalibrate,
            protocol_id=f"DASHBOARD-{protocol_prefix}-{int(time.time())}",
        )
        render_results(result)

        if Confirm.ask("Save synthetic CSV to lumisyn/artifacts/dashboard_runs/?", default=False):
            out_dir = REPO_ROOT / "lumisyn" / "artifacts" / "dashboard_runs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{protocol_prefix}_{int(time.time())}.csv"
            result["df_synthetic"].to_csv(out_path, index=False)
            console.print(f"[green]Saved to {out_path}[/green]")


def non_interactive(args: argparse.Namespace) -> None:
    if args.dataset:
        df, meta = load_known_dataset(args.dataset)
        target_col, task, schema, notes = meta["target"], meta["task"], meta["schema"], None
        protocol_prefix = args.dataset
    elif args.csv:
        df = pd.read_csv(args.csv)
        target_col = args.target or df.columns[-1]
        task = args.task or infer_task(df, target_col)
        schema, notes = infer_schema_for_dataframe(df)
        protocol_prefix = "custom"
    else:
        console.print("[red]Provide --dataset <key> or --csv <path>.[/red]")
        sys.exit(1)

    show_schema_table(schema, notes)
    result = run_pipeline(
        df=df,
        target_col=target_col,
        task=task,
        schema=schema,
        epsilon=args.epsilon,
        epochs=args.epochs,
        enable_recalibration=not args.no_recalibration,
        protocol_id=f"DASHBOARD-{protocol_prefix}-{int(time.time())}",
    )
    render_results(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="L-OTFM terminal dashboard")
    parser.add_argument("--dataset", choices=[d["key"] for d in KNOWN_DATASETS], help="Run a known dataset non-interactively")
    parser.add_argument("--csv", help="Path to a custom CSV to run non-interactively")
    parser.add_argument("--target", help="Target column (custom CSV only)")
    parser.add_argument("--task", choices=["classification", "regression"], help="Task type (custom CSV only)")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--no-recalibration", action="store_true", help="Disable Stage 7b DP marginal recalibration")
    args = parser.parse_args()

    if args.dataset or args.csv:
        non_interactive(args)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
