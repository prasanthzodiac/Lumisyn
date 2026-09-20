import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "report" / "coverage_before_raw.txt"
OUT_PATH = REPO_ROOT / "report" / "COVERAGE_BEFORE.md"

def generate_coverage_before():
    if not RAW_PATH.exists():
        print("Raw coverage not found")
        return

    try:
        with open(RAW_PATH, "r", encoding="utf-16") as f:
            content = f.read()
    except Exception:
        with open(RAW_PATH, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    # Find the coverage table
    lines = content.splitlines()
    table_lines = []
    in_table = False
    for line in lines:
        if line.startswith("Name ") and "Stmts" in line:
            in_table = True
        if in_table:
            table_lines.append(line)
            if line.startswith("TOTAL"):
                break

    # Parse entries
    modules = []
    for line in table_lines[2:-1]: # skip header and separator
        parts = line.split()
        if len(parts) >= 4:
            name = parts[0]
            stmts = parts[1]
            miss = parts[2]
            cover = parts[3]
            modules.append((name, int(stmts), int(miss), cover))

    total_line = table_lines[-1].split() if table_lines else []
    total_stmts = total_line[1] if len(total_line) >= 4 else "0"
    total_miss = total_line[2] if len(total_line) >= 4 else "0"
    total_cov = total_line[3] if len(total_line) >= 4 else "0%"

    weak_zero = [m for m in modules if int(m[3].replace("%", "")) < 50]

    md = []
    md.append("# Baseline Test Coverage Report (Before Fixes)")
    md.append("\n**Timestamp**: 2026-09-20T21:16:10+05:30")
    md.append(f"**Overall Project Coverage**: {total_cov} ({total_stmts} statements, {total_miss} missed)\n")
    md.append("## Release-Path Modules with Zero or Weak Coverage (< 50%)\n")
    md.append("| Module | Statements | Missed | Coverage | Stage Association |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")

    stage_map = {
        "acfm_net\\governance": "Stage 0 (CDGL)",
        "acfm_net\\intelligence": "Stage 1 (UDIE)",
        "acfm_net\\preparation": "Stage 2 (ADPE)",
        "acfm_net\\graph": "Stage 3 (DCGB)",
        "acfm_net\\representation": "Stage 4 (HCARLN)",
        "acfm_net\\privacy\\budget_allocator": "Stage 5 (DAPC)",
        "acfm_net\\generative": "Stage 6 & 8 (Flow)",
        "acfm_net\\repair": "Stage 7 (Constraint Repair)",
        "acfm_net\\evaluation": "Stage 9 (Fidelity / Utility)",
        "backend": "Platform API / Service",
    }

    for name, stmts, miss, cov in weak_zero:
        stage = "Other"
        for k, v in stage_map.items():
            if k in name:
                stage = v
                break
        md.append(f"| `{name}` | {stmts} | {miss} | **{cov}** | {stage} |")

    md.append("\n## Complete Module Coverage Table\n")
    md.append("| Module | Stmts | Miss | Cover |")
    md.append("| :--- | :---: | :---: | :---: |")
    for name, stmts, miss, cov in modules:
        md.append(f"| `{name}` | {stmts} | {miss} | {cov} |")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Generated {OUT_PATH} successfully.")

if __name__ == "__main__":
    generate_coverage_before()
