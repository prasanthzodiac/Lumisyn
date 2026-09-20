import os
import sys
import time
import datetime
import io
import subprocess
import torch

def run_baseline():
    os.makedirs("runs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join("runs", "baseline_test_run.log")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    # Set PYTHONPATH
    env = os.environ.copy()
    framework_dir = os.path.abspath("lumisyn/framework")
    lumisyn_dir = os.path.abspath("lumisyn")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{framework_dir}{os.pathsep}{lumisyn_dir}{os.pathsep}{existing_pp}"

    cmd = [sys.executable, "-m", "pytest", "-v", "lumisyn/backend/tests"]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True
    )
    elapsed = time.time() - t0

    # Also run a quick VRAM measurement test
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    header = f"""======================================================================
LUMISYN BASELINE TEST SUITE EXECUTION LOG
======================================================================
Timestamp: {timestamp}
Device: {device_name}
PyTorch Version: {torch.__version__}
Command: {" ".join(cmd)}
PYTHONPATH: {env["PYTHONPATH"]}
Return Code: {proc.returncode}
Elapsed Time: {elapsed:.2f} seconds
Initial Peak VRAM: {peak_vram_mb:.2f} MB
======================================================================
RAW PYTEST STDOUT:
======================================================================
{proc.stdout}
======================================================================
RAW PYTEST STDERR:
======================================================================
{proc.stderr}
======================================================================
"""
    print(header)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(header)

if __name__ == "__main__":
    run_baseline()
