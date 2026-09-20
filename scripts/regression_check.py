import os
import sys
import numpy as np
import pandas as pd
import math

sys.path.insert(0, os.path.abspath("lumisyn/framework"))

from acfm_net.privacy.prv_accountant import PRVAccountant
from acfm_net.privacy.rdp_accountant import RDPAccountant

def run_regression_check():
    print("================================================================================")
    print(" RUNNING MANDATORY REGRESSION CHECK AGAINST PREVIOUSLY LOGGED VALUES")
    print("================================================================================")

    # 1. RDP and PRV Accountant Check (from runs/gate1b_test_run.log)
    # Logged values:
    #   [Test 4c(i) Report] Exact RDP epsilon: 1.45136809
    #   [Test 4c(i) Report] Exact PRV epsilon: 1.07819611
    rdp = RDPAccountant(target_delta=1e-5)
    eps_rdp = rdp.compute_epsilon(noise_multiplier=1.0, sample_rate=0.01, steps=300)

    prv = PRVAccountant(target_delta=1e-5)
    eps_prv = prv.compute_epsilon(noise_multiplier=1.0, sample_rate=0.01, steps=300)

    logged_rdp = 1.45136809
    logged_prv = 1.07819611

    diff_rdp = abs(eps_rdp - logged_rdp)
    diff_prv = abs(eps_prv - logged_prv)

    print(f"Accountant RDP eps: current={eps_rdp:.8f} | logged={logged_rdp:.8f} | diff={diff_rdp:.2e}")
    print(f"Accountant PRV eps: current={eps_prv:.8f} | logged={logged_prv:.8f} | diff={diff_prv:.2e}")

    assert diff_rdp < 1e-5, f"RDP Accountant regression! diff={diff_rdp}"
    assert diff_prv < 1e-5, f"PRV Accountant regression! diff={diff_prv}"

    print("REGRESSION CHECK PASSED: All rerun values match logged values within floating-point tolerance.")

if __name__ == "__main__":
    run_regression_check()
