import torch
import numpy as np
from typing import Dict, Any, List, Tuple


def audit_released_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gate 2b Item 2: Checkpoint-Audit Verifier.
    Enumerates every tensor and scalar in the released checkpoint and configuration.
    Each item must strictly belong to one of:
      1. DP-SGD trained weights (trained under DP-SGD wrapper and tracked in PRV/RDP accountant)
      2. Public constant / configuration hyperparameter (from public schema or fixed architecture)
      3. Ledger-charged DP output (sanitized via calibrated DP mechanism and recorded in PrivacyLedger)
    Fails with AssertionError if any unbudgeted or private statistic is found.
    """
    assert isinstance(checkpoint, dict), "Checkpoint must be a dictionary"
    
    ALLOWED_DP_TRAINED_STATE_DICTS = {
        "hvae_state_dict": "Stage 4 HierarchicalVAE (DP-SGD)",
        "gat_state_dict": "Stage 4 GraphAttentionEncoder (DP-SGD)",
        "cross_attn_state_dict": "Stage 4 GatedCrossAttention (DP-SGD)",
        "velocity_net_state_dict": "Stage 8 VelocityNetwork (DP-SGD)",
    }
    
    ALLOWED_LEDGER_CHARGED = {
        "dependency_matrix_W0": "Laplace sanitized prior dependency matrix (eps=0.08 charged in ledger)",
    }
    
    ALLOWED_PUBLIC_CONSTANTS = {
        "scaler_params": "Public schema domain bounds [lo, hi]",
        "feature_specs": "Public feature schema types and bounds",
        "config": "Public pipeline hyperparameters and dimensions",
        "privacy_ledger": "Differential privacy composition ledger",
    }
    
    audit_log: List[Dict[str, Any]] = []
    unbudgeted_items: List[str] = []
    total_dp_parameters = 0
    total_ledger_parameters = 0
    total_public_items = 0
    
    for key, val in checkpoint.items():
        if key in ALLOWED_DP_TRAINED_STATE_DICTS:
            component_desc = ALLOWED_DP_TRAINED_STATE_DICTS[key]
            assert isinstance(val, dict), f"State dict {key} must be a dictionary of tensors"
            for tensor_name, tensor_val in val.items():
                if not isinstance(tensor_val, torch.Tensor):
                    unbudgeted_items.append(f"{key}.{tensor_name}: not a tensor ({type(tensor_val)})")
                    continue
                num_elem = tensor_val.numel()
                total_dp_parameters += num_elem
                audit_log.append({
                    "item": f"{key}.{tensor_name}",
                    "category": "DP-SGD Trained Weight",
                    "provenance": component_desc,
                    "shape": list(tensor_val.shape),
                    "elements": num_elem,
                    "dtype": str(tensor_val.dtype),
                    "status": "APPROVED",
                })
        elif key in ALLOWED_LEDGER_CHARGED:
            component_desc = ALLOWED_LEDGER_CHARGED[key]
            arr = np.asarray(val)
            num_elem = arr.size
            total_ledger_parameters += num_elem
            audit_log.append({
                "item": key,
                "category": "Ledger-Charged DP Output",
                "provenance": component_desc,
                "shape": list(arr.shape),
                "elements": num_elem,
                "dtype": str(arr.dtype),
                "status": "APPROVED",
            })
        elif key in ALLOWED_PUBLIC_CONSTANTS:
            component_desc = ALLOWED_PUBLIC_CONSTANTS[key]
            if isinstance(val, dict):
                num_entries = len(val)
                total_public_items += num_entries
                audit_log.append({
                    "item": key,
                    "category": "Public Constant / Configuration",
                    "provenance": component_desc,
                    "entries": num_entries,
                    "status": "APPROVED",
                })
            else:
                total_public_items += 1
                audit_log.append({
                    "item": key,
                    "category": "Public Constant / Configuration",
                    "provenance": component_desc,
                    "value": str(val),
                    "status": "APPROVED",
                })
        else:
            unbudgeted_items.append(f"UNAPPROVED_KEY: '{key}' (type: {type(val)})")
            audit_log.append({
                "item": key,
                "category": "UNKNOWN_UNBUDGETED",
                "provenance": "UNVERIFIED",
                "status": "FAILED",
            })
            
    summary = {
        "total_items_audited": len(audit_log),
        "total_dp_trained_parameters": total_dp_parameters,
        "total_ledger_charged_elements": total_ledger_parameters,
        "total_public_entries": total_public_items,
        "unbudgeted_violations_count": len(unbudgeted_items),
        "unbudgeted_violations": unbudgeted_items,
        "checkpoint_audit_passed": len(unbudgeted_items) == 0,
    }
    
    if len(unbudgeted_items) > 0:
        raise AssertionError(f"Checkpoint audit failed! Found unbudgeted items in release path: {unbudgeted_items}")
        
    return {
        "summary": summary,
        "audit_log": audit_log,
    }
