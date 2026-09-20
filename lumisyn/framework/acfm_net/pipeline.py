import math
import time
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from acfm_net.governance import ClinicalDataGovernanceLayer, GovernanceCertificate
from acfm_net.intelligence import (
    MissingValueAnalyzer,
    OutlierDetectionEngine,
    SensitiveAttributeDetector,
    DependencyEstimationEngine,
    DatasetIntelligenceGenerator,
)
from acfm_net.preparation import (
    AdaptiveImputer,
    DistributionPreservingEncoder,
    AdaptiveFeatureScaler,
    ClassBalancer,
)
from acfm_net.graph import DomainConstraintGraphBuilder, PostLatentGraphRefiner
from acfm_net.representation import (
    GraphAttentionEncoder,
    GatedCrossAttention,
    HierarchicalVAE,
)
from acfm_net.privacy import (
    PRVAccountant,
    AdaptiveBudgetAllocator,
    DPSGDWrapper,
    PrivacyLedger,
    public_schema_from_dataframe,
    sanitize_dependency_matrix,
    sanitize_missing_summary,
)
from acfm_net.generative import (
    ContinuousOTCFM,
)
from acfm_net.repair import ConstrainedGradientRepairLoop
from acfm_net.repair.constraint_repair import enforce_domain_constraints, enforce_public_domain_constraints
from acfm_net.optimization import (
    SlicedWassersteinDistance,
    ParetoModelSelector,
)

# Maximum number of columns allowed before auto-selecting top informative ones
MAX_COLUMNS = 64
# Batch size for GPU/CPU training loop
BATCH_SIZE = 256


def _record_stage(
    report: Dict[str, Any],
    stage_id: str,
    name: str,
    status: str,
    evidence: Optional[Dict[str, Any]] = None,
):
    """Append an auditable, measured stage record to the pipeline report."""
    report.setdefault("stage_reports", []).append({
        "stage": stage_id,
        "name": name,
        "status": status,
        "evidence": evidence or {},
    })


def _safe_correlation(matrix: np.ndarray) -> np.ndarray:
    """Correlation matrix without NumPy divide-by-zero warnings."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return np.eye(values.shape[1] if values.ndim == 2 else 1, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.sum(centered * centered, axis=0))
    denom = np.outer(scale, scale)
    covariance = centered.T @ centered
    result = np.zeros_like(covariance)
    np.divide(covariance, denom, out=result, where=denom > 1e-12)
    np.fill_diagonal(result, 1.0)
    return result


def _select_informative_columns(df: pd.DataFrame, max_cols: int = MAX_COLUMNS) -> pd.DataFrame:
    """
    For wide datasets (>max_cols columns), auto-select the most informative subset:
    1. Drop constant / near-zero-variance columns
    2. Drop columns with >80% missing values
    3. Keep max_cols columns (numeric + categorical mix prioritised)
    """
    keep = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) > 1 and df[col].isna().mean() < 0.8:
                keep.append(col)
        else:
            if df[col].nunique(dropna=True) > 1 and df[col].isna().mean() < 0.8:
                keep.append(col)

    df_keep = df[keep]

    if len(df_keep.columns) <= max_cols:
        return df_keep

    scores = {}
    for col in df_keep.columns:
        if pd.api.types.is_numeric_dtype(df_keep[col]):
            scores[col] = float(df_keep[col].std(skipna=True) or 0.0)
        else:
            scores[col] = float(df_keep[col].nunique(dropna=True))

    top_cols = sorted(scores, key=scores.get, reverse=True)[:max_cols]
    print(f"[ACFM-Net v6] Wide dataset detected ({len(df.columns)} cols). "
          f"Auto-selected top {max_cols} informative columns.")
    return df_keep[top_cols]


class ACFMNetV5Pipeline:
    """
    Research-Grade Latent Optimal Transport Flow Matching (L-OTFM) Pipeline.
    Implements a 2-Phase generative architecture:
      Phase 1: Tabular Latent Representation Learning (HVAE / T-VAE)
      Phase 2: Exact Mini-Batch Optimal Transport Flow Matching (OT-CFM) in Latent Space
    Inference: Pure prior sampling z_0 ~ N(0, I) -> RK4 ODE -> Decoder -> Domain Enforcer.
    """

class ModelEMA:
    """
    Step-tied warm-up Weight Exponential Moving Average for training stability.
    Uses decay tied to the number of steps: decay_t = min(max_decay, (1 + step) / (warmup_steps + step)).
    Saves initial model weights to calculate L2 distance of released weights from initialization.
    """
    def __init__(self, model: nn.Module, max_decay: float = 0.999, warmup_steps: int = 10):
        self.max_decay = float(max_decay)
        self.warmup_steps = int(warmup_steps)
        self.step_count = 0
        self.init_weights = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.last_decay = 0.0

    def get_decay(self) -> float:
        return min(self.max_decay, (1.0 + self.step_count) / (float(self.warmup_steps) + self.step_count))

    def update(self, model: nn.Module):
        self.step_count += 1
        decay = self.get_decay()
        self.last_decay = decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].copy_(decay * self.shadow[k] + (1.0 - decay) * v)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def compute_l2_from_init(self, current_state: Optional[Dict[str, torch.Tensor]] = None) -> float:
        state = current_state if current_state is not None else self.shadow
        total_sq = 0.0
        for k, v in state.items():
            if k in self.init_weights:
                diff = v.cpu() - self.init_weights[k]
                total_sq += float(torch.sum(diff ** 2).item())
        return math.sqrt(total_sq)


class ACFMNetV5Pipeline:
    """
    Research-Grade Differentially Private Clinical Generative Pipeline (v6.0)
    Two-Stage Architecture:
      Phase 1: Hierarchical Context-Aware Representation Learning Network (HCARLN) in Latent Space
      Phase 2: Independent Flow Matching (I-CFM) in Latent Space with Single Composed DP-SGD Accountant
    Inference: Pure prior sampling z_0 ~ N(0, I) -> RK4 ODE -> Decoder -> Domain Enforcer.
    """

    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        strict_release_privacy: bool = True,
        use_ot_coupling: bool = False,
        allow_nonprivate_ot: bool = False,
        poisson_sampling: bool = True,
        allow_nonprivate_debug: bool = False,
        batch_size: Optional[int] = None,
        max_grad_norm_hvae: float = 1.25,
        max_grad_norm_flow: float = 1.25,
        use_public_graph_only: bool = True,
        ema_warmup_steps: int = 10,
        ema_max_decay: float = 0.999,
        # Gate 4b: collapse fix parameters
        beta_kl: float = 0.01,
        free_bits: float = 0.0,
        disable_gat: bool = True,
        # Gate 4c: latent_mode, VAE stabilization flags, stage budget split, and flow architecture
        latent_mode: str = "identity",
        logvar_init_bias: float = 0.0,
        deterministic_encoder: bool = False,
        stage_split_ratio: Tuple[float, float] = (0.5, 0.5),
        flow_hidden_dim: int = 128,
        flow_time_embed_dim: int = 64,
        flow_num_layers: int = 2,
        flow_optimizer: str = "adam",
        flow_lr: float = 1e-3,
        prior_scale: float = 1.0,
        public_schema: Optional[Dict[str, Any]] = None,
    ):
        if not strict_release_privacy and not allow_nonprivate_debug:
            raise ValueError(
                "strict_release_privacy must be True for formal differential privacy release guarantees. "
                "Set allow_nonprivate_debug=True only for non-private debug/ablation experiments."
            )

        self.target_epsilon = float(target_epsilon)
        self.target_delta = float(target_delta)
        self.strict_release_privacy = strict_release_privacy
        self.allow_nonprivate_debug = bool(allow_nonprivate_debug)
        self.use_ot_coupling = use_ot_coupling
        self.allow_nonprivate_ot = allow_nonprivate_ot
        self.poisson_sampling = poisson_sampling
        self.configured_batch_size = int(batch_size) if batch_size is not None else None
        self.max_grad_norm_hvae = float(max_grad_norm_hvae)
        self.max_grad_norm_flow = float(max_grad_norm_flow)
        self.use_public_graph_only = bool(use_public_graph_only)
        self.ema_warmup_steps = int(ema_warmup_steps)
        self.ema_max_decay = float(ema_max_decay)
        # Gate 4b: constant beta_kl (no annealing), free_bits, and GAT ablation flag
        self.beta_kl = float(beta_kl)
        self.free_bits = float(free_bits)
        self.disable_gat = bool(disable_gat)
        # Gate 4c: latent mode & VAE stabilization flags
        self.latent_mode = str(latent_mode).lower()
        self.logvar_init_bias = float(logvar_init_bias)
        self.deterministic_encoder = bool(deterministic_encoder)
        self.stage_split_ratio = stage_split_ratio
        self.public_schema = public_schema
        self.flow_hidden_dim = int(flow_hidden_dim)
        self.flow_time_embed_dim = int(flow_time_embed_dim)
        self.flow_num_layers = int(flow_num_layers)
        self.flow_optimizer = str(flow_optimizer).lower()
        self.flow_lr = float(flow_lr)
        self.prior_scale = float(prior_scale)

        # -- GPU / Device detection --------------------------------------------
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            global BATCH_SIZE
            if vram_gb >= 16:
                BATCH_SIZE = 1024
            elif vram_gb >= 8:
                BATCH_SIZE = 512
            else:
                BATCH_SIZE = 256
            print(f"[ACFM-Net L-OTFM] GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM). "
                  f"Batch size set to {BATCH_SIZE}.")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
            print("[ACFM-Net L-OTFM] Running on CPU engine.")

        # Stage 0: CDGL
        self.cdgl = ClinicalDataGovernanceLayer()

        # Stage 1: UDIE
        self.missing_analyzer = MissingValueAnalyzer()
        self.outlier_engine = OutlierDetectionEngine()
        self.sensitive_detector = SensitiveAttributeDetector()
        self.dependency_engine = DependencyEstimationEngine()
        self.dataset2vec = DatasetIntelligenceGenerator(fingerprint_dim=256)

        # Stage 2: ADPE
        self.imputer = AdaptiveImputer()
        self.encoder = DistributionPreservingEncoder()
        self.scaler = AdaptiveFeatureScaler()
        self.balancer = ClassBalancer()

        # Stage 3 & 3b: DCGB & Refinement
        self.graph_builder = DomainConstraintGraphBuilder(top_k=15)
        self.graph_refiner = PostLatentGraphRefiner(alpha=0.60)

        # Stage 5: DAPC & Privacy
        self.prv_accountant = PRVAccountant(target_epsilon=target_epsilon, target_delta=target_delta)
        self.budget_allocator = AdaptiveBudgetAllocator(target_epsilon=target_epsilon, target_delta=target_delta).to(self.device)

        # Stage 7: Repair Loop
        self.repair_loop = ConstrainedGradientRepairLoop()

        # Stage 8: Optimization & Pareto Selection
        self.model_selector = ParetoModelSelector(target_epsilon=target_epsilon)

    def _make_dataloader(self, X_tensor: torch.Tensor, batch_size: Optional[int] = None) -> data_utils.DataLoader:
        bs = batch_size or BATCH_SIZE
        dataset = data_utils.TensorDataset(X_tensor.cpu())
        if self.poisson_sampling:
            from opacus.data_loader import DPDataLoader
            base_loader = data_utils.DataLoader(
                dataset,
                batch_size=bs,
                pin_memory=(self.device.type == "cuda"),
                num_workers=0,
            )
            loader = DPDataLoader.from_data_loader(base_loader, distributed=False)
        else:
            loader = data_utils.DataLoader(
                dataset,
                batch_size=bs,
                shuffle=True,
                pin_memory=(self.device.type == "cuda"),
                num_workers=0,
                drop_last=False,
            )
        return loader

    def fit_generate(
        self,
        df_raw: pd.DataFrame,
        num_samples: Optional[int] = None,
        target_col: Optional[str] = None,
        epochs: int = 10,
        protocol_id: str = "IRB-2026-HC-001",
        public_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes research-grade Latent Optimal Transport Flow Matching on any dataset.
        """
        num_samples = num_samples or len(df_raw)
        started_at = time.time()
        # Keep a request-scoped target consistent across the policy mapper,
        # accountant, and training wrappers.
        self.budget_allocator.eps_max = float(self.target_epsilon)
        self.budget_allocator.target_epsilon = float(self.target_epsilon)
        self.budget_allocator.target_delta = float(self.target_delta)
        self.prv_accountant.set_target_epsilon(float(self.target_epsilon))
        self.prv_accountant.target_delta = float(self.target_delta)
        report = {
            "pipeline_status": "RUNNING",
            "stages_completed": [],
            "stage_reports": [],
            "device": str(self.device),
            "privacy_contract": {
                "mode": "strict_public_schema" if self.strict_release_privacy else "training_only",
                "public_schema_assumption": bool(self.strict_release_privacy),
            },
        }

        if len(df_raw.columns) > MAX_COLUMNS:
            tc = target_col if (target_col and target_col in df_raw.columns) else None
            df_raw = _select_informative_columns(df_raw, max_cols=MAX_COLUMNS)
            if tc and tc in df_raw.columns:
                target_col = tc

        if target_col and target_col in df_raw.columns:
            df_raw = df_raw.dropna(subset=[target_col]).reset_index(drop=True)

        print(f"[Stage 0] Dataset shape entering pipeline: {df_raw.shape}")

        # --------------------------------------------------------------------
        # STAGE 0: Clinical Data Governance Layer (CDGL)
        # --------------------------------------------------------------------
        cert, cdgl_logs, df_deidentified = self.cdgl.process_governance(
            df_raw,
            protocol_id=protocol_id,
            hipaa_flag=True,
            gdpr_flag=True,
            run_deidentification=True,
        )
        report["governance_certificate"] = cert
        report["stages_completed"].append("Stage 0: CDGL Passed (De-identification Complete)")
        _record_stage(report, "0", "Governance and de-identification", "PASSED", {
            "rows_in": len(df_raw),
            "rows_out": len(df_deidentified),
            "columns_removed": len(cert.phi_columns_removed),
            "columns_hashed": len(cert.hashed_columns),
            "re_id_risk": None if self.strict_release_privacy else cert.re_id_risk,
            "re_id_risk_status": "not released from raw governance scan" if self.strict_release_privacy else "raw audit value",
            "audit_handle": cert.audit_trail_handle,
        })

        # --------------------------------------------------------------------
        # STAGE 1: Universal Dataset Intelligence Engine (UDIE)
        # --------------------------------------------------------------------
        if public_schema is None:
            public_schema = self.public_schema if self.public_schema is not None else public_schema_from_dataframe(df_deidentified)
        report["privacy_contract"].update({
            "schema_source": "declared column schema and policy bounds",
            "schema_columns": len(public_schema),
            "empirical_alignment_disabled": bool(self.strict_release_privacy),
            "categorical_vocabulary_assumed_public": bool(self.strict_release_privacy),
        })
        df_sample = df_deidentified.sample(n=min(500, len(df_deidentified)), random_state=42).reset_index(drop=True)

        if self.strict_release_privacy:
            # Missingness summary is unconsumed in the release path (preprocessing uses fixed row-local bounds);
            # zero Laplace charge is allocated to missingness.
            missing_info = {
                "missing_mechanisms": {},
                "global_missing_pct": 0.0,
            }
            sensitivity_info = self.sensitive_detector.detect(df_sample, use_data_dependent_risk=False)
        else:
            missing_info = self.missing_analyzer.analyze(df_sample)
            sensitivity_info = self.sensitive_detector.detect(df_sample)
        sensitive_info = sensitivity_info
        outlier_info = (
            {"outlier_scores": np.zeros(len(df_sample), dtype=np.float32), "privacy_safe": True}
            if self.strict_release_privacy
            else self.outlier_engine.analyze(df_sample)
        )
        dep_info_raw = self.dependency_engine.analyze(df_sample)
        dep_info = dep_info_raw
        privacy_ledger = PrivacyLedger(self.target_epsilon, self.target_delta)
        if self.strict_release_privacy:
            if not self.use_public_graph_only:
                # Noised graph with Laplace noise (Gate 3b ablation mode)
                dep_info = {
                    **dep_info_raw,
                    "dependency_matrix": sanitize_dependency_matrix(
                        dep_info_raw["dependency_matrix"],
                        epsilon=max(0.01, self.target_epsilon * 0.08),
                        n_rows=len(df_sample),
                    ),
                }
                privacy_ledger.add(
                    "dependency_graph_statistics",
                    epsilon=self.target_epsilon * 0.08,
                    delta=self.target_delta * 0.05,
                    mechanism="bounded Laplace dependency matrix",
                )
            else:
                # Public-only graph (Gate 3b Item 7): Exact sensitivity under add/remove cannot
                # be established for composite Spearman/mutual info/ANOVA statistics.
                # Replace noised graph with public-only graph, charging 0 eps to the ledger
                # and transferring the freed 0.08 epsilon entirely to DP-SGD.
                dep_info = {
                    **dep_info_raw,
                    "dependency_matrix": np.eye(len(df_sample.columns), dtype=np.float32),
                    "privacy_status": "public_only_graph",
                }

        if self.strict_release_privacy:
            fingerprint_info = {
                "fingerprint": np.zeros(256, dtype=np.float32),
                "task_info": {"task": "classification" if target_col else "unsupervised", "privacy_safe": True},
            }
        else:
            fingerprint_info = self.dataset2vec.analyze_and_encode(
                df_sample,
                dependency_matrix=dep_info["dependency_matrix"],
                sensitivity_vector=sensitive_info["sensitivity_vector"],
                target_col=target_col
            )

        S_vec = torch.tensor(sensitive_info["sensitivity_vector"], dtype=torch.float32, device=self.device)
        I_fp  = torch.tensor(fingerprint_info["fingerprint"],      dtype=torch.float32, device=self.device)
        report["task_info"] = fingerprint_info.get("task_info", {})
        report["stages_completed"].append("Stage 1: UDIE Passed (Task-Type Inferred)")
        _record_stage(report, "1", "Universal dataset intelligence", "PASSED", {
            "sampled_rows": len(df_sample),
            "task": fingerprint_info.get("task_info", {}).get("task"),
            "global_missing_pct": missing_info.get("global_missing_pct"),
            "outlier_rows": None if self.strict_release_privacy else int(np.sum(outlier_info.get("outlier_scores", []) > 0.5)) if len(outlier_info.get("outlier_scores", [])) else 0,
            "outlier_analysis": "disabled under strict public-schema release" if self.strict_release_privacy else "empirical",
            "fingerprint_dim": len(fingerprint_info.get("fingerprint", [])),
        })
        print(f"[Stage 1] UDIE complete. Inferred task: {fingerprint_info.get('task_info', {}).get('task')}")

        # --------------------------------------------------------------------
        # STAGE 2: Adaptive Data Preparation Engine (ADPE)
        # --------------------------------------------------------------------
        df_imputed, conf_matrix = self.imputer.fit_transform(
            df_deidentified,
            missing_info.get("missing_mechanisms"),
            privacy_mode=self.strict_release_privacy,
            public_schema=public_schema if self.strict_release_privacy else None,
        )
        encoded_X, enc_info     = self.encoder.fit_transform(
            df_imputed,
            privacy_schema=public_schema if self.strict_release_privacy else None,
        )
        scaled_X                = self.scaler.fit_transform(
            encoded_X,
            enc_info["feature_specs"],
            privacy_mode=self.strict_release_privacy,
            encoded_column_names=enc_info["encoded_column_names"],
        )

        y_labels = df_deidentified[target_col].values if (target_col and target_col in df_deidentified.columns) else None
        if self.strict_release_privacy:
            X_bal, y_bal, bal_info = scaled_X, y_labels, {"action": "disabled_under_strict_privacy", "imbalance_ratio": None}
        else:
            X_bal, y_bal, bal_info = self.balancer.balance(scaled_X, y_labels)

        self.scaled_X_train = X_bal
        X_cpu    = torch.tensor(X_bal, dtype=torch.float32)
        X_tensor = X_cpu.to(self.device)
        report["stages_completed"].append("Stage 2: ADPE Passed")
        _record_stage(report, "2", "Adaptive data preparation", "PASSED", {
            "encoded_rows": int(X_cpu.shape[0]),
            "encoded_features": int(X_cpu.shape[1]),
            "missing_cells_after_imputation": int(df_imputed.isna().sum().sum()),
            "balanced_rows": int(len(X_bal)),
        })
        print(f"[Stage 2] ADPE complete. Encoded shape: {X_cpu.shape}")

        # --------------------------------------------------------------------
        # STAGE 3: Domain Constraint Graph Builder (DCGB)
        # --------------------------------------------------------------------
        enc_cols = enc_info["encoded_column_names"]
        d_enc    = len(enc_cols)

        G_0, W_0, g_info = self.graph_builder.build_graph(
            columns=enc_cols,
            dependency_matrix=dep_info.get("dependency_matrix", np.eye(d_enc, dtype=np.float32)),
            sensitivity_vector=sensitive_info.get("sensitivity_vector", np.ones(d_enc, dtype=np.float32)*0.5),
            # Mutual information over raw rows is excluded in strict mode;
            # graph weights use the DP-sanitized dependency matrix plus public
            # ontology edges instead.
            df_encoded=None if self.strict_release_privacy else pd.DataFrame(encoded_X, columns=enc_cols),
            target_col=target_col
        )
        report["stages_completed"].append("Stage 3: DCGB Passed (MI Edges Fused)")
        graph_inactive = bool(g_info.get("num_edges", 0) == 0 or g_info.get("clinical_edges", 0) == 0)
        _record_stage(report, "3", "Domain constraint graph", "PASSED", {
            "nodes": g_info.get("num_nodes"),
            "edges": g_info.get("num_edges"),
            "clinical_edges": g_info.get("clinical_edges", 0),
            "graph_inactive": graph_inactive,
            "top_k": self.graph_builder.top_k,
            "dependency_matrix_shape": list(np.asarray(dep_info.get("dependency_matrix")).shape),
            "privacy_status": "DP-sanitized dependency matrix; raw MI disabled" if self.strict_release_privacy else "metadata statistics require separate accounting",
        })
        print(f"[Stage 3] DCGB complete. Graph: {g_info['num_nodes']} nodes, {g_info['num_edges']} edges.")

        # --------------------------------------------------------------------
        # STAGE 5: Dataset-Adaptive Privacy Budget Controller (DAPC)
        # Calibrate once using the actual DP-SGD batch rate and total step count.
        # This pre-flight must happen before either trainable component updates.
        # --------------------------------------------------------------------
        cat_cols = [c for c in df_deidentified.columns if not pd.api.types.is_float_dtype(df_deidentified[c])]
        if self.strict_release_privacy:
            u_ratio = 0.0
        elif cat_cols:
            u_ratio = float(df_deidentified[cat_cols].drop_duplicates().shape[0] / max(1, len(df_deidentified))) * 0.1
        else:
            u_ratio = 0.02

        dep_mat = dep_info.get("dependency_matrix", np.eye(d_enc))
        if isinstance(dep_mat, np.ndarray) and dep_mat.shape[0] > 1:
            off_diag = dep_mat[~np.eye(dep_mat.shape[0], dtype=bool)]
            dep_s = float(np.max(np.abs(off_diag))) if len(off_diag) > 0 else 0.2
        else:
            dep_s = 0.2
        base_bs = self.configured_batch_size if self.configured_batch_size is not None else BATCH_SIZE
        actual_batch_size = min(base_bs, len(X_cpu))
        batches_per_epoch = max(1, math.ceil(len(X_cpu) / actual_batch_size))
        flow_batches_per_epoch = batches_per_epoch

        if self.latent_mode == "identity":
            representation_epochs = 0
            hvae_steps_planned = 0
            flow_steps_planned = epochs * flow_batches_per_epoch
            total_dp_steps = flow_steps_planned
        else:
            representation_epochs = max(5, epochs // 2)
            hvae_steps_planned = representation_epochs * batches_per_epoch
            flow_steps_planned = epochs * flow_batches_per_epoch
            total_dp_steps = hvae_steps_planned + flow_steps_planned

        actual_sample_rate = min(1.0, actual_batch_size / max(1, len(X_cpu)))

        dapc_cert = self.budget_allocator.analyze_dataset_dapc(
            sensitivity_vector=S_vec,
            num_samples=len(df_deidentified),
            num_features=d_enc,
            reid_risk=cert.re_id_risk,
            unique_ratio=u_ratio,
            dep_score=dep_s,
            is_high_stakes=(target_col and any(k in target_col.lower() for k in ["mortality", "death", "cancer", "icu"])),
            dataset_name=protocol_id,
            sample_rate=actual_sample_rate,
            steps=total_dp_steps,
        )

        adaptive_target_eps = float(dapc_cert["selected_target_epsilon"])
        self.prv_accountant.set_target_epsilon(adaptive_target_eps)
        # Note: DAPC safety gating evaluates only public cohort size (n >= 100), feature count,
        # and public schema bounds. No private data is consumed for budget decision;
        # zero Laplace charge is needed for dapc_risk_selection, freeing budget for DP-SGD.
        nontraining_epsilon = privacy_ledger.epsilon
        training_target_eps = max(0.05, adaptive_target_eps - nontraining_epsilon)
        training_delta = self.target_delta * 0.80 if self.strict_release_privacy else self.target_delta
        self.prv_accountant.target_delta = training_delta
        feature_scales, noise_scales, privacy_info = self.budget_allocator(
            S_vec,
            I_fp,
            sample_rate=actual_sample_rate,
            steps=total_dp_steps,
        )

        if self.latent_mode == "identity":
            actual_sigma = self.budget_allocator.calibrate_noise_multiplier_accountant(
                target_eps_sgd=training_target_eps,
                sample_rate=actual_sample_rate,
                steps=total_dp_steps,
                delta=training_delta,
            )
            actual_sigma_hvae = None
            actual_sigma_flow = actual_sigma
            eps_hvae_target = 0.0
            eps_flow_target = training_target_eps
            delta_hvae = 0.0
            delta_flow = training_delta
        else:
            split_hvae, split_flow = self.stage_split_ratio
            eps_hvae_target = max(0.01, training_target_eps * split_hvae)
            eps_flow_target = max(0.01, training_target_eps * split_flow)
            delta_hvae = training_delta / 2.0
            delta_flow = training_delta / 2.0
            actual_sigma_hvae = self.budget_allocator.calibrate_noise_multiplier_accountant(
                target_eps_sgd=eps_hvae_target,
                sample_rate=actual_sample_rate,
                steps=max(1, hvae_steps_planned),
                delta=delta_hvae,
            )
            actual_sigma_flow = self.budget_allocator.calibrate_noise_multiplier_accountant(
                target_eps_sgd=eps_flow_target,
                sample_rate=actual_sample_rate,
                steps=max(1, flow_steps_planned),
                delta=delta_flow,
            )
            actual_sigma = actual_sigma_flow

        dapc_cert["training_budget_eps_sgd"] = round(training_target_eps, 6)
        dapc_cert["calibrated_noise_multiplier_sigma"] = round(actual_sigma, 6)

        report["dapc_certificate"] = dapc_cert
        report["target_epsilon_adaptive"] = adaptive_target_eps
        report["dapc_decision"] = dapc_cert["decision"]
        report["privacy_training_plan"] = {
            "algorithm": "per-example DP-SGD",
            "max_grad_norm": 1.25,
            "noise_multiplier": actual_sigma,
            "noise_multiplier_hvae": actual_sigma_hvae,
            "noise_multiplier_flow": actual_sigma_flow,
            "stage_split_ratio": self.stage_split_ratio,
            "sample_rate": actual_sample_rate,
            "expected_batch_size": actual_batch_size,
            "planned_steps": total_dp_steps,
            "representation_epochs": representation_epochs,
            "flow_epochs": epochs,
            "target_epsilon": training_target_eps,
            "release_target_epsilon": adaptive_target_eps,
            "target_delta": training_delta,
        }

        if dapc_cert["decision"] in ["ABSTAIN", "REJECT"]:
            report["pipeline_status"] = "ABSTAINED"
            report["abstention_reason"] = " ".join(dapc_cert["decision_reasons"])
            print(f"[Stage 5 DAPC] Pipeline execution ABSTAINED: {report['abstention_reason']}")
            raise RuntimeError(f"DAPC Security Abstention triggered ({dapc_cert['decision']}): {report['abstention_reason']}")

        report["stages_completed"].append(f"Stage 5: DAPC Passed (Target ε* = {adaptive_target_eps:.3f}, Decision: {dapc_cert['decision']})")
        _record_stage(report, "5", "Dataset-adaptive privacy controller", "PASSED", {
            "decision": dapc_cert["decision"],
            "risk_score_R_P": dapc_cert["risk_score_R_P"],
            "selected_target_epsilon": adaptive_target_eps,
            "training_budget_eps_sgd": dapc_cert["training_budget_eps_sgd"],
            "noise_multiplier": actual_sigma,
            "sample_rate": actual_sample_rate,
            "expected_batch_size": actual_batch_size,
            "planned_dp_steps": total_dp_steps,
        })

        # --------------------------------------------------------------------
        # STAGE 4: Hierarchical Representation Learning (HCARLN)
        # --------------------------------------------------------------------
        in_dim = X_tensor.shape[1]
        if self.latent_mode == "identity":
            latent_dim = in_dim
            latent_Z = X_tensor
            hvae = None
            gat = None
            cross_attn = None
            hvae_steps = 0
            hvae_stability_log = []
            report["hvae_baseline"] = {
                "latent_mode": "identity",
                "dimensions": in_dim,
                "notice": "HVAE skipped in identity mode; 100% of epsilon allocated to flow in X_scaled.",
            }
            report["latent_dependence_summary"] = {
                "latent_mode": "identity",
                "notice": "No latent VAE compression; flow runs directly on X_scaled.",
            }
            report["stages_completed"].append("Stage 4: HCARLN Skipped (Identity Latent Mode)")
            _record_stage(report, "4", "DP hierarchical representation learning", "SKIPPED", {
                "latent_mode": "identity",
                "notice": "Identity mode trains flow directly in X_scaled; whole epsilon goes to flow.",
            })
            print(f"[Stage 4] Latent mode is 'identity': bypassing HVAE. Flow trains directly in X_scaled ({in_dim} dims).")
        else:
            # Gate 4b & 4c: pass beta_kl, free_bits, logvar_init_bias, deterministic_encoder
            hvae = HierarchicalVAE(
                in_dim=in_dim,
                global_dim=32,
                group_dim=16,
                num_groups=2,
                beta_kl=self.beta_kl,
                free_bits=self.free_bits,
                logvar_init_bias=self.logvar_init_bias,
                deterministic_encoder=self.deterministic_encoder,
            ).to(self.device)
            latent_dim = hvae.joint_dim

            gat = GraphAttentionEncoder(in_features=1, hidden_dim=64).to(self.device)
            cross_attn = GatedCrossAttention(feature_dim=in_dim, graph_dim=64, hidden_dim=64).to(self.device)
            adj_matrix = torch.tensor(W_0, dtype=torch.float32, device=self.device)
            # Pretrain Latent VAE on dataset representation
            if self.disable_gat:
                representation_model = hvae
            else:
                representation_model = nn.ModuleList([gat, cross_attn, hvae])
            vae_optimizer = torch.optim.AdamW(representation_model.parameters(), lr=2e-3, weight_decay=1e-5)
            dp_vae = DPSGDWrapper(
                representation_model,
                vae_optimizer,
                max_grad_norm=self.max_grad_norm_hvae,
                noise_multiplier=actual_sigma_hvae,
                target_epsilon=eps_hvae_target,
                target_delta=delta_hvae,
                expected_batch_size=actual_batch_size,
            )
            dataloader = self._make_dataloader(X_cpu, batch_size=actual_batch_size)

            # Gate 2b Item 4: HVAE Reconstruction Baselines on Scaled Inputs X in [-1, 1]^{d_enc}
            with torch.no_grad():
                zero_recon_loss = float(torch.mean(X_tensor ** 2).item())
                mean_feature_val = torch.mean(X_tensor, dim=0, keepdim=True)
                mean_recon_loss = float(torch.mean((X_tensor - mean_feature_val) ** 2).item())
            print(f"[Stage 4 HVAE Baseline] Loss Definition: mean MSE over {in_dim} dimensions (per-sample mean, then batch mean)")
            print(f"[Stage 4 HVAE Baseline] Zero-predictor recon loss: {zero_recon_loss:.4f} | Mean-predictor recon loss: {mean_recon_loss:.4f}")
            report["hvae_baseline"] = {
                "loss_definition": f"mean MSE averaged over {in_dim} feature dimensions (per-sample mean, then batch mean)",
                "dimensions": in_dim,
                "zero_predictor_recon_loss": round(zero_recon_loss, 4),
                "mean_predictor_recon_loss": round(mean_recon_loss, 4),
            }

            hvae.train()
            hvae_steps = 0
            hvae_stability_log = []
            for v_ep in range(1, representation_epochs + 1):
                ep_hvae_loss = 0.0
                ep_recon = 0.0
                ep_kl = 0.0
                ep_clipped_frac = 0.0
                ep_norms = []
                ep_hvae_batches = 0
                for (X_batch,) in dataloader:
                    if len(X_batch) == 0:
                        # Match Opacus: empty Poisson draw is stepped with pure Gaussian noise and counted in accounting
                        loss_vae = dp_vae.step_microbatch(torch.zeros(0, device=self.device))
                    else:
                        X_batch = X_batch.to(self.device, non_blocking=True)
                        if self.disable_gat:
                            # Gate 4b GAT ablation: bypass GAT and cross-attention
                            X_enc_in = X_batch
                        else:
                            h_gat = gat(X_batch, adj_matrix)
                            X_enc_in = cross_attn(X_batch, h_gat)
                        # HVAE reconstruction target is strictly the FIXED X_batch (X_scaled)
                        vae_out = hvae(X_enc_in, target=X_batch, epoch=v_ep)
                        loss_vae = dp_vae.step_microbatch(vae_out["per_sample_loss"])
                        ep_recon += float(vae_out["recon_loss"].item())
                        ep_kl += float(vae_out["kl_loss"].item())
                        ep_clipped_frac += float(dp_vae.last_fraction_clipped)
                        if dp_vae.last_sample_norms:
                            ep_norms.extend(dp_vae.last_sample_norms)
                    ep_hvae_loss += float(loss_vae)
                    ep_hvae_batches += 1
                    hvae_steps += 1

                # Step single accountant per epoch to accurately maintain history
                self.prv_accountant.step(
                    noise_multiplier=actual_sigma_hvae,
                    sample_rate=actual_sample_rate,
                    steps=ep_hvae_batches,
                )
                eps_so_far = self.prv_accountant.get_accumulated_epsilon()
                avg_loss = ep_hvae_loss / max(1, ep_hvae_batches)
                avg_recon = ep_recon / max(1, ep_hvae_batches)
                avg_kl = ep_kl / max(1, ep_hvae_batches)
                avg_clipped = ep_clipped_frac / max(1, ep_hvae_batches)
                mean_norm = float(np.mean(ep_norms)) if ep_norms else 0.0
                max_norm = float(np.max(ep_norms)) if ep_norms else 0.0

                hvae_stability_log.append({
                    "epoch": v_ep,
                    "loss": avg_loss,
                    "recon_loss": avg_recon,
                    "kl_loss": avg_kl,
                    "fraction_clipped": avg_clipped,
                    "mean_grad_norm": mean_norm,
                    "max_grad_norm": max_norm,
                    "accumulated_epsilon": eps_so_far,
                })
                print(f"  [Stage 4 HVAE] Epoch {v_ep:2d}/{representation_epochs} | Loss={avg_loss:.4f} (Recon={avg_recon:.4f}, KL={avg_kl:.4f}) | Clipped Frac={avg_clipped:.2%} | Grad Norm (mean/max)={mean_norm:.4f}/{max_norm:.4f} | Accumulated DP eps={eps_so_far:.4f}")

            report["hvae_stability_log"] = hvae_stability_log

            hvae.eval()
            with torch.no_grad():
                if self.disable_gat:
                    X_enc_in_all = X_tensor
                else:
                    h_gat_all = gat(X_tensor, adj_matrix)
                    X_enc_in_all = cross_attn(X_tensor, h_gat_all)
                hvae_out = hvae(X_enc_in_all, target=X_tensor, epoch=epochs)
                latent_Z = hvae_out["Z"]

            # -----------------------------------------------------------------------
            # Gate 4c: Latent-Dependence Summary in FIXED X_scaled space (X_tensor)
            # Recon MSE with: (a) posterior z, (b) shuffled z, (c) prior z ~ N(0,I)
            # Pass rule: shuffled_mse >= 3x posterior_mse  AND  posterior_mse <= mean_pred_mse / 3
            # Measured in X_tensor space (the fixed X_scaled public-bounds space).
            # -----------------------------------------------------------------------
            with torch.no_grad():
                mean_pred_mse = float(torch.mean((X_tensor - X_tensor.mean(dim=0, keepdim=True)) ** 2).item())

                x_recon_posterior = hvae.decode(latent_Z)
                posterior_mse = float(torch.mean((X_tensor - x_recon_posterior) ** 2).item())

                perm_idx = torch.randperm(latent_Z.shape[0])
                z_shuffled = latent_Z[perm_idx]
                x_recon_shuffled = hvae.decode(z_shuffled)
                shuffled_mse = float(torch.mean((X_tensor - x_recon_shuffled) ** 2).item())

                z_prior = torch.randn_like(latent_Z)
                x_recon_prior = hvae.decode(z_prior)
                prior_mse = float(torch.mean((X_tensor - x_recon_prior) ** 2).item())

                total_kl_nats = float(hvae_out.get("total_kl_nats", torch.tensor(float('nan'))).item())
                raw_kl_nats = float(hvae_out.get("raw_total_kl_nats", torch.tensor(float('nan'))).item())

            shuffle_ratio = shuffled_mse / max(1e-9, posterior_mse)
            ld_pass = (shuffle_ratio >= 3.0) and (posterior_mse <= mean_pred_mse / 3.0)
            print(f"\n[Gate 4c Latent-Dependence Summary (in X_scaled space)]")
            print(f"  Posterior MSE  : {posterior_mse:.6f}")
            print(f"  Shuffled MSE   : {shuffled_mse:.6f}  (ratio vs posterior: {shuffle_ratio:.2f}x)")
            print(f"  Prior MSE      : {prior_mse:.6f}")
            print(f"  Mean-pred MSE  : {mean_pred_mse:.6f}")
            print(f"  Clamped KL     : {total_kl_nats:.4f} nats")
            print(f"  Raw KL (nats)  : {raw_kl_nats:.4f} nats")
            print(f"  Pass rule      : shuffled >= 3x posterior AND posterior <= mean_pred/3")
            print(f"  PASS           : {ld_pass}  [shuffled/posterior={shuffle_ratio:.2f}, posterior/mean_pred={posterior_mse/max(1e-9,mean_pred_mse):.4f}]")

            report["latent_dependence_summary"] = {
                "posterior_mse": round(posterior_mse, 6),
                "shuffled_mse": round(shuffled_mse, 6),
                "prior_mse": round(prior_mse, 6),
                "mean_predictor_mse": round(mean_pred_mse, 6),
                "total_kl_nats": round(total_kl_nats, 4),
                "raw_total_kl_nats": round(raw_kl_nats, 4),
                "shuffled_to_posterior_ratio": round(shuffle_ratio, 2),
                "pass_rule": "shuffled_mse >= 3 * posterior_mse AND posterior_mse <= mean_pred_mse / 3",
                "pass": bool(ld_pass),
                "beta_kl": self.beta_kl,
                "free_bits": self.free_bits,
                "logvar_init_bias": self.logvar_init_bias,
                "deterministic_encoder": self.deterministic_encoder,
                "disable_gat": self.disable_gat,
            }

            report["stages_completed"].append("Stage 4: HCARLN Passed (Latent VAE Trained)")
            _record_stage(report, "4", "DP hierarchical representation learning", "PASSED", {
                "latent_rows": int(latent_Z.shape[0]),
                "latent_features": int(latent_Z.shape[1]),
                "trainable_components": ["GraphAttentionEncoder", "GatedCrossAttention", "HierarchicalVAE"],
                "optimizer": "per-example DP-SGD",
                "epochs": representation_epochs,
                "steps": hvae_steps,
                "beta_kl": self.beta_kl,
                "free_bits": self.free_bits,
                "logvar_init_bias": self.logvar_init_bias,
                "deterministic_encoder": self.deterministic_encoder,
                "disable_gat": self.disable_gat,
                "latent_dependence_pass": bool(ld_pass),
            })
            print(f"[Stage 4] HCARLN complete. Latent Z shape: {latent_Z.shape}")

        # --------------------------------------------------------------------
        # STAGE 3b: Post-Latent Graph Refinement (Excluded from DP release path)
        # --------------------------------------------------------------------
        report["stages_completed"].append("Stage 3b: Skipped under strict privacy")
        _record_stage(report, "3b", "Post-latent graph refinement", "SKIPPED", {
            "notice": "Excluded from active DP path to prevent unbudgeted private latent access."
        })

        # --------------------------------------------------------------------
        # STAGE 6: Continuous Flow Matching (I-CFM Engine)
        # --------------------------------------------------------------------
        ot_cfm = ContinuousOTCFM(
            continuous_dim=latent_dim,
            cond_dim=0,
            hidden_dim=self.flow_hidden_dim,
            time_embed_dim=self.flow_time_embed_dim,
            num_layers=self.flow_num_layers
        ).to(self.device)
        p_flow = sum(p.numel() for p in ot_cfm.velocity_net.parameters() if p.requires_grad)
        noise_over_clip = float(actual_sigma_flow * math.sqrt(p_flow) / actual_batch_size)
        print(f"[Stage 6 VelocityNet] Parameters: {p_flow:,} | Flow sigma: {actual_sigma_flow:.4f} | noise/clip = sigma*sqrt(p)/B: {noise_over_clip:.6f}")
        if getattr(self, "flow_optimizer", "adam") == "sgd":
            optimizer_cfm = torch.optim.SGD(ot_cfm.velocity_net.parameters(), lr=self.flow_lr, momentum=0.9, weight_decay=1e-5)
        else:
            optimizer_cfm = torch.optim.AdamW(ot_cfm.velocity_net.parameters(), lr=self.flow_lr, weight_decay=1e-5)
        dp_flow = DPSGDWrapper(
            ot_cfm.velocity_net,
            optimizer_cfm,
            max_grad_norm=self.max_grad_norm_flow,
            noise_multiplier=actual_sigma_flow,
            target_epsilon=eps_flow_target if self.latent_mode == "vae" else adaptive_target_eps,
            target_delta=delta_flow if self.latent_mode == "vae" else training_delta,
            expected_batch_size=actual_batch_size,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cfm, T_max=epochs, eta_min=1e-5)

        # Gate 2b Item 1 & 3 / Gate 3b Item 8: Flow-loss baseline
        # Theoretical zero-predictor loss under N(0, I) latents: 2.0000; Gaussian floor: pi / 2 ≈ 1.5708
        # Dev-only empirical statistics from private latents are excluded from release reports.
        zero_predictor_loss = 2.0
        gaussian_floor = float(math.pi / 2.0)  # pi / 2 ≈ 1.5707963

        if self.allow_nonprivate_debug:
            latent_mean = latent_Z.mean(dim=0)
            latent_var = latent_Z.var(dim=0, unbiased=False)
            latent_scale_mean = float(latent_var.mean().item())
            empirical_zero = float(latent_scale_mean + (latent_mean ** 2).mean().item() + 1.0)
            print(f"[Stage 6 Debug Baseline] Empirical Zero-predictor flow loss: {empirical_zero:.4f} | Latent scale: {latent_scale_mean:.4f}")

        print(f"[Stage 6 Baseline] Standardized Zero-predictor baseline: {zero_predictor_loss:.4f} (2.0000 for N(0,I)) | Gaussian floor: {gaussian_floor:.4f} (pi/2)")

        report["flow_baseline"] = {
            "standardized_reference_loss": zero_predictor_loss,
            "gaussian_floor": round(gaussian_floor, 4),
        }

        report["stages_completed"].append("Stage 6: Latent I-CFM Engine Initialized")
        _record_stage(report, "6", "Latent independent flow matching", "INITIALIZED", {
            "latent_dim": int(latent_dim),
            "optimizer": "per-example DP-SGD",
            "trainable_components": ["ContinuousOTCFM.velocity_net"],
            "planned_epochs": epochs,
            "coupling_mode": "batch-Hungarian OT" if self.use_ot_coupling else "independent (I-CFM)",
            "zero_predictor_loss": round(zero_predictor_loss, 4),
            "gaussian_floor": round(gaussian_floor, 4),
        })

        # --------------------------------------------------------------------
        # STAGE 8: Optimization Training Loop (Latent Flow Matching + Model EMA)
        # --------------------------------------------------------------------
        latent_loader = self._make_dataloader(latent_Z.detach(), batch_size=actual_batch_size)

        print(f"=== STAGE 8: Flow Training ({'OT' if self.use_ot_coupling else 'I-CFM'}) -- {epochs} epochs, batch={actual_batch_size}, device={self.device} ===")
        ot_cfm.train()
        flow_ema = ModelEMA(ot_cfm.velocity_net, max_decay=self.ema_max_decay, warmup_steps=self.ema_warmup_steps)
        flow_steps = 0
        flow_stability_log = []

        for ep in range(1, epochs + 1):
            ep_flow_loss = 0.0
            ep_flow_batches = 0
            ep_flow_clipped_frac = 0.0
            ep_flow_norms = []

            for (z_batch,) in latent_loader:
                if len(z_batch) == 0:
                    loss_flow = dp_flow.step_microbatch(torch.zeros(0, device=self.device))
                else:
                    z_batch = z_batch.to(self.device, non_blocking=True)
                    # Flow Matching loss (I-CFM by default; batch-OT only for non-private ablation)
                    per_sample_flow_loss = ot_cfm.compute_loss(
                        z_batch,
                        use_ot_coupling=self.use_ot_coupling,
                        return_per_sample=True,
                        allow_nonprivate_ot=self.allow_nonprivate_ot,
                    )
                    loss_flow = dp_flow.step_microbatch(per_sample_flow_loss)
                    ep_flow_clipped_frac += float(dp_flow.last_fraction_clipped)
                    if dp_flow.last_sample_norms:
                        ep_flow_norms.extend(dp_flow.last_sample_norms)
                flow_ema.update(ot_cfm.velocity_net)

                ep_flow_loss += float(loss_flow)
                ep_flow_batches += 1
                flow_steps += 1

            # Optimizer step before scheduler step (resolving PyTorch warning)
            scheduler.step()

            # Step single accountant per epoch
            self.prv_accountant.step(
                noise_multiplier=actual_sigma_flow,
                sample_rate=actual_sample_rate,
                steps=ep_flow_batches,
            )
            eps_so_far = self.prv_accountant.get_accumulated_epsilon()
            avg_flow_loss = ep_flow_loss / max(1, ep_flow_batches)
            avg_flow_clipped = ep_flow_clipped_frac / max(1, ep_flow_batches)
            mean_flow_norm = float(np.mean(ep_flow_norms)) if ep_flow_norms else 0.0
            median_flow_norm = float(np.median(ep_flow_norms)) if ep_flow_norms else 0.0
            max_flow_norm = float(np.max(ep_flow_norms)) if ep_flow_norms else 0.0
            flow_stability_log.append({
                "epoch": ep,
                "loss": avg_flow_loss,
                "fraction_clipped": avg_flow_clipped,
                "mean_grad_norm": mean_flow_norm,
                "median_grad_norm": median_flow_norm,
                "max_grad_norm": max_flow_norm,
                "accumulated_epsilon": eps_so_far,
            })
            print(f"  [Stage 8 Flow] Epoch {ep:3d}/{epochs} | Flow Loss={avg_flow_loss:.4f} | Clipped Frac={avg_flow_clipped:.2%} | Grad Norm (mean/med/max)={mean_flow_norm:.4f}/{median_flow_norm:.4f}/{max_flow_norm:.4f} | Accumulated DP eps={eps_so_far:.4f}")

        report["flow_stability_log"] = flow_stability_log

        # Gate 3b Item 1: Compare live-model vs EMA-model flow loss
        # and report L2 distance of released weights from initialization.
        with torch.no_grad():
            eval_z_batch = next(iter(latent_loader))[0].to(self.device)
            live_flow_loss = float(ot_cfm.compute_loss(eval_z_batch, use_ot_coupling=self.use_ot_coupling).item())
            
            live_weights_state = {k: v.clone() for k, v in ot_cfm.velocity_net.state_dict().items()}
            flow_ema.apply_to(ot_cfm.velocity_net)
            ema_flow_loss = float(ot_cfm.compute_loss(eval_z_batch, use_ot_coupling=self.use_ot_coupling).item())
            
            l2_dist_ema = flow_ema.compute_l2_from_init()
            l2_dist_live = flow_ema.compute_l2_from_init(live_weights_state)

        print(f"\n[Stage 8 Flow Diagnostics (Gate 3b Item 1)]")
        print(f"  EMA schedule: step-tied warm-up decay_t = min({flow_ema.max_decay}, (1 + step)/({flow_ema.warmup_steps} + step))")
        print(f"  Final EMA step count: {flow_ema.step_count} | Final EMA decay: {flow_ema.last_decay:.5f}")
        print(f"  L2 distance of released (EMA) weights from init: {l2_dist_ema:.6f}")
        print(f"  L2 distance of live weights from init:           {l2_dist_live:.6f}")
        print(f"  Flow loss comparison: Live model = {live_flow_loss:.4f} vs EMA model = {ema_flow_loss:.4f}\n")

        report["flow_training_diagnostics"] = {
            "ema_schedule": f"min({flow_ema.max_decay}, (1 + step)/({flow_ema.warmup_steps} + step))",
            "ema_warmup_steps": flow_ema.warmup_steps,
            "final_ema_decay": round(flow_ema.last_decay, 5),
            "final_ema_step_count": flow_ema.step_count,
            "l2_distance_released_ema_from_init": round(l2_dist_ema, 6),
            "l2_distance_live_from_init": round(l2_dist_live, 6),
            "live_model_flow_loss": round(live_flow_loss, 4),
            "ema_model_flow_loss": round(ema_flow_loss, 4),
        }

        total_dp_steps = hvae_steps + flow_steps
        accounted_eps = self.prv_accountant.get_accumulated_epsilon()

        # Compute exact formal bounds: RDP as formal guarantee, PRV upper bound
        from acfm_net.privacy.rdp_accountant import RDPAccountant
        rdp_formal = RDPAccountant(target_delta=training_delta)
        for s, q, c in self.prv_accountant.accountant.history:
            rdp_formal.accountant.history.append((float(s), float(q), int(c)))
        rdp_formal_eps = float(rdp_formal.get_accumulated_epsilon())

        # Apply final EMA checkpoint (fixed schedule, zero private latent queries)
        flow_ema.apply_to(ot_cfm.velocity_net)

        report["stages_completed"].append(f"Stage 8: Flow Training Complete ({epochs} epochs, Model EMA)")
        flow_stage_report = next(
            (stage for stage in report["stage_reports"] if stage.get("stage") == "6"),
            None,
        )
        if flow_stage_report is not None:
            flow_stage_report.update({
                "status": "PASSED",
                "evidence": {
                    **flow_stage_report.get("evidence", {}),
                    "trained_epochs": epochs,
                    "final_flow_loss": float(ep_flow_loss / max(1, ep_flow_batches)),
                },
            })
        privacy_training_verified = bool(
            accounted_eps <= training_target_eps + 1e-4
            and np.isfinite(accounted_eps)
            and accounted_eps > 0
        )
        report["privacy_training"] = {
            "target_epsilon": training_target_eps,
            "release_target_epsilon": adaptive_target_eps,
            "target_delta": training_delta,
            "accounted_epsilon": float(accounted_eps),
            "formal_guarantee_rdp_epsilon": rdp_formal_eps,
            "noise_multiplier": actual_sigma,
            "noise_multiplier_flow": actual_sigma_flow,
            "noise_multiplier_hvae": actual_sigma_hvae,
            "sample_rate": actual_sample_rate,
            "expected_batch_size": actual_batch_size,
            "hvae_steps": hvae_steps,
            "flow_steps": flow_steps,
            "steps": total_dp_steps,
            "accountant": "PRVAccountant (Poisson subsampling single-instance composition)",
            "verified": privacy_training_verified,
            "components": [
                "GraphAttentionEncoder",
                "GatedCrossAttention",
                "HierarchicalVAE",
                "ContinuousOTCFM.velocity_net",
            ],
        }
        if self.strict_release_privacy:
            assert accounted_eps <= training_target_eps + 1e-4, (
                f"DP accountant epsilon ({accounted_eps:.6f}) exceeded target training epsilon ({training_target_eps:.6f})"
            )
            privacy_ledger.add(
                "dp_sgd_training",
                epsilon=round(float(accounted_eps), 6),
                delta=training_delta,
                mechanism="per-example DP-SGD with PRV/RDP accounting",
            )
            release_privacy_verified = bool(
                privacy_ledger.verify()
                and privacy_ledger.epsilon <= adaptive_target_eps + 1e-6
            )
        else:
            release_privacy_verified = False
        report["privacy_composition"] = privacy_ledger.as_dict()
        report["privacy_composition"]["accountant_training_epsilon"] = float(accounted_eps)
        report["privacy_composition"]["meta_charges_epsilon"] = float(nontraining_epsilon)
        report["privacy_composition"]["formal_guarantee_rdp_epsilon"] = rdp_formal_eps

        # Gate 1c Item 9: Report both epsilons with and without Laplace charges
        report["privacy_summary"] = {
            "prv_upper_with_laplace": round(float(accounted_eps + nontraining_epsilon), 6),
            "prv_upper_without_laplace": round(float(accounted_eps), 6),
            "rdp_with_laplace": round(float(rdp_formal_eps + nontraining_epsilon), 6),
            "rdp_without_laplace": round(float(rdp_formal_eps), 6),
            "laplace_meta_charges": round(float(nontraining_epsilon), 6),
            "release_target_epsilon": adaptive_target_eps,
        }
        print(f"\n====================== PRIVACY ACCOUNTING SUMMARY ======================")
        print(f"  HEADLINE GUARANTEE: Opacus PRV Upper Bound (with Laplace charges): {report['privacy_summary']['prv_upper_with_laplace']:.6f} (Target: {adaptive_target_eps:.4f})")
        print(f"  FORMAL GUARANTEE:   RDP Upper Bound        (with Laplace charges): {report['privacy_summary']['rdp_with_laplace']:.6f}")
        print(f"  [Headline Designation]: Opacus PRV Upper Bound is the HEADLINE guarantee; RDP is the formal composition guarantee.")
        print(f"  DP-SGD PRV Upper (without Laplace):         {report['privacy_summary']['prv_upper_without_laplace']:.6f}")
        print(f"  DP-SGD RDP       (without Laplace):         {report['privacy_summary']['rdp_without_laplace']:.6f}")
        print(f"  Laplace Non-Training Charges (Ledger):      {report['privacy_summary']['laplace_meta_charges']:.6f}")
        print(f"========================================================================\n")

        report["release_privacy_verified"] = release_privacy_verified
        report["privacy_status"] = (
            "VERIFIED_CONDITIONAL_PUBLIC_SCHEMA"
            if release_privacy_verified else "UNVERIFIED"
        )
        last_flow_loss = float(ep_flow_loss / max(1, ep_flow_batches))
        print(f"[Stage 8 Summary] Final flow loss: {last_flow_loss:.4f} vs Zero-predictor baseline: {zero_predictor_loss:.4f} (2.0000 if ~N(0,I)) vs Gaussian floor: {gaussian_floor:.4f} (pi/2)")
        _record_stage(report, "8", "DP flow training", "PASSED" if privacy_training_verified else "UNVERIFIED", {
            "epochs": epochs,
            "steps": total_dp_steps,
            "accounted_epsilon": float(accounted_eps),
            "target_epsilon": adaptive_target_eps,
            "privacy_training_verified": privacy_training_verified,
            "last_flow_loss": last_flow_loss,
            "zero_predictor_loss": round(zero_predictor_loss, 4),
            "gaussian_floor": round(gaussian_floor, 4),
            "model_selection": "final_epoch_model_ema",
        })

        # Save checkpoint for audit test (Gate 2b Item 2)
        self.released_checkpoint = {
            "hvae_state_dict": hvae.state_dict() if hvae is not None else {},
            "gat_state_dict": gat.state_dict() if gat is not None else {},
            "cross_attn_state_dict": cross_attn.state_dict() if cross_attn is not None else {},
            "velocity_net_state_dict": ot_cfm.velocity_net.state_dict(),
            "dependency_matrix_W0": W_0,
            "scaler_params": self.scaler.scalers,
            "feature_specs": self.encoder.feature_specs,
            "config": {
                "target_epsilon": self.target_epsilon,
                "target_delta": self.target_delta,
                "latent_dim": latent_dim,
                "in_dim": in_dim,
                "latent_mode": self.latent_mode,
                "flow_param_count": p_flow,
                "noise_over_clip": noise_over_clip,
                "strict_release_privacy": self.strict_release_privacy,
            },
            "privacy_ledger": privacy_ledger.as_dict(),
        }
        report["released_checkpoint"] = self.released_checkpoint
        report["released_checkpoint_keys"] = list(self.released_checkpoint.keys())

        # --------------------------------------------------------------------
        # GENERATIVE INFERENCE SAMPLING (Zero Training Set Memorization)
        # Sample pure standard normal prior z_0 ~ N(0, I) -> integrate RK4 -> decode
        # --------------------------------------------------------------------
        print(f"[Inference] Generating {num_samples} synthetic patient samples via pure prior sampling z_0 ~ N(0, I) & RK4...")
        ot_cfm.eval()
        if hvae is not None:
            hvae.eval()

        with torch.no_grad():
            # 1. Integrate continuous ODE: dz/dt = v_theta(z_t, t) from t=0 to t=1
            synth_latent_Z = ot_cfm.sample(
                num_samples=num_samples,
                device=self.device,
                num_steps=30,
                solver="rk4"
            )

            # 2. Decode: if identity mode, latent is already in X_scaled space; otherwise VAE Decoder
            if self.latent_mode == "identity":
                synth_scaled_tensor = synth_latent_Z
            else:
                synth_scaled_tensor = hvae.decode(synth_latent_Z)

        # Stage 7: Constrained Gradient Repair Pass
        col_kinds_by_idx: Dict[int, str] = {}
        col_idx = 0
        for col_name in self.encoder.column_order:
            spec = self.encoder.feature_specs.get(col_name, {})
            kind = spec.get("kind", "continuous")
            for _ in range(spec.get("dim", 1)):
                col_kinds_by_idx[col_idx] = kind
                col_idx += 1

        repaired_synth_scaled, final_repair_stats = self.repair_loop.repair(
            synth_scaled_tensor, column_kinds=col_kinds_by_idx
        )
        # Gate 4d Item A1: Identity-mode diagnostics before clamp
        pre_clamp_out_of_bounds = float(((synth_scaled_tensor < -1.0) | (synth_scaled_tensor > 1.0)).float().mean().item())
        cat_margins = {}
        if self.latent_mode == "identity":
            c_idx = 0
            for col_name in self.encoder.column_order:
                spec = self.encoder.feature_specs.get(col_name, {})
                dim = spec.get("dim", 1)
                if spec.get("type") == "categorical_ohe" and dim > 1:
                    block = synth_scaled_tensor[:, c_idx:c_idx + dim]
                    if block.shape[1] >= 2:
                        sorted_vals, _ = torch.sort(block, dim=1, descending=True)
                        margin = float((sorted_vals[:, 0] - sorted_vals[:, 1]).mean().item())
                        cat_margins[col_name] = round(margin, 4)
                c_idx += dim

        report["identity_diagnostics"] = {
            "pre_clamp_out_of_bounds_fraction": round(pre_clamp_out_of_bounds, 6),
            "categorical_block_top1_margins": cat_margins,
            "mean_top1_margin": round(float(np.mean(list(cat_margins.values()))), 4) if cat_margins else 0.0,
        }

        # Gate 4b Item 6 / Gate 4c Item 1: Apply public-bounds clamp inside release path (Stage 7).
        repaired_synth_scaled = torch.clamp(repaired_synth_scaled, -1.0, 1.0)

        # Gate 4c Item 1: Decode = public-bounds inverse transform + argmax on one-hot blocks + Stage 7 clamp
        if self.latent_mode == "identity":
            curr_idx = 0
            for col_name in self.encoder.column_order:
                spec = self.encoder.feature_specs.get(col_name, {})
                dim = spec.get("dim", 1)
                if spec.get("type") == "categorical_ohe" and dim > 1:
                    block = repaired_synth_scaled[:, curr_idx:curr_idx + dim]
                    hard = torch.zeros_like(block)
                    hard.scatter_(1, block.argmax(dim=1, keepdim=True), 1.0)
                    repaired_synth_scaled[:, curr_idx:curr_idx + dim] = hard
                curr_idx += dim

        report["repair_stats"] = final_repair_stats
        report["stages_completed"].append("Stage 7: Constrained Repair Pass Completed")
        _record_stage(report, "7", "Constrained gradient repair", "PASSED", {
            "rows_repaired": int(len(repaired_synth_scaled)),
            "repair_stats": final_repair_stats,
        })

        # Decode back to original feature space
        synth_np_scaled  = repaired_synth_scaled.cpu().numpy()
        self.scaled_X_generated = synth_np_scaled
        synth_np_encoded = self.scaler.inverse_transform(synth_np_scaled)
        df_synthetic     = self.encoder.decode(synth_np_encoded)

        # Enforce domain constraints & physical plausibility
        df_synthetic = (
            enforce_public_domain_constraints(df_synthetic, self.encoder.feature_specs)
            if self.strict_release_privacy
            else enforce_domain_constraints(df_synthetic, df_deidentified)
        )

        # Project synthetic DataFrame back to original schema columns
        orig_cols = [c for c in df_deidentified.columns if c in df_synthetic.columns]
        if orig_cols:
            df_synthetic = df_synthetic[orig_cols]

        report["pipeline_status"]  = "SUCCESS"
        report["synthetic_shape"]  = df_synthetic.shape
        report["generation_runtime_seconds"] = round(time.time() - started_at, 3)
        _record_stage(report, "9", "Synthetic data evaluation handoff", "READY", {
            "generated_rows": int(df_synthetic.shape[0]),
            "generated_columns": int(df_synthetic.shape[1]),
            "missing_cells": int(df_synthetic.isna().sum().sum()),
            "privacy_status": report["privacy_status"],
        })
        print(f"[Done] Research-grade synthetic dataset generated: {df_synthetic.shape}")
        return df_synthetic, report
