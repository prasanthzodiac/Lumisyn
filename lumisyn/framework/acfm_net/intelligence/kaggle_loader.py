# ==============================================================================
# QUARANTINED MODULE: DEAD CODE / NON-RELEASE DEVELOPMENT UTILITY
# Excluded from the differential privacy release pipeline and core runtime.
# Retained strictly for historical Kaggle dataset download convenience.
# ==============================================================================
import os
import glob
from typing import Optional, Tuple
import pandas as pd

class KaggleDatasetLoader:
    """
    Automatic Kaggle Dataset Downloader & Loader for ACFM-Net v5.
    Fetches real clinical datasets directly from Kaggle platform (kagglehub / kaggle API).
    """

    KAGGLE_DATASETS = {
        "stroke": "fedesoriano/stroke-prediction-dataset",
        "diabetes": "uciml/pima-indians-diabetes-database",
        "pima": "uciml/pima-indians-diabetes-database",
        "heart": "johnsmith88/heart-disease-dataset",
        "heart_disease": "ronitf/heart-disease-uci",
        "nhanes": "cdc/nhanes",
        "mimic": "mimic-iv-demo"
    }

    def __init__(self, cache_dir: str = "kaggle_datasets"):
        if cache_dir == "kaggle_datasets":
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            cache_dir = os.path.join(project_root, "data", "kaggle_datasets")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load_dataset(self, dataset_key_or_handle: str) -> Tuple[pd.DataFrame, str]:
        """
        Downloads dataset from Kaggle website and returns (DataFrame, dataset_name).
        """
        handle = self.KAGGLE_DATASETS.get(dataset_key_or_handle.lower(), dataset_key_or_handle)
        print(f"[Kaggle Loader] Downloading dataset '{handle}' directly from Kaggle website...")

        download_path = None
        try:
            import kagglehub
            download_path = kagglehub.dataset_download(handle)
            print(f"[Kaggle Loader] Successfully downloaded '{handle}' to {download_path}")
        except Exception as e:
            print(f"[Kaggle Loader] kagglehub warning: {e}. Searching local cache...")

        # Find CSV files in downloaded path or cache_dir
        search_dirs = [download_path, self.cache_dir] if download_path else [self.cache_dir]
        csv_files = []
        for s_dir in search_dirs:
            if s_dir and os.path.exists(s_dir):
                csv_files.extend(glob.glob(os.path.join(s_dir, "*.csv")))
                csv_files.extend(glob.glob(os.path.join(s_dir, "**", "*.csv"), recursive=True))

        if not csv_files:
            raise FileNotFoundError(f"Could not find downloaded CSV files for Kaggle dataset handle '{handle}'.")

        # Pick primary CSV file (largest or first match)
        primary_csv = max(csv_files, key=os.path.getsize)
        print(f"[Kaggle Loader] Loading primary dataset CSV: {os.path.basename(primary_csv)}")

        df = pd.read_csv(primary_csv)
        print(f"[Kaggle Loader] Loaded DataFrame shape: {df.shape}")
        return df, handle
