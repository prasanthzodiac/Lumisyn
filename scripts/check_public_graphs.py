import sys
sys.path.insert(0, "lumisyn/framework")
import pandas as pd
import numpy as np
from acfm_net.graph.builder import DomainConstraintGraphBuilder
from acfm_net.intelligence.sensitive import SensitiveAttributeDetector

datasets = {
    "Adult": "lumisyn/data/kaggle_datasets/adult.csv",
    "Cardio": "lumisyn/data/kaggle_datasets/cardio_train.csv",
    "Pima": "lumisyn/data/kaggle_datasets/pima_diabetes.csv",
    "Wisconsin": "lumisyn/data/kaggle_datasets/wisconsin_clean.csv",
}

detector = SensitiveAttributeDetector()
builder = DomainConstraintGraphBuilder(corr_threshold=0.10, edge_prune_threshold=0.10)

print("=== Stage 3 (DCGB) Public Graph Edge Report ===")
results = {}
for name, path in datasets.items():
    df = pd.read_csv(path, nrows=50)
    cols = list(df.columns)
    d = len(cols)
    sens = detector.detect(df, use_data_dependent_risk=False)["sensitivity_vector"]
    # Pure public graph: no private dependency matrix (identity), no private df_encoded
    G, W, info = builder.build_graph(
        columns=cols,
        dependency_matrix=np.eye(d, dtype=np.float32),
        sensitivity_vector=sens,
        df_encoded=None
    )
    edge_list = [(cols[u], cols[v], round(data.get("weight", 0.0), 3)) for u, v, data in G.edges(data=True)]
    results[name] = {
        "num_nodes": info["num_nodes"],
        "num_edges": info["num_edges"],
        "clinical_edges": info["clinical_edges"],
        "edges": edge_list,
        "graph_inactive": (info["num_edges"] == 0 or info["clinical_edges"] == 0),
    }
    print(f"\nDataset: {name}")
    print(f"  Nodes: {info['num_nodes']}")
    print(f"  Total Edges: {info['num_edges']}")
    print(f"  Clinical Hard Edges: {info['clinical_edges']}")
    print(f"  Graph Inactive: {results[name]['graph_inactive']}")
    if edge_list:
        print("  Sample Edges:")
        for u, v, w in edge_list[:10]:
            print(f"    - {u} <---> {v} (w={w})")
    else:
        print("  No public domain constraint edges found.")
