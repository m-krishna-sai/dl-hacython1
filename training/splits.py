"""
TRUSTMED Patient-Level Stratified Splitting
Ensures zero data leakage between train, validation, and test splits.
All slices, multi-views, or multi-images from the same patient/subject remain strictly in the same split.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

def split_by_patient(
    samples: List[Dict[str, Any]],
    patient_key: str = "patient_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partitions samples into train, val, and test splits strictly grouped by patient ID.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    random.seed(seed)

    # Group sample records by patient ID
    patient_to_samples: Dict[str, List[Dict[str, Any]]] = {}
    for item in samples:
        pid = str(item.get(patient_key, "UNKNOWN"))
        if pid not in patient_to_samples:
            patient_to_samples[pid] = []
        patient_to_samples[pid].append(item)

    unique_patients = list(patient_to_samples.keys())
    random.shuffle(unique_patients)

    n_total = len(unique_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pids = set(unique_patients[:n_train])
    val_pids = set(unique_patients[n_train:n_train + n_val])
    test_pids = set(unique_patients[n_train + n_val:])

    # Verify disjoint patient sets
    assert len(train_pids.intersection(val_pids)) == 0, "Patient leakage between train and val"
    assert len(train_pids.intersection(test_pids)) == 0, "Patient leakage between train and test"
    assert len(val_pids.intersection(test_pids)) == 0, "Patient leakage between val and test"

    train_samples = [s for pid in train_pids for s in patient_to_samples[pid]]
    val_samples = [s for pid in val_pids for s in patient_to_samples[pid]]
    test_samples = [s for pid in test_pids for s in patient_to_samples[pid]]

    return train_samples, val_samples, test_samples

def generate_all_splits(base_dir: Path):
    """Generates and persists train/val/test splits for all datasets."""
    datasets = ["lung", "brain", "skin", "retina"]
    splits_summary = {}

    for ds in datasets:
        meta_file = base_dir / "data" / ds / "metadata.json"
        if not meta_file.exists():
            continue
        with open(meta_file, "r") as f:
            samples = json.load(f)

        train, val, test = split_by_patient(samples, patient_key="patient_id")

        splits_dir = base_dir / "data" / ds / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        with open(splits_dir / "train.json", "w") as f:
            json.dump(train, f, indent=2)
        with open(splits_dir / "val.json", "w") as f:
            json.dump(val, f, indent=2)
        with open(splits_dir / "test.json", "w") as f:
            json.dump(test, f, indent=2)

        splits_summary[ds] = {
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "train_patients": len(set(s["patient_id"] for s in train)),
            "val_patients": len(set(s["patient_id"] for s in val)),
            "test_patients": len(set(s["patient_id"] for s in test)),
        }

    print("Patient-level splits generated successfully:")
    for ds, counts in splits_summary.items():
        print(f"  [{ds.upper()}] Samples: {counts['train_samples']} train / {counts['val_samples']} val / {counts['test_samples']} test (Patients: {counts['train_patients']}/{counts['val_patients']}/{counts['test_patients']})")
    return splits_summary

if __name__ == "__main__":
    ws = Path(__file__).resolve().parent.parent
    generate_all_splits(ws)
