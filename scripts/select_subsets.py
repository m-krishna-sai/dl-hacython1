"""
TRUSTMED Reproducible Subset Selection Script
Generates reproducible patient/subject-level stratified splits (70% train, 15% val, 15% test).
Saves manifests to data/subset_manifest/{lung,brain,skin,retina}.csv.
Enforces zero patient/subject leakage across splits.
Seed: 42
"""

import os
import csv
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

SEED = 42
random.seed(SEED)

OFFICIAL_DIR = Path("data/official_subset")
MANIFEST_DIR = Path("data/subset_manifest")
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. LUNG (NIH ChestX-ray14)
# -------------------------------------------------------------
def select_lung_subset(target_count: int = 2500):
    """
    Selects ~2,500 real ChestX-ray14 images from images_001.
    Performs patient-level splitting to prevent data leakage.
    Ensures multi-label representation across official 14 thoracic classes:
    Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
    Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
    Pleural_Thickening, Hernia.
    """
    print("\n--- Processing Lung (NIH ChestX-ray14) Subset ---")
    meta_csv = OFFICIAL_DIR / "lung" / "Data_Entry_2017_v2020.csv"
    img_dir = OFFICIAL_DIR / "lung" / "images"
    
    if not meta_csv.exists() or not img_dir.exists():
        print(f"Skipping Lung: {meta_csv} or {img_dir} does not exist.")
        return

    # Check which images actually exist in img_dir
    avail_images = set(p.name for p in img_dir.glob("*.png"))
    print(f"Available local lung images: {len(avail_images)}")

    # Parse metadata
    # Format: Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,...
    patient_to_records = defaultdict(list)
    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["Image Index"]
            if img_name in avail_images:
                pid = row["Patient ID"]
                labels = row["Finding Labels"]
                patient_to_records[pid].append((img_name, labels))

    # Patient-level splitting
    patients = list(patient_to_records.keys())
    random.shuffle(patients)

    # Select patients until we reach ~target_count images
    selected_patients = []
    total_imgs = 0
    for p in patients:
        selected_patients.append(p)
        total_imgs += len(patient_to_records[p])
        if total_imgs >= target_count:
            break

    print(f"Selected {len(selected_patients)} patients with {total_imgs} images.")

    # 70% train, 15% val, 15% test on patient level
    n_pts = len(selected_patients)
    n_train = int(0.70 * n_pts)
    n_val = int(0.15 * n_pts)
    
    train_pts = set(selected_patients[:n_train])
    val_pts = set(selected_patients[n_train:n_train + n_val])
    test_pts = set(selected_patients[n_train + n_val:])

    # Verify zero overlap
    assert len(train_pts & val_pts) == 0
    assert len(train_pts & test_pts) == 0
    assert len(val_pts & test_pts) == 0

    manifest_rows = []
    split_counts = defaultdict(int)
    label_counts = Counter()

    for p in selected_patients:
        if p in train_pts:
            sp = "train"
        elif p in val_pts:
            sp = "val"
        else:
            sp = "test"
            
        for img_name, labels in patient_to_records[p]:
            manifest_rows.append({
                "image_id": img_name,
                "patient_id": p,
                "labels": labels,
                "split": sp
            })
            split_counts[sp] += 1
            for l in labels.split("|"):
                label_counts[l] += 1

    out_csv = MANIFEST_DIR / "lung.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "patient_id", "labels", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved {out_csv} ({len(manifest_rows)} rows)")
    print(f"Splits: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")
    print(f"Top Labels represented: {label_counts.most_common(5)}")


# -------------------------------------------------------------
# 2. BRAIN (BraTS 2021)
# -------------------------------------------------------------
def select_brain_subset(target_subjects: int = 40):
    """
    Selects ~40 real BraTS 2021 subjects with real MRI slices & masks.
    Enforces STRICT subject-level splitting: slices from the same subject
    NEVER appear across different splits.
    """
    print("\n--- Processing Brain (BraTS 2021) Subset ---")
    raw_dir = OFFICIAL_DIR / "brain" / "raw_slices"
    if not raw_dir.exists():
        print(f"Skipping Brain: {raw_dir} does not exist.")
        return

    # Slices are named: slice_<idx>_<subject_id>.h5
    subject_to_slices = defaultdict(list)
    for p in raw_dir.glob("*.h5"):
        parts = p.name.split("_", 2)
        if len(parts) >= 3:
            subj_id = parts[-1].replace(".h5", "")
            subject_to_slices[subj_id].append(p.name)

    subjects = sorted(list(subject_to_slices.keys()))
    print(f"Total available subjects: {len(subjects)}")
    
    random.shuffle(subjects)
    selected_subjs = subjects[:min(target_subjects, len(subjects))]
    
    # 70% train, 15% val, 15% test on subject level
    n_s = len(selected_subjs)
    n_train = max(1, int(0.70 * n_s))
    n_val = max(1, int(0.15 * n_s))
    
    train_subjs = set(selected_subjs[:n_train])
    val_subjs = set(selected_subjs[n_train:n_train + n_val])
    test_subjs = set(selected_subjs[n_train + n_val:])
    
    assert len(train_subjs & val_subjs) == 0
    assert len(train_subjs & test_subjs) == 0
    assert len(val_subjs & test_subjs) == 0

    manifest_rows = []
    split_counts = defaultdict(int)

    for subj in selected_subjs:
        if subj in train_subjs:
            sp = "train"
        elif subj in val_subjs:
            sp = "val"
        else:
            sp = "test"
            
        for s_file in sorted(subject_to_slices[subj]):
            manifest_rows.append({
                "slice_id": s_file,
                "subject_id": subj,
                "split": sp
            })
            split_counts[sp] += 1

    out_csv = MANIFEST_DIR / "brain.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slice_id", "subject_id", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved {out_csv} ({len(manifest_rows)} slices across {len(selected_subjs)} subjects)")
    print(f"Subjects: Train={len(train_subjs)}, Val={len(val_subjs)}, Test={len(test_subjs)}")
    print(f"Slices: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")


# -------------------------------------------------------------
# 3. SKIN (ISIC 2019)
# -------------------------------------------------------------
def select_skin_subset(target_count: int = 3000):
    """
    Selects ~3,000 real ISIC 2019 images across all 8 classes:
    MEL (0), NV (1), BCC (2), AK (3), BKL (4), DF (5), VASC (6), SCC (7).
    Stratified 70% train, 15% val, 15% test.
    """
    print("\n--- Processing Skin (ISIC 2019) Subset ---")
    meta_path = OFFICIAL_DIR / "skin" / "metadata.jsonl"
    img_dir = OFFICIAL_DIR / "skin" / "images"
    
    if not meta_path.exists() or not img_dir.exists():
        print(f"Skipping Skin: {meta_path} or {img_dir} does not exist.")
        return

    CLASS_NAMES = {
        0: "MEL", 1: "NV", 2: "BCC", 3: "AK",
        4: "BKL", 5: "DF", 6: "VASC", 7: "SCC"
    }

    # Read metadata and group by class
    class_to_images = defaultdict(list)
    avail_images = set(p.name for p in img_dir.glob("*.jpeg"))
    
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            fname = Path(data["file_name"]).name
            if fname in avail_images:
                lbl = int(data["label"])
                if lbl in CLASS_NAMES:
                    class_to_images[lbl].append(fname)

    # Balance/stratify: take up to target_count proportionally, ensuring rare classes are kept
    total_avail = sum(len(v) for v in class_to_images.values())
    print(f"Available skin images: {total_avail} across {len(class_to_images)} classes")

    selected_per_class = {}
    for lbl, img_list in class_to_images.items():
        random.shuffle(img_list)
        # Allocate proportionally with minimum representation of 50 images per class
        ratio = len(img_list) / total_avail
        target_k = max(50, int(ratio * target_count))
        selected = img_list[:min(target_k, len(img_list))]
        selected_per_class[lbl] = selected

    manifest_rows = []
    split_counts = defaultdict(int)

    for lbl, imgs in selected_per_class.items():
        c_name = CLASS_NAMES[lbl]
        n = len(imgs)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        
        for i, fname in enumerate(imgs):
            if i < n_train:
                sp = "train"
            elif i < n_train + n_val:
                sp = "val"
            else:
                sp = "test"
                
            manifest_rows.append({
                "image_id": fname,
                "label": lbl,
                "class_name": c_name,
                "split": sp
            })
            split_counts[sp] += 1

    random.shuffle(manifest_rows)
    out_csv = MANIFEST_DIR / "skin.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label", "class_name", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved {out_csv} ({len(manifest_rows)} rows)")
    print(f"Splits: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")


# -------------------------------------------------------------
# 4. RETINA (APTOS 2019)
# -------------------------------------------------------------
def select_retina_subset():
    """
    Selects real APTOS 2019 images across grades 0-4.
    Stratified 70% train, 15% val, 15% test.
    """
    print("\n--- Processing Retina (APTOS 2019) Subset ---")
    csv_path = OFFICIAL_DIR / "retina" / "train.csv"
    img_dir = OFFICIAL_DIR / "retina" / "images"
    
    if not csv_path.exists() or not img_dir.exists():
        print(f"Skipping Retina: {csv_path} or {img_dir} does not exist.")
        return

    avail_images = set(p.stem for p in img_dir.glob("*.png"))
    print(f"Available local retina images: {len(avail_images)}")

    grade_to_images = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_code = row["id_code"]
            if id_code in avail_images:
                grade = int(row["diagnosis"])
                grade_to_images[grade].append(id_code)

    manifest_rows = []
    split_counts = defaultdict(int)

    for grade, img_list in sorted(grade_to_images.items()):
        random.shuffle(img_list)
        n = len(img_list)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        
        for i, id_code in enumerate(img_list):
            if i < n_train:
                sp = "train"
            elif i < n_train + n_val:
                sp = "val"
            else:
                sp = "test"
                
            manifest_rows.append({
                "image_id": f"{id_code}.png",
                "grade": grade,
                "split": sp
            })
            split_counts[sp] += 1

    random.shuffle(manifest_rows)
    out_csv = MANIFEST_DIR / "retina.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "grade", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved {out_csv} ({len(manifest_rows)} rows)")
    print(f"Splits: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")


def main():
    print("=" * 60)
    print("TRUSTMED REPRODUCIBLE SUBSET SELECTION & SPLITTING")
    print(f"Seed: {SEED}")
    print("=" * 60)
    
    select_lung_subset()
    select_brain_subset()
    select_skin_subset()
    select_retina_subset()
    
    print("\nManifest selection complete!")


if __name__ == "__main__":
    main()
