"""
TRUSTMED Dataset Download Script
Downloads real samples from original public medical datasets:
- NIH ChestX-ray14 (Lung)
- BraTS 2021 (Brain)
- ISIC 2019 (Skin)
- APTOS 2019 Blindness Detection (Retina)

Strictly respects the 7 GB total dataset storage budget (preferred 5-7 GB).
"""

import os
import sys
import shutil
import urllib.request
import zipfile
import tarfile
import json
import csv
from pathlib import Path
from typing import Dict, Tuple

# Configuration
STORAGE_BUDGET_GB = 7.0
BASE_DIR = Path("data/official_subset")

# Download URLs
URLS = {
    "lung_meta": "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/Data_Entry_2017_v2020.csv",
    "lung_zip": "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/images_001.zip",
    "brain_tar": "https://huggingface.co/datasets/Unknown6197/Brats2021_sliced/resolve/main/sliced_data.tar.gz",
    "skin_zip": "https://huggingface.co/datasets/MKZuziak/ISIC_2019_224/resolve/main/data.zip",
    "retina_csv": "https://huggingface.co/datasets/Tejaswini628/aptos-fundus-images/resolve/main/data/train.csv",
    "retina_img_base": "https://huggingface.co/datasets/Tejaswini628/aptos-fundus-images/resolve/main/data/train_images/"
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TrustMed/1.0"}


def get_dir_size_bytes(path: Path) -> int:
    """Calculate total size of files in a directory in bytes."""
    if not path.exists():
        return 0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def download_file_with_progress(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    req = urllib.request.Request(url, headers=HEADERS)
    print(f"[{desc}] Starting download: {url}")
    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            last_pct = -1
            
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = int(downloaded * 100 / total_size)
                    if pct % 10 == 0 and pct != last_pct:
                        print(f"[{desc}] {pct}% ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)")
                        last_pct = pct
                        
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            print(f"[{desc}] Completed: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
            return True
    except Exception as e:
        print(f"[{desc}] Error downloading {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False
    return False


def check_disk_space() -> Tuple[float, float]:
    """Return total and free disk space in GB for the current working drive."""
    total, used, free = shutil.disk_usage(os.getcwd())
    return total / 1e9, free / 1e9


def download_skin() -> Tuple[int, float]:
    """Download ISIC 2019 dataset subset (~150 MB)."""
    skin_dir = BASE_DIR / "skin"
    img_dir = skin_dir / "images"
    meta_path = skin_dir / "metadata.jsonl"
    
    # Check if already present
    if img_dir.exists() and len(list(img_dir.glob("*.jpeg"))) > 2000 and meta_path.exists():
        count = len(list(img_dir.glob("*.jpeg")))
        size_gb = get_dir_size_bytes(skin_dir) / 1e9
        print(f"[Skin] Already downloaded: {count} images, {size_gb:.2f} GB")
        return count, size_gb
        
    skin_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if cached in huggingface hub
    cache_zip = Path(os.path.expanduser("~/.cache/huggingface/hub/datasets--MKZuziak--ISIC_2019_224/snapshots/37e933395b4c36f987019275dbd95e4f247ce27c/data.zip"))
    zip_dest = skin_dir / "data.zip"
    
    if cache_zip.exists():
        print(f"[Skin] Using cached zip from {cache_zip}")
        zip_to_extract = cache_zip
    else:
        print("[Skin] Downloading ISIC 2019 resized dataset zip...")
        success = download_file_with_progress(URLS["skin_zip"], zip_dest, desc="Skin ISIC")
        if not success:
            raise RuntimeError("Failed to download ISIC 2019 dataset")
        zip_to_extract = zip_dest

    print("[Skin] Extracting ISIC 2019 images and metadata...")
    with zipfile.ZipFile(zip_to_extract) as zf:
        # Extract metadata.jsonl
        zf.extract("data/metadata.jsonl", skin_dir)
        extracted_meta = skin_dir / "data" / "metadata.jsonl"
        if extracted_meta.exists():
            if meta_path.exists():
                meta_path.unlink()
            extracted_meta.rename(meta_path)
            shutil.rmtree(skin_dir / "data", ignore_errors=True)
            
        # Extract images flattening structure into images/
        count = 0
        for member in zf.namelist():
            if member.endswith(".jpeg"):
                fname = Path(member).name
                with zf.open(member) as source, open(img_dir / fname, "wb") as target:
                    shutil.copyfileobj(source, target)
                count += 1
                if count % 5000 == 0:
                    print(f"[Skin] Extracted {count} images...")

    if zip_dest.exists():
        zip_dest.unlink()
        
    size_gb = get_dir_size_bytes(skin_dir) / 1e9
    print(f"[Skin] Complete: {count} images extracted, {size_gb:.2f} GB")
    return count, size_gb


def download_brain() -> Tuple[int, float]:
    """Download BraTS 2021 sliced MRI dataset (~500 MB download, ~0.9 GB extracted)."""
    brain_dir = BASE_DIR / "brain"
    raw_dir = brain_dir / "raw_slices"
    
    if raw_dir.exists() and len(list(raw_dir.glob("*.h5"))) > 1000:
        subjects = set(p.name.split("_", 2)[-1].replace(".h5", "") for p in raw_dir.glob("*.h5"))
        size_gb = get_dir_size_bytes(brain_dir) / 1e9
        print(f"[Brain] Already downloaded: {len(subjects)} subjects, {size_gb:.2f} GB")
        return len(subjects), size_gb
        
    brain_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    tar_dest = brain_dir / "sliced_data.tar.gz"
    print("[Brain] Downloading BraTS 2021 sliced dataset...")
    success = download_file_with_progress(URLS["brain_tar"], tar_dest, desc="Brain BraTS")
    if not success:
        raise RuntimeError("Failed to download BraTS 2021 dataset")
        
    print("[Brain] Extracting BraTS 2021 slices...")
    with tarfile.open(tar_dest, "r:gz") as tar:
        count = 0
        for member in tar:
            if member.name.endswith(".h5"):
                fname = Path(member.name).name
                member.name = fname
                tar.extract(member, raw_dir)
                count += 1
                if count % 1000 == 0:
                    print(f"[Brain] Extracted {count} slices...")
                    
    # Clean up archive to respect storage budget
    if tar_dest.exists():
        tar_dest.unlink()

    # Retain target 50 complete subjects to stay strictly within 5-7 GB budget
    all_files = list(raw_dir.glob("*.h5"))
    all_subjs = sorted(list(set(p.name.split("_", 2)[-1].replace(".h5", "") for p in all_files)))
    target_50_subjs = set(all_subjs[:50])
    for p in all_files:
        subj = p.name.split("_", 2)[-1].replace(".h5", "")
        if subj not in target_50_subjs:
            p.unlink()
        
    subjects = set(p.name.split("_", 2)[-1].replace(".h5", "") for p in raw_dir.glob("*.h5"))
    count = len(list(raw_dir.glob("*.h5")))
    size_gb = get_dir_size_bytes(brain_dir) / 1e9
    print(f"[Brain] Complete: {len(subjects)} subjects ({count} slices), {size_gb:.2f} GB")
    return len(subjects), size_gb


def download_lung() -> Tuple[int, float]:
    """Download NIH ChestX-ray14 images_001.zip (~1.87 GB download, ~1.95 GB extracted)."""
    lung_dir = BASE_DIR / "lung"
    img_dir = lung_dir / "images"
    meta_path = lung_dir / "Data_Entry_2017_v2020.csv"
    
    if img_dir.exists() and len(list(img_dir.glob("*.png"))) > 3000 and meta_path.exists():
        count = len(list(img_dir.glob("*.png")))
        size_gb = get_dir_size_bytes(lung_dir) / 1e9
        print(f"[Lung] Already downloaded: {count} images, {size_gb:.2f} GB")
        return count, size_gb
        
    lung_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Download metadata CSV
    if not meta_path.exists():
        print("[Lung] Downloading Data_Entry_2017_v2020.csv...")
        download_file_with_progress(URLS["lung_meta"], meta_path, desc="Lung Metadata")
        
    # Download images_001.zip
    zip_dest = lung_dir / "images_001.zip"
    print("[Lung] Downloading NIH ChestX-ray14 images_001.zip (~1.87 GB)...")
    success = download_file_with_progress(URLS["lung_zip"], zip_dest, desc="Lung ChestX-ray14")
    if not success:
        raise RuntimeError("Failed to download NIH ChestX-ray14 archive")
        
    print("[Lung] Extracting ChestX-ray14 images...")
    with zipfile.ZipFile(zip_dest) as zf:
        count = 0
        for member in zf.namelist():
            if member.endswith(".png"):
                fname = Path(member).name
                with zf.open(member) as source, open(img_dir / fname, "wb") as target:
                    shutil.copyfileobj(source, target)
                count += 1
                if count % 1000 == 0:
                    print(f"[Lung] Extracted {count} images...")
                    
    # Clean up archive to stay strictly within 7 GB budget
    if zip_dest.exists():
        zip_dest.unlink()
        
    size_gb = get_dir_size_bytes(lung_dir) / 1e9
    print(f"[Lung] Complete: {count} images, {size_gb:.2f} GB")
    return count, size_gb


def download_retina() -> Tuple[int, float]:
    """
    Download APTOS 2019 Blindness Detection dataset.
    Downloads metadata train.csv, then downloads a stratified subset of ~1,150 real high-res images (~2.8 GB)
    to keep total dataset storage within 5-7 GB.
    """
    retina_dir = BASE_DIR / "retina"
    img_dir = retina_dir / "images"
    csv_path = retina_dir / "train.csv"
    
    retina_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Download train.csv
    if not csv_path.exists():
        print("[Retina] Downloading APTOS 2019 train.csv...")
        download_file_with_progress(URLS["retina_csv"], csv_path, desc="Retina CSV")
        
    # Read metadata and select stratified subset across all 5 grades
    # Target: ~300 Grade 0, ~200 Grade 1, ~300 Grade 2, ~150 Grade 3, ~200 Grade 4 = 1,150 images
    samples_per_grade = {0: 300, 1: 200, 2: 300, 3: 150, 4: 200}
    
    import random
    random.seed(42)
    
    grade_map = {0: [], 1: [], 2: [], 3: [], 4: []}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = int(row["diagnosis"])
            if g in grade_map:
                grade_map[g].append(row["id_code"])
                
    selected_ids = []
    for g, target_n in samples_per_grade.items():
        avail = grade_map[g]
        random.shuffle(avail)
        chosen = avail[:min(target_n, len(avail))]
        selected_ids.extend(chosen)
        print(f"[Retina] Selected {len(chosen)} / {len(avail)} images for Grade {g}")
        
    print(f"[Retina] Total selected images for download: {len(selected_ids)}")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Download selected images concurrently
    count = 0
    total = len(selected_ids)
    
    def fetch_img(id_code: str) -> bool:
        img_path = img_dir / f"{id_code}.png"
        if img_path.exists() and img_path.stat().st_size > 1000:
            return True
        url = f"{URLS['retina_img_base']}{id_code}.png"
        req = urllib.request.Request(url, headers=HEADERS)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp, open(img_path, "wb") as out:
                    shutil.copyfileobj(resp, out)
                return True
            except Exception as e:
                if attempt == 2:
                    print(f"[Retina] Failed {id_code}.png: {e}")
                import time; time.sleep(0.5)
        return False

    print(f"[Retina] Downloading {total} images using 12 concurrent workers...")
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch_img, code): code for code in selected_ids}
        for fut in as_completed(futures):
            if fut.result():
                count += 1
            if count % 100 == 0 or count == total:
                print(f"[Retina] Progress: {count}/{total} images ({count*100//total}%)")
            
    size_gb = get_dir_size_bytes(retina_dir) / 1e9
    print(f"[Retina] Complete: {count} images, {size_gb:.2f} GB")
    return count, size_gb


def main():
    print("=" * 60)
    print("TRUSTMED AUTOMATIC DATASET DOWNLOAD PIPELINE")
    print(f"Target Storage Budget: {STORAGE_BUDGET_GB:.1f} GB (Preferred: 5.0 - 7.0 GB)")
    print("=" * 60)
    
    total_disk, free_disk = check_disk_space()
    print(f"Disk check: Total {total_disk:.1f} GB, Free {free_disk:.1f} GB")
    if free_disk < 15.0:
        print(f"WARNING: Free disk space ({free_disk:.1f} GB) is low. At least 15 GB recommended for temporary files.")
        
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Download Skin (ISIC 2019) ~0.15 GB
    print("\n--- 1. DOWNLOADING SKIN (ISIC 2019) ---")
    skin_samples, skin_gb = download_skin()
    
    # 2. Download Brain (BraTS 2021) ~0.95 GB
    print("\n--- 2. DOWNLOADING BRAIN (BraTS 2021) ---")
    brain_subjects, brain_gb = download_brain()
    
    # 3. Download Lung (NIH ChestX-ray14) ~1.95 GB
    print("\n--- 3. DOWNLOADING LUNG (NIH ChestX-ray14) ---")
    lung_samples, lung_gb = download_lung()
    
    # 4. Download Retina (APTOS 2019) ~2.80 GB
    print("\n--- 4. DOWNLOADING RETINA (APTOS 2019) ---")
    retina_samples, retina_gb = download_retina()
    
    # Total calculation
    total_storage_gb = skin_gb + brain_gb + lung_gb + retina_gb
    budget_status = "PASS" if total_storage_gb <= STORAGE_BUDGET_GB else "EXCEEDED"
    
    print("\n" + "=" * 60)
    print("TRUSTMED DATASET DOWNLOAD REPORT")
    print("=" * 60)
    print("\nLung:")
    print("Target samples: 2,000-3,000")
    print(f"Downloaded samples: {lung_samples}")
    print(f"Storage: {lung_gb:.2f} GB")
    
    print("\nBrain:")
    print("Target subjects: 30-50")
    print(f"Downloaded subjects: {brain_subjects}")
    print(f"Storage: {brain_gb:.2f} GB")
    
    print("\nSkin:")
    print("Target samples: 2,500-3,500")
    print(f"Downloaded samples: {skin_samples}")
    print(f"Storage: {skin_gb:.2f} GB")
    
    print("\nRetina:")
    print("Target samples: 1,000-1,500")
    print(f"Downloaded samples: {retina_samples}")
    print(f"Storage: {retina_gb:.2f} GB")
    
    print("\n" + "-" * 32)
    print(f"TOTAL DATASET STORAGE: {total_storage_gb:.2f} GB")
    print("-" * 32)
    print(f"Budget: {STORAGE_BUDGET_GB:.1f} GB")
    print(f"Status: {budget_status}")
    print("=" * 60)
    
    if budget_status == "EXCEEDED":
        print("ERROR: Total dataset storage exceeded budget! Clean up unneeded files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
