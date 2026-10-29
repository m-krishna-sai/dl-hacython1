"""
TRUSTMED Real-Time Performance & Resource Benchmark
Measures actual:
- Model startup loading time
- Preprocessing latency
- Baseline inference latency
- Trust Engine latency (MC Dropout, Temperature scaling, OOD, Stability)
- Grad-CAM heatmap generation latency
- Total turnaround latency across batch sizes: 1, 2, 3, 4 images
- Memory usage (RSS)
Saves results to reports/benchmark_report.json.
"""

import time
import sys
import json
import psutil
import torch
from pathlib import Path
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.models.loader import get_registry
from backend.api.inference import process_single_image, process_batch

def run_benchmark():
    print("==================================================")
    print("   TRUSTMED REAL-TIME PERFORMANCE BENCHMARK       ")
    print("   Target CPU: Intel Core i5-1334U (Laptop CPU)   ")
    print("==================================================")

    # 1. Model Loading Benchmark
    t0 = time.perf_counter()
    registry = get_registry()
    registry.load_all_models()
    load_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"[OK] Startup Model Loading Time: {load_time_ms} ms")

    # Load test image
    demo_dir = WORKSPACE_ROOT / "data" / "demo_cases"
    img_path = demo_dir / "case_a_reliable_cxr.png"
    img = Image.open(img_path).convert("RGB")

    # Warmup pass
    _ = process_single_image(img, "warmup.png")

    # 2. Granular Stage Latency Measurement (Single Image)
    single_res = process_single_image(img, "benchmark_cxr.png")
    latencies = single_res["latency_ms"]
    print("\nSingle Image Stage Breakdown:")
    for stage, ms in latencies.items():
        print(f"  - {stage.capitalize()}: {ms} ms")

    # 3. Batch Scaling Benchmark (1, 2, 3, 4 images)
    print("\nBatch Scaling Benchmark (1 - 4 images):")
    batch_benchmarks = {}
    for batch_size in [1, 2, 3, 4]:
        batch_images = [(img, f"scan_{i+1}.png") for i in range(batch_size)]
        t_b0 = time.perf_counter()
        b_res = process_batch(batch_images)
        total_b_ms = round((time.perf_counter() - t_b0) * 1000, 1)
        avg_per_img = round(total_b_ms / batch_size, 1)
        batch_benchmarks[f"{batch_size}_images"] = {
            "total_latency_ms": total_b_ms,
            "average_per_image_ms": avg_per_img,
            "throughput_images_per_sec": round(1000.0 / avg_per_img, 2)
        }
        print(f"  * Batch {batch_size} Image(s): Total {total_b_ms} ms ({avg_per_img} ms/image, {round(1000.0/avg_per_img, 1)} img/s)")

    # 4. Memory Footprint
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_mb = round(mem_info.rss / (1024 * 1024), 1)
    print(f"\nResident Set Size (RAM): {rss_mb} MB")

    report = {
        "benchmark_timestamp": "2026-09-12T13:38:00Z",
        "device": str(registry.device),
        "cpu_info": "Intel Core i5-1334U (12GB RAM, Iris Xe Graphics)",
        "model_loading_latency_ms": load_time_ms,
        "single_image_stage_breakdown_ms": latencies,
        "batch_scaling": batch_benchmarks,
        "memory_rss_mb": rss_mb
    }

    reports_dir = WORKSPACE_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / "benchmark_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nBenchmark completed successfully! Saved to {report_file}")
    return report

if __name__ == "__main__":
    run_benchmark()
