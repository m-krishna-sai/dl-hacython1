"""
TRUSTMED System Health & Diagnostic Endpoints
"""

import sys
import psutil
import torch
from pathlib import Path
from typing import Dict, Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.models.loader import get_registry

def get_system_health() -> Dict[str, Any]:
    registry = get_registry()
    mem = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=None)

    return {
        "status": "HEALTHY",
        "service": "TrustMed Multi-Specialty Medical AI",
        "version": "1.0.0",
        "device": str(registry.device),
        "models_cached": list(registry.models.keys()),
        "all_models_loaded": registry.is_loaded,
        "specialties_supported": ["lung", "brain", "skin", "retina"],
        "system_resources": {
            "ram_used_gb": round((mem.total - mem.available) / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_percent": mem.percent,
            "cpu_percent": cpu_pct
        }
    }
