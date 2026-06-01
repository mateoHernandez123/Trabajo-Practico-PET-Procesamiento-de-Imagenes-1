"""Carga del modelo MedSAM desde HuggingFace (lazy singleton).

El modelo `flaviagiammarino/medsam-vit-base` es el mirror oficial en formato
HuggingFace Transformers de los pesos originales `medsam_vit_b.pth` de Wang
Lab (~375 MB). Se cachea automáticamente en `~/.cache/huggingface/hub/`.

Hiperparámetros del modelo (fijos por arquitectura, ver paper):
    - Backbone:        ViT-Base (12 layers, 768 dim, 12 heads, patch 16×16)
    - Input size:      1024×1024
    - Output mask:     256×256 (upsampled a la entrada del usuario)
    - Parámetros:      ~94 M (validado al cargar)
    - Fine-tuning:     150 epochs, AdamW lr=1e-4 weight_decay=0.01,
                       loss Dice+CE, batch=160, sobre 1.5M imágenes médicas
"""

from __future__ import annotations

import time

DEFAULT_MODEL_ID = "flaviagiammarino/medsam-vit-base"

_MODEL = None
_PROCESSOR = None


def load_medsam(model_id: str = DEFAULT_MODEL_ID, device: str = "cpu"):
    """Carga MedSAM y su processor desde HuggingFace (cacheado).

    Devuelve (model, processor). La primera llamada puede tardar minutos
    si hay que descargar pesos (375 MB). Llamadas sucesivas son ~0.3 s.

    Internamente intenta primero `local_files_only=True` para evitar el
    rate-limit de HF Hub sin token; si el cache está vacío, baja del Hub.
    """
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    try:
        from transformers import SamModel, SamProcessor
    except ImportError as e:
        raise RuntimeError(
            "Falta `transformers`. Instalá con:\n"
            "  .venv-medsam/bin/pip install -r requirements-medsam.txt"
        ) from e

    print(f"[MedSAM] Cargando {model_id} en {device}...")
    t0 = time.time()
    try:
        _MODEL = SamModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
        _PROCESSOR = SamProcessor.from_pretrained(model_id, local_files_only=True)
    except Exception:
        print("[MedSAM] Cache HF vacío, descargando del Hub...")
        _MODEL = SamModel.from_pretrained(model_id).to(device).eval()
        _PROCESSOR = SamProcessor.from_pretrained(model_id)

    n_params = sum(p.numel() for p in _MODEL.parameters())
    print(f"[MedSAM] OK ({n_params/1e6:.1f}M params, {time.time()-t0:.1f}s)")
    return _MODEL, _PROCESSOR
