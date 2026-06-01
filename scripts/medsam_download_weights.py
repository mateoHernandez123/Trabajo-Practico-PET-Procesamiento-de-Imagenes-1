#!/usr/bin/env python3
"""Descarga (o pre-cachea) los pesos pre-entrenados de MedSAM.

Usa el mirror oficial en HuggingFace `flaviagiammarino/medsam-vit-base`
(que es la conversión a formato HF Transformers de los pesos
originales `medsam_vit_b.pth` de Wang Lab — ~375 MB).

Se cachean en ~/.cache/huggingface/hub/, ese es el comportamiento
estándar de transformers (no se almacenan dentro del proyecto).

Uso:
    .venv-medsam/bin/python scripts/medsam_download_weights.py
    .venv-medsam/bin/python scripts/medsam_download_weights.py --model-id <otro-id-HF>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_MODEL_ID = "flaviagiammarino/medsam-vit-base"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Repo HuggingFace con los pesos de MedSAM (default: {DEFAULT_MODEL_ID}).",
    )
    args = parser.parse_args()

    try:
        from transformers import SamModel, SamProcessor
    except ImportError:
        print("ERROR: falta `transformers`. Instalá con:", file=sys.stderr)
        print("  .venv-medsam/bin/pip install -r requirements-medsam.txt", file=sys.stderr)
        return 1

    print(f"[1/2] Descargando modelo: {args.model_id}")
    t0 = time.time()
    model = SamModel.from_pretrained(args.model_id)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      OK. Parámetros: {n_params/1e6:.1f}M  |  tiempo: {time.time()-t0:.1f}s")

    print(f"[2/2] Descargando processor (image_processor + tokenizer config)...")
    t0 = time.time()
    processor = SamProcessor.from_pretrained(args.model_id)
    print(f"      OK. Tipo: {type(processor).__name__}  |  tiempo: {time.time()-t0:.1f}s")

    cache = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"\nCache HuggingFace: {cache}")
    print("Modelo listo. Probá la inferencia con:")
    print("  .venv-medsam/bin/python scripts/medsam_run_inference.py --help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
