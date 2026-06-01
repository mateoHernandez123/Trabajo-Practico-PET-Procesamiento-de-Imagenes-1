#!/usr/bin/env python3
"""Runner CLI de MedSAM sobre fotos/imágenes 2D médicas.

Thin orchestrator del paquete `scripts/medsam`. Permite procesar una
imagen o una carpeta completa y vuelca los resultados (mask, overlay,
features.csv, meta.json) en la carpeta de salida indicada.

Estrategias de auto-prompt (--bbox):
    auto        Detecta anotación verde válida → bbox del círculo.
                Sin anotación → fallback "full image" con 5% margen.
    full        bbox = imagen completa con 5% margen. Útil para PET con
                hot spot claro contra fondo oscuro/blanco.
    x1,y1,x2,y2 Coords explícitas en píxeles de la imagen original.

Ejemplo:
    .venv-medsam/bin/python scripts/medsam_run.py \\
        --input  imagenes/clinicas_referencia/whatsapp_2026-05-30 \\
        --output resultados_medsam/whatsapp_2026-05-30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Agregar `scripts/` al path para que `import medsam` funcione cuando se
# ejecuta directamente como `python scripts/medsam_run.py`.
sys.path.insert(0, str(Path(__file__).parent))

from medsam import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_THRESHOLD,
    load_medsam,
    run_one,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path, help="Imagen o carpeta de entrada")
    parser.add_argument("--output", required=True, type=Path, help="Carpeta de salida")
    parser.add_argument("--bbox", default="auto", help="auto | full | x1,y1,x2,y2 (default: auto)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Threshold sigmoid (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--device", default="cpu", help="cpu | cuda (default: cpu)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--no-overlay", action="store_true", help="No generar PNG con overlay")
    parser.add_argument("--limit", type=int, default=0, help="Procesar solo N imágenes (0 = todas)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: no existe {args.input}", file=sys.stderr)
        return 1

    if args.input.is_dir():
        exts = {".jpg", ".jpeg", ".png"}
        files = sorted([p for p in args.input.rglob("*") if p.suffix.lower() in exts])
        if args.limit:
            files = files[: args.limit]
    else:
        files = [args.input]

    if not files:
        print(f"ERROR: ningún archivo de imagen en {args.input}", file=sys.stderr)
        return 1

    model, processor = load_medsam(args.model_id, args.device)

    print(f"\n[MedSAM] Procesando {len(files)} imagen(es) → {args.output}")
    print(f"[MedSAM] bbox={args.bbox}  threshold={args.threshold}  device={args.device}\n")

    all_meta = []
    t_total_start = time.time()
    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp.name}")
        try:
            meta = run_one(
                fp, args.output, args.bbox, args.threshold, args.device,
                model, processor, save_overlay=not args.no_overlay,
                model_id=args.model_id,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        print(
            f"    bbox={meta['bbox']['source']}  "
            f"mask={meta['inference']['mask_pixels']}px "
            f"({meta['inference']['mask_fraction']*100:.2f}%)  "
            f"features={meta['n_features']}  "
            f"t={meta['timing']['total_s']:.2f}s"
        )
        all_meta.append(meta)

    summary_path = args.output / "_summary.json"
    summary_path.write_text(json.dumps({
        "files_processed": len(all_meta),
        "files_total": len(files),
        "elapsed_s": round(time.time() - t_total_start, 1),
        "per_file": all_meta,
    }, indent=2, ensure_ascii=False))
    print(f"\n[MedSAM] Listo. Resumen: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
