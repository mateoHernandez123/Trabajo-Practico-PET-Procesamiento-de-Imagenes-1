#!/usr/bin/env python3
"""Genera paneles original/máscara/overlay por modalidad.

Lee las imágenes originales + sus outputs MedSAM y compone un panel
side-by-side por categoría (CT cerebro, CT tórax, PET cuerpo, PET cerebro),
útil para el README y la documentación.

Uso:
    .venv-medsam/bin/python scripts/medsam_build_panels.py \\
        --input-dir   imagenes/clinicas_referencia/whatsapp_2026-05-30 \\
        --results-dir resultados_medsam/whatsapp_2026-05-30 \\
        --output-dir  docs/figuras/medsam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from medsam.visualize import build_modality_panel  # noqa: E402


# Mapeo: prefijo del filename → (título del panel, output filename,
#                                cuántas filas mostrar, paso para subsampling)
PANELS = [
    ("ct_cerebro_coronal_anotado", "Panel - CT cerebro coronal (con anotacion verde)",
     "panel_ct_cerebro_coronal_anotado.png", 2, 1),
    ("ct_cerebro_sagital_anotado", "Panel - CT cerebro sagital (con anotacion verde)",
     "panel_ct_cerebro_sagital_anotado.png", 1, 1),
    ("ct_cerebro_axial",           "Panel - CT cerebro axial",
     "panel_ct_cerebro_axial.png", 4, 1),
    ("ct_craneo_basal_axial",      "Panel - CT craneo basal axial",
     "panel_ct_craneo_basal_axial.png", 2, 1),
    ("ct_torax_axial_panel",       "Panel - CT torax axial (muestra de 6 fotos)",
     "panel_ct_torax_axial.png", 6, 5),  # 27 imágenes → mostramos cada 5ª, 6 filas
    ("pet_cuerpo_mip",             "Panel - PET cuerpo completo MIP (muestra de 5 fotos)",
     "panel_pet_cuerpo_mip.png", 5, 2),  # 10 → cada 2ª = 5 filas
    ("pet_cerebro_axial_hotspot",  "Panel - PET cerebral axial con hot spot",
     "panel_pet_cerebro_hotspot.png", 3, 1),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Carpeta con las fotos originales (.jpeg)")
    parser.add_argument("--results-dir", required=True, type=Path,
                        help="Carpeta con los outputs MedSAM (*_mask.png, *_overlay.png)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Carpeta de salida para los paneles PNG")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for prefix, title, out_name, max_rows, step in PANELS:
        # Filtrar imágenes que matchean el prefijo
        candidates = sorted(args.input_dir.glob(f"{prefix}*.jpeg"))
        if not candidates:
            print(f"  - SKIP {prefix}: 0 imágenes")
            continue

        # Subsamplear y limitar
        sampled = candidates[::step][:max_rows]

        rows = []
        for img_path in sampled:
            stem = img_path.stem
            mask_path = args.results_dir / f"{stem}_mask.png"
            overlay_path = args.results_dir / f"{stem}_overlay.png"
            if not (mask_path.exists() and overlay_path.exists()):
                print(f"  - SKIP {stem}: faltan outputs MedSAM")
                continue
            rows.append({
                "label": stem,
                "image": img_path,
                "mask": mask_path,
                "overlay": overlay_path,
            })

        if not rows:
            print(f"  - SKIP {prefix}: 0 filas válidas")
            continue

        out_path = args.output_dir / out_name
        build_modality_panel(rows, out_path, title)
        print(f"  - {prefix}: {len(rows)} filas → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
