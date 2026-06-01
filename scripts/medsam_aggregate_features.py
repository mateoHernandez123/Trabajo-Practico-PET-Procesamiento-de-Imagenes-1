#!/usr/bin/env python3
"""Agrega todos los `*_features.csv` de una carpeta en un CSV maestro.

Recorre los `_features.csv` por imagen que genera `medsam_run.py`, le
agrega la columna `image_id` (= nombre de archivo de la imagen original)
y produce un único CSV con todas las features de todas las imágenes.

Uso:
    .venv-medsam/bin/python scripts/medsam_aggregate_features.py \\
        --results-dir resultados_medsam/whatsapp_2026-05-30 \\
        --output      resultados_medsam/whatsapp_2026-05-30/_all_features.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from medsam.features import CSV_FIELDNAMES  # noqa: E402

AGG_FIELDNAMES = ["image_id", "modality_group"] + CSV_FIELDNAMES


def _modality_group(image_stem: str) -> str:
    """Deriva la categoría a partir del prefijo del nombre semántico."""
    parts = image_stem.split("_")
    if len(parts) < 2:
        return "unknown"
    # Heurística simple: primeras 2-3 palabras del filename
    if image_stem.startswith("ct_cerebro_coronal"):
        return "ct_cerebro_coronal"
    if image_stem.startswith("ct_cerebro_sagital"):
        return "ct_cerebro_sagital"
    if image_stem.startswith("ct_cerebro_axial"):
        return "ct_cerebro_axial"
    if image_stem.startswith("ct_craneo_basal"):
        return "ct_craneo_basal"
    if image_stem.startswith("ct_torax"):
        return "ct_torax"
    if image_stem.startswith("pet_cuerpo"):
        return "pet_cuerpo"
    if image_stem.startswith("pet_cerebro"):
        return "pet_cerebro"
    return "otro"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    feat_files = sorted(args.results_dir.glob("*_features.csv"))
    if not feat_files:
        print(f"ERROR: no se encontraron _features.csv en {args.results_dir}", file=sys.stderr)
        return 1

    total_rows = 0
    with args.output.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=AGG_FIELDNAMES)
        writer.writeheader()
        for fp in feat_files:
            image_stem = fp.stem.replace("_features", "")
            modality = _modality_group(image_stem)
            with fp.open(encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    out_row = {"image_id": image_stem, "modality_group": modality, **row}
                    writer.writerow(out_row)
                    total_rows += 1

    print(f"OK: {len(feat_files)} imágenes → {total_rows} features agregadas en {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
