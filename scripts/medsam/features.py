"""Extracción y persistencia de features morfológicas de la máscara MedSAM.

Reutiliza `compute_features` de `segment_pet.py` (Pista 1 clásica) para
mantener consistencia: las mismas features se calculan tanto sobre la
máscara generada por umbralización de PET (Pista 1) como sobre la máscara
generada por MedSAM (Pista 2). Esto permite comparar resultados.

Features extraídas (por blob detectado):
    area_px           Área en píxeles
    perimeter_px      Perímetro en píxeles
    centroid          (x, y) en píxeles
    bbox              (x, y, w, h)
    axis_major_px     Eje mayor de la elipse equivalente
    axis_minor_px     Eje menor de la elipse equivalente
    orientation_deg   Orientación del eje mayor en grados
    eccentricity      e ∈ [0, 1)  (0 = círculo)
    compactness       4π · area / perim²   (1 = círculo perfecto)
    mean_intensity    Media del valor en gris dentro del blob
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Reutilizamos el cálculo de features del pipeline clásico de Pista 1
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from segment_pet import compute_features  # noqa: E402


CSV_FIELDNAMES = [
    "id", "label_id", "area_px", "perimeter_px",
    "centroid_x", "centroid_y",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "axis_major_px", "axis_minor_px",
    "orientation_deg", "eccentricity",
    "compactness", "mean_intensity",
]


def write_features_csv(path: Path, features: list[dict]) -> None:
    """Vuelca features (lista de dicts) a un CSV plano con columnas estándar."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for feat in features:
            cx, cy = feat["centroid"]
            x, y, w, h = feat["bbox"]
            writer.writerow({
                "id": feat["id"],
                "label_id": feat["label_id"],
                "area_px": feat["area_px"],
                "perimeter_px": round(feat["perimeter_px"], 3),
                "centroid_x": round(cx, 2),
                "centroid_y": round(cy, 2),
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "axis_major_px": round(feat["axis_major_px"], 3),
                "axis_minor_px": round(feat["axis_minor_px"], 3),
                "orientation_deg": round(feat["orientation_deg"], 2),
                "eccentricity": round(feat["eccentricity"], 4),
                "compactness": round(feat["compactness"], 4),
                "mean_intensity": round(feat["mean_intensity"], 2),
            })


__all__ = ["compute_features", "write_features_csv", "CSV_FIELDNAMES"]
