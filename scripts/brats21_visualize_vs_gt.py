"""Visualización comparativa BraTS: ground truth vs predicción.

Produce un PNG con 2 filas (ground truth | predicción) × 3 columnas (axial,
coronal, sagital). Se usa la modalidad FLAIR como fondo (la más útil
clínicamente para ver tumor en MRI).

Sobre cada slice se superpone la segmentación en color:
    azul   = NCR/NET (etiqueta 1)
    verde  = ED      (etiqueta 2)
    rojo   = ET      (etiqueta 4)

Uso:
    python scripts/brats21_visualize_vs_gt.py \
        --case  data/msd_brats/BRATS_001 \
        --pred  resultados_brats21/msd_brats_001_fold0/BRATS_001_seg.nii.gz \
        --gt    data/msd_brats/BRATS_001/BRATS_001_seg.nii.gz \
        --out   resultados_brats21/msd_brats_001_fold0/preview_vs_gt.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def find_flair(case_dir: Path) -> Path:
    cand = list(case_dir.glob("*_flair.nii*"))
    if not cand:
        raise FileNotFoundError(f"No se encontró FLAIR en {case_dir}")
    return cand[0]


def central_slice_in_tumor(seg: np.ndarray, axis: int) -> int:
    """Slice del eje `axis` donde el tumor tiene mayor área."""
    tumor = seg > 0
    if not tumor.any():
        return seg.shape[axis] // 2
    axes_other = tuple(i for i in range(3) if i != axis)
    counts = tumor.sum(axis=axes_other)
    return int(np.argmax(counts))


def slice_along(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[axis] = idx
    return vol[tuple(sl)]


def overlay(ax, base: np.ndarray, seg: np.ndarray, title: str) -> None:
    ax.imshow(base.T, cmap="gray", origin="lower")
    color = np.zeros(base.shape + (4,), dtype=np.float32)
    color[seg == 2] = [0.10, 0.85, 0.10, 0.50]
    color[seg == 1] = [0.10, 0.40, 0.95, 0.55]
    color[seg == 4] = [0.95, 0.20, 0.20, 0.70]
    ax.imshow(np.transpose(color, (1, 0, 2)), origin="lower")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True, type=Path)
    p.add_argument("--pred", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--title", default=None, help="Título de la figura")
    args = p.parse_args()

    flair = nib.load(str(find_flair(args.case))).get_fdata().astype(np.float32)
    pred = nib.load(str(args.pred)).get_fdata().astype(np.int32)
    gt = nib.load(str(args.gt)).get_fdata().astype(np.int32)

    cz = central_slice_in_tumor(gt, 0)
    cy = central_slice_in_tumor(gt, 1)
    cx = central_slice_in_tumor(gt, 2)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    planes = [("axial", 0, cz), ("coronal", 1, cy), ("sagital", 2, cx)]
    for col, (name, axis, idx) in enumerate(planes):
        base = slice_along(flair, axis, idx)
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)
        overlay(axes[0, col], base, slice_along(gt, axis, idx),   f"GT  - {name} (slice {idx})")
        overlay(axes[1, col], base, slice_along(pred, axis, idx), f"Pred - {name} (slice {idx})")

    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(0.10, 0.40, 0.95), markersize=12, label='NCR/NET (1)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(0.10, 0.85, 0.10), markersize=12, label='ED (2)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(0.95, 0.20, 0.20), markersize=12, label='ET (4)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False, fontsize=11)
    title = args.title or f"Ground truth vs predicción — {args.case.name}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out), dpi=120, bbox_inches="tight")
    print(f"[ok] preview -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
