"""Visualiza la segmentación BraTS21 sobre las modalidades MRI.

Toma el caso de entrada y la segmentación generada, y produce un PNG con
3 filas (axial central, coronal central, sagital central) x 4 columnas
(t1, t1ce, t2, flair) mostrando la imagen base + el contorno de las clases
WT/TC/ET en color.

Uso:
    python scripts/brats21_visualize.py \
        --case data/brats21_synth/BraTS_synth_001 \
        --seg  resultados_brats21/real_fold0/BraTS_synth_001_seg.nii.gz \
        --out  resultados_brats21/real_fold0/preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

MODALITIES = ("t1", "t1ce", "t2", "flair")


def find_modality(case_dir: Path, mod: str) -> Path:
    cand = list(case_dir.glob(f"*_{mod}.nii*"))
    if not cand:
        raise FileNotFoundError(f"No se encontró {mod} en {case_dir}")
    return cand[0]


def central_slice(vol: np.ndarray, axis: int) -> int:
    nz = np.where(vol > 0)
    if len(nz[axis]) == 0:
        return vol.shape[axis] // 2
    return int(np.median(nz[axis]))


def overlay_seg(ax, base: np.ndarray, seg: np.ndarray, title: str) -> None:
    ax.imshow(base.T, cmap="gray", origin="lower")
    color_layer = np.zeros(base.shape + (4,), dtype=np.float32)
    color_layer[seg == 2] = [0.10, 0.80, 0.10, 0.45]
    color_layer[seg == 1] = [0.10, 0.40, 0.95, 0.55]
    color_layer[seg == 4] = [0.95, 0.20, 0.20, 0.70]
    ax.imshow(np.transpose(color_layer, (1, 0, 2)), origin="lower")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True, type=Path)
    p.add_argument("--seg", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    case_dir = args.case.resolve()
    mods = {m: nib.load(str(find_modality(case_dir, m))).get_fdata().astype(np.float32) for m in MODALITIES}
    seg = nib.load(str(args.seg.resolve())).get_fdata().astype(np.uint8)

    ref = mods["flair"]
    cz = central_slice(ref, 0)
    cy = central_slice(ref, 1)
    cx = central_slice(ref, 2)

    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    plane_specs = [
        ("axial",    lambda v: v[cz, :, :],  lambda s: s[cz, :, :]),
        ("coronal",  lambda v: v[:, cy, :],  lambda s: s[:, cy, :]),
        ("sagital",  lambda v: v[:, :, cx],  lambda s: s[:, :, cx]),
    ]
    for r, (plane_name, take_v, take_s) in enumerate(plane_specs):
        seg_slice = take_s(seg)
        for c, mod in enumerate(MODALITIES):
            base = take_v(mods[mod])
            base = (base - base.min()) / (base.max() - base.min() + 1e-8)
            overlay_seg(axes[r, c], base, seg_slice, f"{plane_name} - {mod}")

    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(0.10, 0.40, 0.95), markersize=12, label='NCR/NET (1)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(0.10, 0.80, 0.10), markersize=12, label='ED (2)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(0.95, 0.20, 0.20), markersize=12, label='ET (4)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False, fontsize=11)
    fig.suptitle(f"BraTS21 inference — {case_dir.name}", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out), dpi=110, bbox_inches="tight")
    print(f"[ok] preview -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
