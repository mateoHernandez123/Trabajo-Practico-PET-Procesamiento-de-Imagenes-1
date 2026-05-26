"""Calcula Dice score por sub-región BraTS entre una predicción y un ground truth.

Sub-regiones evaluadas (estándar del challenge BraTS):
    WT  = Whole Tumor       = unión de NCR + ED + ET   (etiquetas 1 + 2 + 4)
    TC  = Tumor Core        = NCR + ET                  (etiquetas 1 + 4)
    ET  = Enhancing Tumor   = ET solamente              (etiqueta 4)

Dice(A, B) = 2 · |A ∩ B| / (|A| + |B|)

Uso:
    python scripts/brats21_eval_dice.py \
        --pred resultados_brats21/msd_brats_001_fold0/BRATS_001_seg.nii.gz \
        --gt   data/msd_brats/BRATS_001/BRATS_001_seg.nii.gz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def to_subregions(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Devuelve máscaras booleanas para WT, TC y ET a partir de etiquetas BraTS."""
    return {
        "WT": (labels == 1) | (labels == 2) | (labels == 4),
        "TC": (labels == 1) | (labels == 4),
        "ET": labels == 4,
    }


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred.sum() + gt.sum() + 1e-12))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", type=Path, required=True, help="NIfTI con segmentación predicha")
    p.add_argument("--gt", type=Path, required=True, help="NIfTI con ground truth")
    p.add_argument("--json", type=Path, default=None,
                   help="opcional: guardar resultados como JSON en esta ruta")
    args = p.parse_args()

    pred_arr = nib.load(str(args.pred)).get_fdata().astype(np.int32)
    gt_arr = nib.load(str(args.gt)).get_fdata().astype(np.int32)

    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"Shapes distintos: pred={pred_arr.shape} gt={gt_arr.shape}")

    print(f"[pred] {args.pred.name}  shape={pred_arr.shape}  unique={np.unique(pred_arr).tolist()}")
    print(f"[gt  ] {args.gt.name}    shape={gt_arr.shape}  unique={np.unique(gt_arr).tolist()}")

    pred_sub = to_subregions(pred_arr)
    gt_sub = to_subregions(gt_arr)

    results = {}
    print("\n  Sub-región |     Dice | Voxels pred | Voxels GT  | Intersección")
    print("  ---------- + -------- + ----------- + ---------- + -------------")
    for name in ("WT", "TC", "ET"):
        d = dice_score(pred_sub[name], gt_sub[name])
        n_pred = int(pred_sub[name].sum())
        n_gt = int(gt_sub[name].sum())
        n_inter = int(np.logical_and(pred_sub[name], gt_sub[name]).sum())
        results[name] = {"dice": d, "voxels_pred": n_pred, "voxels_gt": n_gt, "intersection": n_inter}
        print(f"  {name:10s} | {d:8.4f} | {n_pred:>11d} | {n_gt:>10d} | {n_inter:>12d}")

    mean_dice = float(np.mean([results[k]["dice"] for k in ("WT", "TC", "ET")]))
    print(f"\n  Dice promedio (WT, TC, ET): {mean_dice:.4f}")
    results["mean"] = mean_dice

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[save] resultados -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
