"""Genera un caso sintético con la estructura BraTS21 (4 modalidades MRI).

El dataset oficial BraTS 2021 está detrás de un registro en Synapse, así que
para poder ejecutar la inferencia end-to-end sin descargarlo se construye
acá un volumen 3D plausible (esfera elipsoidal con ruido) en las 4 modalidades
estándar:

    BraTS_synth_001/
        BraTS_synth_001_flair.nii.gz
        BraTS_synth_001_t1.nii.gz
        BraTS_synth_001_t1ce.nii.gz
        BraTS_synth_001_t2.nii.gz

Los volúmenes son de 240x240x155 (mismas dimensiones que BraTS) con espaciado
(1, 1, 1) mm. El "cerebro" sintético es una elipsoide centrada con intensidades
ligeramente distintas por modalidad y un pequeño "tumor" hiperintenso.

Esto NO es un caso real ni los resultados de la inferencia tendrán sentido
clínico; sirve únicamente para validar que el pipeline corre end-to-end.

Uso:
    python scripts/brats21_make_synthetic_case.py [--out data/brats21_synth]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def make_brain(shape=(240, 240, 155), seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    H, W, D = shape

    zz, yy, xx = np.mgrid[0:H, 0:W, 0:D].astype(np.float32)
    cz, cy, cx = H / 2, W / 2, D / 2
    rad = ((zz - cz) / (H * 0.42)) ** 2 + ((yy - cy) / (W * 0.42)) ** 2 + ((xx - cx) / (D * 0.42)) ** 2
    brain_mask = (rad <= 1.0).astype(np.float32)

    tcz, tcy, tcx = H / 2 + 20, W / 2 - 10, D / 2 + 5
    trad = ((zz - tcz) / 8) ** 2 + ((yy - tcy) / 10) ** 2 + ((xx - tcx) / 5) ** 2
    tumor_mask = ((trad <= 1.0) & (brain_mask > 0)).astype(np.float32)

    return brain_mask, tumor_mask


def synth_modality(brain_mask: np.ndarray, tumor_mask: np.ndarray,
                   base: float, tumor_delta: float, noise_std: float,
                   seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = brain_mask * base
    vol = vol + tumor_mask * tumor_delta
    vol = vol + rng.normal(0.0, noise_std, vol.shape).astype(np.float32)
    vol = np.where(brain_mask > 0, vol, 0.0)
    return vol.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/brats21_synth"),
        help="carpeta donde escribir los NIfTI del caso sintético",
    )
    parser.add_argument(
        "--case-id",
        default="BraTS_synth_001",
        help="ID del caso (se usa como nombre de carpeta y prefijo de archivos)",
    )
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=[240, 240, 155],
        help="shape del volumen (H W D), por defecto el real de BraTS",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    case_dir: Path = (args.out / args.case_id).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)

    brain_mask, tumor_mask = make_brain(tuple(args.shape), seed=args.seed)

    modality_specs = {
        "t1":    dict(base=200.0, tumor_delta=-30.0, noise_std=4.0),
        "t1ce":  dict(base=210.0, tumor_delta=+80.0, noise_std=4.0),
        "t2":    dict(base=160.0, tumor_delta=+120.0, noise_std=6.0),
        "flair": dict(base=140.0, tumor_delta=+150.0, noise_std=6.0),
    }

    affine = np.eye(4, dtype=np.float32)

    for i, (mod, spec) in enumerate(modality_specs.items()):
        vol = synth_modality(
            brain_mask, tumor_mask,
            base=spec["base"],
            tumor_delta=spec["tumor_delta"],
            noise_std=spec["noise_std"],
            seed=args.seed + i,
        )
        out_file = case_dir / f"{args.case_id}_{mod}.nii.gz"
        nib.save(nib.Nifti1Image(vol, affine), str(out_file))
        print(f"[ok] {out_file}  shape={vol.shape}  range=[{vol.min():.1f}, {vol.max():.1f}]")

    print(f"\nCaso sintético generado en: {case_dir}")
    print("Estructura compatible con BraTS21 (t1, t1ce, t2, flair).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
