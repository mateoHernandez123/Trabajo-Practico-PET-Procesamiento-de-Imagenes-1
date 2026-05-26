"""Separa un caso de Medical Decathlon Task01_BrainTumour al formato BraTS.

Medical Decathlon empaqueta las 4 modalidades MRI en un único NIfTI 4D donde la
cuarta dimensión es el canal (orden: FLAIR, T1w, T1ce, T2w). BraTS21 y nuestro
runner esperan 4 archivos separados con sufijo `_t1`, `_t1ce`, `_t2`, `_flair`.

Este script:
1. Lee el 4D NIfTI de imagesTr/<case>.nii.gz
2. Lo separa en 4 archivos 3D con el sufijo correcto
3. Copia el label (labelsTr/<case>.nii.gz) como `_seg.nii.gz`
4. Remapea las etiquetas del Decathlon al esquema BraTS estándar:
       Decathlon: 0=fondo, 1=edema, 2=non-enhancing, 3=enhancing
       BraTS:     0=fondo, 1=NCR/NET, 2=edema, 4=enhancing tumor

Uso:
    python scripts/brats21_split_msd_case.py \
        --case BRATS_001 \
        --raw  data/msd_brats/_raw_msd \
        --out  data/msd_brats/BRATS_001

Salida:
    data/msd_brats/BRATS_001/
    ├── BRATS_001_t1.nii.gz
    ├── BRATS_001_t1ce.nii.gz
    ├── BRATS_001_t2.nii.gz
    ├── BRATS_001_flair.nii.gz
    └── BRATS_001_seg.nii.gz   # ground truth ya en formato BraTS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


DECATHLON_CHANNEL_ORDER = ["flair", "t1", "t1ce", "t2"]
DECATHLON_TO_BRATS_LABEL = {0: 0, 1: 2, 2: 1, 3: 4}


def split_image(image_4d_path: Path, case_id: str, out_dir: Path) -> None:
    img = nib.load(str(image_4d_path))
    data = img.get_fdata()
    print(f"[load] {image_4d_path.name}  shape={data.shape}  dtype={data.dtype}")

    if data.ndim != 4 or data.shape[-1] != 4:
        raise ValueError(f"Esperaba un NIfTI 4D con 4 canales en la última dim; encontré {data.shape}")

    affine = img.affine
    header = img.header

    for ch_idx, modality in enumerate(DECATHLON_CHANNEL_ORDER):
        vol = data[..., ch_idx].astype(np.float32)
        out_path = out_dir / f"{case_id}_{modality}.nii.gz"
        out_img = nib.Nifti1Image(vol, affine, header)
        out_img.set_data_dtype(np.float32)
        nib.save(out_img, str(out_path))
        print(f"  [{modality:5s}] -> {out_path.name}  shape={vol.shape}  rango=[{vol.min():.1f},{vol.max():.1f}]")


def remap_label(label_path: Path, case_id: str, out_dir: Path) -> None:
    img = nib.load(str(label_path))
    data = img.get_fdata().astype(np.int32)
    print(f"[load] {label_path.name}  shape={data.shape}  unique={np.unique(data).tolist()}")

    out_data = np.zeros_like(data, dtype=np.uint8)
    for decathlon_val, brats_val in DECATHLON_TO_BRATS_LABEL.items():
        out_data[data == decathlon_val] = brats_val

    out_path = out_dir / f"{case_id}_seg.nii.gz"
    out_img = nib.Nifti1Image(out_data, img.affine, img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(out_path))

    counts = {int(v): int((out_data == v).sum()) for v in np.unique(out_data)}
    print(f"  [seg ] -> {out_path.name}  voxels por etiqueta: {counts}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True, help="ID del caso (ej BRATS_001)")
    p.add_argument("--raw", type=Path, required=True,
                   help="carpeta _raw_msd con imagesTr/ y labelsTr/")
    p.add_argument("--out", type=Path, required=True,
                   help="carpeta destino para los 5 archivos BraTS-style")
    args = p.parse_args()

    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_4d = args.raw / "imagesTr" / f"{args.case}.nii.gz"
    label_3d = args.raw / "labelsTr" / f"{args.case}.nii.gz"

    if not image_4d.exists():
        raise FileNotFoundError(f"No existe {image_4d}")
    if not label_3d.exists():
        raise FileNotFoundError(f"No existe {label_3d}")

    split_image(image_4d, args.case, out_dir)
    remap_label(label_3d, args.case, out_dir)

    print(f"\n[ok] caso {args.case} listo en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
