"""Descarga selectiva de un caso de Medical Decathlon Task01_BrainTumour.

El dataset completo pesa ~7 GB y vive en un tar monolítico, pero un solo caso
ocupa apenas ~30 MB (4 modalidades + label). Como tar es un formato secuencial,
podemos hacer streaming-extract: abrir la URL como un stream HTTP, recorrer las
entradas del tar a medida que se descargan, guardar solo las dos que nos
interesan (imagesTr/<case>.nii.gz y labelsTr/<case>.nii.gz) y cortar la
descarga apenas tenemos ambas.

Resultado: descargamos ~50–200 MB en lugar de 7 GB.

Uso:
    python scripts/brats21_download_msd_case.py \
        --case BRATS_001 \
        --out  data/msd_brats

Salida:
    data/msd_brats/
    ├── BRATS_001/
    │   ├── BRATS_001_t1.nii.gz       # splitter aparte (brats21_split_msd_case.py)
    │   ├── BRATS_001_t1ce.nii.gz
    │   ├── BRATS_001_t2.nii.gz
    │   └── BRATS_001_flair.nii.gz
    └── _raw_msd/
        ├── imagesTr/BRATS_001.nii.gz  # 4D NIfTI original (4 modalidades stacked)
        └── labelsTr/BRATS_001.nii.gz  # 3D ground truth

Este script SOLO descarga los archivos crudos. Para producir el formato BraTS
(4 archivos separados con sufijo _t1/_t1ce/_t2/_flair) hay que correr después
`brats21_split_msd_case.py`.

Referencias:
- Antonelli, M. et al. (2022). The Medical Segmentation Decathlon. Nat Commun 13, 4128.
- Dataset: http://medicaldecathlon.com → Task01_BrainTumour
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import requests


MSD_TAR_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"


def stream_extract(case_id: str, out_dir: Path, timeout: int = 60) -> dict[str, Path]:
    """Hace streaming-extract del tar bajando sólo lo necesario para `case_id`.

    Devuelve un dict {kind: ruta_local} con kind ∈ {'image', 'label'}.
    Lanza si tras recorrer todo el tar no se encontró el caso.
    """
    raw_dir = out_dir / "_raw_msd"
    (raw_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    targets = {
        f"Task01_BrainTumour/imagesTr/{case_id}.nii.gz": ("image", raw_dir / "imagesTr" / f"{case_id}.nii.gz"),
        f"Task01_BrainTumour/labelsTr/{case_id}.nii.gz": ("label", raw_dir / "labelsTr" / f"{case_id}.nii.gz"),
    }
    found: dict[str, Path] = {}

    print(f"[stream] {MSD_TAR_URL}")
    print(f"[stream] busco {len(targets)} archivos para caso '{case_id}'")

    with requests.get(MSD_TAR_URL, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode="r|") as tar:
            for member in tar:
                if member.name in targets:
                    kind, dst = targets[member.name]
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    print(f"[hit ] {member.name}  ({member.size/1e6:.1f} MB)")
                    with open(dst, "wb") as out:
                        while True:
                            chunk = f.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                    found[kind] = dst
                    if len(found) == len(targets):
                        print("[done] todos los archivos descargados, corto la conexión")
                        break

    if len(found) != len(targets):
        missing = set(t[0] for t in targets.values()) - set(found.keys())
        raise RuntimeError(f"No se encontraron en el tar: {missing}")
    return found


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="BRATS_001", help="ID del caso (ej: BRATS_001, BRATS_002, ...)")
    p.add_argument("--out", type=Path, default=Path("data/msd_brats"),
                   help="carpeta destino")
    p.add_argument("--timeout", type=int, default=120,
                   help="timeout HTTP en segundos (default 120)")
    args = p.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        found = stream_extract(args.case, out_dir, timeout=args.timeout)
    except Exception as exc:
        print(f"\nERROR durante la descarga: {exc}", file=sys.stderr)
        return 1

    print("\nArchivos descargados:")
    for kind, path in found.items():
        print(f"  {kind:6s}: {path}  ({path.stat().st_size/1e6:.1f} MB)")
    print(f"\nPróximo paso: separar el NIfTI 4D en 4 archivos BraTS-style con:")
    print(f"  python scripts/brats21_split_msd_case.py --case {args.case} --raw {out_dir/'_raw_msd'} --out {out_dir/args.case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
