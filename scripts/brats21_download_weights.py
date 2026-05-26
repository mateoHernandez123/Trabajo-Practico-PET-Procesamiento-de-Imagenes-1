"""Descarga los pesos pre-entrenados del modelo BraTS21 desde Google Drive.

Los pesos son los que el autor del repositorio Alxaline/BraTS21 publicó como
"final_weights_brats21.zip" en:
https://drive.google.com/file/d/1Xt2rdD60IeEwcd8-yiMZHZkI0udcXgc7/view?usp=sharing

El zip pesa ~3 GB y al descomprimir contiene 10 carpetas (5 folds x 2 modelos)
con un best_model.pth y un config.yaml en cada una.

Uso:
    python scripts/brats21_download_weights.py [--out external/BraTS21/checkpoints]

Si la descarga automática falla (Google Drive a veces bloquea), descargá el
archivo manualmente desde el navegador y pasalo con --from-local <zip>.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path


GDRIVE_FILE_ID = "1Xt2rdD60IeEwcd8-yiMZHZkI0udcXgc7"
ZIP_NAME = "final_weights_brats21.zip"


def download_from_gdrive(out_zip: Path) -> bool:
    try:
        import gdown
    except ImportError:
        print("ERROR: falta 'gdown'. Instalá con: pip install gdown", file=sys.stderr)
        return False

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] gdrive id={GDRIVE_FILE_ID} -> {out_zip}")
    try:
        gdown.download(id=GDRIVE_FILE_ID, output=str(out_zip), quiet=False, resume=True)
    except Exception as exc:
        print(f"[download] falló: {exc}", file=sys.stderr)
        return False
    return out_zip.exists() and out_zip.stat().st_size > 1024 * 1024


def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("external/BraTS21/checkpoints"),
        help="carpeta donde colocar los pesos extraídos",
    )
    parser.add_argument(
        "--from-local",
        type=Path,
        default=None,
        help="usar un .zip ya descargado en lugar de bajarlo de Google Drive",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="no borrar el .zip al finalizar",
    )
    args = parser.parse_args()

    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_local is not None:
        zip_path = args.from_local.resolve()
        if not zip_path.exists():
            print(f"ERROR: no existe {zip_path}", file=sys.stderr)
            return 2
    else:
        zip_path = out_dir / ZIP_NAME
        if zip_path.exists():
            print(f"[skip] ya existe {zip_path}, no se vuelve a descargar")
        else:
            ok = download_from_gdrive(zip_path)
            if not ok:
                print(
                    "\nLa descarga automática falló. Opciones:\n"
                    f"  1. Abrir en navegador: https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view\n"
                    f"  2. Guardar como '{ZIP_NAME}' en cualquier carpeta\n"
                    "  3. Volver a correr este script con --from-local <ruta_al_zip>",
                    file=sys.stderr,
                )
                return 1

    unzip(zip_path, out_dir)

    if not args.keep_zip and args.from_local is None:
        try:
            os.remove(zip_path)
            print(f"[clean] borrado {zip_path}")
        except OSError:
            pass

    print(f"\nListo. Pesos disponibles en {out_dir}")
    print("Contenido encontrado:")
    for sub in sorted(out_dir.glob("**/best_model.pth")):
        print(f"  - {sub.relative_to(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
