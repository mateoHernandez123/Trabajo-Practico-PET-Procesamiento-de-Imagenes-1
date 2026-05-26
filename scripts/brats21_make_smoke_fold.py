"""Smoke test del runner de inferencia BraTS21 con pesos aleatorios.

Crea un fold "fake" con:
    - config.yaml: hiperparámetros del modelo `equiunet_assp_evocor` con
      width=48, num_classes=3, act=leakyrelu, norm=group (el del paper)
    - best_model.pth: pesos aleatorios (modelo inicializado, no entrenado)

Después corre `brats21_run_inference.py` apuntando a ese fold sobre el caso
sintético.

Sirve únicamente para verificar que el pipeline (carga config, instancia
modelo, carga state_dict, sliding-window inference, conversión WT/TC/ET a
etiquetas BraTS, guardado NIfTI) funciona. La segmentación resultante no
tendrá sentido alguno porque el modelo no fue entrenado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BRATS21_DIR = REPO_ROOT / "external" / "BraTS21"
sys.path.insert(0, str(BRATS21_DIR))

from networks.equiunet2021 import EquiUnetASSPEvo


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "external" / "BraTS21" / "checkpoints" / "smoke_test" / "fold0")
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--num-classes", type=int, default=3)
    args = p.parse_args()

    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model": "equiunet_assp_evocor",
        "width": args.width,
        "num_classes": args.num_classes,
        "norm": "group",
        "act": "leakyrelu",
        "dropout": 0.0,
        "remove_outliers": True,
        "patch_size": [128, 128, 128],
    }
    cfg_path = out_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"[ok] config -> {cfg_path}")

    features = [args.width * 2 ** i for i in range(4)]
    model = EquiUnetASSPEvo(
        inplanes=4,
        num_classes=args.num_classes,
        features=features,
        norm_layer="group",
        act="leakyrelu",
        deep_supervision=True,
        dropout=0.0,
    )
    state = {"model": model.state_dict()}
    ckpt_path = out_dir / "best_model.pth"
    torch.save(state, ckpt_path)
    print(f"[ok] best_model.pth (pesos aleatorios) -> {ckpt_path}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] parametros del modelo: {n_params/1e6:.2f} M")
    print(f"[info] tamaño checkpoint: {ckpt_path.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
