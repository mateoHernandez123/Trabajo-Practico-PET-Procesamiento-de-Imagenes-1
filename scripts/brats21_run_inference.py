"""Inferencia BraTS21 standalone (CPU + Windows friendly).

Bypasea el `Engine` y `main_inference.py` del repositorio original (que asumen
CUDA y dependencias UNIX como `resource`) y arma una inferencia mínima usando
sólo el modelo `EquiUnetASSPEvo` del repo.

Pipeline:
    1. Lee un config.yaml de un fold pre-entrenado.
    2. Construye el modelo con los hiperparámetros del config.
    3. Carga los pesos best_model.pth.
    4. Lee las 4 modalidades (t1, t1ce, t2, flair) del caso de entrada.
    5. Normaliza z-score por canal (sólo voxels no-cero).
    6. Hace sliding window inference 3D con MONAI.
    7. Threshold + selecciona la mayor componente conexa por canal.
    8. Convierte (WT, TC, ET) -> etiquetas BraTS (0/1/2/4) y guarda como NIfTI.

Uso (un solo fold, lo más simple):
    python scripts/brats21_run_inference.py \
        --config external/BraTS21/checkpoints/final_weights_brats21/baseline_equiunet_assp_evocor/fold0_ns/config.yaml \
        --input data/brats21_synth/BraTS_synth_001 \
        --output resultados_brats21 \
        --roi 96 96 96

Uso (ensemble de varios folds, promediando logits):
    python scripts/brats21_run_inference.py \
        --config <fold0/config.yaml> <fold1/config.yaml> ... \
        --input <case_dir> \
        --output <out_dir>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BRATS21_DIR = REPO_ROOT / "external" / "BraTS21"
sys.path.insert(0, str(BRATS21_DIR))

from networks.equiunet2020 import EquiUnet, AttEquiUnet  # noqa: E402
from networks.equiunet2021 import EquiUnetASSPEvo  # noqa: E402

from monai.inferers import sliding_window_inference  # noqa: E402
from monai.transforms import KeepLargestConnectedComponent  # noqa: E402

MODALITIES = ("t1", "t1ce", "t2", "flair")


def build_model_from_config(cfg: dict) -> torch.nn.Module:
    """Reproduce la lógica de `get_model` de `src/definer.py` para los
    modelos efectivamente usados por los pesos publicados."""
    name = cfg["model"]
    width = int(cfg.get("width", 48))
    num_classes = int(cfg.get("num_classes", 3))
    norm = cfg.get("norm", "group")
    act = cfg.get("act", "leakyrelu")
    dropout = float(cfg.get("dropout", 0.0))
    features = [width * 2 ** i for i in range(4)]
    kwargs = dict(
        inplanes=4,
        num_classes=num_classes,
        features=features,
        norm_layer=norm,
        act=act,
        deep_supervision=True,
        dropout=dropout,
    )
    if name == "equiunet":
        return EquiUnet(**kwargs)
    if name == "equiunet_ref":
        kwargs["refinement"] = True
        return EquiUnet(**kwargs)
    if name == "att_equiunet":
        return AttEquiUnet(**kwargs)
    if name in ("equiunet_assp_evo", "equiunet_assp_evocor"):
        return EquiUnetASSPEvo(**kwargs)
    if name == "equiunet_assp_evo_ref":
        kwargs["refinement"] = True
        return EquiUnetASSPEvo(**kwargs)
    raise ValueError(f"Modelo no soportado por este runner: {name!r}")


def load_modality(case_dir: Path, case_id: str, modality: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    cand = list(case_dir.glob(f"{case_id}_{modality}.nii*"))
    if not cand:
        cand = list(case_dir.glob(f"*_{modality}.nii*"))
    if not cand:
        raise FileNotFoundError(f"No se encontró modalidad {modality!r} en {case_dir}")
    img = nib.load(str(cand[0]))
    return img.get_fdata().astype(np.float32), img


def zscore_nonzero(vol: np.ndarray) -> np.ndarray:
    mask = vol != 0
    if mask.sum() == 0:
        return vol
    mean = vol[mask].mean()
    std = vol[mask].std() + 1e-8
    out = np.zeros_like(vol)
    out[mask] = (vol[mask] - mean) / std
    return out


def crop_to_nonzero(volumes: list[np.ndarray]) -> tuple[list[np.ndarray], tuple[slice, slice, slice]]:
    union = np.zeros_like(volumes[0], dtype=bool)
    for v in volumes:
        union |= (v != 0)
    if not union.any():
        sl = (slice(0, volumes[0].shape[0]), slice(0, volumes[0].shape[1]), slice(0, volumes[0].shape[2]))
        return volumes, sl
    coords = np.array(np.where(union))
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1) + 1
    sl = tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))
    return [v[sl] for v in volumes], sl


def take_deep_supervision_main(out) -> torch.Tensor:
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def model_predictor(model: torch.nn.Module):
    def _fn(x: torch.Tensor) -> torch.Tensor:
        return take_deep_supervision_main(model(x))
    return _fn


def convert_wtTcEt_to_brats(mask3: np.ndarray) -> np.ndarray:
    """mask3 con canales (WT, TC, ET). Devuelve etiquetas {0, 1, 2, 4}.

    - 0 = fondo
    - 1 = NCR/NET = TC y no-ET
    - 2 = ED      = WT y no-TC
    - 4 = ET      = ET
    """
    wt = mask3[0] > 0.5
    tc = mask3[1] > 0.5
    et = mask3[2] > 0.5
    out = np.zeros_like(wt, dtype=np.uint8)
    out[wt & ~tc] = 2
    out[tc & ~et] = 1
    out[et] = 4
    return out


def run_inference(args: argparse.Namespace) -> int:
    device = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

    case_dir = args.input.resolve()
    case_id = args.case_id or case_dir.name
    print(f"[case] {case_id}  ({case_dir})")

    print("[load] modalidades...")
    raw_volumes, raw_imgs = [], []
    for mod in MODALITIES:
        vol, img = load_modality(case_dir, case_id, mod)
        raw_volumes.append(vol)
        raw_imgs.append(img)
        print(f"  - {mod:5s}  shape={vol.shape}  rango=[{vol.min():.1f},{vol.max():.1f}]")

    cropped, sl = crop_to_nonzero(raw_volumes)
    print(f"[crop] foreground bbox: {sl}, shape={cropped[0].shape}")

    normed = [zscore_nonzero(v) for v in cropped]
    x = np.stack(normed, axis=0)[None].astype(np.float32)
    x_t = torch.from_numpy(x).to(device)
    print(f"[input] tensor shape: {tuple(x_t.shape)}")

    configs = [yaml.safe_load(open(c, "r")) for c in args.config]
    models = []
    for cfg, cfg_path in zip(configs, args.config):
        cfg_path = Path(cfg_path)
        ckpt_path = cfg_path.parent / "best_model.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No se encontró best_model.pth junto a {cfg_path}")
        print(f"[model] construyendo {cfg['model']} y cargando {ckpt_path}")
        model = build_model_from_config(cfg)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] keys faltantes: {len(missing)} (ej: {missing[:3]})")
        if unexpected:
            print(f"  [warn] keys inesperadas: {len(unexpected)} (ej: {unexpected[:3]})")
        model.to(device).eval()
        models.append(model)

    roi = tuple(args.roi)
    print(f"[infer] sliding window roi={roi}, sw_batch_size={args.sw_batch_size}, overlap={args.overlap}")
    start = time.time()
    probs_sum = None
    with torch.no_grad():
        for i, model in enumerate(models):
            logits = sliding_window_inference(
                inputs=x_t,
                roi_size=roi,
                sw_batch_size=args.sw_batch_size,
                predictor=model_predictor(model),
                overlap=args.overlap,
                mode="gaussian",
                device=device,
            )
            probs = torch.sigmoid(logits)
            probs_sum = probs if probs_sum is None else probs_sum + probs
            print(f"  fold {i+1}/{len(models)} listo")
    probs = (probs_sum / len(models))[0]
    elapsed = time.time() - start
    print(f"[infer] total: {elapsed:.1f}s")

    bin_mask = (probs >= args.threshold).cpu().numpy().astype(np.uint8)
    if args.cleaning_areas:
        klc = KeepLargestConnectedComponent(applied_labels=[1], num_components=1)
        for c in range(bin_mask.shape[0]):
            comp = klc(torch.from_numpy(bin_mask[c:c+1]).long()).numpy()
            bin_mask[c] = comp[0]

    full_mask3 = np.zeros((3,) + raw_volumes[0].shape, dtype=np.uint8)
    full_mask3[:, sl[0], sl[1], sl[2]] = bin_mask
    seg = convert_wtTcEt_to_brats(full_mask3)

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_seg = out_dir / f"{case_id}_seg.nii.gz"
    nib.save(nib.Nifti1Image(seg, raw_imgs[0].affine, raw_imgs[0].header), str(out_seg))
    print(f"[save] segmentación: {out_seg}")

    if args.save_probs:
        for c, ch_name in enumerate(["WT", "TC", "ET"]):
            full_prob = np.zeros(raw_volumes[0].shape, dtype=np.float32)
            full_prob[sl[0], sl[1], sl[2]] = probs[c].cpu().numpy().astype(np.float32)
            out_prob = out_dir / f"{case_id}_prob_{ch_name}.nii.gz"
            nib.save(nib.Nifti1Image(full_prob, raw_imgs[0].affine), str(out_prob))
            print(f"[save] prob {ch_name}: {out_prob}")

    voxels = {1: int((seg == 1).sum()), 2: int((seg == 2).sum()), 4: int((seg == 4).sum())}
    total = int(seg.size)
    print("\n[result] voxels por clase:")
    print(f"  fondo  (0): {total - sum(voxels.values()):>10d}")
    print(f"  NCR/NET(1): {voxels[1]:>10d}")
    print(f"  ED     (2): {voxels[2]:>10d}")
    print(f"  ET     (4): {voxels[4]:>10d}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", nargs="+", required=True, type=Path,
                   help="ruta(s) a config.yaml de fold(s) pre-entrenado(s)")
    p.add_argument("--input", required=True, type=Path,
                   help="carpeta del caso con t1/t1ce/t2/flair NIfTI")
    p.add_argument("--output", required=True, type=Path,
                   help="carpeta destino para la segmentación")
    p.add_argument("--case-id", default=None,
                   help="ID del paciente (default: nombre de la carpeta de input)")
    p.add_argument("--roi", nargs=3, type=int, default=[128, 128, 128],
                   help="tamaño del patch sliding window (default 128 128 128)")
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cleaning-areas", action="store_true",
                   help="quedarse con la mayor componente conexa por canal")
    p.add_argument("--save-probs", action="store_true",
                   help="además de la seg, guardar las prob por canal (WT/TC/ET)")
    args = p.parse_args()
    return run_inference(args)


if __name__ == "__main__":
    raise SystemExit(main())
