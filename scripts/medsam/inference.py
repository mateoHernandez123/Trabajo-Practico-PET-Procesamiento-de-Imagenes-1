"""Forward de MedSAM + orquestación por imagen.

`medsam_predict` hace 1 forward y devuelve una máscara binaria H×W al
tamaño original de la imagen. `run_one` orquesta todo el pipeline para
una sola foto: preprocesar → resolver bbox → predecir → extraer features
→ persistir mask/overlay/csv/json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .bbox_strategies import resolve_bbox
from .features import compute_features, write_features_csv
from .preprocess import preprocess_photo
from .visualize import build_overlay


DEFAULT_INPUT_SIZE = 1024     # Fijo por arquitectura del ViT encoder de MedSAM
DEFAULT_THRESHOLD = 0.5       # Sigmoid → binario; default del paper

# Sufijos de archivos de salida
SUFFIX_MASK = "_mask.png"
SUFFIX_OVERLAY = "_overlay.png"
SUFFIX_META = "_meta.json"
SUFFIX_FEATURES = "_features.csv"


@torch.no_grad()
def medsam_predict(
    img_rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    model,
    processor,
    device: str = "cpu",
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Devuelve máscara binaria (H, W) uint8 0/255 al tamaño de `img_rgb`.

    Notas:
      - El processor reescala internamente a 1024×1024 (input del ViT encoder).
      - `multimask_output=False` pide UNA máscara (no las 3 alternativas de SAM).
      - `post_process_masks` lleva la máscara de 256×256 al tamaño original.
      - `threshold` se aplica vía el bool del tensor devuelto por HF (ya es bool).
    """
    inputs = processor(
        img_rgb,
        input_boxes=[[list(bbox_xyxy)]],  # batch=1, n_boxes=1, [x1,y1,x2,y2]
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs, multimask_output=False)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    mask = masks[0][0, 0].numpy()  # (H, W) bool
    return (mask.astype(np.uint8)) * 255


def run_one(
    input_path: Path,
    output_dir: Path,
    bbox_strategy: str,
    threshold: float,
    device: str,
    model,
    processor,
    save_overlay: bool = True,
    model_id: str | None = None,
) -> dict:
    """Procesa una imagen y persiste mask / overlay / features.csv / meta.json.

    Devuelve el dict de meta con todos los datos (lo mismo que se persiste).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.replace(" ", "_")

    # 1. Pre-procesado
    t0 = time.time()
    img_clean, info = preprocess_photo(input_path, mask_annotations=True, crop_scan=False)
    img_orig = cv2.cvtColor(cv2.imread(str(input_path)), cv2.COLOR_BGR2RGB)
    t_preproc = time.time() - t0

    # 2. Bbox prompt
    bbox, bbox_source = resolve_bbox(img_clean, img_orig, bbox_strategy)

    # 3. Inferencia
    t0 = time.time()
    mask = medsam_predict(img_clean, bbox, model, processor, device=device, threshold=threshold)
    t_infer = time.time() - t0

    mask_pixels = int((mask > 0).sum())
    mask_fraction = mask_pixels / (mask.shape[0] * mask.shape[1])

    # 4. Persistir máscara
    mask_path = output_dir / f"{stem}{SUFFIX_MASK}"
    cv2.imwrite(str(mask_path), mask)

    # 5. Features morfológicas (mismo cálculo que Pista 1)
    gray = cv2.cvtColor(img_clean, cv2.COLOR_RGB2GRAY)
    features, _labels = compute_features(mask, gray)
    features_path = output_dir / f"{stem}{SUFFIX_FEATURES}"
    write_features_csv(features_path, features)

    # 6. Overlay (opcional)
    overlay_path = None
    if save_overlay:
        overlay = build_overlay(img_clean, mask, bbox, features)
        overlay_path = output_dir / f"{stem}{SUFFIX_OVERLAY}"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 7. Meta JSON
    meta = {
        "input": str(input_path),
        "output_mask": str(mask_path),
        "output_overlay": str(overlay_path) if overlay_path else None,
        "output_features_csv": str(features_path),
        "n_features": len(features),
        "preprocess": {
            "original_size": info.original_size,
            "final_size": info.final_size,
            "green_pixels_masked": info.green_pixels_masked,
            "steps_applied": info.steps_applied,
        },
        "bbox": {
            "coords_xyxy": list(bbox),
            "source": bbox_source,
            "strategy_requested": bbox_strategy,
        },
        "inference": {
            "model_id": model_id,
            "device": device,
            "threshold": threshold,
            "input_size": DEFAULT_INPUT_SIZE,
            "mask_pixels": mask_pixels,
            "mask_fraction": round(mask_fraction, 6),
        },
        "timing": {
            "preprocess_s": round(t_preproc, 3),
            "inference_s": round(t_infer, 3),
            "total_s": round(t_preproc + t_infer, 3),
        },
    }
    meta_path = output_dir / f"{stem}{SUFFIX_META}"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return meta
