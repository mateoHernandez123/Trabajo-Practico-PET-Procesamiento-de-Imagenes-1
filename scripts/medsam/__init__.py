"""Paquete MedSAM para Pista 2 del TP de Procesamiento de Imágenes I.

MedSAM (Ma et al., Nature Communications 2024) es la adaptación médica del
Segment Anything Model de Meta, fine-tuneada sobre 1.5M pares imagen-máscara
de 10 modalidades (CT, MRI, PET, RX, US, endoscopía, dermatoscopía,
patología, OCT, fondo de ojo).

Este paquete provee una API simple para correr MedSAM sobre fotos de
celular de placas/pantallas médicas y extraer features morfológicas
reutilizando el pipeline de la Pista 1 clásica.

Uso típico (CLI):
    .venv-medsam/bin/python scripts/medsam_run.py \\
        --input imagenes/clinicas_referencia/whatsapp_2026-05-30 \\
        --output resultados_medsam/whatsapp_2026-05-30

Uso como librería:
    from medsam import load_medsam, run_one
    model, processor = load_medsam(device="cpu")
    result = run_one(image_path, output_dir, "auto", 0.5, "cpu", model, processor)

Submódulos:
    model            — carga del modelo desde HuggingFace (singleton)
    preprocess       — detección + inpaint de anotación verde
    bbox_strategies  — auto-bbox a partir de anotación, full-image o coords
    inference        — forward de MedSAM + post-procesado
    features         — extracción de features morfológicas + writer CSV
    visualize        — overlay + paneles comparativos
"""

from .model import DEFAULT_MODEL_ID, load_medsam
from .preprocess import (
    PreprocessInfo,
    detect_green_annotation_mask,
    inpaint_annotation,
    load_rgb,
    preprocess_photo,
)
from .bbox_strategies import (
    bbox_from_green_annotation,
    bbox_full_image,
    resolve_bbox,
)
from .inference import DEFAULT_INPUT_SIZE, DEFAULT_THRESHOLD, medsam_predict, run_one
from .features import write_features_csv
from .visualize import build_overlay, build_modality_panel

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_THRESHOLD",
    "PreprocessInfo",
    "load_medsam",
    "load_rgb",
    "preprocess_photo",
    "detect_green_annotation_mask",
    "inpaint_annotation",
    "bbox_from_green_annotation",
    "bbox_full_image",
    "resolve_bbox",
    "medsam_predict",
    "run_one",
    "write_features_csv",
    "build_overlay",
    "build_modality_panel",
]
