"""Estrategias para resolver el bounding-box prompt que MedSAM necesita.

MedSAM no segmenta por sí solo: hay que decirle "fijate dentro de este
rectángulo qué objeto destacar". Implementamos tres estrategias:

    auto       Si la imagen original tiene un círculo verde de anotación
               válido, usamos su bbox. Si no, fallback a full image.

    full       bbox = imagen completa con margen interno del 5%. Útil para
               PET donde la lesión es el único hot spot brillante.

    manual     coords explícitas `x1,y1,x2,y2` pasadas por CLI.
"""

from __future__ import annotations

import numpy as np

from .preprocess import detect_green_annotation_mask


# Auto-bbox "full image" usa un margen interior del 5% para excluir bordes
FULL_BBOX_MARGIN_FRAC = 0.05


def bbox_from_green_annotation(img_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Si la imagen tiene un círculo de marcado verde, devuelve su bbox.

    Devuelve `(x1, y1, x2, y2)` o `None` si no se detecta anotación válida.
    """
    mask = detect_green_annotation_mask(img_rgb)
    if int(mask.sum()) == 0:
        return None
    ys, xs = np.where(mask > 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def bbox_full_image(
    img_rgb: np.ndarray,
    margin_frac: float = FULL_BBOX_MARGIN_FRAC,
) -> tuple[int, int, int, int]:
    """Bbox que cubre toda la imagen con un margen interior `margin_frac`."""
    h, w = img_rgb.shape[:2]
    mx = int(w * margin_frac)
    my = int(h * margin_frac)
    return (mx, my, w - mx, h - my)


def resolve_bbox(
    img_rgb_clean: np.ndarray,
    img_rgb_orig: np.ndarray,
    strategy: str,
) -> tuple[tuple[int, int, int, int], str]:
    """Devuelve `(bbox, source)` donde source identifica de qué estrategia salió."""
    if strategy == "auto":
        # 1) anotación verde sobre la imagen ORIGINAL (no la inpainted)
        bbox = bbox_from_green_annotation(img_rgb_orig)
        if bbox is not None:
            return bbox, "green_annotation"
        # 2) fallback: imagen completa
        return bbox_full_image(img_rgb_clean), "full_image_fallback"

    if strategy == "full":
        return bbox_full_image(img_rgb_clean), "full_image"

    # Coords explícitas "x1,y1,x2,y2"
    try:
        coords = tuple(int(v.strip()) for v in strategy.split(","))
        if len(coords) != 4:
            raise ValueError
    except (ValueError, AttributeError):
        raise ValueError(
            f"--bbox inválido: {strategy!r}. Debe ser auto, full, o x1,y1,x2,y2"
        )
    return coords, "manual"
