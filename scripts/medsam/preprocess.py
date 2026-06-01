"""Pre-procesado de fotos de celular de placas/pantallas médicas.

Pipeline:
    1. Cargar imagen como RGB (cv2 lee en BGR, convertimos).
    2. Detectar anotaciones verdes (círculos de marcado del radiólogo) en HSV.
    3. Validar la detección por tamaño y bbox (filtra falsos positivos por
       reflejos o por el halo verde del colormap PET).
    4. Si hay anotación válida, inpaint para que MedSAM no la vea.
    5. (Opcional) Auto-recortar la región del scan dentro del frame.

Hiperparámetros (constantes al tope del módulo) están todos documentados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# Detección de anotación verde (HSV de OpenCV: H 0–179, S 0–255, V 0–255)
# Acotado a verde puro/neón (hue ~60) con saturación alta para evitar:
#   - cian/teal del colormap PET (hue ~85-110)
#   - tonos amarillos/orange (hue ~15-30, saturados)
GREEN_HSV_LOWER = np.array([40, 120, 80], dtype=np.uint8)
GREEN_HSV_UPPER = np.array([75, 255, 255], dtype=np.uint8)

# Validación post-detección (filtra falsos positivos por ruido o reflejos)
GREEN_MIN_TOTAL_PX = 800           # Píxeles totales mínimos en la imagen
GREEN_MIN_COMPONENT_PX = 400       # Tamaño mínimo del componente más grande
GREEN_MAX_BBOX_FRAC = 0.5          # El bbox NO debe cubrir > X% del frame

# Inpainting (rellena los píxeles verdes con info del entorno)
INPAINT_RADIUS_PX = 7              # Radio en píxeles que mira `cv2.inpaint`
INPAINT_METHOD = cv2.INPAINT_TELEA # Alternativa: cv2.INPAINT_NS (Navier-Stokes)

# Dilatación de la máscara verde antes de inpaint (cubre antialiasing del borde)
GREEN_DILATE_KERNEL = 5
GREEN_DILATE_ITERATIONS = 2

# Auto-crop del scan dentro de la foto (cuando hay borde negro/blanco amplio)
SCAN_CROP_THRESHOLD = 30           # Píxeles más oscuros que esto = fondo (gris)
SCAN_CROP_MIN_AREA_FRAC = 0.15     # El bbox del scan recortado debe ocupar ≥ X% del frame


@dataclass
class PreprocessInfo:
    """Metadata del pre-procesado (para debugging y documentación)."""
    original_size: tuple[int, int] = (0, 0)      # (W, H)
    green_pixels_masked: int = 0
    green_fraction: float = 0.0
    scan_bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h), None = no cropped
    final_size: tuple[int, int] = (0, 0)         # (W, H)
    steps_applied: list[str] = field(default_factory=list)


def load_rgb(path: Path | str) -> np.ndarray:
    """Lee una imagen del disco y devuelve un array RGB uint8 (H, W, 3)."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def detect_green_annotation_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Devuelve una máscara binaria (uint8 0/255) de los píxeles verdes.

    Devuelve máscara vacía si:
       - no hay píxeles verdes,
       - hay muy pocos (ruido),
       - el componente más grande es muy chico (< GREEN_MIN_COMPONENT_PX),
       - o el bbox del componente cubre demasiada área (> GREEN_MAX_BBOX_FRAC).
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(img_hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)

    total = int((raw > 0).sum())
    if total < GREEN_MIN_TOTAL_PX:
        return np.zeros_like(raw)

    num, _labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if num <= 1:
        return np.zeros_like(raw)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < GREEN_MIN_COMPONENT_PX:
        return np.zeros_like(raw)

    w = int(stats[largest, cv2.CC_STAT_WIDTH])
    h = int(stats[largest, cv2.CC_STAT_HEIGHT])
    frame_area = img_rgb.shape[0] * img_rgb.shape[1]
    if (w * h) / frame_area > GREEN_MAX_BBOX_FRAC:
        return np.zeros_like(raw)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GREEN_DILATE_KERNEL,) * 2)
    return cv2.dilate(raw, kernel, iterations=GREEN_DILATE_ITERATIONS)


def inpaint_annotation(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rellena los píxeles enmascarados usando información del entorno."""
    if int(mask.sum()) == 0:
        return img_rgb
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_inpainted = cv2.inpaint(img_bgr, mask, INPAINT_RADIUS_PX, INPAINT_METHOD)
    return cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB)


def auto_crop_scan(img_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Recorta el bbox del 'scan' dentro de la foto, descartando bordes oscuros.

    Heurística: convierte a gris, umbraliza > SCAN_CROP_THRESHOLD, toma el
    bbox del componente conexo más grande. Si ese bbox cubre menos de
    SCAN_CROP_MIN_AREA_FRAC del frame, no recorta.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, SCAN_CROP_THRESHOLD, 255, cv2.THRESH_BINARY)
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return img_rgb, None

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    w = int(stats[largest, cv2.CC_STAT_WIDTH])
    h = int(stats[largest, cv2.CC_STAT_HEIGHT])

    frame_area = img_rgb.shape[0] * img_rgb.shape[1]
    if (w * h) / frame_area < SCAN_CROP_MIN_AREA_FRAC:
        return img_rgb, None

    return img_rgb[y : y + h, x : x + w].copy(), (x, y, w, h)


def preprocess_photo(
    path: Path | str,
    mask_annotations: bool = True,
    crop_scan: bool = False,
) -> tuple[np.ndarray, PreprocessInfo]:
    """Pipeline completo. Devuelve (img_rgb_limpia, info)."""
    info = PreprocessInfo()
    img = load_rgb(path)
    info.original_size = (img.shape[1], img.shape[0])
    info.steps_applied.append("load_rgb")

    if mask_annotations:
        mask = detect_green_annotation_mask(img)
        info.green_pixels_masked = int(mask.sum() // 255)
        info.green_fraction = info.green_pixels_masked / (img.shape[0] * img.shape[1])
        if info.green_pixels_masked > 0:
            img = inpaint_annotation(img, mask)
            info.steps_applied.append(f"inpaint_green(n_px={info.green_pixels_masked})")

    if crop_scan:
        img, bbox = auto_crop_scan(img)
        info.scan_bbox = bbox
        if bbox is not None:
            info.steps_applied.append(f"crop_scan(bbox={bbox})")

    info.final_size = (img.shape[1], img.shape[0])
    return img, info
