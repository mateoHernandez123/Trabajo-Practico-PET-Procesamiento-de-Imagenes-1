"""Generación de overlays y paneles comparativos.

Funciones:
    build_overlay         Superpone máscara, bbox-prompt y bbox-features
                          sobre una imagen RGB. Devuelve array RGB.
    build_modality_panel  Construye un panel original / máscara / overlay
                          side-by-side para una lista de imágenes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# Colores para el overlay (RGB)
COLOR_MASK_FILL = (255, 255, 0)       # Amarillo translúcido
COLOR_BBOX_PROMPT = (0, 100, 255)     # Azul (bbox usado como prompt MedSAM)
COLOR_BBOX_FEATURE = (255, 0, 255)    # Magenta (bbox por feature detectada)

MASK_FILL_ALPHA = 0.3                 # Transparencia del relleno amarillo
BBOX_PROMPT_THICKNESS = 3
BBOX_FEATURE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2


def build_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_prompt: tuple[int, int, int, int],
    features: list[dict],
) -> np.ndarray:
    """Devuelve la imagen RGB con la máscara, el bbox-prompt y los bbox-feature."""
    overlay = img_rgb.copy()
    overlay_color = np.zeros_like(overlay)
    overlay_color[mask > 0] = COLOR_MASK_FILL
    overlay = cv2.addWeighted(
        overlay, 1.0 - MASK_FILL_ALPHA, overlay_color, MASK_FILL_ALPHA, 0
    )
    cv2.rectangle(
        overlay,
        (bbox_prompt[0], bbox_prompt[1]),
        (bbox_prompt[2], bbox_prompt[3]),
        COLOR_BBOX_PROMPT,
        BBOX_PROMPT_THICKNESS,
    )
    for f in features:
        x, y, w, h = f["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BBOX_FEATURE, BBOX_FEATURE_THICKNESS)
        cv2.putText(
            overlay,
            f"#{f['id']}",
            (x, max(y - 6, 12)),
            FONT, FONT_SCALE, COLOR_BBOX_FEATURE, FONT_THICKNESS,
        )
    return overlay


def _thumb(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Reescala una imagen RGB para que el lado mayor sea `max_dim`."""
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale == 1.0:
        return img
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def _pad_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Centra `img` en un canvas blanco (target_h, target_w, 3)."""
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    h, w = img.shape[:2]
    y0 = (target_h - h) // 2
    x0 = (target_w - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = img
    return canvas


def build_modality_panel(
    rows: list[dict],
    output_path: Path,
    title: str,
    cell_max_dim: int = 320,
    label_height: int = 28,
) -> None:
    """Construye un panel original / máscara / overlay para una modalidad.

    rows = [
        {
            "label":   "ct_torax_axial_panel_01",
            "image":   <ruta a la foto ORIGINAL>,
            "mask":    <ruta al _mask.png>,
            "overlay": <ruta al _overlay.png>,
        },
        ...
    ]
    """
    if not rows:
        raise ValueError("rows vacío: no hay nada para graficar")

    # Cargar y armar grid
    triplets = []
    for r in rows:
        img = cv2.cvtColor(cv2.imread(str(r["image"])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(r["mask"]), cv2.IMREAD_GRAYSCALE)
        # Convertir mask a 3 canales para que se pueda colocar en el panel
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        overlay = cv2.cvtColor(cv2.imread(str(r["overlay"])), cv2.COLOR_BGR2RGB)
        triplets.append((r["label"], img, mask_rgb, overlay))

    cell_w = cell_max_dim
    cell_h = cell_max_dim
    cols = 3                       # original | mask | overlay
    n_rows = len(triplets)
    title_h = 50
    col_header_h = 25

    canvas_w = cols * cell_w + 20
    canvas_h = title_h + col_header_h + n_rows * (cell_h + label_height) + 10
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # Título
    cv2.putText(canvas, title, (10, 35), FONT, 1.0, (0, 0, 0), 2)

    # Headers de columna
    for c, hdr in enumerate(["ORIGINAL", "MASCARA MedSAM", "OVERLAY"]):
        x0 = 10 + c * cell_w
        cv2.putText(canvas, hdr, (x0 + 5, title_h + 18), FONT, 0.55, (50, 50, 50), 1)

    # Filas
    for r_idx, (label, img, mask_rgb, overlay) in enumerate(triplets):
        y0 = title_h + col_header_h + r_idx * (cell_h + label_height)
        cv2.putText(canvas, label, (10, y0 + 18), FONT, 0.5, (0, 0, 0), 1)
        for c, cell_img in enumerate([img, mask_rgb, overlay]):
            thumb = _thumb(cell_img, cell_max_dim - 10)
            padded = _pad_to(thumb, cell_h, cell_w)
            xx = 10 + c * cell_w
            yy = y0 + label_height
            canvas[yy : yy + cell_h, xx : xx + cell_w] = padded

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
