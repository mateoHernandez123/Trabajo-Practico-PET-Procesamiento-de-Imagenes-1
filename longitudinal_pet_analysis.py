"""
Análisis longitudinal de tumores cerebrales en imágenes PET.

Permite comparar estudios PET del MISMO paciente en distintos momentos
para evaluar la evolución tumoral: crecimiento, respuesta al tratamiento
o estabilidad.

Pipeline:
    1. Carga de imágenes PET de múltiples timepoints.
    2. Segmentación tumoral (nnU-Net JuST_BrainPET o métodos clásicos).
    3. Registro espacial entre timepoints (alineación).
    4. Comparación de máscaras tumorales a lo largo del tiempo.
    5. Cálculo de métricas longitudinales (volumen, Dice, RECIST).
    6. Generación de visualizaciones y reportes.

Modos de segmentación:
    - nnunet   : nnU-Net v1 con JuST_BrainPET (Task169_BrainTumorPET).
                 Requiere NIfTI 3D + nnU-Net instalado.
    - classical: K-Means + morfología (funciona con PNG/JPG 2D).

Uso:
    # Demo con datos sintéticos (no requiere imágenes externas):
    python longitudinal_pet_analysis.py --generate-demo

    # Analizar imágenes PET de un paciente:
    python longitudinal_pet_analysis.py carpeta_paciente/

    # Analizar con nnU-Net (requiere instalación previa):
    python longitudinal_pet_analysis.py carpeta_paciente/ --method nnunet

    # Imagen PET existente del proyecto + timepoints sintéticos:
    python longitudinal_pet_analysis.py imagenes/pet_cuerpo_completo.png --generate-demo

Dependencias base: numpy, opencv-python, matplotlib, scikit-image, pandas.
Opcionales (nnU-Net): torch, nnunet (v1), nibabel, SimpleITK.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

NNUNET_AVAILABLE = shutil.which("nnUNet_predict") is not None

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEMO_SIZE = 256
BRAIN_RADIUS_FRAC = 0.38
TUMOR_DEFAULT_CENTER = (170, 120)
TUMOR_DEFAULT_AXES = (22, 16)
TUMOR_BASE_INTENSITY = 225

KMEANS_K = 4
MIN_LESION_AREA = 30
ERODE_KERNEL = 3
ERODE_ITERATIONS = 1
DILATE_KERNEL = 5
DILATE_ITERATIONS = 2
MORPH_KERNEL = 5

# Criterios RECIST simplificados (basados en cambio de área/volumen)
RECIST_CR_THRESHOLD = -0.90   # Complete Response: >90% reducción
RECIST_PR_THRESHOLD = -0.30   # Partial Response:  >30% reducción
RECIST_PD_THRESHOLD = 0.20    # Progressive Disease: >20% aumento

CMAP_GROWTH = (0, 0, 255)     # Rojo — crecimiento tumoral
CMAP_SHRINK = (0, 200, 0)     # Verde — reducción tumoral
CMAP_STABLE = (255, 200, 0)   # Amarillo — tumor estable


# ---------------------------------------------------------------------------
# Generación de datos demo (cerebro PET sintético)
# ---------------------------------------------------------------------------

def _make_brain_pet(
    size: int = DEMO_SIZE,
    tumor_center: tuple[int, int] = TUMOR_DEFAULT_CENTER,
    tumor_axes: tuple[int, int] = TUMOR_DEFAULT_AXES,
    tumor_intensity: int = TUMOR_BASE_INTENSITY,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera un corte axial PET cerebral sintético con un tumor.

    Retorna (imagen_gray, máscara_tumor).
    Convención: mayor intensidad = mayor captación (tumor brilla).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    img = np.full((size, size), 8, dtype=np.float32)
    cx, cy = size // 2, size // 2
    brain_r = int(size * BRAIN_RADIUS_FRAC)

    # Parénquima cerebral base
    brain_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(brain_mask, (cx, cy), brain_r, 255, -1)
    img[brain_mask > 0] = 90

    # Corteza (anillo externo, mayor captación que sustancia blanca)
    cortex = np.zeros_like(brain_mask)
    cv2.circle(cortex, (cx, cy), brain_r, 255, -1)
    inner = np.zeros_like(brain_mask)
    cv2.circle(inner, (cx, cy), int(brain_r * 0.72), 255, -1)
    cortex = cv2.subtract(cortex, inner)
    img[cortex > 0] = 120

    # Núcleos basales (captación moderada-alta, bilateral)
    for dx in [-18, 18]:
        cv2.ellipse(img, (cx + dx, cy + 3), (7, 10), 0, 0, 360, 135, -1)

    # Ventrículos (baja captación)
    cv2.ellipse(img, (cx, cy - 2), (8, 18), 0, 0, 360, 35, -1)

    # Tumor (alta captación)
    tumor_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(tumor_mask, tumor_center, tumor_axes, 20, 0, 360, 255, -1)
    img[tumor_mask > 0] = tumor_intensity

    # Heterogeneidad tumoral
    tumor_ys, tumor_xs = np.where(tumor_mask > 0)
    if len(tumor_ys) > 0:
        het_noise = rng.normal(0, 12, size=len(tumor_ys))
        img[tumor_ys, tumor_xs] = np.clip(
            img[tumor_ys, tumor_xs] + het_noise, tumor_intensity - 30, 255
        )

    # Textura cerebral (ruido de fondo)
    tissue_noise = rng.normal(0, 4, img.shape).astype(np.float32)
    img += tissue_noise
    img = np.clip(img, 0, 255)

    # Suavizado PET (resolución limitada del scanner)
    img = cv2.GaussianBlur(img, (7, 7), 1.8)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img, tumor_mask


def generate_demo_data(
    out_dir: Path,
    base_image_path: Path | None = None,
) -> list[dict]:
    """Genera datos longitudinales sintéticos con 4 timepoints.

    Escenario clínico simulado:
        T0 (Enero):  Tumor detectado — línea base
        T1 (Abril):  Tumor crece ~20% (sin tratamiento aún)
        T2 (Julio):  Tratamiento iniciado, tumor se reduce ~15%
        T3 (Octubre): Buena respuesta, tumor se reduce ~40% vs peak

    Retorna lista de dicts: {date, image, mask, label, path_img, path_mask}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2025)

    base_date = datetime(2025, 1, 15)
    scenarios = [
        {"label": "T0_baseline",    "months": 0,  "axes_scale": 1.00, "intensity_delta": 0},
        {"label": "T1_crecimiento", "months": 3,  "axes_scale": 1.20, "intensity_delta": 8},
        {"label": "T2_tratamiento", "months": 6,  "axes_scale": 1.02, "intensity_delta": -5},
        {"label": "T3_respuesta",   "months": 9,  "axes_scale": 0.65, "intensity_delta": -20},
    ]

    timepoints: list[dict] = []

    for sc in scenarios:
        date = base_date + timedelta(days=sc["months"] * 30)
        ax_a = int(TUMOR_DEFAULT_AXES[0] * sc["axes_scale"])
        ax_b = int(TUMOR_DEFAULT_AXES[1] * sc["axes_scale"])
        intensity = np.clip(TUMOR_BASE_INTENSITY + sc["intensity_delta"], 150, 255)

        img, mask = _make_brain_pet(
            tumor_axes=(max(ax_a, 3), max(ax_b, 3)),
            tumor_intensity=int(intensity),
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )

        fname_img = f"{sc['label']}_{date.strftime('%Y-%m')}.png"
        fname_mask = f"{sc['label']}_{date.strftime('%Y-%m')}_mask.png"
        path_img = out_dir / fname_img
        path_mask = out_dir / fname_mask

        cv2.imwrite(str(path_img), img)
        cv2.imwrite(str(path_mask), mask)

        timepoints.append({
            "date": date,
            "image": img,
            "mask": mask,
            "label": sc["label"],
            "path_img": path_img,
            "path_mask": path_mask,
        })

    metadata = {
        "patient_id": "DEMO_001",
        "modality": "PET_sintetico",
        "description": "Datos sintéticos para demostración de análisis longitudinal",
        "timepoints": [
            {
                "label": tp["label"],
                "date": tp["date"].isoformat(),
                "image": tp["path_img"].name,
                "mask": tp["path_mask"].name,
            }
            for tp in timepoints
        ],
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Demo generado en: {out_dir}")
    print(f"  Timepoints: {len(timepoints)}")
    for tp in timepoints:
        area = int(np.sum(tp["mask"] > 0))
        print(f"    {tp['label']:20s}  fecha={tp['date'].strftime('%Y-%m-%d')}  "
              f"área_tumor={area}px")

    return timepoints


# ---------------------------------------------------------------------------
# Carga de datos reales
# ---------------------------------------------------------------------------

def load_timepoints_from_dir(
    patient_dir: Path,
    mask_suffix: str = "_mask",
) -> list[dict]:
    """Carga timepoints desde un directorio de paciente.

    Espera pares de archivos: imagen + máscara (opcional).
    Si no hay máscara, se segmenta con métodos clásicos.
    """
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    nifti_ext = {".nii", ".gz"}

    image_files = sorted([
        p for p in patient_dir.iterdir()
        if p.suffix.lower() in extensions
        and mask_suffix not in p.stem
        and p.stem != "metadata"
    ])

    if not image_files:
        nifti_files = sorted([
            p for p in patient_dir.iterdir()
            if p.name.endswith(".nii.gz") or p.suffix == ".nii"
        ])
        if nifti_files:
            return _load_nifti_timepoints(nifti_files, patient_dir)
        raise FileNotFoundError(
            f"No se encontraron imágenes en {patient_dir}\n"
            f"Formatos soportados: {extensions} o NIfTI (.nii.gz)"
        )

    timepoints: list[dict] = []
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  WARN: No se pudo leer {img_path.name}, omitiendo.")
            continue

        mask_path = img_path.with_stem(img_path.stem + mask_suffix)
        if not mask_path.exists():
            for ext in extensions:
                candidate = img_path.with_stem(img_path.stem + mask_suffix).with_suffix(ext)
                if candidate.exists():
                    mask_path = candidate
                    break

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            mask = segment_classical(img)

        date = _extract_date_from_name(img_path.stem, i)

        timepoints.append({
            "date": date,
            "image": img,
            "mask": mask,
            "label": img_path.stem,
            "path_img": img_path,
            "path_mask": mask_path if mask_path.exists() else None,
        })

    timepoints.sort(key=lambda t: t["date"])
    return timepoints


def _load_nifti_timepoints(
    nifti_files: list[Path],
    patient_dir: Path,
) -> list[dict]:
    """Carga timepoints desde archivos NIfTI 3D."""
    if not HAS_NIBABEL:
        raise ImportError(
            "Se requiere nibabel para cargar NIfTI.\n"
            "Instalar: pip install nibabel"
        )

    timepoints = []
    for i, nii_path in enumerate(nifti_files):
        nii = nib.load(str(nii_path))
        vol = np.asanyarray(nii.dataobj)
        mid_slice = vol.shape[2] // 2
        img_2d = vol[:, :, mid_slice].astype(np.float32)
        img_2d = ((img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8) * 255)
        img_2d = img_2d.astype(np.uint8)

        mask = segment_classical(img_2d)
        date = _extract_date_from_name(nii_path.stem, i)

        timepoints.append({
            "date": date,
            "image": img_2d,
            "mask": mask,
            "label": nii_path.stem,
            "path_img": nii_path,
            "path_mask": None,
            "nifti": nii,
            "spacing": nii.header.get_zooms(),
        })

    timepoints.sort(key=lambda t: t["date"])
    return timepoints


def _extract_date_from_name(stem: str, index: int) -> datetime:
    """Intenta extraer una fecha del nombre del archivo."""
    import re
    m = re.search(r"(\d{4})[-_](\d{2})(?:[-_](\d{2}))?", stem)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        d = int(m.group(3)) if m.group(3) else 15
        try:
            return datetime(y, mo, d)
        except ValueError:
            pass
    return datetime(2025, 1 + index * 3, 15)


# ---------------------------------------------------------------------------
# Segmentación clásica (fallback sin nnU-Net)
# ---------------------------------------------------------------------------

def segment_classical(gray: np.ndarray) -> np.ndarray:
    """Segmenta tumores en PET cerebral usando K-Means + morfología."""
    denoised = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Máscara cerebral (excluir fondo)
    _, brain = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain = cv2.morphologyEx(brain, cv2.MORPH_CLOSE, kernel, iterations=3)
    brain = cv2.morphologyEx(brain, cv2.MORPH_OPEN, kernel, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(brain, connectivity=8)
    if num > 1:
        largest = max(range(1, num), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        brain = np.where(labels == largest, 255, 0).astype(np.uint8)

    # K-Means: cluster más brillante = tumor (PET cerebral)
    ys, xs = np.where(brain > 0)
    if ys.size < KMEANS_K:
        return np.zeros_like(gray)

    pixels = denoised[ys, xs].reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, km_labels, centers = cv2.kmeans(
        pixels, KMEANS_K, None, criteria, 5, cv2.KMEANS_PP_CENTERS,
    )
    km_labels = km_labels.flatten()
    centers_flat = centers.flatten()

    hot_cluster = int(np.argmax(centers_flat))
    mask_raw = np.zeros_like(gray)
    mask_raw[ys[km_labels == hot_cluster], xs[km_labels == hot_cluster]] = 255

    # Morfología
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (ERODE_KERNEL, ERODE_KERNEL))
    mask = cv2.erode(mask_raw, k_erode, iterations=ERODE_ITERATIONS)

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (DILATE_KERNEL, DILATE_KERNEL))
    mask = cv2.dilate(mask, k_dilate, iterations=DILATE_ITERATIONS)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Filtro por área mínima
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_LESION_AREA:
            result[labels == i] = 255

    return result


# ---------------------------------------------------------------------------
# Segmentación con nnU-Net (JuST_BrainPET)
# ---------------------------------------------------------------------------

def segment_nnunet(
    input_dir: Path,
    output_dir: Path,
    task: str = "Task169_BrainTumorPET",
    config: str = "3d_fullres",
) -> Path:
    """Ejecuta nnU-Net v1 inference para PET cerebral.

    Requiere:
        - nnU-Net v1 instalado: pip install git+https://github.com/MIC-DKFZ/nnUNet.git@nnunetv1
        - Modelo descargado: nnUNet_download_pretrained_model Task169_BrainTumorPET
        - Variables de entorno: nnUNet_raw, nnUNet_preprocessed, nnUNet_results
        - Imágenes de entrada en NIfTI (.nii.gz) dentro de input_dir

    Args:
        input_dir:  Carpeta con archivos NIfTI de entrada.
        output_dir: Carpeta donde se guardarán las segmentaciones.
        task:       Nombre del task nnU-Net.
        config:     Configuración del modelo (2d, 3d_fullres, etc.).

    Returns:
        Path a la carpeta de salida con las segmentaciones.
    """
    if not NNUNET_AVAILABLE:
        raise RuntimeError(
            "nnU-Net no está instalado o no se encuentra en el PATH.\n\n"
            "Para instalar nnU-Net v1 con JuST_BrainPET:\n"
            "  1. pip install torch  (con CUDA si tienes GPU)\n"
            "  2. pip install git+https://github.com/MIC-DKFZ/nnUNet.git@nnunetv1\n"
            "  3. Configurar variables de entorno:\n"
            "     export nnUNet_raw_data_base=/path/to/nnUNet_raw\n"
            "     export nnUNet_preprocessed=/path/to/nnUNet_preprocessed\n"
            "     export RESULTS_FOLDER=/path/to/nnUNet_results\n"
            "  4. nnUNet_download_pretrained_model Task169_BrainTumorPET\n"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNet_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-t", task,
        "-m", config,
    ]

    print(f"  Ejecutando nnU-Net: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nnU-Net falló (código {result.returncode}):\n{result.stderr}"
        )

    print(f"  Segmentaciones guardadas en: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Registro espacial (alineación entre timepoints)
# ---------------------------------------------------------------------------

def register_2d(
    fixed: np.ndarray,
    moving: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Registro rígido 2D usando correlación de fase.

    Alinea la imagen `moving` al espacio de `fixed`.
    Retorna (imagen_registrada, matriz_de_transformación).
    """
    fixed_f = fixed.astype(np.float64)
    moving_f = moving.astype(np.float64)

    (dx, dy), response = cv2.phaseCorrelate(fixed_f, moving_f)

    rows, cols = fixed.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    registered = cv2.warpAffine(moving, M, (cols, rows))

    return registered, M


def register_mask(mask: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Aplica transformación a una máscara (nearest-neighbor)."""
    rows, cols = mask.shape
    registered = cv2.warpAffine(
        mask, transform, (cols, rows),
        flags=cv2.INTER_NEAREST,
    )
    return registered


def register_all_to_baseline(
    timepoints: list[dict],
) -> list[dict]:
    """Registra todos los timepoints al espacio del baseline (T0)."""
    if len(timepoints) < 2:
        return timepoints

    baseline = timepoints[0]["image"]
    result = [timepoints[0].copy()]
    result[0]["registered"] = True

    for tp in timepoints[1:]:
        reg_img, M = register_2d(baseline, tp["image"])
        reg_mask = register_mask(tp["mask"], M)

        tp_copy = tp.copy()
        tp_copy["image_original"] = tp["image"]
        tp_copy["mask_original"] = tp["mask"]
        tp_copy["image"] = reg_img
        tp_copy["mask"] = reg_mask
        tp_copy["transform"] = M
        tp_copy["registered"] = True
        result.append(tp_copy)

    return result


# ---------------------------------------------------------------------------
# Métricas longitudinales
# ---------------------------------------------------------------------------

def tumor_area(mask: np.ndarray) -> int:
    """Área tumoral en píxeles."""
    return int(np.sum(mask > 0))


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice Similarity Coefficient entre dos máscaras binarias."""
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    intersection = int(np.sum(a & b))
    total = int(np.sum(a) + np.sum(b))
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def jaccard_index(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Índice de Jaccard (IoU) entre dos máscaras."""
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    intersection = int(np.sum(a & b))
    union = int(np.sum(a | b))
    if union == 0:
        return 1.0
    return intersection / union


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Centroide (x, y) de la máscara tumoral."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)))


def centroid_displacement(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float | None:
    """Desplazamiento euclidiano del centroide tumoral entre dos masks."""
    c_a = centroid(mask_a)
    c_b = centroid(mask_b)
    if c_a is None or c_b is None:
        return None
    return math.sqrt((c_a[0] - c_b[0]) ** 2 + (c_a[1] - c_b[1]) ** 2)


def new_tumor_regions(mask_prev: np.ndarray, mask_curr: np.ndarray) -> np.ndarray:
    """Píxeles tumorales nuevos (presentes en curr pero no en prev)."""
    prev = (mask_prev > 0).astype(np.uint8)
    curr = (mask_curr > 0).astype(np.uint8)
    return (curr & ~prev) * 255


def disappeared_regions(mask_prev: np.ndarray, mask_curr: np.ndarray) -> np.ndarray:
    """Píxeles tumorales que desaparecieron (en prev pero no en curr)."""
    prev = (mask_prev > 0).astype(np.uint8)
    curr = (mask_curr > 0).astype(np.uint8)
    return (prev & ~curr) * 255


def recist_classification(area_change_pct: float) -> str:
    """Clasificación RECIST simplificada basada en cambio de área."""
    if area_change_pct <= RECIST_CR_THRESHOLD * 100:
        return "CR (Respuesta Completa)"
    elif area_change_pct <= RECIST_PR_THRESHOLD * 100:
        return "PR (Respuesta Parcial)"
    elif area_change_pct >= RECIST_PD_THRESHOLD * 100:
        return "PD (Enfermedad Progresiva)"
    return "SD (Enfermedad Estable)"


def mean_tumor_intensity(image: np.ndarray, mask: np.ndarray) -> float | None:
    """Intensidad media del tumor (proxy de actividad metabólica)."""
    pixels = image[mask > 0]
    if len(pixels) == 0:
        return None
    return float(np.mean(pixels))


def compute_longitudinal_metrics(
    timepoints: list[dict],
) -> list[dict]:
    """Calcula métricas longitudinales entre timepoints consecutivos.

    Retorna lista de dicts con métricas por par de timepoints.
    """
    metrics: list[dict] = []

    baseline_area = tumor_area(timepoints[0]["mask"])

    for i in range(len(timepoints)):
        tp = timepoints[i]
        area = tumor_area(tp["mask"])
        c = centroid(tp["mask"])
        intensity = mean_tumor_intensity(tp["image"], tp["mask"])

        m = {
            "timepoint": tp["label"],
            "date": tp["date"],
            "area_px": area,
            "centroid_x": c[0] if c else None,
            "centroid_y": c[1] if c else None,
            "mean_intensity": intensity,
        }

        if i == 0:
            m["area_change_abs"] = 0
            m["area_change_pct"] = 0.0
            m["dice_vs_prev"] = None
            m["jaccard_vs_prev"] = None
            m["centroid_disp"] = None
            m["new_area_px"] = 0
            m["disappeared_area_px"] = 0
            m["recist"] = "Baseline"
            m["area_change_vs_baseline_pct"] = 0.0
        else:
            prev = timepoints[i - 1]
            prev_area = tumor_area(prev["mask"])

            area_change_abs = area - prev_area
            area_change_pct = (
                (area_change_abs / prev_area * 100) if prev_area > 0 else 0.0
            )
            area_change_vs_bl = (
                ((area - baseline_area) / baseline_area * 100)
                if baseline_area > 0 else 0.0
            )

            m["area_change_abs"] = area_change_abs
            m["area_change_pct"] = round(area_change_pct, 1)
            m["dice_vs_prev"] = round(dice_coefficient(prev["mask"], tp["mask"]), 3)
            m["jaccard_vs_prev"] = round(jaccard_index(prev["mask"], tp["mask"]), 3)
            m["centroid_disp"] = round(centroid_displacement(prev["mask"], tp["mask"]) or 0, 1)
            m["new_area_px"] = int(np.sum(new_tumor_regions(prev["mask"], tp["mask"]) > 0))
            m["disappeared_area_px"] = int(np.sum(disappeared_regions(prev["mask"], tp["mask"]) > 0))
            m["recist"] = recist_classification(area_change_vs_bl)
            m["area_change_vs_baseline_pct"] = round(area_change_vs_bl, 1)

        metrics.append(m)

    return metrics


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def _overlay_mask(gray: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.35):
    """Superpone máscara coloreada sobre imagen en escala de grises."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)


def plot_timepoints_grid(
    timepoints: list[dict],
    metrics: list[dict],
    save_path: Path,
) -> None:
    """Grilla con imagen + overlay de segmentación para cada timepoint."""
    n = len(timepoints)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (tp, m) in enumerate(zip(timepoints, metrics)):
        # Fila 1: imagen con overlay
        overlay = _overlay_mask(tp["image"], tp["mask"], color=(0, 255, 0))
        axes[0][i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        date_str = tp["date"].strftime("%Y-%m-%d")
        axes[0][i].set_title(
            f"{tp['label']}\n{date_str}\nÁrea: {m['area_px']} px",
            fontsize=9,
        )
        axes[0][i].axis("off")

        # Fila 2: máscara binaria
        axes[1][i].imshow(tp["mask"], cmap="gray")
        change_str = ""
        if i > 0:
            sign = "+" if m["area_change_pct"] >= 0 else ""
            change_str = f"Cambio: {sign}{m['area_change_pct']}%\n{m['recist']}"
        else:
            change_str = "BASELINE"
        axes[1][i].set_title(change_str, fontsize=9)
        axes[1][i].axis("off")

    plt.suptitle("Evolución temporal — Segmentación tumoral PET", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_volume_timeline(
    metrics: list[dict],
    save_path: Path,
) -> None:
    """Gráfico de línea: área tumoral vs tiempo."""
    dates = [m["date"] for m in metrics]
    areas = [m["area_px"] for m in metrics]
    labels = [m["timepoint"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(dates, areas, "o-", color="#2196F3", linewidth=2.5, markersize=10,
            markerfacecolor="white", markeredgewidth=2)

    for i, (d, a, lbl) in enumerate(zip(dates, areas, labels)):
        offset_y = max(areas) * 0.04
        ax.annotate(
            f"{a} px\n{lbl}",
            (d, a),
            textcoords="offset points",
            xytext=(0, 12 + (15 if i % 2 else 0)),
            ha="center", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    # Resaltar tendencia
    if len(areas) >= 2:
        peak_idx = int(np.argmax(areas))
        ax.axvline(dates[peak_idx], color="red", linestyle="--", alpha=0.3,
                   label=f"Pico: {areas[peak_idx]} px")

    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Área tumoral (píxeles)", fontsize=11)
    ax.set_title("Evolución del volumen tumoral a lo largo del tiempo",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_change_heatmaps(
    timepoints: list[dict],
    save_path: Path,
) -> None:
    """Heatmaps de cambio tumoral entre timepoints consecutivos."""
    n_pairs = len(timepoints) - 1
    if n_pairs < 1:
        return

    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]

    for i in range(n_pairs):
        prev_mask = timepoints[i]["mask"]
        curr_mask = timepoints[i + 1]["mask"]
        base_img = timepoints[i + 1]["image"]

        new_px = new_tumor_regions(prev_mask, curr_mask)
        gone_px = disappeared_regions(prev_mask, curr_mask)
        stable = ((prev_mask > 0) & (curr_mask > 0)).astype(np.uint8) * 255

        rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
        rgb[new_px > 0] = CMAP_GROWTH       # Rojo: crecimiento
        rgb[gone_px > 0] = CMAP_SHRINK      # Verde: reducción
        rgb[stable > 0] = CMAP_STABLE       # Amarillo: estable

        axes[i].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        t_prev = timepoints[i]["label"]
        t_curr = timepoints[i + 1]["label"]
        axes[i].set_title(f"{t_prev} >> {t_curr}", fontsize=10)
        axes[i].axis("off")

    # Leyenda manual
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(1, 0, 0), label="Crecimiento"),
        Patch(facecolor=(0, 0.78, 0), label="Reducción"),
        Patch(facecolor=(1, 0.78, 0), label="Estable"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, frameon=True)

    plt.suptitle("Mapa de cambios tumorales entre timepoints", fontsize=13,
                 fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_dashboard(
    metrics: list[dict],
    save_path: Path,
) -> None:
    """Dashboard con métricas longitudinales."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dates = [m["date"] for m in metrics]
    areas = [m["area_px"] for m in metrics]

    # 1. Área tumoral
    ax = axes[0][0]
    colors = []
    for m in metrics:
        if m["area_change_pct"] > 0:
            colors.append("#F44336")
        elif m["area_change_pct"] < 0:
            colors.append("#4CAF50")
        else:
            colors.append("#2196F3")
    ax.bar(range(len(metrics)), areas, color=colors, alpha=0.8, edgecolor="gray")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m["timepoint"] for m in metrics], fontsize=8, rotation=30)
    ax.set_ylabel("Área (px)")
    ax.set_title("Área tumoral por timepoint")
    ax.grid(axis="y", alpha=0.3)

    # 2. Cambio porcentual vs baseline
    ax = axes[0][1]
    pct_changes = [m["area_change_vs_baseline_pct"] for m in metrics]
    bar_colors = ["#4CAF50" if p <= 0 else "#F44336" for p in pct_changes]
    ax.bar(range(len(metrics)), pct_changes, color=bar_colors, alpha=0.8,
           edgecolor="gray")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-30, color="orange", linewidth=0.8, linestyle="--", alpha=0.5,
               label="RECIST PR (-30%)")
    ax.axhline(20, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
               label="RECIST PD (+20%)")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m["timepoint"] for m in metrics], fontsize=8, rotation=30)
    ax.set_ylabel("Cambio vs baseline (%)")
    ax.set_title("Cambio porcentual respecto al baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 3. Dice Similarity
    ax = axes[1][0]
    dice_vals = [m["dice_vs_prev"] for m in metrics if m["dice_vs_prev"] is not None]
    dice_labels = [m["timepoint"] for m in metrics if m["dice_vs_prev"] is not None]
    if dice_vals:
        ax.plot(range(len(dice_vals)), dice_vals, "s-", color="#9C27B0",
                linewidth=2, markersize=8)
        ax.set_xticks(range(len(dice_vals)))
        ax.set_xticklabels(dice_labels, fontsize=8, rotation=30)
        ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Dice Coefficient")
    ax.set_title("Similitud Dice vs timepoint anterior")
    ax.grid(True, alpha=0.3)

    # 4. Tabla resumen
    ax = axes[1][1]
    ax.axis("off")
    col_labels = ["Timepoint", "Área", "Cambio%", "RECIST"]
    cell_text = []
    for m in metrics:
        sign = "+" if m["area_change_pct"] > 0 else ""
        cell_text.append([
            m["timepoint"],
            str(m["area_px"]),
            f"{sign}{m['area_change_pct']}%" if m["area_change_pct"] != 0 else "—",
            m["recist"],
        ])
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j, label in enumerate(col_labels):
        table[0, j].set_facecolor("#E3F2FD")
    ax.set_title("Resumen de evaluación RECIST", fontsize=11, pad=20)

    plt.suptitle("Dashboard de análisis longitudinal", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def save_metrics_csv(path: Path, metrics: list[dict]) -> None:
    """Guarda métricas longitudinales en CSV."""
    headers = [
        "timepoint", "date", "area_px",
        "area_change_abs", "area_change_pct", "area_change_vs_baseline_pct",
        "dice_vs_prev", "jaccard_vs_prev",
        "centroid_x", "centroid_y", "centroid_disp",
        "new_area_px", "disappeared_area_px",
        "mean_intensity", "recist",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for m in metrics:
            row = m.copy()
            row["date"] = row["date"].strftime("%Y-%m-%d")
            writer.writerow(row)


def save_text_report(path: Path, metrics: list[dict], patient_id: str) -> None:
    """Genera reporte de texto legible."""
    lines = [
        "=" * 70,
        f"  REPORTE DE ANÁLISIS LONGITUDINAL — {patient_id}",
        f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
        f"  Numero de timepoints: {len(metrics)}",
        f"  Periodo: {metrics[0]['date'].strftime('%Y-%m-%d')} -- "
        f"{metrics[-1]['date'].strftime('%Y-%m-%d')}",
        "",
    ]

    # Evolución
    lines.append("  EVOLUCIÓN TUMORAL")
    lines.append("  " + "-" * 60)
    for m in metrics:
        sign = "+" if m["area_change_pct"] > 0 else ""
        lines.append(
            f"  {m['timepoint']:22s}  "
            f"Fecha: {m['date'].strftime('%Y-%m-%d')}  "
            f"Área: {m['area_px']:>5d} px  "
            f"Cambio: {sign}{m['area_change_pct']:>6.1f}%  "
            f"{m['recist']}"
        )

    # Resumen
    baseline_area = metrics[0]["area_px"]
    final_area = metrics[-1]["area_px"]
    if baseline_area > 0:
        total_change = (final_area - baseline_area) / baseline_area * 100
    else:
        total_change = 0.0

    peak_area = max(m["area_px"] for m in metrics)
    peak_tp = next(m for m in metrics if m["area_px"] == peak_area)

    lines.extend([
        "",
        "  RESUMEN",
        "  " + "-" * 60,
        f"  Área baseline:         {baseline_area} px",
        f"  Área final:            {final_area} px",
        f"  Cambio total:          {total_change:+.1f}%",
        f"  Pico tumoral:          {peak_area} px ({peak_tp['timepoint']})",
        f"  Evaluacion final:      {metrics[-1]['recist']}",
    ])

    if total_change < -30:
        lines.append("\n  >> El tumor muestra BUENA RESPUESTA al tratamiento.")
    elif total_change > 20:
        lines.append("\n  >> El tumor muestra PROGRESION. Reevaluar tratamiento.")
    else:
        lines.append("\n  >> El tumor se mantiene ESTABLE.")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    path.write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_longitudinal_analysis(
    timepoints: list[dict],
    output_dir: Path,
    patient_id: str = "paciente",
    method: str = "classical",
    show: bool = True,
) -> list[dict]:
    """Ejecuta el pipeline completo de análisis longitudinal.

    Args:
        timepoints: Lista de dicts con keys: date, image, mask, label
        output_dir: Directorio para guardar resultados.
        patient_id: Identificador del paciente.
        method: Método de segmentación ('classical' o 'nnunet').
        show: Mostrar plots interactivos.

    Returns:
        Lista de métricas longitudinales.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  ANÁLISIS LONGITUDINAL — {patient_id}")
    print(f"  Timepoints: {len(timepoints)}")
    print(f"  Método: {method}")
    print(f"{'=' * 65}")

    # Segmentación (si las máscaras no están precalculadas)
    for tp in timepoints:
        if tp["mask"] is None or np.sum(tp["mask"] > 0) == 0:
            print(f"\n  Segmentando {tp['label']}...")
            tp["mask"] = segment_classical(tp["image"])

    # Registro al baseline
    print("\n  Registrando timepoints al baseline...")
    timepoints = register_all_to_baseline(timepoints)

    # Métricas
    print("\n  Calculando métricas longitudinales...")
    metrics = compute_longitudinal_metrics(timepoints)

    # Guardar segmentaciones individuales
    tp_dir = output_dir / "timepoints"
    tp_dir.mkdir(exist_ok=True)
    for tp in timepoints:
        cv2.imwrite(str(tp_dir / f"{tp['label']}_imagen.png"), tp["image"])
        cv2.imwrite(str(tp_dir / f"{tp['label']}_mascara.png"), tp["mask"])
        overlay = _overlay_mask(tp["image"], tp["mask"])
        cv2.imwrite(
            str(tp_dir / f"{tp['label']}_overlay.png"),
            overlay,
        )

    # Guardar heatmaps de cambio individuales
    heatmap_dir = output_dir / "heatmaps_cambio"
    heatmap_dir.mkdir(exist_ok=True)
    for i in range(1, len(timepoints)):
        prev = timepoints[i - 1]
        curr = timepoints[i]
        new_px = new_tumor_regions(prev["mask"], curr["mask"])
        gone_px = disappeared_regions(prev["mask"], curr["mask"])
        stable = ((prev["mask"] > 0) & (curr["mask"] > 0)).astype(np.uint8) * 255

        heatmap = cv2.cvtColor(curr["image"], cv2.COLOR_GRAY2RGB)
        heatmap[new_px > 0] = CMAP_GROWTH
        heatmap[gone_px > 0] = CMAP_SHRINK
        heatmap[stable > 0] = CMAP_STABLE
        cv2.imwrite(
            str(heatmap_dir / f"cambio_{prev['label']}_a_{curr['label']}.png"),
            heatmap,
        )

    # Visualizaciones
    print("\n  Generando visualizaciones...")
    plot_timepoints_grid(timepoints, metrics,
                         output_dir / "comparacion_temporal.png")
    plot_volume_timeline(metrics,
                         output_dir / "timeline_volumen.png")
    plot_change_heatmaps(timepoints,
                         output_dir / "heatmaps_cambio_resumen.png")
    plot_metrics_dashboard(metrics,
                           output_dir / "dashboard_metricas.png")

    # CSV y reporte
    save_metrics_csv(output_dir / "metricas_longitudinales.csv", metrics)
    save_text_report(output_dir / "reporte.txt", metrics, patient_id)

    print(f"\n  Resultados guardados en: {output_dir}")
    print(f"  Archivos generados:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(output_dir)}")

    if show:
        plt.show()

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Análisis longitudinal de tumores en PET cerebral.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s --generate-demo
  %(prog)s carpeta_paciente/
  %(prog)s img_t0.png img_t1.png img_t2.png
  %(prog)s carpeta_nifti/ --method nnunet

Instalación de nnU-Net (opcional):
  pip install torch
  pip install git+https://github.com/MIC-DKFZ/nnUNet.git@nnunetv1
  nnUNet_download_pretrained_model Task169_BrainTumorPET
""",
    )
    p.add_argument(
        "paths", nargs="*", default=[],
        help="Ruta a directorio de paciente o lista de imágenes PET.",
    )
    p.add_argument(
        "--generate-demo", action="store_true",
        help="Genera datos sintéticos y ejecuta el pipeline de demostración.",
    )
    p.add_argument(
        "--method", choices=["classical", "nnunet"], default="classical",
        help="Método de segmentación (default: classical).",
    )
    p.add_argument(
        "--patient-id", default=None,
        help="Identificador del paciente (default: nombre del directorio).",
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="No abrir ventanas matplotlib.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(__file__).parent / "resultados_longitudinal"

    if args.generate_demo:
        print("\n  Generando datos de demostración...")
        demo_dir = out_root / "demo_datos"
        timepoints = generate_demo_data(demo_dir)

        run_longitudinal_analysis(
            timepoints,
            output_dir=out_root / "demo_resultados",
            patient_id="DEMO_001",
            method="classical",
            show=not args.no_show,
        )
        return 0

    if not args.paths:
        print("Error: especificar --generate-demo o ruta a imágenes/directorio.")
        print("Uso: python longitudinal_pet_analysis.py --generate-demo")
        print("     python longitudinal_pet_analysis.py carpeta_paciente/")
        return 1

    path = Path(args.paths[0])

    if path.is_dir():
        patient_id = args.patient_id or path.name
        timepoints = load_timepoints_from_dir(path)
    elif len(args.paths) >= 2:
        patient_id = args.patient_id or "paciente"
        timepoints = []
        for i, p_str in enumerate(args.paths):
            p = Path(p_str)
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  WARN: No se pudo leer {p}, omitiendo.")
                continue
            mask = segment_classical(img)
            date = _extract_date_from_name(p.stem, i)
            timepoints.append({
                "date": date,
                "image": img,
                "mask": mask,
                "label": p.stem,
                "path_img": p,
                "path_mask": None,
            })
        timepoints.sort(key=lambda t: t["date"])
    elif path.is_file():
        print(f"Se necesitan al menos 2 imágenes para análisis longitudinal.")
        print(f"Uso: python longitudinal_pet_analysis.py img_t0.png img_t1.png")
        print(f"     python longitudinal_pet_analysis.py --generate-demo")
        return 1
    else:
        print(f"No se encontró: {path}")
        return 1

    if len(timepoints) < 2:
        print("Se necesitan al menos 2 timepoints para análisis longitudinal.")
        return 1

    run_longitudinal_analysis(
        timepoints,
        output_dir=out_root / (args.patient_id or path.name),
        patient_id=patient_id,
        method=args.method,
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
