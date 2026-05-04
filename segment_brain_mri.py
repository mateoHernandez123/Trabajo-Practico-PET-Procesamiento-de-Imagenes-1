"""
Segmentación y detección de tumores cerebrales en imágenes MRI.

Aplica tres métodos de segmentación sobre el Brain Tumor MRI Dataset:
    1. K-Means — clustering directo en intensidades de píxel.
    2. SuperPixel (SLIC) + Clustering — agrupa superpíxeles por intensidad
       media para detectar regiones tumorales.
    3. Region Growing — crecimiento desde semillas en zonas de alta intensidad.

Pipeline por imagen:
    1. Preprocesamiento (CLAHE + suavizado).
    2. Extracción de máscara cerebral (Otsu + morfología).
    3. Detección de bordes (Canny).
    4. Segmentación (los tres métodos).
    5. Post-procesamiento morfológico + filtros.
    6. Caracterización (features geométricas).
    7. Visualización y guardado.

Uso:
    # Procesar una sola imagen:
    python3 segment_brain_mri.py dataset/Testing/glioma/Te-gl_0010.jpg

    # Procesar un directorio completo (batch):
    python3 segment_brain_mri.py dataset/Testing/glioma/ --batch --max-images 10

    # Procesar TODO el dataset de testing:
    python3 segment_brain_mri.py dataset/Testing/ --batch --max-images 20

    # Elegir método:
    python3 segment_brain_mri.py imagen.jpg --method superpixel

Dependencias: numpy, opencv-python, matplotlib, scikit-image.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic, mark_boundaries

# ---------------------------------------------------------------------------
# Constantes de segmentación
# ---------------------------------------------------------------------------

# SuperPixel SLIC
SLIC_N_SEGMENTS = 200
SLIC_COMPACTNESS = 20
SLIC_SIGMA = 1.5

# K-Means
KMEANS_K = 4
KMEANS_ATTEMPTS = 5

# Region Growing
HOT_PERCENTILE = 85.0
REGION_GROW_TOLERANCE = 20
MIN_LESION_AREA = 50

# Morfología
MORPH_KERNEL = 5
ERODE_KERNEL = 3
ERODE_ITERATIONS = 1
DILATE_KERNEL = 5
DILATE_ITERATIONS = 2

# Filtro por forma (descarta fondo/regiones no tumorales)
ORGAN_MIN_AREA = 2000
ORGAN_MIN_COMPACTNESS = 0.60
ORGAN_MIN_SOLIDITY = 0.85

# Canny
CANNY_LOW = 30
CANNY_HIGH = 100
CROP_PAD = 6

# Imagen de entrada
TARGET_SIZE = 256


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------

def load_and_prepare(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga una imagen MRI, la redimensiona y devuelve (gray, color_rgb)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return gray, rgb


def preprocess(gray: np.ndarray) -> np.ndarray:
    """CLAHE para mejorar contraste local + suavizado leve."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (5, 5), sigmaX=1.0)
    return denoised


def brain_mask(gray: np.ndarray) -> np.ndarray:
    """Máscara binaria del cerebro (elimina fondo negro)."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    mask = np.zeros_like(th)
    if num <= 1:
        return th
    largest = max(range(1, num), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    mask[labels == largest] = 255
    return mask


def detect_edges(denoised: np.ndarray, brain: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(denoised, CANNY_LOW, CANNY_HIGH)
    edges[brain == 0] = 0
    return edges


# ---------------------------------------------------------------------------
# Método 1 — K-Means sobre intensidades
# ---------------------------------------------------------------------------

def kmeans_segmentation(
    gray: np.ndarray,
    brain: np.ndarray,
    k: int = KMEANS_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-Means en intensidades. El cluster más brillante (tumor en MRI con
    contraste) se selecciona como candidato tumoral."""
    ys, xs = np.where(brain > 0)
    if ys.size < k:
        empty = np.zeros_like(gray)
        return empty, empty.astype(np.int32) - 1, np.array([])

    pixels = gray[ys, xs].reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, labels_flat, centers = cv2.kmeans(
        pixels, k, None, criteria, KMEANS_ATTEMPTS, cv2.KMEANS_PP_CENTERS,
    )
    labels_flat = labels_flat.flatten()
    centers_flat = centers.flatten()

    cluster_map = np.full(gray.shape, -1, dtype=np.int32)
    cluster_map[ys, xs] = labels_flat

    # En MRI con contraste, el tumor suele ser la región más BRILLANTE.
    hot_cluster = int(np.argmax(centers_flat))
    mask_hot = np.where(cluster_map == hot_cluster, 255, 0).astype(np.uint8)

    order = np.argsort(centers_flat)
    return mask_hot, cluster_map, centers_flat[order]


def cluster_visual(cluster_map: np.ndarray, brain: np.ndarray) -> np.ndarray:
    palette = np.array([
        [30, 30, 30],
        [220, 30, 30],
        [255, 160, 0],
        [50, 130, 255],
        [80, 200, 80],
        [180, 80, 220],
    ], dtype=np.uint8)
    rgb = np.zeros((*cluster_map.shape, 3), dtype=np.uint8)
    unique = sorted({int(v) for v in np.unique(cluster_map) if v >= 0})
    for idx, lbl in enumerate(unique):
        colour = palette[(idx + 1) % len(palette)]
        rgb[cluster_map == lbl] = colour
    rgb[brain == 0] = (0, 0, 0)
    return rgb


# ---------------------------------------------------------------------------
# Método 2 — SuperPixel (SLIC) + Clustering
# ---------------------------------------------------------------------------

def superpixel_segmentation(
    gray: np.ndarray,
    rgb: np.ndarray,
    brain: np.ndarray,
    n_segments: int = SLIC_N_SEGMENTS,
    compactness: float = SLIC_COMPACTNESS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SLIC superpíxeles → features por superpíxel → K-Means → tumor mask.

    Estrategia: se calculan features ricas por superpíxel (intensidad media,
    desviación, posición, gradiente medio) y se agrupan con K-Means en 6
    clusters.  Se seleccionan como tumorales los superpíxeles cuya intensidad
    media supera ampliamente la media cerebral, descartando los que pertenecen
    al fondo.

    Devuelve (mask_tumor, superpixel_labels, superpixel_boundaries_rgb).
    """
    segments = slic(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=SLIC_SIGMA,
        start_label=0,
        channel_axis=2,
    )

    unique_labels = np.unique(segments)
    n_sp = len(unique_labels)

    # Calcular gradiente para distinguir regiones con textura/bordes tumorales
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)

    h, w = gray.shape
    sp_mean_intensity = np.zeros(n_sp, dtype=np.float32)
    sp_features = np.zeros((n_sp, 5), dtype=np.float32)
    sp_brain_ratio = np.zeros(n_sp, dtype=np.float32)

    for i, lbl in enumerate(unique_labels):
        sp_mask = segments == lbl
        sp_pixels = gray[sp_mask]
        sp_mean_intensity[i] = sp_pixels.mean()
        sp_features[i, 0] = sp_pixels.mean()
        sp_features[i, 1] = sp_pixels.std()
        sp_features[i, 2] = gradient_mag[sp_mask].mean()
        ys, xs = np.where(sp_mask)
        sp_features[i, 3] = ys.mean() / h
        sp_features[i, 4] = xs.mean() / w
        sp_brain_ratio[i] = brain[sp_mask].sum() / (255.0 * sp_mask.sum()) if sp_mask.sum() > 0 else 0

    # Normalizar cada columna a [0, 1]
    feat_norm = sp_features.copy()
    for col in range(feat_norm.shape[1]):
        col_min = feat_norm[:, col].min()
        col_max = feat_norm[:, col].max()
        if col_max > col_min:
            feat_norm[:, col] = (feat_norm[:, col] - col_min) / (col_max - col_min)

    # Ponderar: intensidad tiene 3× peso para enfatizar regiones brillantes
    weights = np.array([3.0, 1.0, 1.0, 0.5, 0.5], dtype=np.float32)
    feat_weighted = feat_norm * weights[np.newaxis, :]

    # K-Means sobre features ponderados de superpíxeles (6 clusters)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.5)
    k = min(6, n_sp)
    _, sp_labels, sp_centers = cv2.kmeans(
        feat_weighted, k, None, criteria, 15, cv2.KMEANS_PP_CENTERS,
    )
    sp_labels = sp_labels.flatten()

    # Estadísticas por cluster
    brain_mean = gray[brain > 0].mean() if brain.sum() > 0 else 128.0
    brain_std = gray[brain > 0].std() if brain.sum() > 0 else 30.0

    # Seleccionar como tumoral: clusters cuya intensidad media de superpíxeles
    # supere (media_cerebral + 0.8 * std) Y que estén dentro del cerebro.
    intensity_threshold = brain_mean + 0.8 * brain_std
    tumor_mask = np.zeros_like(gray, dtype=np.uint8)

    for c in range(k):
        members = np.where(sp_labels == c)[0]
        if len(members) == 0:
            continue
        cluster_intensity = np.mean(sp_mean_intensity[members])
        cluster_brain = np.mean(sp_brain_ratio[members])

        if cluster_intensity >= intensity_threshold and cluster_brain > 0.5:
            for m in members:
                lbl = unique_labels[m]
                if sp_brain_ratio[m] > 0.3:
                    tumor_mask[segments == lbl] = 255

    tumor_mask[brain == 0] = 0

    boundaries_rgb = (mark_boundaries(rgb, segments, color=(1, 1, 0)) * 255).astype(np.uint8)

    return tumor_mask, segments, boundaries_rgb


# ---------------------------------------------------------------------------
# Método 3 — Region Growing
# ---------------------------------------------------------------------------

def hot_candidates_mri(
    enhanced: np.ndarray,
    brain: np.ndarray,
) -> tuple[np.ndarray, float]:
    """En MRI con contraste, los tumores son BRILLANTES (alto valor de gris)."""
    values = enhanced[brain > 0]
    if values.size == 0:
        raise RuntimeError("Máscara cerebral vacía.")
    thr = float(np.percentile(values, HOT_PERCENTILE))
    candidates = ((enhanced >= thr) & (brain > 0)).astype(np.uint8) * 255
    return candidates, thr


def seeds_from_candidates(candidates: np.ndarray) -> list[tuple[int, int]]:
    num, _, stats, centroids = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    seeds: list[tuple[int, int]] = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < MIN_LESION_AREA:
            continue
        cx, cy = centroids[i]
        seeds.append((int(round(cy)), int(round(cx))))
    return seeds


def region_growing(
    image: np.ndarray,
    seeds: list[tuple[int, int]],
    brain: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    h, w = image.shape
    visited = np.zeros_like(image, dtype=bool)
    result = np.zeros_like(image, dtype=np.uint8)
    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for sy, sx in seeds:
        if not (0 <= sy < h and 0 <= sx < w):
            continue
        if visited[sy, sx] or brain[sy, sx] == 0:
            continue

        seed_value = int(image[sy, sx])
        queue: deque[tuple[int, int]] = deque([(sy, sx)])
        visited[sy, sx] = True

        while queue:
            y, x = queue.popleft()
            result[y, x] = 255
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if visited[ny, nx] or brain[ny, nx] == 0:
                    continue
                if abs(int(image[ny, nx]) - seed_value) <= tolerance:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return result


def segment_region_mri(
    enhanced: np.ndarray,
    brain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, np.ndarray]]:
    candidates, thr = hot_candidates_mri(enhanced, brain)
    seeds = seeds_from_candidates(candidates)
    grown = region_growing(enhanced, seeds, brain, REGION_GROW_TOLERANCE)
    final, morph_steps = postprocess(grown)
    return final, candidates, grown, thr, morph_steps


# ---------------------------------------------------------------------------
# Post-procesamiento
# ---------------------------------------------------------------------------

def shape_filter(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Filtra componentes que parecen fondo/estructura normal (muy grandes,
    compactos y sólidos) en vez de tumor."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    removed: list[dict] = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_LESION_AREA:
            continue

        if area <= ORGAN_MIN_AREA:
            result[labels == i] = 255
            continue

        component = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, closed=True)
        compactness = (4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        is_organ = (compactness > ORGAN_MIN_COMPACTNESS
                    and solidity > ORGAN_MIN_SOLIDITY)

        if is_organ:
            removed.append({
                "label": i, "area": area,
                "compactness": round(compactness, 3),
                "solidity": round(solidity, 3),
            })
        else:
            result[labels == i] = 255

    return result, removed


def postprocess(mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    steps: dict[str, np.ndarray] = {"raw": mask.copy()}

    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (ERODE_KERNEL, ERODE_KERNEL))
    eroded = cv2.erode(mask, k_erode, iterations=ERODE_ITERATIONS)
    steps["eroded"] = eroded.copy()

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (DILATE_KERNEL, DILATE_KERNEL))
    dilated = cv2.dilate(eroded, k_dilate, iterations=DILATE_ITERATIONS)
    steps["dilated"] = dilated.copy()

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_KERNEL, MORPH_KERNEL))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, k_close, iterations=2)
    steps["closed"] = closed.copy()

    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    area_filtered = np.zeros_like(closed)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_LESION_AREA:
            area_filtered[labels == i] = 255
    steps["area_filtered"] = area_filtered.copy()

    shape_filtered, removed = shape_filter(area_filtered)
    steps["shape_filtered"] = shape_filtered.copy()

    if removed:
        print(f"  Filtro forma: {len(removed)} componente(s) descartado(s):")
        for r in removed:
            print(f"    - área={r['area']}px  comp={r['compactness']}  "
                  f"sol={r['solidity']}")

    return shape_filtered, steps


# ---------------------------------------------------------------------------
# Caracterización
# ---------------------------------------------------------------------------

def compute_features(
    mask: np.ndarray,
    gray: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    features: list[dict] = []
    next_id = 1

    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_LESION_AREA:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]

        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, closed=True))

        if len(contour) >= 5:
            (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
            axis_major = max(axis_a, axis_b)
            axis_minor = min(axis_a, axis_b)
            a, b = axis_major / 2, axis_minor / 2
            eccentricity = math.sqrt(1 - (b / a) ** 2) if a > 0 else 0.0
        else:
            axis_major = float(max(w, h))
            axis_minor = float(min(w, h))
            angle = 0.0
            eccentricity = 0.0

        compactness = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        mean_intensity = float(gray[labels == label].mean())

        features.append({
            "id": next_id,
            "label_id": label,
            "area_px": area,
            "perimeter_px": perimeter,
            "centroid": (float(cx), float(cy)),
            "bbox": (x, y, w, h),
            "axis_major_px": float(axis_major),
            "axis_minor_px": float(axis_minor),
            "orientation_deg": float(angle),
            "eccentricity": float(eccentricity),
            "compactness": float(compactness),
            "mean_intensity": mean_intensity,
        })
        next_id += 1

    return features, labels


def draw_characterization(
    gray: np.ndarray,
    mask: np.ndarray,
    features: list[dict],
) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    accepted_label_ids = {f["label_id"] for f in features}
    if accepted_label_ids:
        num, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        accepted_mask = np.where(np.isin(labels, list(accepted_label_ids)),
                                 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(accepted_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (0, 0, 255), 1)

    for f in features:
        x, y, w, h = f["bbox"]
        cx, cy = f["centroid"]
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 200, 0), 1)
        cv2.circle(rgb, (int(round(cx)), int(round(cy))), 2, (255, 0, 255), -1)
        cv2.putText(
            rgb, str(f["id"]),
            (x, max(8, y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA,
        )
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def extract_crops(
    gray: np.ndarray,
    labels: np.ndarray,
    features: list[dict],
    pad: int = CROP_PAD,
) -> list[np.ndarray]:
    h_img, w_img = gray.shape
    crops: list[np.ndarray] = []
    for f in features:
        x, y, w, h = f["bbox"]
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w_img, x + w + pad)
        y1 = min(h_img, y + h + pad)
        roi = gray[y0:y1, x0:x1].copy()
        roi_mask = (labels[y0:y1, x0:x1] == f["label_id"])
        roi = np.where(roi_mask, roi, 0).astype(np.uint8)
        crops.append(roi)
    return crops


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_features_csv(path: Path, features: list[dict]) -> None:
    headers = [
        "id", "area_px", "perimeter_px",
        "centroid_x", "centroid_y",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "axis_major_px", "axis_minor_px",
        "orientation_deg", "eccentricity", "compactness",
        "mean_intensity",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for f in features:
            cx, cy = f["centroid"]
            x, y, w, h = f["bbox"]
            writer.writerow([
                f["id"], f["area_px"], f"{f['perimeter_px']:.2f}",
                f"{cx:.1f}", f"{cy:.1f}",
                x, y, w, h,
                f"{f['axis_major_px']:.2f}", f"{f['axis_minor_px']:.2f}",
                f"{f['orientation_deg']:.2f}",
                f"{f['eccentricity']:.3f}", f"{f['compactness']:.3f}",
                f"{f['mean_intensity']:.1f}",
            ])


def save_morphology_steps(out_dir: Path, steps: dict[str, np.ndarray]) -> None:
    morph_dir = out_dir / "morfologia"
    morph_dir.mkdir(parents=True, exist_ok=True)
    for name, img in steps.items():
        cv2.imwrite(str(morph_dir / f"{name}.png"), img)


def save_outputs(
    out_dir: Path,
    edges: np.ndarray,
    mask: np.ndarray,
    characterization: np.ndarray,
    crops: list[np.ndarray],
    features: list[dict],
    morph_steps: dict[str, np.ndarray] | None = None,
    extra_images: dict[str, np.ndarray] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "edges.png"), edges)
    cv2.imwrite(str(out_dir / "mask_binary.png"), mask)
    cv2.imwrite(
        str(out_dir / "characterization.png"),
        cv2.cvtColor(characterization, cv2.COLOR_RGB2BGR),
    )
    if extra_images:
        for name, img in extra_images.items():
            if img.ndim == 3:
                cv2.imwrite(str(out_dir / f"{name}.png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(out_dir / f"{name}.png"), img)

    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    for f, crop in zip(features, crops):
        cv2.imwrite(str(crops_dir / f"tumor_{f['id']:02d}.png"), crop)
    save_features_csv(out_dir / "features.csv", features)
    if morph_steps is not None:
        save_morphology_steps(out_dir, morph_steps)


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def print_features_table(method: str, features: list[dict]) -> None:
    n = len(features)
    print(f"\n  === {method}: {n} tumor{'es' if n != 1 else ''} detectado{'s' if n != 1 else ''} ===")
    if not features:
        return
    header = (
        f"  {'ID':>3} {'Área':>5} {'Perím':>7} {'Centroide':>14} "
        f"{'BBox':>16} {'Ejes':>12} {'Excent':>7} {'Compact':>8} {'I.med':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for f in features:
        cx, cy = f["centroid"]
        x, y, w, h = f["bbox"]
        print(
            f"  {f['id']:>3} {f['area_px']:>5} {f['perimeter_px']:>7.1f} "
            f"({cx:>5.1f},{cy:>5.1f}) "
            f"({x:>3},{y:>3},{w:>3},{h:>3}) "
            f"{f['axis_major_px']:>5.1f}/{f['axis_minor_px']:<5.1f} "
            f"{f['eccentricity']:>7.3f} {f['compactness']:>8.3f} "
            f"{f['mean_intensity']:>6.1f}"
        )
    total_area = sum(f["area_px"] for f in features)
    print(f"  {'':>3} ───── Área total tumoral: {total_area} px")


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def show_full_pipeline(
    gray: np.ndarray,
    rgb: np.ndarray,
    enhanced: np.ndarray,
    brain: np.ndarray,
    edges: np.ndarray,
    results: dict[str, dict],
    image_name: str,
) -> None:
    """Grilla visual con todos los métodos lado a lado."""
    methods = list(results.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(3, max(4, n_methods + 1), figsize=(5 * max(4, n_methods + 1), 13))

    # Fila 1: preprocesamiento
    row0_panels = [
        (gray, "Original (gray)", "gray"),
        (enhanced, "CLAHE + Gaussiano", "gray"),
        (brain, "Máscara cerebral", "gray"),
        (edges, "Bordes (Canny)", "gray"),
    ]
    for ax, (img, title, cmap) in zip(axes[0], row0_panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[0][len(row0_panels):]:
        ax.axis("off")

    # Fila 2: máscaras por método
    for idx, method in enumerate(methods):
        data = results[method]
        ax = axes[1][idx]
        ax.imshow(data["mask"], cmap="gray")
        n_tumors = len(data["features"])
        ax.set_title(f"{method}\nmáscara ({n_tumors} tumor{'es' if n_tumors != 1 else ''})", fontsize=9)
        ax.axis("off")
    # Extra visual si hay superpixel
    if "superpixel" in results and "boundaries" in results["superpixel"]:
        ax = axes[1][n_methods]
        ax.imshow(results["superpixel"]["boundaries"])
        ax.set_title("SuperPixel SLIC\n(fronteras)", fontsize=9)
        ax.axis("off")
    for ax in axes[1][n_methods + 1:]:
        ax.axis("off")

    # Fila 3: caracterización
    for idx, method in enumerate(methods):
        data = results[method]
        ax = axes[2][idx]
        ax.imshow(data["characterization"])
        ax.set_title(f"{method}\ncaracterización", fontsize=9)
        ax.axis("off")
    if "kmeans" in results and "cluster_rgb" in results["kmeans"]:
        ax = axes[2][n_methods]
        ax.imshow(results["kmeans"]["cluster_rgb"])
        ax.set_title(f"K-Means clusters\n(K={KMEANS_K})", fontsize=9)
        ax.axis("off")
    for ax in axes[2][n_methods + 1:]:
        ax.axis("off")

    plt.suptitle(f"Brain Tumor MRI — {image_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def show_morphology_steps(
    steps: dict[str, np.ndarray],
    method_label: str,
) -> None:
    titles = [
        ("raw",            "1. Máscara cruda"),
        ("eroded",         f"2. Erosión ({ERODE_KERNEL}×{ERODE_KERNEL}, {ERODE_ITERATIONS}it)"),
        ("dilated",        f"3. Dilatación ({DILATE_KERNEL}×{DILATE_KERNEL}, {DILATE_ITERATIONS}it)"),
        ("closed",         f"4. Cierre ({MORPH_KERNEL}×{MORPH_KERNEL})"),
        ("area_filtered",  f"5. Filtro área (≥{MIN_LESION_AREA}px)"),
        ("shape_filtered", "6. Filtro forma"),
    ]
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    for ax, (key, label) in zip(axes, titles):
        ax.imshow(steps[key], cmap="gray")
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    plt.suptitle(f"Morfología — {method_label}", fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Pipeline por imagen
# ---------------------------------------------------------------------------

def process_single_image(
    path: Path,
    methods: list[str],
    out_root: Path,
    show: bool = True,
) -> dict[str, list[dict]]:
    """Procesa una imagen MRI con los métodos seleccionados.
    Devuelve dict con features por método."""

    image_name = path.stem
    category = path.parent.name
    print(f"\n{'='*60}")
    print(f"Imagen: {path.name}  |  Categoría: {category}")
    print(f"{'='*60}")

    gray, rgb = load_and_prepare(path)
    enhanced = preprocess(gray)
    brain = brain_mask(enhanced)
    edges = detect_edges(enhanced, brain)

    out_base = out_root / category / image_name
    all_results: dict[str, dict] = {}
    all_features: dict[str, list[dict]] = {}

    for method in methods:
        print(f"\n  [{method}]")

        if method == "kmeans":
            mask_raw, cluster_map, centers = kmeans_segmentation(enhanced, brain)
            mask_final, morph_steps = postprocess(mask_raw)
            features, labels = compute_features(mask_final, gray)
            characterization = draw_characterization(gray, mask_final, features)
            crops = extract_crops(gray, labels, features)
            c_rgb = cluster_visual(cluster_map, brain)

            save_outputs(
                out_base / method, edges, mask_final, characterization,
                crops, features, morph_steps,
                extra_images={"cluster_visual": c_rgb},
            )
            all_results[method] = {
                "mask": mask_final, "features": features,
                "characterization": characterization,
                "morph_steps": morph_steps,
                "cluster_rgb": c_rgb,
            }

        elif method == "superpixel":
            mask_raw, sp_labels, boundaries = superpixel_segmentation(
                enhanced, rgb, brain)
            mask_final, morph_steps = postprocess(mask_raw)
            features, labels = compute_features(mask_final, gray)
            characterization = draw_characterization(gray, mask_final, features)
            crops = extract_crops(gray, labels, features)

            save_outputs(
                out_base / method, edges, mask_final, characterization,
                crops, features, morph_steps,
                extra_images={"superpixel_boundaries": boundaries},
            )
            all_results[method] = {
                "mask": mask_final, "features": features,
                "characterization": characterization,
                "morph_steps": morph_steps,
                "boundaries": boundaries,
            }

        elif method == "region":
            mask_final, candidates, grown, thr, morph_steps = segment_region_mri(
                enhanced, brain)
            features, labels = compute_features(mask_final, gray)
            characterization = draw_characterization(gray, mask_final, features)
            crops = extract_crops(gray, labels, features)

            save_outputs(
                out_base / method, edges, mask_final, characterization,
                crops, features, morph_steps,
            )
            all_results[method] = {
                "mask": mask_final, "features": features,
                "characterization": characterization,
                "morph_steps": morph_steps,
            }

        print_features_table(method, features)
        all_features[method] = features

    # Resumen de detección por imagen
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │ RESUMEN DE DETECCIÓN — {image_name:<21s}│")
    print(f"  ├─────────────────────┬───────────────────────┤")
    print(f"  │ {'Método':<20s}│ {'Tumores detectados':>21s} │")
    print(f"  ├─────────────────────┼───────────────────────┤")
    for method in methods:
        n = len(all_features[method])
        print(f"  │ {method:<20s}│ {n:>21d} │")
    print(f"  └─────────────────────┴───────────────────────┘")

    if show:
        show_full_pipeline(gray, rgb, enhanced, brain, edges,
                           all_results, f"{category}/{image_name}")
        for method in methods:
            if "morph_steps" in all_results[method]:
                show_morphology_steps(all_results[method]["morph_steps"],
                                      f"{method} — {image_name}")

    return all_features


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def collect_images(root: Path, max_images: int = 0) -> list[Path]:
    """Recolecta imágenes .jpg/.png de un directorio (recursivo).
    Excluye la categoría 'notumor'."""
    extensions = {".jpg", ".jpeg", ".png"}
    images: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in extensions and "notumor" not in p.parts:
            images.append(p)
    if max_images > 0:
        images = images[:max_images]
    return images


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segmentación de tumores cerebrales en MRI con SuperPixel + Clustering.")
    p.add_argument("path",
                   help="Ruta a imagen o directorio de imágenes MRI.")
    p.add_argument("--method",
                   choices=["kmeans", "superpixel", "region", "all"],
                   default="all",
                   help="Método de segmentación (default: all)")
    p.add_argument("--batch", action="store_true",
                   help="Procesar todas las imágenes del directorio.")
    p.add_argument("--max-images", type=int, default=0,
                   help="Límite de imágenes en modo batch (0=sin límite).")
    p.add_argument("--no-show", action="store_true",
                   help="No mostrar plots (útil para batch).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.path)

    if args.method == "all":
        methods = ["kmeans", "superpixel", "region"]
    else:
        methods = [args.method]

    out_root = Path(__file__).parent / "resultados_mri"

    if args.batch or path.is_dir():
        images = collect_images(path, args.max_images)
        if not images:
            print(f"No se encontraron imágenes en {path}")
            return 1
        print(f"Procesando {len(images)} imágenes con métodos: {methods}")

        batch_results: list[dict] = []
        for img_path in images:
            try:
                feats = process_single_image(img_path, methods, out_root,
                                             show=not args.no_show)
                row = {
                    "image": img_path.name,
                    "category": img_path.parent.name,
                }
                for m in methods:
                    row[f"{m}_tumors"] = len(feats.get(m, []))
                    row[f"{m}_area_px"] = sum(
                        f["area_px"] for f in feats.get(m, []))
                batch_results.append(row)
            except Exception as e:
                print(f"  ERROR en {img_path.name}: {e}")

        print_batch_summary(batch_results, methods)
        save_batch_summary_csv(out_root / "resumen_batch.csv",
                               batch_results, methods)
    else:
        if not path.is_file():
            print(f"No se encontró: {path}")
            return 1
        process_single_image(path, methods, out_root, show=not args.no_show)

    print(f"\nResultados en: {out_root}/")
    return 0


def print_batch_summary(
    results: list[dict],
    methods: list[str],
) -> None:
    """Tabla resumen final del batch con conteo de tumores por imagen."""
    print(f"\n{'='*80}")
    print(f"  RESUMEN GLOBAL — {len(results)} imágenes procesadas")
    print(f"{'='*80}")

    # Tabla por imagen
    method_headers = "".join(f"│ {m:>10s} " for m in methods)
    print(f"\n  {'Imagen':<25s} {'Categoría':<14s}" + method_headers + "│")
    print("  " + "─" * (25 + 14 + len(methods) * 13 + 1))
    for r in results:
        cols = "".join(f"│ {r.get(f'{m}_tumors', 0):>10d} " for m in methods)
        print(f"  {r['image']:<25s} {r['category']:<14s}" + cols + "│")

    # Estadísticas agregadas por categoría
    categories = sorted({r["category"] for r in results})
    print(f"\n  {'─'*60}")
    print(f"  ESTADÍSTICAS POR CATEGORÍA")
    print(f"  {'─'*60}")
    for cat in categories:
        cat_rows = [r for r in results if r["category"] == cat]
        print(f"\n  [{cat}] ({len(cat_rows)} imágenes)")
        for m in methods:
            counts = [r.get(f"{m}_tumors", 0) for r in cat_rows]
            areas = [r.get(f"{m}_area_px", 0) for r in cat_rows]
            detected = sum(1 for c in counts if c > 0)
            print(f"    {m:<12s}: "
                  f"detección {detected}/{len(cat_rows)} "
                  f"({100*detected/len(cat_rows):.0f}%)  "
                  f"tumores/img={np.mean(counts):.1f}  "
                  f"área_media={np.mean(areas):.0f}px")

    # Totales
    print(f"\n  {'─'*60}")
    print(f"  TOTALES")
    print(f"  {'─'*60}")
    for m in methods:
        total_tumors = sum(r.get(f"{m}_tumors", 0) for r in results)
        detected = sum(1 for r in results if r.get(f"{m}_tumors", 0) > 0)
        print(f"    {m:<12s}: {total_tumors} tumores en {detected}/{len(results)} imágenes")


def save_batch_summary_csv(
    path: Path,
    results: list[dict],
    methods: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["image", "category"]
    for m in methods:
        headers.extend([f"{m}_tumors", f"{m}_area_px"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, 0) for k in headers})
    print(f"\n  Resumen CSV guardado en: {path}")


if __name__ == "__main__":
    sys.exit(main())
