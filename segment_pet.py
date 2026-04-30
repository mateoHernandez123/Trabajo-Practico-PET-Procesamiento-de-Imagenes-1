"""
Segmentación y caracterización de objetos en PET de cuerpo completo.

Cumple la consigna del trabajo:
    1. Pre-procesar la imagen.
    2. Obtener los bordes.
    3. Obtener bounding box de cada objeto.
    4. Obtener features (área, perímetro, centroide, ejes, orientación,
       excentricidad, compacidad, intensidad media).
    5. Generar máscara binaria con post-procesamiento morfológico
       (erosión + dilatación explícitas).
    6. Generar, a partir de la máscara, un recorte por objeto.

Ofrece DOS métodos de segmentación intercambiables (--method):
    - region   : Umbralización por percentil + Region Growing (BFS).
    - kmeans   : Clustering K-Means en intensidades (cluster más caliente).
    - both     : Ejecuta los dos y produce salidas comparativas.

Post-procesamiento morfológico:
    Tras la segmentación, se aplica un pipeline de morfología explícita:
        1. Erosión  — separa regiones débilmente conectadas y elimina ruido
           y blobs pequeños de captación fisiológica (ej. cerebro).
        2. Dilatación — recupera los bordes del tumor (más iteraciones que la
           erosión para capturar píxeles de borde con menor captación).
        3. Cierre — sella huecos internos residuales.
        4. Filtro por área — descarta componentes menores al umbral.
        5. Filtro por forma — descarta componentes con perfil de órgano
           (grandes + compactos + sólidos) usando compacidad y solidez.
    Se generan imágenes intermedias de cada paso en resultados/<método>/morfologia/.

Filtro anatómico opcional (--filter-anatomy):
    Descarta componentes con área excesiva o ubicados en la franja superior
    de la imagen (cabeza) o muy inferior (vejiga). Es una HEURÍSTICA, no
    una solución rigurosa: el método correcto requeriría un atlas anatómico
    o exclusión manual por ROI.

Uso:
    python3 segment_pet.py [imagen] [--method region|kmeans|both]
                           [--filter-anatomy] [--no-show]

Dependencias: numpy, opencv-python, matplotlib.
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


# Segmentación
HOT_PERCENTILE = 90.0          # Top 10% de intensidad dentro del cuerpo
REGION_GROW_TOLERANCE = 25     # Tolerancia en intensidad para crecer
KMEANS_K = 4                   # Número de clusters para K-Means
KMEANS_ATTEMPTS = 5            # Reinicios de K-Means (mejora estabilidad)
MIN_LESION_AREA = 15           # Área mínima (px) para conservar un componente
MORPH_KERNEL = 3               # Tamaño del kernel de morfología (cierre final)
BODY_BG_THRESHOLD = 240        # Píxeles > este valor se consideran fondo blanco
CANNY_LOW = 40                 # Umbral inferior de Canny
CANNY_HIGH = 120               # Umbral superior de Canny
CROP_PAD = 4                   # Margen alrededor de cada recorte

# Morfología explícita (erosión + dilatación)
ERODE_KERNEL = 3               # Tamaño del kernel de erosión
ERODE_ITERATIONS = 2           # Iteraciones de erosión (separa regiones, elimina ruido)
DILATE_KERNEL = 3              # Tamaño del kernel de dilatación
DILATE_ITERATIONS = 3          # Iteraciones de dilatación (recupera bordes del tumor)

# Filtro por forma (discriminación órgano vs tumor)
# Un componente se clasifica como órgano si cumple TODAS estas condiciones:
#   área > ORGAN_MIN_AREA  AND  compacidad > ORGAN_MIN_COMPACTNESS  AND  solidez > ORGAN_MIN_SOLIDITY
# Órganos (cerebro, hígado): grandes, redondeados, contorno suave
# Tumores: más pequeños y/o bordes irregulares
ORGAN_MIN_AREA = 350           # Área mínima (px) para evaluar si es órgano
ORGAN_MIN_COMPACTNESS = 0.40   # Compacidad mínima (4πA/P²) para perfil de órgano
ORGAN_MIN_SOLIDITY = 0.65      # Solidez mínima (área/convex_hull) para perfil de órgano

# Filtro anatómico (heurístico, opcional con --filter-anatomy)
EXCLUDE_TOP_FRACTION = 0.30    # Centroides en el 30% superior → cabeza
EXCLUDE_BOTTOM_FRACTION = 0.93 # Centroides en el 7% inferior → vejiga
MAX_OBJECT_AREA = 500          # Componentes > este área → órganos (no lesión)


# ---------------------------------------------------------------------------
# Pre-procesamiento y máscara del cuerpo
# ---------------------------------------------------------------------------

def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return img


def preprocess(gray: np.ndarray) -> np.ndarray:
    # Mediana elimina ruido sal-y-pimienta sin difuminar bordes.
    # Gaussiano leve suaviza el grano residual del detector.
    denoised = cv2.medianBlur(gray, 3)
    denoised = cv2.GaussianBlur(denoised, (3, 3), sigmaX=0.8)
    return denoised


def body_mask(gray: np.ndarray) -> np.ndarray:
    th = np.where(gray < BODY_BG_THRESHOLD, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    mask = np.zeros_like(th)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 500:
            mask[labels == i] = 255
    return mask


def detect_edges(denoised: np.ndarray, body: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(denoised, CANNY_LOW, CANNY_HIGH)
    edges[body == 0] = 0
    return edges


# ---------------------------------------------------------------------------
# Método A — Region Growing
# ---------------------------------------------------------------------------

def hot_candidates(inverted: np.ndarray, body: np.ndarray) -> tuple[np.ndarray, float]:
    values = inverted[body > 0]
    if values.size == 0:
        raise RuntimeError("Máscara de cuerpo vacía; revisa la imagen de entrada.")
    thr = float(np.percentile(values, HOT_PERCENTILE))
    candidates = ((inverted >= thr) & (body > 0)).astype(np.uint8) * 255
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
    body: np.ndarray,
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
        if visited[sy, sx] or body[sy, sx] == 0:
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
                if visited[ny, nx] or body[ny, nx] == 0:
                    continue
                if abs(int(image[ny, nx]) - seed_value) <= tolerance:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return result


def segment_region(
    inverted: np.ndarray,
    body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, np.ndarray]]:
    """Devuelve (mask_final, candidatos, region_growing_raw, threshold, morph_steps)."""
    candidates, thr = hot_candidates(inverted, body)
    seeds = seeds_from_candidates(candidates)
    grown = region_growing(inverted, seeds, body, REGION_GROW_TOLERANCE)
    final, morph_steps = postprocess(grown)
    return final, candidates, grown, thr, morph_steps


# ---------------------------------------------------------------------------
# Método B — K-Means
# ---------------------------------------------------------------------------

def kmeans_segmentation(
    gray: np.ndarray,
    body: np.ndarray,
    k: int = KMEANS_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica K-Means a las intensidades de los píxeles del cuerpo y devuelve
    (mask_hot, cluster_map, centers_sorted). El cluster "hot" es el de menor
    intensidad media (en PET hot=dark)."""
    ys, xs = np.where(body > 0)
    if ys.size < k:
        empty = np.zeros_like(gray)
        return empty, empty.astype(np.int32) - 1, np.array([])

    pixels = gray[ys, xs].reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels_flat, centers = cv2.kmeans(
        pixels, k, None, criteria, KMEANS_ATTEMPTS, cv2.KMEANS_PP_CENTERS,
    )
    labels_flat = labels_flat.flatten()
    centers_flat = centers.flatten()

    # Mapa 2D de clusters (-1 fuera del cuerpo, 0..k-1 dentro).
    cluster_map = np.full(gray.shape, -1, dtype=np.int32)
    cluster_map[ys, xs] = labels_flat

    # En PET "hot=dark" el cluster con MENOR media de intensidad es el más caliente.
    hot_cluster = int(np.argmin(centers_flat))
    mask_hot = np.where(cluster_map == hot_cluster, 255, 0).astype(np.uint8)

    # Devolvemos centros ordenados de oscuro a claro para visualización.
    order = np.argsort(centers_flat)
    return mask_hot, cluster_map, centers_flat[order]


def cluster_visual(cluster_map: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Imagen RGB que pinta cada cluster con un color distinto (qualitativo)."""
    palette = np.array([
        [255, 255, 255],   # fondo
        [220, 30, 30],     # cluster 0 (más caliente)
        [255, 140, 0],
        [50, 130, 255],
        [80, 200, 80],
        [180, 80, 220],
    ], dtype=np.uint8)

    rgb = np.full((*cluster_map.shape, 3), 255, dtype=np.uint8)
    rgb[body == 0] = (255, 255, 255)
    if cluster_map.max() < 0:
        return rgb

    # Reasignamos índices según media de intensidad creciente para que el
    # cluster 0 sea siempre el más oscuro/caliente.
    return _colourise_clusters(cluster_map, body, palette)


def _colourise_clusters(cluster_map, body, palette):
    rgb = np.full((*cluster_map.shape, 3), 255, dtype=np.uint8)
    unique = sorted({int(v) for v in np.unique(cluster_map) if v >= 0})
    for idx, lbl in enumerate(unique):
        colour = palette[(idx + 1) % len(palette)]
        rgb[cluster_map == lbl] = colour
    rgb[body == 0] = (255, 255, 255)
    return rgb


def segment_kmeans(
    gray: np.ndarray,
    body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Devuelve (mask_final, mask_raw, cluster_map, centers_sorted, morph_steps)."""
    raw, cluster_map, centers = kmeans_segmentation(gray, body, KMEANS_K)
    final, morph_steps = postprocess(raw)
    return final, raw, cluster_map, centers, morph_steps


# ---------------------------------------------------------------------------
# Post-procesamiento común
# ---------------------------------------------------------------------------

def shape_filter(
    mask: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Elimina componentes con perfil de órgano (grandes + compactos + sólidos).

    Los órganos (cerebro, hígado, riñones) en PET presentan captación
    fisiológica que NO es patológica.  Se distinguen de los tumores por:
        - Área grande (> ORGAN_MIN_AREA)
        - Alta compacidad (forma redondeada, 4πA/P² alto)
        - Alta solidez (contorno suave, pocos huecos → área ≈ convex hull)

    Los tumores, incluso los grandes, tienden a tener bordes más irregulares
    (menor compacidad y/o menor solidez) que los órganos sanos.

    Componentes con área ≤ ORGAN_MIN_AREA se conservan sin análisis de forma.
    """
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
    """Limpieza morfológica con erosión y dilatación explícitas + filtro por forma.

    Pipeline:
        1. Erosión — elimina conexiones espurias, ruido fino y blobs pequeños
           de captación fisiológica (p.ej. cerebro en Region Growing).
        2. Dilatación — recupera los bordes del tumor y rellena micro-huecos.
           Usar más iteraciones de dilatación que de erosión produce una
           expansión neta que captura píxeles de borde con menor captación.
        3. Cierre — sella huecos internos que persistan tras la dilatación.
        4. Filtro por área — descarta componentes menores a MIN_LESION_AREA.
        5. Filtro por forma — descarta componentes con perfil de órgano
           (grandes + compactos + sólidos), como el cerebro en K-Means.

    Retorna (máscara_final, diccionario_de_pasos_intermedios).
    """
    steps: dict[str, np.ndarray] = {"raw": mask.copy()}

    # 1. Erosión explícita
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (ERODE_KERNEL, ERODE_KERNEL))
    eroded = cv2.erode(mask, k_erode, iterations=ERODE_ITERATIONS)
    steps["eroded"] = eroded.copy()

    # 2. Dilatación explícita
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (DILATE_KERNEL, DILATE_KERNEL))
    dilated = cv2.dilate(eroded, k_dilate, iterations=DILATE_ITERATIONS)
    steps["dilated"] = dilated.copy()

    # 3. Cierre morfológico para sellar huecos internos restantes
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_KERNEL, MORPH_KERNEL))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, k_close, iterations=1)
    steps["closed"] = closed.copy()

    # 4. Filtro por área mínima
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    area_filtered = np.zeros_like(closed)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_LESION_AREA:
            area_filtered[labels == i] = 255
    steps["area_filtered"] = area_filtered.copy()

    # 5. Filtro por forma (descarta órganos: grandes + compactos + sólidos)
    shape_filtered, removed = shape_filter(area_filtered)
    steps["shape_filtered"] = shape_filtered.copy()

    if removed:
        print(f"  Filtro por forma: {len(removed)} componente(s) descartado(s) como órgano:")
        for r in removed:
            print(f"    - área={r['area']}px  compacidad={r['compactness']}  "
                  f"solidez={r['solidity']}")

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
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, closed=True))

        # cv2.fitEllipse requiere ≥ 5 puntos. Devuelve ejes COMPLETOS.
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


def anatomical_filter(
    features: list[dict],
    image_shape: tuple[int, int],
) -> tuple[list[dict], list[tuple[int, list[str]]]]:
    """Filtro heurístico para descartar captación fisiológica:
        - Componentes con área > MAX_OBJECT_AREA (cerebro, hígado, riñones).
        - Centroides en la franja superior (cabeza) o inferior (vejiga).
    Devuelve (features_aceptados, lista_excluidos_con_motivo).
    """
    h, _ = image_shape
    top_y = h * EXCLUDE_TOP_FRACTION
    bottom_y = h * EXCLUDE_BOTTOM_FRACTION

    accepted: list[dict] = []
    excluded: list[tuple[int, list[str]]] = []

    for f in features:
        cx, cy = f["centroid"]
        reasons: list[str] = []
        if f["area_px"] > MAX_OBJECT_AREA:
            reasons.append(f"area>{MAX_OBJECT_AREA}")
        if cy < top_y:
            reasons.append("franja_superior(cabeza)")
        if cy > bottom_y:
            reasons.append("franja_inferior(vejiga)")
        if reasons:
            excluded.append((f["id"], reasons))
        else:
            accepted.append(f)

    for new_id, f in enumerate(accepted, start=1):
        f["id"] = new_id
    return accepted, excluded


def draw_characterization(
    gray: np.ndarray,
    mask: np.ndarray,
    features: list[dict],
) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Dibujamos solo los contornos de los objetos ACEPTADOS para no confundir
    # con los que el filtro descartó.
    accepted_label_ids = {f["label_id"] for f in features}
    if accepted_label_ids:
        # Reconstruimos una máscara solo con componentes aceptados.
        num, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        accepted_mask = np.where(np.isin(labels, list(accepted_label_ids)),
                                 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(accepted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (0, 0, 255), 1)

    for f in features:
        x, y, w, h = f["bbox"]
        cx, cy = f["centroid"]
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 200, 0), 1)
        cv2.circle(rgb, (int(round(cx)), int(round(cy))), 2, (255, 0, 255), -1)
        cv2.putText(
            rgb, str(f["id"]),
            (x, max(8, y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA,
        )
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def extract_crops(
    gray: np.ndarray,
    labels: np.ndarray,
    features: list[dict],
    pad: int = CROP_PAD,
    apply_mask: bool = True,
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
        if apply_mask:
            roi_mask = (labels[y0:y1, x0:x1] == f["label_id"])
            roi = np.where(roi_mask, roi, 255).astype(np.uint8)
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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "edges.png"), edges)
    cv2.imwrite(str(out_dir / "mask_binary.png"), mask)
    cv2.imwrite(
        str(out_dir / "characterization.png"),
        cv2.cvtColor(characterization, cv2.COLOR_RGB2BGR),
    )
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    for f, crop in zip(features, crops):
        cv2.imwrite(str(crops_dir / f"object_{f['id']:02d}.png"), crop)
    save_features_csv(out_dir / "features.csv", features)
    if morph_steps is not None:
        save_morphology_steps(out_dir, morph_steps)


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def print_features_table(method: str, features: list[dict],
                         excluded: list[tuple[int, list[str]]] | None = None) -> None:
    print(f"\n=== Método: {method} ===")
    if not features:
        print("  No se detectaron objetos.")
        return
    header = (
        f"{'ID':>3} {'Área':>5} {'Perím.':>7} {'Centroide (x,y)':>18} "
        f"{'BBox (x,y,w,h)':>20} {'Ejes M/m':>14} {'Orient°':>8} "
        f"{'Excent.':>8} {'Compact.':>9} {'I.media':>8}"
    )
    print(header)
    print("-" * len(header))
    for f in features:
        cx, cy = f["centroid"]
        x, y, w, h = f["bbox"]
        print(
            f"{f['id']:>3} {f['area_px']:>5} {f['perimeter_px']:>7.1f} "
            f"({cx:>6.1f},{cy:>6.1f})    "
            f"({x:>3},{y:>3},{w:>3},{h:>3})  "
            f"{f['axis_major_px']:>5.1f}/{f['axis_minor_px']:<5.1f}  "
            f"{f['orientation_deg']:>7.1f} "
            f"{f['eccentricity']:>8.3f} {f['compactness']:>9.3f} "
            f"{f['mean_intensity']:>8.1f}"
        )
    if excluded:
        print(f"\n  Excluidos por filtro anatómico ({len(excluded)}):")
        for orig_id, reasons in excluded:
            print(f"    - obj original #{orig_id}: {', '.join(reasons)}")


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def show_pipeline_single(
    gray: np.ndarray,
    denoised: np.ndarray,
    body: np.ndarray,
    edges: np.ndarray,
    intermediate: np.ndarray,
    intermediate_title: str,
    mask: np.ndarray,
    characterization: np.ndarray,
    method_label: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    panels = [
        (gray,             "1. Original",                              "gray"),
        (denoised,         "2. Pre-procesado (mediana + gauss)",       "gray"),
        (body,             "3. Máscara del cuerpo",                    "gray"),
        (edges,            f"4. Bordes (Canny {CANNY_LOW}/{CANNY_HIGH})", "gray"),
        (intermediate,     f"5. {intermediate_title}",                 None),
        (mask,             f"6. Máscara binaria final\n({method_label})", "gray"),
        (characterization, "7. Caracterización (bbox + centroide + ID)", None),
        (None,             "8. Crops → out/.../crops/*.png",           None),
    ]
    for ax, (img, title, cmap) in zip(axes.flat, panels):
        if img is None:
            ax.text(0.5, 0.5, "ver carpeta out/",
                    ha="center", va="center", fontsize=11, color="gray")
        elif cmap == "gray":
            ax.imshow(img, cmap="gray")
        elif cmap is None and img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_methods_comparison(
    gray: np.ndarray,
    edges: np.ndarray,
    mask_region: np.ndarray, char_region: np.ndarray,
    mask_kmeans: np.ndarray, char_kmeans: np.ndarray,
    cluster_rgb: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    panels_top = [
        (gray,        "Original",                "gray"),
        (edges,       f"Bordes (Canny)",         "gray"),
        (cluster_rgb, f"K-Means (K={KMEANS_K}) — clusters", None),
        (mask_region, "Region Growing — máscara","gray"),
    ]
    panels_bottom = [
        (char_region,  "Region Growing — caracterización", None),
        (mask_kmeans,  "K-Means — máscara",                 "gray"),
        (char_kmeans,  "K-Means — caracterización",         None),
        (None,         "Crops en out/region/ y out/kmeans/", None),
    ]
    for ax, (img, title, cmap) in zip(axes[0], panels_top):
        if cmap == "gray":
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax, (img, title, cmap) in zip(axes[1], panels_bottom):
        if img is None:
            ax.text(0.5, 0.5, "ver carpetas out/",
                    ha="center", va="center", fontsize=11, color="gray")
        elif cmap == "gray":
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.suptitle("Comparativa Region Growing vs K-Means", fontsize=12)
    plt.tight_layout()
    plt.show()


def show_morphology_steps(
    steps: dict[str, np.ndarray],
    method_label: str,
) -> None:
    """Muestra los pasos intermedios de la morfología: raw → erosión → dilatación → cierre → filtro área → filtro forma."""
    titles = [
        ("raw",            "1. Máscara cruda\n(segmentación)"),
        ("eroded",         f"2. Erosión\n(kernel {ERODE_KERNEL}×{ERODE_KERNEL}, "
                           f"{ERODE_ITERATIONS} iter)"),
        ("dilated",        f"3. Dilatación\n(kernel {DILATE_KERNEL}×{DILATE_KERNEL}, "
                           f"{DILATE_ITERATIONS} iter)"),
        ("closed",         f"4. Cierre\n(kernel {MORPH_KERNEL}×{MORPH_KERNEL}, 1 iter)"),
        ("area_filtered",  f"5. Filtro por área\n(≥ {MIN_LESION_AREA} px)"),
        ("shape_filtered", "6. Filtro por forma\n(descarta órganos)"),
    ]
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))
    for ax, (key, label) in zip(axes, titles):
        ax.imshow(steps[key], cmap="gray")
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    plt.suptitle(f"Pipeline morfológico — {method_label}", fontsize=12)
    plt.tight_layout()
    plt.show()


def show_crops_gallery(crops: list[np.ndarray], features: list[dict], title: str) -> None:
    if not crops:
        return
    n = len(crops)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows))
    axes = np.atleast_2d(axes)
    for idx, (crop, f) in enumerate(zip(crops, features)):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"#{f['id']}  A={f['area_px']}px", fontsize=8)
        ax.axis("off")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def characterize_and_save(
    name: str,
    mask: np.ndarray,
    gray: np.ndarray,
    edges: np.ndarray,
    apply_filter: bool,
    out_root: Path,
    morph_steps: dict[str, np.ndarray] | None = None,
):
    features, labels = compute_features(mask, gray)
    excluded: list[tuple[int, list[str]]] = []
    if apply_filter:
        features, excluded = anatomical_filter(features, gray.shape)
    characterization = draw_characterization(gray, mask, features)
    crops = extract_crops(gray, labels, features)
    save_outputs(out_root / name, edges, mask, characterization, crops, features,
                 morph_steps)
    return features, labels, characterization, crops, excluded


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("path", nargs="?", default=None,
                   help="Ruta a la imagen PET (por defecto: 'Pasted image.png' junto al script)")
    p.add_argument("--method", choices=["region", "kmeans", "both"], default="both",
                   help="Método de segmentación")
    p.add_argument("--filter-anatomy", action="store_true",
                   help="Activa el filtro heurístico para descartar cabeza/órganos grandes")
    p.add_argument("--no-show", action="store_true",
                   help="No abre ventanas matplotlib (útil para batch)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    default_path = Path(__file__).parent / "imagenes" / "pet_cuerpo_completo.png"
    path = Path(args.path) if args.path else default_path

    gray = load_grayscale(path)
    denoised = preprocess(gray)
    inverted = cv2.bitwise_not(denoised)
    body = body_mask(denoised)
    edges = detect_edges(denoised, body)

    out_root = Path(__file__).parent / "resultados"
    print(f"Imagen: {path}   métodos: {args.method}   filtro_anatomia: {args.filter_anatomy}")

    region_pack = kmeans_pack = None

    if args.method in ("region", "both"):
        mask_r, candidates, grown, thr, morph_r = segment_region(inverted, body)
        feats_r, labels_r, char_r, crops_r, excl_r = characterize_and_save(
            "region", mask_r, gray, edges, args.filter_anatomy, out_root, morph_r,
        )
        print_features_table("Region Growing", feats_r, excl_r)
        region_pack = (mask_r, candidates, grown, thr, feats_r, char_r, crops_r, morph_r)

    if args.method in ("kmeans", "both"):
        mask_k, raw_k, cluster_map, centers, morph_k = segment_kmeans(gray, body)
        feats_k, labels_k, char_k, crops_k, excl_k = characterize_and_save(
            "kmeans", mask_k, gray, edges, args.filter_anatomy, out_root, morph_k,
        )
        cluster_rgb = cluster_visual(cluster_map, body)
        print(f"\nK-Means centros (intensidad media, ordenado): "
              f"{['%.1f' % c for c in centers]}")
        print_features_table("K-Means", feats_k, excl_k)
        kmeans_pack = (mask_k, raw_k, cluster_map, cluster_rgb, feats_k, char_k, crops_k, morph_k)

    print(f"\nSalidas escritas en: {out_root}/{{region,kmeans}}/")

    if args.no_show:
        return 0

    if args.method == "region":
        mask_r, candidates, grown, _, feats_r, char_r, crops_r, morph_r = region_pack
        show_pipeline_single(gray, denoised, body, edges,
                             grown, "Region Growing (raw)",
                             mask_r, char_r, "Region Growing")
        show_morphology_steps(morph_r, "Region Growing")
        show_crops_gallery(crops_r, feats_r, "Recortes — Region Growing")

    elif args.method == "kmeans":
        mask_k, raw_k, cluster_map, cluster_rgb, feats_k, char_k, crops_k, morph_k = kmeans_pack
        show_pipeline_single(gray, denoised, body, edges,
                             cluster_rgb, f"K-Means (K={KMEANS_K})",
                             mask_k, char_k, "K-Means")
        show_morphology_steps(morph_k, "K-Means")
        show_crops_gallery(crops_k, feats_k, "Recortes — K-Means")

    else:  # both
        mask_r, _, _, _, feats_r, char_r, crops_r, morph_r = region_pack
        mask_k, _, _, cluster_rgb, feats_k, char_k, crops_k, morph_k = kmeans_pack
        show_methods_comparison(gray, edges, mask_r, char_r,
                                mask_k, char_k, cluster_rgb)
        show_morphology_steps(morph_r, "Region Growing")
        show_morphology_steps(morph_k, "K-Means")
        show_crops_gallery(crops_r, feats_r, "Recortes — Region Growing")
        show_crops_gallery(crops_k, feats_k, "Recortes — K-Means")

    return 0


if __name__ == "__main__":
    sys.exit(main())
