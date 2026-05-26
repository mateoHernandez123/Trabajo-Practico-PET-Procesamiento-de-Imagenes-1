# Trabajo Práctico — Detección y Caracterización de Tumores en Imágenes Médicas

**Materia:** Procesamiento de Imágenes I  
**Integrantes:** Mateo Hernandez, Felipe Lucero  
**Repositorio en GitHub:** [github.com/mateoHernandez123/Trabajo-Practico-PET-Morfologia](https://github.com/mateoHernandez123/Trabajo-Practico-PET-Morfologia)

El TP aborda el problema de **detección y caracterización de tumores en imágenes médicas** combinando dos enfoques complementarios:

| Pista | Modalidad | Técnica | Estado |
|-------|-----------|---------|--------|
| **1. PET de cuerpo completo** | Imagen 2D `.png` | Procesamiento clásico (Region Growing, K-Means, morfología, filtro por forma) | Implementado en `master` |
| **2. MRI cerebral multimodal** | Volumen 3D `.nii.gz` (4 modalidades: T1, T1ce, T2, FLAIR) | **Modelo pre-entrenado** [BraTS21](https://github.com/Alxaline/BraTS21) (U-Net 3D, ensamble de hasta 10 folds del paper MICCAI 2021) | Implementado en rama `feat/brats21-pretrained-integration` |

Las dos pistas atacan el mismo problema (segmentar tumor sobre tejido sano) en dominios distintos. La pista 1 muestra el dominio de las técnicas clásicas vistas en la materia; la pista 2 demuestra cómo, **siguiendo la recomendación de la cátedra**, se puede mejorar la precisión apoyándose en un modelo deep-learning ya entrenado por terceros, sin necesidad de entrenar nada propio.

## Cómo ejecutar — Pista 1 (PET clásico)

```bash
pip install -r requirements.txt
python3 segment_pet.py
```

Instrucciones detalladas (venv, Windows/Linux, Git Bash): [docs/Readme.md](docs/Readme.md).  
Respuestas y justificaciones de la consigna: [docs/doc.md](docs/doc.md).

La carpeta `resultados/` se genera al ejecutar el script. La imagen de entrada debe estar en `imagenes/pet_cuerpo_completo.png` (ver [docs/Readme.md](docs/Readme.md) para usar otra ruta).

## Cómo ejecutar — Pista 2 (Brain MRI con BraTS21)

```bash
git checkout feat/brats21-pretrained-integration

# 1. Entorno aislado
python -m venv .venv-brats21
.venv-brats21/Scripts/python.exe -m pip install -U pip wheel setuptools
.venv-brats21/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-brats21/Scripts/python.exe -m pip install -r requirements-brats21.txt

# 2. Descargar pesos pre-entrenados (~617 MB, 10 folds del paper)
.venv-brats21/Scripts/python.exe scripts/brats21_download_weights.py

# 3. Descargar un caso real del Medical Decathlon (streaming, ~11 MB)
.venv-brats21/Scripts/python.exe scripts/brats21_download_msd_case.py --case BRATS_001
.venv-brats21/Scripts/python.exe scripts/brats21_split_msd_case.py \
    --case BRATS_001 \
    --raw  data/msd_brats/_raw_msd \
    --out  data/msd_brats/BRATS_001

# 4. Correr inferencia con 1 fold (rápido, ~48 s CPU)
.venv-brats21/Scripts/python.exe scripts/brats21_run_inference.py \
    --config external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor/fold0_ns/config.yaml \
    --input  data/msd_brats/BRATS_001 \
    --output resultados_brats21/msd_brats_001_fold0 \
    --roi 96 96 96 --overlap 0.25 --cleaning-areas

# 5. Evaluar contra ground truth (Dice por sub-región)
.venv-brats21/Scripts/python.exe scripts/brats21_eval_dice.py \
    --pred resultados_brats21/msd_brats_001_fold0/BRATS_001_seg.nii.gz \
    --gt   data/msd_brats/BRATS_001/BRATS_001_seg.nii.gz

# 6. Visualizar GT vs predicción
.venv-brats21/Scripts/python.exe scripts/brats21_visualize_vs_gt.py \
    --case data/msd_brats/BRATS_001 \
    --pred resultados_brats21/msd_brats_001_fold0/BRATS_001_seg.nii.gz \
    --gt   data/msd_brats/BRATS_001/BRATS_001_seg.nii.gz \
    --out  resultados_brats21/msd_brats_001_fold0/preview_vs_gt.png
```

Documentación completa (datasets recomendados, formato de entrada, resultados medidos, limitaciones): [docs/brats21.md](docs/brats21.md).

---

# Pista 1 — Detección clásica en PET de cuerpo completo

## Imagen de entrada

Imagen PET de cuerpo completo utilizada como escena de interés. Las zonas oscuras representan alta actividad metabólica (hot spots).

<p align="center">
  <img src="imagenes/pet_cuerpo_completo.png" alt="Imagen PET de entrada — cuerpo completo" width="200">
</p>

**Uso en el código:** se carga en escala de grises desde `imagenes/pet_cuerpo_completo.png` y es la base del pipeline completo.

---

## Resultados visuales (qué muestra cada imagen y qué técnica justifica)

### 1. Bordes detectados (Canny)

<p align="center">
  <img src="resultados/region/edges.png" alt="Bordes detectados con Canny" width="200">
</p>

**Qué es:** bordes detectados con Canny (umbrales 40/120) sobre la imagen preprocesada, restringidos a la silueta del cuerpo.  
**Qué justifica:** visualizar los gradientes de intensidad presentes en la imagen; los bordes son más marcados en las zonas de transición entre tejido con captación y tejido normal.

### 2. Máscara binaria — Region Growing

<p align="center">
  <img src="resultados/region/mask_binary.png" alt="Máscara binaria — Region Growing" width="200">
</p>

**Qué es:** máscara binaria obtenida por umbralización por percentil 90 + crecimiento de regiones (BFS con tolerancia 25) + post-procesamiento morfológico (erosión + dilatación + filtro por forma).  
**Qué justifica:** solo quedan los tumores. El cerebro y otros órganos con captación fisiológica fueron eliminados por la combinación de erosión fuerte (2 iteraciones) y filtro por forma.

### 3. Máscara binaria — K-Means

<p align="center">
  <img src="resultados/kmeans/mask_binary.png" alt="Máscara binaria — K-Means" width="200">
</p>

**Qué es:** máscara binaria obtenida por K-Means (K=4 clusters) seleccionando el cluster más oscuro + post-procesamiento morfológico (erosión + dilatación + filtro por forma).  
**Qué justifica:** el filtro por forma descartó la enorme región del cerebro que K-Means capturaba, dejando solo las lesiones focales.

---

## Pipeline morfológico (erosión + dilatación + filtro por forma)

Tras la segmentación, se aplica un pipeline de morfología matemática con **operaciones explícitas** para aislar los tumores descartando captación fisiológica:

| Paso | Operación | Efecto |
|------|-----------|--------|
| 1 | **Erosión** (kernel 3×3, 2 iter) | Separa regiones débilmente conectadas, elimina ruido y blobs pequeños de captación fisiológica (ej. cerebro en Region Growing) |
| 2 | **Dilatación** (kernel 3×3, 3 iter) | Recupera bordes del tumor; la asimetría (3 iter vs 2) captura píxeles de borde con menor captación |
| 3 | **Cierre** (kernel 3×3, 1 iter) | Sella huecos internos residuales |
| 4 | **Filtro por área** (≥ 15 px) | Descarta artefactos pequeños |
| 5 | **Filtro por forma** | Descarta componentes con perfil de órgano (grandes + compactos + sólidos) |

### Filtro por forma — discriminación órgano vs tumor

Los órganos (cerebro, hígado) presentan captación fisiológica normal en PET. Para distinguirlos de tumores **sin depender de la posición**, se analizan métricas de forma:

| Métrica | Órganos | Tumores |
|---------|---------|---------|
| **Compacidad** (4πA/P²) | Alta (> 0.40): forma redondeada | Variable: bordes irregulares |
| **Solidez** (A/A_convex_hull) | Alta (> 0.65): contorno suave | Variable: más concavidades |
| **Área** | Grande (> 350 px) | Menor |

Un componente se descarta como órgano si cumple **todas** las condiciones. Esto es independiente de la posición: funciona sin importar dónde estén los tumores en el cuerpo.

### Pasos morfológicos — Region Growing

<p align="center">
  <img src="resultados/region/morfologia/raw.png" alt="Region Growing — máscara cruda" width="130">
  <img src="resultados/region/morfologia/eroded.png" alt="Region Growing — erosión" width="130">
  <img src="resultados/region/morfologia/dilated.png" alt="Region Growing — dilatación" width="130">
  <img src="resultados/region/morfologia/closed.png" alt="Region Growing — cierre" width="130">
  <img src="resultados/region/morfologia/area_filtered.png" alt="Region Growing — filtro área" width="130">
  <img src="resultados/region/morfologia/shape_filtered.png" alt="Region Growing — filtro forma" width="130">
</p>

De izquierda a derecha: máscara cruda → erosión (elimina cerebro chico) → dilatación (recupera bordes) → cierre → filtro por área → **filtro por forma** (descarta órganos).

### Pasos morfológicos — K-Means

<p align="center">
  <img src="resultados/kmeans/morfologia/raw.png" alt="K-Means — máscara cruda" width="130">
  <img src="resultados/kmeans/morfologia/eroded.png" alt="K-Means — erosión" width="130">
  <img src="resultados/kmeans/morfologia/dilated.png" alt="K-Means — dilatación" width="130">
  <img src="resultados/kmeans/morfologia/closed.png" alt="K-Means — cierre" width="130">
  <img src="resultados/kmeans/morfologia/area_filtered.png" alt="K-Means — filtro área" width="130">
  <img src="resultados/kmeans/morfologia/shape_filtered.png" alt="K-Means — filtro forma" width="130">
</p>

De izquierda a derecha: máscara cruda (cerebro enorme) → erosión → dilatación → cierre → filtro por área → **filtro por forma** (cerebro descartado, solo tumores).

---

### 4. Caracterización — Region Growing

<p align="center">
  <img src="resultados/region/characterization.png" alt="Caracterización Region Growing — bounding box, centroide e ID" width="200">
</p>

**Qué es:** imagen original con bounding boxes (verde), centroides (magenta) e IDs (azul) de cada tumor detectado por Region Growing.  
**Qué justifica:** los bounding boxes solo marcan lesiones en la zona de cadera y piernas. El cerebro no aparece porque fue descartado por el pipeline morfológico.

### 5. Caracterización — K-Means

<p align="center">
  <img src="resultados/kmeans/characterization.png" alt="Caracterización K-Means — bounding box, centroide e ID" width="200">
</p>

**Qué es:** imagen original con bounding boxes, centroides e IDs de cada tumor detectado por K-Means.  
**Qué justifica:** solo se marcan las lesiones focales. La captación fisiológica del cerebro fue eliminada.

### 6. Comparativa de métodos

<p align="center">
  <img src="resultados/comparison_filtered.png" alt="Comparativa Region Growing vs K-Means" width="700">
</p>

**Qué es:** panel comparativo que muestra el pipeline completo de ambos métodos side-by-side.  
**Qué justifica:** permite evaluar las diferencias entre Region Growing y K-Means en cuanto a la cantidad, tamaño y ubicación de las lesiones detectadas.

### 7. Recortes individuales — Region Growing

<p align="center">
  <img src="resultados/crops_region_filtered.png" alt="Galería de recortes — Region Growing" width="500">
</p>

**Qué es:** galería de recortes donde cada tumor aparece aislado sobre fondo blanco.  
**Qué justifica:** cada crop muestra exclusivamente los píxeles de la lesión, sin incluir cerebro ni otros órganos.

### 8. Recortes individuales — K-Means

<p align="center">
  <img src="resultados/crops_kmeans_filtered.png" alt="Galería de recortes — K-Means" width="500">
</p>

**Qué es:** galería de recortes de los tumores detectados por K-Means.  
**Qué justifica:** misma técnica de extracción, distinto método de segmentación.

---

## Features detectadas

### Region Growing

| ID  | Área | Perímetro | Centroide (x, y) | BBox (x, y, w, h)  | Ejes M/m      | Orient.° | Excent. | Compact. | I. media |
| --- | ---- | --------- | ---------------- | ------------------ | ------------- | -------- | ------- | -------- | -------- |
| 1   | 249  | 56.77     | (76.6, 158.9)    | (67, 150, 20, 18)  | 19.22 / 15.26 | 120.14   | 0.608   | 0.971    | 18.5     |
| 2   | 203  | 54.77     | (78.2, 179.6)    | (70, 170, 18, 19)  | 19.37 / 12.66 | 147.41   | 0.757   | 0.850    | 15.8     |
| 3   | 185  | 52.28     | (107.2, 185.7)   | (102, 175, 12, 22) | 20.61 / 10.23 | 165.32   | 0.868   | 0.850    | 21.1     |

### K-Means

| ID  | Área | Perímetro | Centroide (x, y) | BBox (x, y, w, h)  | Ejes M/m      | Orient.° | Excent. | Compact. | I. media |
| --- | ---- | --------- | ---------------- | ------------------ | ------------- | -------- | ------- | -------- | -------- |
| 1   | 256  | 64.53     | (107.1, 185.4)   | (100, 173, 15, 25) | 24.91 / 12.39 | 166.38   | 0.868   | 0.773    | 42.5     |
| 2   | 25   | 16.97     | (51.0, 195.0)    | (48, 192, 7, 7)    | 5.25 / 5.25   | 0.00     | 0.000   | 1.091    | 48.2     |
| 3   | 25   | 16.97     | (111.0, 203.0)   | (108, 200, 7, 7)   | 5.25 / 5.25   | 0.00     | 0.000   | 1.091    | 52.1     |

### Componentes descartados por filtro por forma

| Método | Área (px) | Compacidad | Solidez | Motivo |
|--------|-----------|------------|---------|--------|
| Region Growing | 438 | 0.584 | 0.902 | Cerebro (grande + compacto + sólido) |
| K-Means | 1296 | 0.755 | 0.976 | Cerebro (grande + compacto + sólido) |
| K-Means | 595 | 0.459 | 0.849 | Órgano (grande + compacto + sólido) |

---

# Pista 2 — Detección de tumor cerebral en MRI con modelo pre-entrenado

> Disponible en la rama `feat/brats21-pretrained-integration`.  
> Documentación detallada: [docs/brats21.md](docs/brats21.md).

## Motivación

La cátedra recomendó usar **modelos pre-entrenados** para mejorar la precisión de la detección de tumores cerebrales sin tener que entrenar uno propio (lo cual requeriría miles de imágenes anotadas, GPU y semanas de tiempo). El modelo [Alxaline/BraTS21](https://github.com/Alxaline/BraTS21) — solución del autor al desafío RSNA/ASNR/MICCAI Brain Tumor Segmentation 2021 — es exactamente esa herramienta: una U-Net 3D entrenada sobre 1.251 pacientes con tumor cerebral, con Dice promedio reportado de **0.88** sobre el set de validación oficial.

## Qué se integró

| Componente | Función |
|-----------|---------|
| `external/BraTS21/` | Repositorio upstream clonado completo, sin modificar |
| `external/BraTS21/checkpoints/` | 10 folds pre-entrenados del paper (5 con criterio Dice + 5 con criterio Jaccard), ~617 MB |
| `scripts/brats21_download_weights.py` | Descarga automática de los pesos desde Google Drive |
| `scripts/brats21_download_msd_case.py` | Descarga selectiva de un caso real de Medical Decathlon (streaming-tar, ~11 MB en lugar de los 7 GB del tar completo) |
| `scripts/brats21_split_msd_case.py` | Convierte el NIfTI 4D del Decathlon al formato BraTS estándar (4 archivos `_t1`/`_t1ce`/`_t2`/`_flair`) y remapea labels |
| `scripts/brats21_make_synthetic_case.py` | Genera un caso MRI sintético para validar el pipeline antes de usar datos reales |
| `scripts/brats21_make_smoke_fold.py` | Fold con pesos aleatorios para smoke test rápido sin descargar pesos |
| `scripts/brats21_run_inference.py` | Runner CPU end-to-end (carga config + pesos, sliding window 3D, post-procesamiento, guardado NIfTI). Soporta single fold o ensamble |
| `scripts/brats21_eval_dice.py` | Computa Dice score por sub-región (WT, TC, ET) entre predicción y ground truth |
| `scripts/brats21_visualize.py` | Genera PNG con la segmentación superpuesta sobre las 4 modalidades |
| `scripts/brats21_visualize_vs_gt.py` | Genera PNG comparativo ground truth vs predicción en 3 planos |
| `requirements-brats21.txt` | Dependencias modernas (PyTorch 2.12 CPU + MONAI 1.3, compatibles con Python 3.11) |
| `docs/brats21.md` | Documentación completa de la pista 2 |
| `docs/figuras/brats21/` | Figuras versionadas de resultados sobre el caso real BRATS_001 |

## Formato de entrada esperado

El runner espera una carpeta con las **4 modalidades MRI estándar de BraTS** en NIfTI:

```
<carpeta_del_caso>/
├── <case_id>_t1.nii.gz       # T1-weighted
├── <case_id>_t1ce.nii.gz     # T1 con contraste (gadolinio)
├── <case_id>_t2.nii.gz       # T2-weighted
└── <case_id>_flair.nii.gz    # FLAIR
```

Datasets públicos compatibles directamente con este formato (ver [docs/brats21.md](docs/brats21.md) para detalles):
- **Medical Decathlon Task01_BrainTumour** (recomendado para empezar)
- **Kaggle BraTS 2021**
- **Synapse BraTS oficial** (requiere registro)

## Salida que produce

| Archivo | Contenido |
|---------|-----------|
| `resultados_brats21/<corrida>/<case_id>_seg.nii.gz` | Segmentación con etiquetas BraTS estándar: 0=fondo, 1=NCR/NET (necrosis), 2=ED (edema), 4=ET (enhancing tumor) |
| `resultados_brats21/<corrida>/preview.png` | Visualización: 3 planos centrales × 4 modalidades, máscara superpuesta en color |
| `resultados_brats21/<corrida>/<case_id>_prob_{WT,TC,ET}.nii.gz` | (opcional, con `--save-probs`) Mapas de probabilidad por subregión |

Las tres subregiones segmentadas son las que define el challenge BraTS:
- **WT (Whole Tumor)** = todo lo que es tumor = unión de NCR + ED + ET
- **TC (Tumor Core)** = núcleo activo = NCR + ET
- **ET (Enhancing Tumor)** = sólo la parte que capta contraste

## Material clínico de referencia (no usado directamente por el modelo)

En `imagenes/clinicas_referencia/` quedan guardadas **14 imágenes clínicas reales** aportadas durante el desarrollo (CT torácicas y PET de cuerpo completo del centro IMAXE, Buenos Aires). **Estas imágenes no se procesan con BraTS21** porque son CT/PET 2D de tórax/cuerpo entero, mientras que BraTS21 requiere MRI 3D cerebral en 4 modalidades. Se conservan como referencia visual del tipo de estudios médicos reales que existen en producción, y eventualmente podrían usarse para extender la **Pista 1 (PET clásico)** con casos adicionales si se digitalizan a un formato compatible.

## Resultado real sobre un caso de Medical Decathlon

Se descargó el caso `BRATS_001` del [Medical Segmentation Decathlon Task01_BrainTumour](http://medicaldecathlon.com/) (subconjunto público de BraTS, descarga directa sin registro) y se corrió la inferencia con el fold0 pre-entrenado del paper sobre este equipo (Windows 11, sin GPU).

<p align="center">
  <img src="docs/figuras/brats21/msd_brats_001_fold0_vs_gt.png" alt="BraTS_001: ground truth vs predicción de BraTS21" width="800">
</p>

**Comparativa ground truth vs predicción del modelo** sobre FLAIR, en los 3 planos centrales del tumor (axial, coronal, sagital).

| Variante | Tiempo CPU | Dice WT | Dice TC | Dice ET | Dice mean |
|----------|-----------:|--------:|--------:|--------:|----------:|
| **1 fold** (ROI 96, overlap 0.25) | **48.6 s** | **0.9119** | 0.6288 | **0.8942** | **0.8117** |
| 1 fold (ROI 128, overlap 0.5 — receta del paper) | 76.4 s | 0.9036 | 0.6157 | 0.8921 | 0.8038 |
| **Ensamble 10 folds** (paper config) | 496.7 s | **0.9121** | 0.6273 | **0.8943** | 0.8112 |
| Paper (ensamble 10 folds + TTA, validación oficial BraTS 2021) | — | 0.925 | 0.876 | 0.840 | **0.881** |

Nuestro **WT** y **ET** están casi en lo reportado por el paper (diferencias < 0.02). El **TC** sale más bajo en este caso particular porque el modelo confunde parte del edema (color verde) con necrosis (color azul) — visible en la figura de arriba: el rojo (ET) coincide casi perfecto, pero el reparto entre verde y azul fuera del rojo está sesgado. Esto pasa también con el ensamble completo, indicando que es una característica del caso `BRATS_001`, no de un fold mal entrenado. Detalles y análisis completos en [docs/brats21.md sección 9](docs/brats21.md#92-caso-real-del-medical-decathlon--con-dice-score-contra-ground-truth).

## Comparativa de las dos pistas

| Criterio | Pista 1 (PET clásico) | Pista 2 (BraTS21 pre-entrenado) |
|----------|----------------------|--------------------------------|
| Técnica | Region Growing / K-Means + morfología + filtro por forma | U-Net 3D con 16.6M parámetros + ensamble multi-fold |
| Anotación necesaria | Ninguna (sin supervisión) | Ninguna (modelo ya entrenado en BraTS 2021) |
| Modalidad | PET 2D, 1 canal | MRI 3D, 4 canales |
| Tiempo de procesamiento (CPU) | < 5 s por imagen | 48 s (1 fold, ROI 96); 76 s (1 fold, ROI 128); ~8 min (ensamble 10 folds) |
| Métricas validables | Visual y por features (área, compacidad, etc.) | **Dice score vs ground truth** (medido: 0.91 WT, 0.89 ET) |
| Hiperparámetros sensibles | Percentil, tolerancia, kernels morfológicos | Threshold (0.5), ROI sliding-window, overlap |

---

## Estructura del proyecto

| Ruta | Contenido | Pista |
|------|-----------|------|
| `README.md` | Este archivo: resumen, figuras y estructura | — |
| `segment_pet.py` | Pipeline PET clásico: preprocesamiento, bordes, segmentación, morfología, filtro por forma, features, recortes | 1 |
| `requirements.txt` | Dependencias pista 1 (numpy, opencv-python, matplotlib) | 1 |
| `imagenes/` | Entrada pista 1; por defecto `pet_cuerpo_completo.png` | 1 |
| `resultados/` | PNG, CSV, recortes y pasos morfológicos de la pista 1 (generados al ejecutar) | 1 |
| `resultados/<m>/morfologia/` | Imágenes intermedias: erosión, dilatación, cierre, filtro área, filtro forma | 1 |
| `docs/Readme.md` | Instalación, entorno virtual y salidas | 1 |
| `docs/doc.md` | Informe / respuestas a la consigna; incluye sección sobre la extensión deep-learning | 1 + 2 |
| `external/BraTS21/` | Repo upstream pre-entrenado, clonado sin modificar | 2 |
| `scripts/brats21_*.py` | Scripts de descarga, generación de caso sintético, inferencia y visualización | 2 |
| `requirements-brats21.txt` | Dependencias pista 2 (torch 2.12 CPU, MONAI 1.3) | 2 |
| `docs/brats21.md` | Documentación completa de la pista 2 | 2 |
| `.venv-brats21/` | (ignorado) venv aislado para la pista 2 | 2 |
| `data/brats21_synth/` | (ignorado) caso sintético generado en pista 2 | 2 |
| `resultados_brats21/` | (ignorado) salidas NIfTI + PNG de la inferencia BraTS21 | 2 |
| `.gitignore` | Excluye venv/, pesos descargados, datos y resultados generados | — |

Parámetros útiles en código:
- **Pista 1** (`segment_pet.py`): `HOT_PERCENTILE`, `REGION_GROW_TOLERANCE`, `KMEANS_K`, `MIN_LESION_AREA`, `ERODE_KERNEL`, `ERODE_ITERATIONS`, `DILATE_KERNEL`, `DILATE_ITERATIONS`, `ORGAN_MIN_AREA`, `ORGAN_MIN_COMPACTNESS`, `ORGAN_MIN_SOLIDITY`, `MORPH_KERNEL`, `CANNY_LOW`/`CANNY_HIGH`, `CROP_PAD`.
- **Pista 2** (`scripts/brats21_run_inference.py`): `--roi` (tamaño del patch sliding window), `--overlap`, `--threshold`, `--cleaning-areas`, `--save-probs`.

---

## Clonar o actualizar desde GitHub

```bash
git clone git@github.com:mateoHernandez123/Trabajo-Practico-PET-Morfologia.git
cd Trabajo-Practico-PET-Morfologia

# Para trabajar en la pista 1 (PET clásico):
git checkout master

# Para trabajar en la pista 2 (Brain MRI con BraTS21):
git checkout feat/brats21-pretrained-integration
```
