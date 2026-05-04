# Instrucciones de instalación y ejecución

## Requisitos previos

- **Python 3.10** o superior
- **pip** (gestor de paquetes de Python)
- **Git** (para clonar el repositorio)

---

## Clonar el repositorio

```bash
git clone git@github.com:mateoHernandez123/Trabajo-Practico-PET-Procesamiento-de-Imagenes-1.git
cd Trabajo-Practico-PET-Procesamiento-de-Imagenes-1
```

---

## Crear entorno virtual (recomendado)

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (CMD)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Windows (Git Bash / PowerShell)

```bash
python -m venv .venv
source .venv/Scripts/activate
```

---

## Instalar dependencias

```bash
pip install -r requirements.txt
```

Paquetes utilizados:

| Paquete         | Uso                                                                           |
| --------------- | ----------------------------------------------------------------------------- |
| `numpy`         | Operaciones matriciales y cálculos numéricos                                  |
| `opencv-python` | Lectura de imágenes, filtros, morfología, Canny, K-Means, componentes conexas |
| `matplotlib`    | Visualización del pipeline y galería de recortes                              |
| `scikit-image`  | SuperPixel SLIC (segmentación MRI cerebral)                                   |

---

## Imagen de entrada

La imagen PET debe estar en `imagenes/pet_cuerpo_completo.png`. Para usar otra imagen:

```bash
python3 segment_pet.py ruta/a/mi_imagen.png
```

---

## Ejecutar el script

```bash
# Ambos métodos (por defecto)
python3 segment_pet.py

# Solo Region Growing
python3 segment_pet.py --method region

# Solo K-Means
python3 segment_pet.py --method kmeans

# Con filtro anatómico
python3 segment_pet.py --filter-anatomy

# Sin ventanas matplotlib (modo batch)
python3 segment_pet.py --no-show

# Combinación de flags
python3 segment_pet.py --method both --filter-anatomy --no-show
```

---

## Salidas generadas

Todas las salidas se guardan en `resultados/`:

```
resultados/
├── region/
│   ├── edges.png              # Bordes (Canny)
│   ├── mask_binary.png        # Máscara binaria final
│   ├── characterization.png   # BBox + centroide + ID
│   ├── features.csv           # Tabla de features
│   ├── morfologia/            # Pasos intermedios de morfología
│   │   ├── raw.png            # Máscara cruda (antes de morfología)
│   │   ├── eroded.png         # Después de erosión
│   │   ├── dilated.png        # Después de dilatación
│   │   ├── closed.png         # Después de cierre
│   │   ├── area_filtered.png  # Después de filtrado por área
│   │   └── shape_filtered.png # Después de filtro por forma (sin órganos)
│   └── crops/                 # Recortes individuales
│       ├── object_01.png
│       └── ...
└── kmeans/
    ├── edges.png
    ├── mask_binary.png
    ├── characterization.png
    ├── features.csv
    ├── morfologia/
    │   ├── raw.png
    │   ├── eroded.png
    │   ├── dilated.png
    │   ├── closed.png
    │   ├── area_filtered.png
    │   └── shape_filtered.png
    └── crops/
        ├── object_01.png
        └── ...
```

---

## Argumentos del CLI

| Argumento          | Tipo                           | Default                            | Descripción                        |
| ------------------ | ------------------------------ | ---------------------------------- | ---------------------------------- |
| `path`             | posicional (opcional)          | `imagenes/pet_cuerpo_completo.png` | Ruta a la imagen PET               |
| `--method`         | `region` \| `kmeans` \| `both` | `both`                             | Método de segmentación             |
| `--filter-anatomy` | flag                           | desactivado                        | Activa filtro heurístico anatómico |
| `--no-show`        | flag                           | desactivado                        | No abre ventanas de matplotlib     |

---

---

## Descargar el dataset de MRI cerebral

El script `segment_brain_mri.py` requiere el [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset):

```bash
mkdir -p dataset
wget -O dataset/brain-tumor-mri-dataset.zip \
  "https://zenodo.org/records/12735702/files/brain-tumor-mri-dataset.zip?download=1"
cd dataset && unzip brain-tumor-mri-dataset.zip && cd ..
```

Estructura resultante:

```
dataset/
├── Training/
│   ├── glioma/      (1,321 imágenes)
│   ├── meningioma/  (1,339 imágenes)
│   ├── notumor/     (1,595 imágenes)
│   └── pituitary/   (1,457 imágenes)
└── Testing/
    ├── glioma/      (300 imágenes)
    ├── meningioma/  (306 imágenes)
    ├── notumor/     (405 imágenes)
    └── pituitary/   (300 imágenes)
```

---

## Ejecutar el script de MRI cerebral

```bash
# Una imagen, todos los métodos (K-Means + SuperPixel + Region Growing):
python3 segment_brain_mri.py dataset/Testing/glioma/Te-gl_0010.jpg

# Solo SuperPixel:
python3 segment_brain_mri.py dataset/Testing/glioma/Te-gl_0010.jpg --method superpixel

# Batch (10 imágenes de meningioma, sin plots):
python3 segment_brain_mri.py dataset/Testing/meningioma/ --batch --max-images 10 --no-show

# Todo el testing:
python3 segment_brain_mri.py dataset/Testing/ --batch --no-show
```

### Argumentos del CLI — MRI cerebral

| Argumento       | Tipo                                         | Default | Descripción                             |
| --------------- | -------------------------------------------- | ------- | --------------------------------------- |
| `path`          | posicional                                   | —       | Ruta a imagen o directorio              |
| `--method`      | `kmeans` \| `superpixel` \| `region` \| `all`| `all`   | Método de segmentación                  |
| `--batch`       | flag                                         | off     | Procesar todas las imágenes del dir     |
| `--max-images`  | entero                                       | 0       | Límite en modo batch (0 = sin límite)   |
| `--no-show`     | flag                                         | off     | No abre ventanas de matplotlib          |

### Salidas generadas — MRI cerebral

Se guardan en `resultados_mri/<categoría>/<imagen>/<método>/`:

- `edges.png` — bordes (Canny)
- `mask_binary.png` — máscara binaria final
- `characterization.png` — BBox + centroide + ID de cada tumor
- `features.csv` — features geométricas de cada tumor
- `morfologia/` — pasos intermedios (raw, eroded, dilated, closed, area_filtered, shape_filtered)
- `crops/` — recorte individual por tumor detectado
- `cluster_visual.png` (K-Means) — clusters coloreados
- `superpixel_boundaries.png` (SuperPixel) — fronteras SLIC

En modo batch se genera además `resultados_mri/resumen_batch.csv` con el conteo de tumores por imagen y método.

---

## Parámetros ajustables en código

Las constantes al inicio de `segment_pet.py` permiten calibrar el pipeline sin modificar la lógica:

| Constante                  | Valor    | Descripción                                |
| -------------------------- | -------- | ------------------------------------------ |
| `HOT_PERCENTILE`           | 90.0     | Percentil para candidatos (Region Growing) |
| `REGION_GROW_TOLERANCE`    | 25       | Tolerancia de intensidad para BFS          |
| `KMEANS_K`                 | 4        | Número de clusters                         |
| `MIN_LESION_AREA`          | 15       | Área mínima (px) para conservar componente |
| `MORPH_KERNEL`             | 3        | Tamaño del kernel de cierre final          |
| `ERODE_KERNEL`             | 3        | Tamaño del kernel de erosión               |
| `ERODE_ITERATIONS`         | 2        | Iteraciones de erosión                     |
| `DILATE_KERNEL`            | 3        | Tamaño del kernel de dilatación            |
| `DILATE_ITERATIONS`        | 3        | Iteraciones de dilatación                  |
| `ORGAN_MIN_AREA`           | 350      | Área mínima para evaluar si es órgano      |
| `ORGAN_MIN_COMPACTNESS`    | 0.40     | Compacidad mínima para perfil de órgano    |
| `ORGAN_MIN_SOLIDITY`       | 0.65     | Solidez mínima para perfil de órgano       |
| `CANNY_LOW` / `CANNY_HIGH` | 40 / 120 | Umbrales de Canny                          |
| `CROP_PAD`                 | 4        | Margen alrededor de cada recorte           |
| `MAX_OBJECT_AREA`          | 500      | Área máxima para filtro anatómico          |
