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
│   ├── mask_binary.png        # Máscara binaria
│   ├── characterization.png   # BBox + centroide + ID
│   ├── features.csv           # Tabla de features
│   └── crops/                 # Recortes individuales
│       ├── object_01.png
│       └── ...
└── kmeans/
    ├── edges.png
    ├── mask_binary.png
    ├── characterization.png
    ├── features.csv
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

## Parámetros ajustables en código

Las constantes al inicio de `segment_pet.py` permiten calibrar el pipeline sin modificar la lógica:

| Constante                  | Valor    | Descripción                                |
| -------------------------- | -------- | ------------------------------------------ |
| `HOT_PERCENTILE`           | 90.0     | Percentil para candidatos (Region Growing) |
| `REGION_GROW_TOLERANCE`    | 25       | Tolerancia de intensidad para BFS          |
| `KMEANS_K`                 | 4        | Número de clusters                         |
| `MIN_LESION_AREA`          | 15       | Área mínima (px) para conservar componente |
| `CANNY_LOW` / `CANNY_HIGH` | 40 / 120 | Umbrales de Canny                          |
| `CROP_PAD`                 | 4        | Margen alrededor de cada recorte           |
| `MAX_OBJECT_AREA`          | 500      | Área máxima para filtro anatómico          |
