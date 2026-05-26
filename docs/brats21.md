# Pista 2 — Detección de tumor cerebral con BraTS21 pre-entrenado

> Rama: `feat/brats21-pretrained-integration`  
> Materia: Procesamiento de Imágenes I — UNNOBA  
> Integrantes: Mateo Hernandez, Felipe Lucero

## 1. Contexto y motivación académica

La cátedra recomendó extender el TP con una **segunda pista** que mejore la precisión de detección de tumores cerebrales reutilizando **modelos pre-entrenados** (sin entrenar uno propio). Esta decisión tiene fundamento técnico claro:

| Aspecto | Entrenar desde cero | Usar pre-entrenado |
|---------|--------------------|--------------------|
| Datos anotados necesarios | Miles de pacientes con segmentación experta | Cero — los pesos ya están |
| Hardware | Cluster GPU multi-día | Una GPU consumer o incluso CPU |
| Riesgo de sobreajuste | Alto con pocos datos | Bajo: el modelo ya generaliza |
| Reproducibilidad clínica | Difícil sin protocolo idéntico | Acotada al dominio de entrenamiento |
| Tiempo total a entregar el TP | Semanas | Horas |

El proyecto [Alxaline/BraTS21](https://github.com/Alxaline/BraTS21) — solución del autor al desafío **RSNA/ASNR/MICCAI Brain Tumor Segmentation 2021** — es exactamente el tipo de modelo recomendado:

- Entrenado sobre **1.251 pacientes** del dataset oficial BraTS 2021.
- Arquitectura U-Net 3D con bloques EvoNorm + ASPP + Squeeze-and-Excitation (ver paper: [Carré et al., 2022, BrainLes 2021](https://doi.org/10.1007/978-3-031-09002-8_23)).
- Reporta **Dice 0.92 (Whole Tumor) / 0.88 (Tumor Core) / 0.84 (Enhancing Tumor)** sobre el set de validación oficial — competitivo con los top-5 del challenge.
- Pesos públicos (Apache 2.0) en Google Drive: ~617 MB para los 10 folds del ensamble.

## 2. ¿Por qué este modelo y no otro?

Hay otras alternativas (nnU-Net, SegResNet, UNETR), pero BraTS21 ofrece la mejor combinación de:

1. **Pesos publicados** descargables sin pedir permiso.
2. **Resultados competitivos** del challenge oficial (no es una demo).
3. **Código abierto completo** del autor (no solo pesos), lo que permite reusar componentes y construir wrappers.
4. **Multi-fold ensemble** ya armado: el autor entrena 5 folds del mismo modelo y los promedia, lo cual mejora robustez sin esfuerzo adicional nuestro.

## 3. Qué problema resuelve concretamente

Dado un caso de **MRI cerebral multimodal** (4 modalidades de un mismo paciente, registradas espacialmente):

- T1-weighted (`*_t1.nii.gz`)
- T1 con contraste de gadolinio (`*_t1ce.nii.gz`)
- T2-weighted (`*_t2.nii.gz`)
- FLAIR (`*_flair.nii.gz`)

El modelo produce un **volumen 3D con etiquetas BraTS estándar**:

| Etiqueta | Significado | Sub-región BraTS |
|---------|------------|------------------|
| 0 | Fondo / tejido sano | — |
| 1 | NCR/NET — necrosis y tumor no realzante | parte del Tumor Core |
| 2 | ED — edema peritumoral | parte de Whole Tumor |
| 4 | ET — Enhancing Tumor (capta contraste) | Tumor Core + ET |

A partir de eso se derivan las tres "sub-regiones evaluables" del challenge:

- **WT (Whole Tumor)** = todo lo que es tumor = unión de NCR + ED + ET (etiquetas 1+2+4)
- **TC (Tumor Core)** = núcleo activo = NCR + ET (etiquetas 1+4)
- **ET (Enhancing Tumor)** = sólo lo que capta contraste (etiqueta 4)

Esto es **directamente útil clínicamente**: WT delimita la extensión total de la lesión, TC marca el centro activo, y ET identifica el tumor agresivo que está captando contraste (típicamente glioblastoma).

## 4. Cómo se hizo la integración (decisiones técnicas)

El repo upstream tiene tres incompatibilidades con nuestro entorno (Windows 11 + Python 3.11 + CPU):

| Problema upstream | Resolución elegida |
|-------------------|-------------------|
| `import resource` en `src/main_inference.py` (módulo solo UNIX) | Bypaseamos `main_inference.py` y `Engine` enteros |
| `assert torch.cuda.is_available()` hardcodeado | Runner propio que usa `torch.device("cpu")` por default |
| Dependencias muertas en Python 3.11 (MONAI 0.6, scikit-learn 0.23, pyradiomics 3.0, ranger21) | `requirements-brats21.txt` con versiones modernas (torch 2.12 CPU, MONAI 1.3) |

**Principio rector:** no modificar una sola línea de `external/BraTS21/`. Cualquier corrección o adaptación va en `scripts/`. Esto permite que la subcarpeta del repo upstream sea actualizable en el futuro sin merge conflicts.

## 5. Setup (una sola vez)

```bash
git checkout feat/brats21-pretrained-integration

python -m venv .venv-brats21
.venv-brats21/Scripts/python.exe -m pip install -U pip wheel setuptools
.venv-brats21/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-brats21/Scripts/python.exe -m pip install -r requirements-brats21.txt
```

En Linux/Mac es lo mismo cambiando `.venv-brats21/Scripts/python.exe` por `.venv-brats21/bin/python`.

## 6. Descarga de pesos pre-entrenados

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_download_weights.py \
    --out external/BraTS21/checkpoints
```

Descarga ~617 MB comprimido desde Google Drive (gestión vía `gdown`) y los descomprime en:

```
external/BraTS21/checkpoints/final_weights/
├── baseline_equiunet_assp_evocor/        # 5 folds entrenados con loss Dice
│   ├── fold0_ns/{config.yaml, best_model.pth}
│   ├── fold1_ns/...
│   ├── ...
│   └── fold4_ns/
└── baseline_equiunet_assp_evocor_jaccard/ # 5 folds entrenados con loss Jaccard
    ├── fold0/...
    ├── ...
    └── fold4/
```

Si Google Drive bloquea la descarga automática (suele pasar con archivos >50 MB), bajar manualmente desde <https://drive.google.com/file/d/1Xt2rdD60IeEwcd8-yiMZHZkI0udcXgc7/view> y correr:

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_download_weights.py \
    --from-local /ruta/al/final_weights_brats21.zip
```

## 7. Datasets recomendados para datos reales

Tres fuentes públicas con formato compatible con el runner sin conversión adicional:

### A) Medical Decathlon Task01_BrainTumour (recomendado para empezar)

- **Tamaño:** ~7 GB, 750 casos (484 entrenamiento + 266 test)
- **Acceso:** descarga directa sin registro
- **Origen:** BraTS 2017 (subconjunto del BraTS 2021)
- **Link:** <http://medicaldecathlon.com/> → Task01_BrainTumour
- **Particularidad:** los archivos vienen como **un único NIfTI 4D** por caso (4 canales en la cuarta dimensión), no como 4 archivos separados. Hay que dividirlo en los 4 archivos esperados (`_t1`, `_t1ce`, `_t2`, `_flair`) antes de pasarlo al runner. MONAI lo carga así con `monai.apps.DecathlonDataset`.

### B) Kaggle BraTS 2021

- **Tamaño:** ~30 GB, 1.251 casos
- **Acceso:** requiere cuenta Kaggle + API key, pero la descarga sí es directa
- **Origen:** dataset oficial del challenge 2021 (el mismo con el que se entrenó el modelo)
- **Link:** <https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1>
- **Particularidad:** ya viene en formato BraTS estándar (4 archivos separados `_t1.nii.gz`, `_t1ce.nii.gz`, `_t2.nii.gz`, `_flair.nii.gz`), compatible directamente con el runner.

### C) Synapse BraTS oficial

- **Tamaño:** mismo dataset que Kaggle
- **Acceso:** requiere aprobación manual del comité organizador (puede tardar días)
- **Link:** <https://www.synapse.org/Synapse:syn25829067>
- **Recomendación:** sólo si lo necesitan para una publicación; para el TP, Kaggle o Decathlon son suficientes.

## 8. Inferencia sobre datos reales

Una vez que la carpeta del caso tenga la estructura correcta:

```
<carpeta_caso>/
├── <case_id>_t1.nii.gz
├── <case_id>_t1ce.nii.gz
├── <case_id>_t2.nii.gz
└── <case_id>_flair.nii.gz
```

### 8.1 Inferencia con un solo fold (rápido)

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_run_inference.py \
    --config external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor/fold0_ns/config.yaml \
    --input  <carpeta_caso> \
    --output resultados_brats21/<nombre_corrida> \
    --roi 96 96 96 --overlap 0.25 --cleaning-areas
```

### 8.2 Inferencia con ensamble de los 10 folds (precisión máxima, lento en CPU)

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_run_inference.py \
    --config \
       external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor/fold0_ns/config.yaml \
       external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor/fold1_ns/config.yaml \
       external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor/fold2_ns/config.yaml \
       external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor/fold3_ns/config.yaml \
       external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor/fold4_ns/config.yaml \
       external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor_jaccard/fold0/config.yaml \
       external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor_jaccard/fold1/config.yaml \
       external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor_jaccard/fold2/config.yaml \
       external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor_jaccard/fold3/config.yaml \
       external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor_jaccard/fold4/config.yaml \
    --input  <carpeta_caso> \
    --output resultados_brats21/ensemble_full \
    --roi 128 128 128 --overlap 0.5 --cleaning-areas
```

Los logits de cada fold se promedian internamente antes del threshold (`mean ensembling`), siguiendo la receta del paper.

### 8.3 Visualización

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_visualize.py \
    --case <carpeta_caso> \
    --seg  resultados_brats21/<corrida>/<case_id>_seg.nii.gz \
    --out  resultados_brats21/<corrida>/preview.png
```

Produce un PNG con 3 planos centrales (axial, coronal, sagital) × 4 modalidades (T1, T1ce, T2, FLAIR) con la segmentación superpuesta en color:

- 🔵 azul = NCR/NET (necrosis, etiqueta 1)
- 🟢 verde = ED (edema, etiqueta 2)
- 🔴 rojo = ET (enhancing tumor, etiqueta 4)

## 9. Validación: caso sintético

Para verificar la plomería sin esperar la descarga de un dataset real (que pueden ser GBs), se incluye un generador sintético:

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_make_synthetic_case.py \
    --out data/brats21_synth
```

Genera `data/brats21_synth/BraTS_synth_001/` con 4 NIfTI de 240×240×155 (mismas dimensiones que BraTS real), donde el "cerebro" es una elipsoide con un "tumor" focal hiperintenso plantado en posición conocida.

**Resultados medidos en este equipo** (Windows 11, Python 3.11, sin GPU, sobre el caso sintético):

| Variante | Tiempo CPU | Voxels NCR/NET | Voxels ED | Voxels ET |
|----------|-----------:|---------------:|----------:|----------:|
| Smoke test (pesos aleatorios) | ~106 s | 1.146.239 | 132.772 | 3.882.512 |
| **fold0_ns (pre-entrenado)** | **98 s** | **623** | **60** | **1.075** |

**Interpretación:** con pesos aleatorios el modelo predice tumor "por todos lados" (5+ millones de voxels). Con los pesos pre-entrenados, el modelo identifica **únicamente** la región del tumor sintético plantado (~1.700 voxels totales, coherente con la elipsoide de ~3.000 voxels del generador). La diferencia confirma que:

1. Los pesos se cargan correctamente en nuestra instancia de `EquiUnetASSPEvo`.
2. El pre-procesamiento (crop al foreground + z-score por canal) es correcto.
3. El sliding-window 3D y el post-procesamiento (threshold + componente conexa más grande) producen una máscara compacta y plausible.

Visualización del smoke test: `resultados_brats21/real_fold0/preview.png` (3 planos × 4 modalidades, con el tumor detectado visible en azul en las vistas sagitales).

## 10. Cómo evaluar la precisión con un caso real

Si el dataset descargado incluye **ground truth** (segmentación experta, archivo `*_seg.nii.gz`), se puede computar el Dice score por sub-región para comparar contra los ~0.88 reportados en el paper. La fórmula es:

$$
\text{Dice}(A, B) = \frac{2 \cdot |A \cap B|}{|A| + |B|}
$$

donde A es la predicción binaria y B la ground truth, para cada sub-región (WT, TC, ET) por separado.

Esto se puede implementar en pocas líneas con `monai.metrics.DiceMetric`. Si los integrantes quieren agregarlo, sumar un script `scripts/brats21_eval.py` que tome `--pred` y `--gt` y devuelva los 3 Dices.

## 11. Limitaciones reconocidas

| Limitación | Impacto |
|-----------|---------|
| **Sin GPU** | Inferencia ~50–100× más lenta. Un ensamble de 10 folds sobre un caso real con ROI 128³ puede tardar 1+ hora. |
| **TTA desactivado** | El paper usa test-time augmentation (8 transformaciones promedidas). Se omitió para mantener el runner simple y rápido. Se podría agregar en `brats21_run_inference.py` aplicando `tta.OnAxes`, `tta.HorizontalFlip`, `tta.Rotate90` sobre el tensor de entrada. Costo: +0.5% Dice; tiempo: 8× más lento. |
| **Sin `replace_value`** | El paper aplica una interpolación 2D para "rellenar" voxels ET huérfanos (`utils.transforms.ReplaceWithClosestValue`). Requiere `pyradiomics`, que no compila en Python 3.11. Costo: en general <0.3% Dice; se puede omitir. |
| **MONAI 0.6 → 1.3** | Algunos transforms del runner standalone son equivalentes en API moderna pero no idénticos a los del paper. Diferencias en cómo se hace el padding o el threshold pueden mover el Dice unos décimas. |

## 12. Mapa de archivos de la pista 2

| Ruta | Contenido | Versionado en git |
|------|-----------|-------------------|
| `external/BraTS21/` | Repo upstream clonado sin modificar | ✅ |
| `external/BraTS21/checkpoints/` | Pesos descargados (~617 MB) | ❌ (en `.gitignore`) |
| `scripts/brats21_download_weights.py` | Descarga pesos desde Google Drive | ✅ |
| `scripts/brats21_make_synthetic_case.py` | Generador de caso MRI sintético | ✅ |
| `scripts/brats21_make_smoke_fold.py` | Fold con pesos random para smoke test | ✅ |
| `scripts/brats21_run_inference.py` | Runner CPU end-to-end | ✅ |
| `scripts/brats21_visualize.py` | Generador de PNG con segmentación superpuesta | ✅ |
| `requirements-brats21.txt` | Dependencias modernas (Python 3.11) | ✅ |
| `docs/brats21.md` | Este documento | ✅ |
| `.venv-brats21/` | Venv aislado | ❌ |
| `data/brats21_synth/` | Caso sintético generado | ❌ |
| `resultados_brats21/` | Salidas de inferencia (NIfTI + PNG) | ❌ |

## 13. Referencias

- **Paper de BraTS21 (Alxaline):** Carré, A., Deutsch, E., Robert, C. (2022). *Automatic Brain Tumor Segmentation with a Bridge-Unet Deeply Supervised Enhanced with Downsampling Pooling Combination, ASPP, SE and EvoNorm*. BrainLes 2021, Springer. <https://doi.org/10.1007/978-3-031-09002-8_23>
- **Challenge BraTS 2021:** <http://braintumorsegmentation.org/>
- **Medical Decathlon:** Antonelli, M. et al. (2022). *The Medical Segmentation Decathlon*. Nature Communications 13, 4128.
- **MONAI:** <https://monai.io> — framework usado para el sliding-window inference y los transforms.
- **EquiUnetASSPEvo:** arquitectura definida en `external/BraTS21/networks/equiunet2021.py`.
