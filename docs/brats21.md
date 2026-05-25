# Integración de BraTS21 (Alxaline) en este proyecto

Esta rama (`feat/brats21-pretrained-integration`) clona el proyecto
[Alxaline/BraTS21](https://github.com/Alxaline/BraTS21) — solución del autor
al desafío RSNA/ASNR/MICCAI Brain Tumor Segmentation 2021 — y lo deja listo
para correr inferencia en este repo, con scripts wrappers que evitan los
supuestos de CUDA/Linux del original.

> **Importante.** Este repo principal es de **PET 2D con técnicas clásicas**.
> BraTS21 es **MRI 3D multimodal con U-Net 3D entrenada en PyTorch**. Son
> tareas y modalidades distintas: el modelo BraTS21 **no se puede aplicar
> directamente** a la imagen `imagenes/pet_cuerpo_completo.png`. Lo que sí
> se hace acá es integrar el código pre-entrenado para poder correr una
> inferencia BraTS-style completa sobre un caso MRI (real o sintético).

## ¿Qué se hizo en esta rama?

1. Se clonó el repo upstream en `external/BraTS21/` (sin `.git` interno).
2. Se creó un venv aislado `.venv-brats21/` con versiones modernas
   (PyTorch 2.12 CPU, MONAI 1.3, Python 3.11) — el repo original pedía
   torch 1.9 / MONAI 0.6 / Python 3.7, hoy ya no instalables en Python 3.11.
3. Se evitan los `.cuda()` hardcodeados del `Engine` y `main_inference.py`
   del repo original creando un runner standalone:

   - `scripts/brats21_download_weights.py`: descarga los pesos pre-entrenados
     (3 GB) desde Google Drive vía `gdown`.
   - `scripts/brats21_make_synthetic_case.py`: genera un caso MRI sintético
     con las 4 modalidades en `.nii.gz` (sirve para probar el pipeline sin
     bajar el dataset oficial, que está detrás de Synapse).
   - `scripts/brats21_make_smoke_fold.py`: arma un "fold" con pesos aleatorios
     y un `config.yaml` válido, útil para verificar la plomería sin esperar
     la descarga.
   - `scripts/brats21_run_inference.py`: inferencia end-to-end en CPU
     (carga config + pesos, normaliza z-score por canal, sliding window 3D
     con MONAI, threshold + componente conexa más grande, conversión
     WT/TC/ET → etiquetas BraTS 0/1/2/4, guardado NIfTI).

## Setup

Una sola vez:

```bash
python -m venv .venv-brats21
.venv-brats21/Scripts/python.exe -m pip install -U pip wheel setuptools
.venv-brats21/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-brats21/Scripts/python.exe -m pip install -r requirements-brats21.txt
```

En Linux/Mac es igual pero con `source .venv-brats21/bin/activate` o usando
el binario `python` del venv directamente.

## Ejecución completa (lo que se probó en esta rama)

### 1) Generar caso sintético

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_make_synthetic_case.py \
    --out data/brats21_synth
```

Crea `data/brats21_synth/BraTS_synth_001/` con `*_t1.nii.gz`, `*_t1ce.nii.gz`,
`*_t2.nii.gz`, `*_flair.nii.gz` (240×240×155 cada uno).

### 2a) Smoke test (sin descargar pesos)

Arma un fold con pesos aleatorios y corre la inferencia para validar el
pipeline:

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_make_smoke_fold.py
.venv-brats21/Scripts/python.exe scripts/brats21_run_inference.py \
    --config external/BraTS21/checkpoints/smoke_test/fold0/config.yaml \
    --input  data/brats21_synth/BraTS_synth_001 \
    --output resultados_brats21/smoke_test \
    --roi 96 96 96 --overlap 0.25 --cleaning-areas
```

Resultado real medido en este equipo (sin GPU): **~106 s** para un volumen
recortado 201×201×130 con sliding window de 96³ y overlap 0.25.
La segmentación generada (`*_seg.nii.gz`) no tiene sentido clínico porque
el modelo no fue entrenado; sólo demuestra que toda la cadena funciona.

### 2b) Inferencia con pesos pre-entrenados reales

Descargar los pesos publicados por el autor (~617 MB comprimido, 10 folds,
Google Drive):

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_download_weights.py \
    --out external/BraTS21/checkpoints
```

Si Google Drive bloquea la descarga automática (común en archivos grandes),
bajar manualmente desde
<https://drive.google.com/file/d/1Xt2rdD60IeEwcd8-yiMZHZkI0udcXgc7/view>
y correr:

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_download_weights.py \
    --from-local /ruta/al/final_weights_brats21.zip
```

Una vez descomprimido queda la estructura:

```
external/BraTS21/checkpoints/final_weights/
├── baseline_equiunet_assp_evocor/        # 5 folds (criterio dice)
│   ├── fold0_ns/{config.yaml, best_model.pth}
│   ├── fold1_ns/...  ...  fold4_ns/
└── baseline_equiunet_assp_evocor_jaccard/ # 5 folds (criterio jaccard)
    ├── fold0/...  ...  fold4/
```

Para correr con un solo fold (rápido, lo que se probó acá):

```bash
.venv-brats21/Scripts/python.exe scripts/brats21_run_inference.py \
    --config external/BraTS21/checkpoints/final_weights/baseline_equiunet_assp_evocor/fold0_ns/config.yaml \
    --input  data/brats21_synth/BraTS_synth_001 \
    --output resultados_brats21/real_fold0 \
    --roi 96 96 96 --overlap 0.25 --cleaning-areas
```

Para ensamble completo (los 10 folds del paper), pasar las 10 `config.yaml`
en `--config`. En CPU esto tarda **muchísimo** (~100 s/fold con `roi=96`,
mucho más con `roi=128 overlap=0.5`); recomendable sólo en GPU.

### Resultados reales medidos en este equipo

Sobre el caso sintético `BraTS_synth_001` (240×240×155, 1 "tumor" elipsoidal
de ~3000 voxels plantado), con `roi=96, overlap=0.25, --cleaning-areas`:

| Variante                  | Tiempo CPU | Voxels NCR/NET | Voxels ED | Voxels ET |
|---------------------------|-----------:|---------------:|----------:|----------:|
| Smoke test (pesos random) |     ~106 s |      1.146.239 |   132.772 | 3.882.512 |
| **fold0_ns** (real)       |    **98 s**|            623 |        60 |     1.075 |

La diferencia confirma que los pesos pre-entrenados se cargan correctamente:
el modelo entrenado identifica exclusivamente la región del "tumor"
plantado (~1.700 voxels totales contra los ~8.000.000 de fondo), mientras
que con pesos aleatorios la salida es ruido distribuido por todo el volumen.

Visualización generada con `scripts/brats21_visualize.py` en
`resultados_brats21/real_fold0/preview.png` (3 planos centrales × 4
modalidades, con la segmentación superpuesta en color).

## Limitaciones reconocidas

- **Sin GPU.** Este equipo no tiene NVIDIA, así que toda la inferencia se hace
  en CPU. Tiempos realistas: ~100 s/fold con `roi=96` en sintético; con `roi=128`
  y un caso BraTS real, esperar **5-15 min por fold**.
- **Sin dataset BraTS oficial.** Requiere registro en Synapse. Se reemplaza
  con caso sintético para demo.
- **TTA desactivado.** El runner no implementa el test-time augmentation
  (flips + rotaciones por eje) del paper. Se podría agregar si se necesita
  el último decimal de Dice; para un demo no aporta.
- **Sin `replace_value` post-procesamiento.** El runner sólo aplica threshold
  y "mayor componente conexa". El paper usa además interpolación 2D de
  etiquetas ET huérfanas (`utils.transforms.ReplaceWithClosestValue`); no se
  portó porque requiere `pyradiomics`/dependencias que no compilan en 3.11.
- **MONAI 0.6 → 1.3.** El runner standalone esquiva los wrappers MONAI
  obsoletos del repo (transforms con `threshold_values=True`, `AddChannel()`,
  etc.) usando las APIs equivalentes modernas.

## Estructura agregada por esta rama

| Ruta | Contenido |
|------|-----------|
| `external/BraTS21/` | Repo upstream clonado (sin `.git`), sin modificar |
| `external/BraTS21/checkpoints/` | (ignorado en git) pesos descargados |
| `scripts/brats21_download_weights.py` | Descarga pesos desde Google Drive |
| `scripts/brats21_make_synthetic_case.py` | Genera caso MRI sintético |
| `scripts/brats21_make_smoke_fold.py` | Fold con pesos aleatorios para smoke test |
| `scripts/brats21_run_inference.py` | Runner CPU end-to-end (independiente del Engine) |
| `scripts/brats21_visualize.py` | Genera PNG con la seg superpuesta sobre las modalidades |
| `requirements-brats21.txt` | Dependencias modernas para Python 3.11 + CPU |
| `data/brats21_synth/` | (ignorado en git) caso sintético generado |
| `resultados_brats21/` | (ignorado en git) salidas NIfTI de la inferencia |
| `docs/brats21.md` | Este documento |

## Si más adelante se quiere correr en GPU

El runner ya soporta GPU sin cambios: alcanza con instalar el wheel de torch
con CUDA y cambiar el `device` en `scripts/brats21_run_inference.py` (línea
donde dice `device = torch.device("cpu")`). Para usar el `Engine` y la
inferencia oficial con todo el TTA, hay que también:

1. Reinstalar `monai==0.6.0` (sólo posible con Python 3.7).
2. Resolver los imports rotos (`ranger21`, `pyradiomics`).
3. O bien usar la imagen Docker oficial: `docker pull alxaline/brats21:latest`.
