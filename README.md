# Trabajo Práctico — Extracción y Caracterización de Objetos en PET (con Morfología)

**Materia:** Procesamiento de Imágenes I  
**Integrantes:** Mateo Hernandez, Felipe Lucero  
**Repositorio en GitHub:** [github.com/mateoHernandez123/Trabajo-Practico-PET-Morfologia](https://github.com/mateoHernandez123/Trabajo-Practico-PET-Morfologia)

Este trabajo implementa un pipeline en Python: preprocesamiento (mediana + gaussiano), detección de bordes (Canny), segmentación con dos métodos intercambiables (Region Growing y K-Means), **post-procesamiento morfológico con erosión y dilatación explícitas** y **filtro por forma** para aislar tumores descartando captación fisiológica (cerebro, órganos), extracción de features (área, perímetro, centroide, ejes, orientación, excentricidad, compacidad, intensidad media), generación de máscara binaria y recortes individuales por objeto.

## Cómo ejecutar

```bash
pip install -r requirements.txt
python3 segment_pet.py
```

Instrucciones detalladas (venv, Windows/Linux, Git Bash): [docs/Readme.md](docs/Readme.md).  
Respuestas y justificaciones de la consigna: [docs/doc.md](docs/doc.md).

La carpeta `resultados/` se genera al ejecutar el script. La imagen de entrada debe estar en `imagenes/pet_cuerpo_completo.png` (ver [docs/Readme.md](docs/Readme.md) para usar otra ruta).

---

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

## Estructura del proyecto

| Ruta                         | Contenido                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------ |
| `README.md`                  | Este archivo: resumen, figuras y estructura                                                      |
| `segment_pet.py`             | Pipeline: preprocesamiento, bordes, segmentación, morfología, filtro por forma, features, recortes |
| `requirements.txt`           | Dependencias (numpy, opencv-python, matplotlib)                                                  |
| `imagenes/`                  | Carpeta de entrada; por defecto `pet_cuerpo_completo.png`                                        |
| `resultados/`                | PNG, CSV, recortes y pasos morfológicos generados al ejecutar                                    |
| `resultados/<m>/morfologia/` | Imágenes intermedias: erosión, dilatación, cierre, filtro área, filtro forma                     |
| `docs/Readme.md`             | Instalación, entorno virtual y salidas                                                           |
| `docs/doc.md`                | Informe / respuestas a la consigna                                                               |
| `.gitignore`                 | Excluye venv/, cachés de Python e ignorados de IDE                                               |

Parámetros útiles en código: `HOT_PERCENTILE`, `REGION_GROW_TOLERANCE`, `KMEANS_K`, `MIN_LESION_AREA`, `ERODE_KERNEL`, `ERODE_ITERATIONS`, `DILATE_KERNEL`, `DILATE_ITERATIONS`, `ORGAN_MIN_AREA`, `ORGAN_MIN_COMPACTNESS`, `ORGAN_MIN_SOLIDITY`, `MORPH_KERNEL`, `CANNY_LOW`/`CANNY_HIGH`, `CROP_PAD` en `segment_pet.py`.

---

## Clonar o actualizar desde GitHub

```bash
git clone git@github.com:mateoHernandez123/Trabajo-Practico-PET-Morfologia.git
cd Trabajo-Practico-PET-Morfologia
```
