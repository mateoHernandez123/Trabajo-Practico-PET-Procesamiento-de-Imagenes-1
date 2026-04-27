# Trabajo Práctico — Extracción y Caracterización de Objetos en PET

**Materia:** Procesamiento de Imágenes I  
**Integrantes:** Mateo Hernandez, Felipe Lucero  
**Repositorio en GitHub:** [github.com/mateoHernandez123/Trabajo-Practico-PET-Procesamiento-de-Imagenes-1](https://github.com/mateoHernandez123/Trabajo-Practico-PET-Procesamiento-de-Imagenes-1)

Este trabajo implementa un pipeline en Python: preprocesamiento (mediana + gaussiano), detección de bordes (Canny), segmentación con dos métodos intercambiables (Region Growing y K-Means), extracción de features (área, perímetro, centroide, ejes, orientación, excentricidad, compacidad, intensidad media), generación de máscara binaria y recortes individuales por objeto.

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

**Qué es:** máscara binaria obtenida por umbralización por percentil 90 + crecimiento de regiones (BFS con tolerancia 25) + post-procesamiento morfológico.  
**Qué justifica:** la segmentación por Region Growing permite controlar la tolerancia de crecimiento. Las regiones blancas son los objetos de interés detectados (lesiones con alta captación).

### 3. Máscara binaria — K-Means

<p align="center">
  <img src="resultados/kmeans/mask_binary.png" alt="Máscara binaria — K-Means" width="200">
</p>

**Qué es:** máscara binaria obtenida por K-Means (K=4 clusters) seleccionando el cluster más oscuro (mayor captación metabólica) + post-procesamiento morfológico.  
**Qué justifica:** la segmentación por clustering no supervisado separa automáticamente niveles de intensidad sin requerir umbrales manuales.

### 4. Caracterización — Region Growing

<p align="center">
  <img src="resultados/region/characterization.png" alt="Caracterización Region Growing — bounding box, centroide e ID" width="200">
</p>

**Qué es:** imagen original con bounding boxes (verde), centroides (magenta) e IDs (azul) de cada objeto detectado por Region Growing.  
**Qué justifica:** visualización directa de las features geométricas sobre la imagen para validar que la segmentación captura las lesiones correctas.

### 5. Caracterización — K-Means

<p align="center">
  <img src="resultados/kmeans/characterization.png" alt="Caracterización K-Means — bounding box, centroide e ID" width="200">
</p>

**Qué es:** imagen original con bounding boxes, centroides e IDs de cada objeto detectado por K-Means.  
**Qué justifica:** permite comparar visualmente qué lesiones captura cada método y verificar la correspondencia con la tabla de features.

### 6. Comparativa de métodos (con filtro anatómico)

<p align="center">
  <img src="resultados/comparison_filtered.png" alt="Comparativa Region Growing vs K-Means con filtro anatómico" width="700">
</p>

**Qué es:** panel comparativo que muestra el pipeline completo de ambos métodos side-by-side: imagen original, bordes, clusters K-Means, máscaras y caracterizaciones.  
**Qué justifica:** permite evaluar en un solo vistazo las diferencias entre Region Growing y K-Means en cuanto a la cantidad, tamaño y ubicación de las lesiones detectadas.

### 7. Recortes individuales — Region Growing

<p align="center">
  <img src="resultados/crops_region_filtered.png" alt="Galería de recortes — Region Growing" width="500">
</p>

**Qué es:** galería de recortes donde cada objeto aparece aislado sobre fondo blanco, extraído de la imagen original usando la máscara del componente conexo correspondiente.  
**Qué justifica:** la consigna pide generar, a partir de la máscara, un recorte que contenga solo el objeto. Cada crop muestra exclusivamente los píxeles de la lesión.

### 8. Recortes individuales — K-Means

<p align="center">
  <img src="resultados/crops_kmeans_filtered.png" alt="Galería de recortes — K-Means" width="500">
</p>

**Qué es:** galería de recortes de los objetos detectados por K-Means.  
**Qué justifica:** misma técnica de extracción, distinto método de segmentación. Permite comparar qué regiones se aíslan con cada enfoque.

---

## Features detectadas

### Region Growing

| ID  | Área | Perímetro | Centroide (x, y) | BBox (x, y, w, h)  | Ejes M/m      | Orient.° | Excent. | Compact. | I. media |
| --- | ---- | --------- | ---------------- | ------------------ | ------------- | -------- | ------- | -------- | -------- |
| 1   | 209  | 52.87     | (76.7, 158.8)    | (68, 151, 18, 16)  | 17.77 / 13.51 | 118.69   | 0.650   | 0.940    | 13.9     |
| 2   | 170  | 56.77     | (77.7, 179.1)    | (68, 170, 19, 18)  | 20.60 / 10.86 | 139.74   | 0.850   | 0.663    | 10.7     |
| 3   | 149  | 49.80     | (107.2, 185.7)   | (102, 176, 11, 20) | 20.16 / 8.43  | 165.73   | 0.908   | 0.755    | 12.8     |

### K-Means

| ID  | Área | Perímetro | Centroide (x, y) | BBox (x, y, w, h) | Ejes M/m      | Orient.° | Excent. | Compact. | I. media |
| --- | ---- | --------- | ---------------- | ----------------- | ------------- | -------- | ------- | -------- | -------- |
| 1   | 217  | 72.18     | (106.8, 184.7)   | (99, 169, 15, 29) | 27.98 / 10.48 | 162.53   | 0.927   | 0.523    | 23.5     |
| 2   | 31   | 20.73     | (50.3, 196.2)    | (48, 192, 6, 9)   | 8.42 / 4.04   | 11.17    | 0.877   | 0.907    | 33.5     |
| 3   | 26   | 18.14     | (111.3, 202.5)   | (108, 200, 7, 6)  | 5.97 / 4.56   | 57.66    | 0.646   | 0.993    | 46.3     |

---

## Estructura del proyecto

| Ruta               | Contenido                                                                            |
| ------------------ | ------------------------------------------------------------------------------------ |
| `README.md`        | Este archivo: resumen, figuras y estructura                                          |
| `segment_pet.py`   | Pipeline único: preprocesamiento, bordes, segmentación, features, máscaras, recortes |
| `requirements.txt` | Dependencias (numpy, opencv-python, matplotlib)                                      |
| `imagenes/`        | Carpeta de entrada; por defecto `pet_cuerpo_completo.png`                            |
| `resultados/`      | PNG, CSV y recortes generados al ejecutar                                            |
| `docs/Readme.md`   | Instalación, entorno virtual y salidas                                               |
| `docs/doc.md`      | Informe / respuestas a la consigna                                                   |
| `.gitignore`       | Excluye venv/, cachés de Python e ignorados de IDE                                   |

Parámetros útiles en código: `HOT_PERCENTILE`, `REGION_GROW_TOLERANCE`, `KMEANS_K`, `MIN_LESION_AREA`, `CANNY_LOW`/`CANNY_HIGH`, `CROP_PAD`, `MAX_OBJECT_AREA` en `segment_pet.py`.

---

## Clonar o actualizar desde GitHub

```bash
git clone git@github.com:mateoHernandez123/Trabajo-Practico-PET-Procesamiento-de-Imagenes-1.git
cd Trabajo-Practico-PET-Procesamiento-de-Imagenes-1
```
