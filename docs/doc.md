# Respuestas y justificaciones de la consigna

## Consigna

> A partir de la imagen de cada proyecto, extraer el o los objetos de interés presentes en la escena con el objetivo de caracterizarlos.

La imagen elegida es un **PET de cuerpo completo** (Tomografía por Emisión de Positrones). En este tipo de imágenes, las zonas de alta actividad metabólica (tumores, inflamaciones) aparecen como regiones **oscuras** sobre un fondo claro. El objetivo es aislar y medir esas lesiones.

---

## 1. Pre-procesamiento

> _Pre procesar la imagen con los elementos que sean necesarios para generar una extracción lo más limpia posible._

### ¿Qué se hizo?

Se aplica un pipeline de dos filtros en cascada:

1. **Filtro de mediana (3×3):** elimina ruido impulsivo (sal-y-pimienta) que es común en imágenes médicas nucleares, sin difuminar los bordes de las lesiones.
2. **Filtro gaussiano (3×3, σ=0.8):** suaviza el grano residual del detector PET con una intensidad leve para no perder detalle.

Además se genera una **máscara del cuerpo** para separar la silueta del paciente del fondo blanco de la imagen. Esto se logra con:

- Umbralización inversa (píxeles < 240 → cuerpo)
- Cierre morfológico con kernel elíptico 5×5 (3 iteraciones) para cerrar huecos internos
- Apertura morfológica (1 iteración) para eliminar ruido externo
- Filtrado de componentes conexas: solo se conservan las de área ≥ 500 px

### ¿Por qué?

Sin la máscara del cuerpo, los bordes y la segmentación detectarían artefactos en el fondo. La mediana es preferible al gaussiano como primer paso porque preserva los bordes finos de las lesiones pequeñas. El gaussiano posterior solo atenúa el grano de alta frecuencia que la mediana no elimina.

---

## 2. Detección de bordes

> _Obtener los bordes._

### ¿Qué se hizo?

Se usa el detector de **Canny** con umbrales 40 (inferior) y 120 (superior) sobre la imagen pre-procesada. Los bordes se restringen a la zona del cuerpo aplicando la máscara: todo píxel fuera de la silueta se fuerza a 0.

### ¿Por qué estos umbrales?

- **Umbral bajo (40):** suficiente para capturar bordes de lesiones con contraste moderado respecto al tejido circundante.
- **Umbral alto (120):** evita que gradientes suaves del fondo se interpreten como bordes.
- La relación 1:3 sigue la recomendación de Canny para un buen balance entre detección y supresión de falsos positivos.

### Salida

`resultados/<método>/edges.png` — imagen binaria donde blanco = borde detectado.

---

## 3. Bounding Box

> _Obtener bounding box._

### ¿Qué se hizo?

Para cada componente conexa en la máscara binaria final, se calcula el **rectángulo contenedor mínimo alineado a los ejes** (axis-aligned bounding box) usando `cv2.connectedComponentsWithStats`:

- `CC_STAT_LEFT` → coordenada x
- `CC_STAT_TOP` → coordenada y
- `CC_STAT_WIDTH` → ancho
- `CC_STAT_HEIGHT` → alto

Se dibuja sobre la imagen original en **verde** junto con el **centroide** (punto magenta) y el **ID** del objeto (texto azul).

### ¿Por qué axis-aligned y no rotado?

Para lesiones pequeñas en PET, un bounding box alineado a los ejes es suficiente: las lesiones raramente tienen orientación dominante a esta resolución. Un bounding box rotado (`cv2.minAreaRect`) aportaría complejidad sin beneficio claro.

### Salida

`resultados/<método>/characterization.png` — imagen con bounding boxes, centroides e IDs dibujados.

---

## 4. Features (características)

> _Obtener features (área, ejes, centroide, etc.)_

### ¿Qué se calculó?

| Feature               | Cómo se calcula                                                      | Interpretación                       |
| --------------------- | -------------------------------------------------------------------- | ------------------------------------ |
| **Área (px)**         | Conteo de píxeles del componente conexo                              | Tamaño de la lesión                  |
| **Perímetro (px)**    | `cv2.arcLength()` sobre el contorno cerrado                          | Irregularidad del borde              |
| **Centroide (x, y)**  | Centro de masa geométrico del componente                             | Ubicación espacial                   |
| **BBox (x, y, w, h)** | Rectángulo contenedor del componente                                 | Extensión espacial                   |
| **Eje mayor (px)**    | Eje mayor de la elipse ajustada (`cv2.fitEllipse`)                   | Dimensión principal                  |
| **Eje menor (px)**    | Eje menor de la elipse ajustada                                      | Dimensión secundaria                 |
| **Orientación (°)**   | Ángulo de la elipse ajustada                                         | Dirección del eje mayor              |
| **Excentricidad**     | \( e = \sqrt{1 - (b/a)^2} \), donde a=semieje mayor, b=semieje menor | 0 = circular, →1 = elongado          |
| **Compacidad**        | \( C = 4\pi \cdot A / P^2 \)                                         | 1 = círculo perfecto, <1 = irregular |
| **Intensidad media**  | Promedio de valores de gris dentro del componente                    | Nivel de captación metabólica        |

### ¿Por qué estas features?

- **Área + perímetro + compacidad:** caracterizan la forma. Una lesión maligna tiende a tener bordes más irregulares (compacidad baja).
- **Ejes + orientación + excentricidad:** describen la geometría de la elipse ajustada. Lesiones con alta excentricidad son elongadas (posible infiltración en un eje).
- **Centroide + BBox:** localizan la lesión en el cuerpo.
- **Intensidad media:** proxy de la actividad metabólica (en PET, menor valor de gris = mayor captación).

### Nota sobre `cv2.fitEllipse`

Requiere al menos 5 puntos en el contorno. Para componentes más pequeños, los ejes se aproximan con ancho/alto del bounding box y la excentricidad queda en 0.

### Salida

`resultados/<método>/features.csv` — tabla CSV con una fila por objeto.

---

## 5. Máscara binaria — post-procesamiento morfológico

> _Generar máscara binaria._

### ¿Qué se hizo?

La máscara binaria es el resultado de la segmentación después de un pipeline de **morfología matemática explícita** con erosión, dilatación y **filtro por forma**:

1. **Segmentación** (Region Growing o K-Means) → máscara cruda con ruido y captación fisiológica (cerebro, órganos)
2. **Erosión explícita** (`cv2.erode`, kernel elíptico 3×3, 2 iteraciones) → elimina conexiones espurias, separa tumores que se tocan, remueve ruido fino y elimina blobs pequeños de captación fisiológica (como el cerebro en Region Growing, que es un blob chico y no sobrevive 2 iteraciones de erosión)
3. **Dilatación explícita** (`cv2.dilate`, kernel elíptico 3×3, 3 iteraciones) → recupera los bordes del tumor que la erosión removió. Al usar más iteraciones de dilatación que de erosión (3 vs 2), se produce una **expansión neta de ~1 píxel** que captura píxeles de borde con menor captación metabólica
4. **Cierre morfológico** (kernel elíptico 3×3, 1 iteración) → sella huecos internos residuales
5. **Filtrado por área** (≥ 15 px) → descarta componentes demasiado pequeños para ser lesiones
6. **Filtro por forma** → descarta componentes con perfil de **órgano** (grandes + compactos + sólidos), como el cerebro en K-Means que es demasiado grande para que la erosión lo elimine

### ¿Por qué erosión y dilatación explícitas?

Usar erosión y dilatación como operaciones independientes permite **controlar cada transformación por separado**:

- **Erosión (2 iter):** actúa como filtro de separación y limpieza. Elimina puentes de 1-3 píxeles entre regiones adyacentes, remueve ruido y destruye blobs pequeños de captación fisiológica. Los tumores reales son suficientemente grandes para sobrevivir.
- **Dilatación (3 iter):** actúa como filtro de recuperación y expansión. La asimetría intencional (3 iter dilatación vs 2 iter erosión) produce una ganancia neta de ~1 píxel, deseable porque los píxeles de borde suelen tener menor captación.

### Filtro por forma — discriminación órgano vs tumor

El cerebro, hígado y riñones presentan captación fisiológica normal en PET que NO es patológica. Para distinguirlos de tumores sin depender de la posición (el tumor puede estar en cualquier parte del cuerpo), se analizan **métricas de forma** de cada componente:

| Métrica | Fórmula | Órganos | Tumores |
|---------|---------|---------|---------|
| **Compacidad** | \( 4\pi A / P^2 \) | Alta (> 0.40): forma redondeada | Variable: bordes irregulares |
| **Solidez** | \( A / A_{convex\_hull} \) | Alta (> 0.65): contorno suave | Variable: más concavidades |
| **Área** | Conteo de píxeles | Grande (> 350 px) | Menor |

Un componente se clasifica como **órgano** (y se descarta) si cumple **todas** estas condiciones:
- Área > 350 px
- Compacidad > 0.40
- Solidez > 0.65

Componentes con área ≤ 350 px se conservan sin análisis de forma (son demasiado pequeños para ser órganos).

### ¿Por qué funciona?

- **Cerebro:** región grande, redondeada (alta compacidad), contorno suave (alta solidez) → se descarta.
- **Tumor:** puede ser grande pero tiene bordes irregulares (baja compacidad) o concavidades (baja solidez) → se conserva.
- **Independiente de posición:** no importa si el tumor está en la cabeza, tronco o piernas. La discriminación es puramente por forma.

### Pipeline visualizado

Cada paso se guarda como imagen individual en `resultados/<método>/morfologia/`:

| Archivo | Contenido |
|---------|-----------|
| `raw.png` | Máscara directa de la segmentación (antes de cualquier morfología) |
| `eroded.png` | Después de la erosión (regiones separadas, cerebro chico eliminado) |
| `dilated.png` | Después de la dilatación (bordes recuperados, expansión neta) |
| `closed.png` | Después del cierre (huecos internos sellados) |
| `area_filtered.png` | Después del filtro por área (componentes < 15 px descartados) |
| `shape_filtered.png` | Máscara final (órganos descartados por forma) |

### Resultado

- **Blanco (255):** píxeles que pertenecen a un objeto de interés (lesión/tumor)
- **Negro (0):** fondo (tejido normal + órganos con captación fisiológica + fondo de imagen)

### Salida

- `resultados/<método>/mask_binary.png` — máscara final (solo tumores)
- `resultados/<método>/morfologia/` — imágenes de cada paso intermedio

---

## 6. Recorte del objeto original

> _Generar a partir de la máscara un recorte de la imagen original que sólo contenga el objeto._

### ¿Qué se hizo?

Para cada objeto detectado se genera un recorte (crop) individual:

1. Se toma el **bounding box** del componente conexo
2. Se agrega un **margen de 4 píxeles** (parámetro `CROP_PAD`) en cada dirección, acotado a los límites de la imagen
3. Se extrae el ROI (región de interés) de la **imagen original** en escala de grises
4. Se aplica la **máscara del componente**: los píxeles que no pertenecen al objeto se reemplazan por blanco (255)
5. El resultado es un recorte donde solo se ve la lesión aislada sobre fondo blanco

### ¿Por qué aplicar la máscara sobre el crop?

Sin la máscara, el recorte incluiría tejido circundante dentro del bounding box. Al aplicar la máscara, cada crop muestra **exclusivamente** los píxeles de la lesión, lo cual facilita análisis posteriores (por ejemplo, histograma de intensidades solo de la lesión, o entrada a un clasificador).

### Salida

`resultados/<método>/crops/object_XX.png` — un archivo PNG por objeto detectado.

---

## Comparación de métodos

### Region Growing

| Aspecto         | Detalle                                                                     |
| --------------- | --------------------------------------------------------------------------- |
| **Enfoque**     | Umbral por percentil + BFS desde semillas                                   |
| **Ventaja**     | Control fino sobre la tolerancia de crecimiento                             |
| **Limitación**  | Sensible a la elección de percentil y tolerancia; puede sub/sobre-segmentar |
| **Cuándo usar** | Cuando se conoce aproximadamente el rango de intensidad de las lesiones     |

### K-Means

| Aspecto         | Detalle                                                                    |
| --------------- | -------------------------------------------------------------------------- |
| **Enfoque**     | Clustering no supervisado en el espacio de intensidades                    |
| **Ventaja**     | No requiere umbrales manuales; separa automáticamente niveles de captación |
| **Limitación**  | El número K es un hiperparámetro; puede agrupar tejidos distintos          |
| **Cuándo usar** | Como exploración inicial cuando no se tienen umbrales de referencia        |

### Filtro anatómico (heurístico)

El flag `--filter-anatomy` descarta componentes por ubicación y tamaño:

- **Área > 500 px:** probablemente órganos (cerebro, hígado, riñones)
- **Centroide en el 30% superior:** captación cerebral (normal en PET)
- **Centroide debajo del 93%:** vejiga (acumula trazador)

> **Limitación:** es una heurística basada en posición relativa. Un enfoque riguroso requeriría atlas anatómico o delimitación manual de ROIs.

---

## 7. Extensión: detección de tumores cerebrales con modelo pre-entrenado

> Implementada en la rama `feat/brats21-pretrained-integration`.  
> Documentación técnica completa: [brats21.md](brats21.md).

### 7.1 ¿Por qué esta extensión?

La cátedra recomendó incorporar el uso de **modelos pre-entrenados** para mejorar la tasa de detección de tumores cerebrales, en lugar de entrenar uno propio. La justificación es directa:

| Criterio | Pista 1 (PET clásico) | Pista 2 (Brain MRI con modelo pre-entrenado) |
|----------|----------------------|---------------------------------------------|
| **Disponibilidad de datos anotados** | No requiere anotaciones (no supervisado) | Cero — el modelo ya fue entrenado por terceros |
| **Reproducibilidad de criterios clínicos** | Depende de umbrales y kernels elegidos | Depende del corpus de entrenamiento (BraTS 2021, 1.251 pacientes) |
| **Precisión sobre el dominio objetivo** | Buena para lesiones focales contrastadas | Estado del arte sobre BraTS: Dice ~0.88 promedio |
| **Esfuerzo de desarrollo** | Implementación completa | Integración + wrappers |
| **Generalización a casos no vistos** | Limitada (parámetros calibrados a la imagen ejemplo) | Alta (entrenado sobre 1.251 casos heterogéneos) |

La idea no es **reemplazar** la Pista 1 sino **complementarla**: la pista 1 demuestra dominio de las técnicas clásicas de la materia (morfología, segmentación por crecimiento de regiones, clustering, caracterización por forma); la pista 2 muestra cómo, cuando el problema lo justifica y existe un modelo pre-entrenado adecuado, se puede obtener un salto cualitativo en precisión sin entrar al complejo proceso de entrenar deep learning desde cero.

### 7.2 ¿Qué modelo se eligió y por qué?

[Alxaline/BraTS21](https://github.com/Alxaline/BraTS21) — solución del autor al desafío **RSNA/ASNR/MICCAI Brain Tumor Segmentation 2021** (publicación: Carré et al., 2022, BrainLes 2021). Se eligió por cumplir simultáneamente todos estos criterios:

1. **Pesos publicados** (Apache 2.0) descargables sin pedir permiso (~617 MB para 10 folds).
2. **Resultados competitivos** del challenge oficial: Dice 0.92 (WT) / 0.88 (TC) / 0.84 (ET) sobre la validación oficial — del nivel de los top-5 del challenge.
3. **Código abierto completo** del autor, lo que permite construir wrappers sin modificar el upstream.
4. **Ensamble multi-fold** ya armado: 5 folds con criterio Dice + 5 con criterio Jaccard, promediables para mejorar robustez.

### 7.3 ¿Qué problema resuelve concretamente?

Dado un caso de **MRI cerebral multimodal** del paciente (4 volúmenes 3D registrados espacialmente: T1, T1ce, T2, FLAIR), produce un volumen de etiquetas con la segmentación de las tres sub-regiones tumorales que define el challenge BraTS:

- **WT (Whole Tumor)** = unión de necrosis, edema y tumor realzante → delimita la lesión completa
- **TC (Tumor Core)** = necrosis + tumor realzante → marca el núcleo activo
- **ET (Enhancing Tumor)** = solo el tumor que capta contraste → identifica la parte más agresiva (típicamente glioblastoma)

### 7.4 Decisiones de diseño en la integración

Tres incompatibilidades del repo upstream con nuestro entorno (Windows 11, Python 3.11, sin GPU) y cómo se resolvieron:

| Problema upstream | Solución elegida |
|-------------------|------------------|
| `import resource` (módulo solo UNIX) en `src/main_inference.py` | Bypasear `main_inference.py` y `Engine` enteros con un runner standalone propio |
| `assert torch.cuda.is_available()` hardcodeado | Runner que usa `torch.device("cpu")` por default y permite cambiar a `"cuda"` con un flag |
| Dependencias muertas (MONAI 0.6, scikit-learn 0.23, pyradiomics 3.0, ranger21) no instalables en Python 3.11 | Nuevo `requirements-brats21.txt` con versiones modernas (torch 2.12 CPU, MONAI 1.3) en un venv aislado |

**Principio rector:** no se modifica una sola línea de `external/BraTS21/`. Todas las adaptaciones viven en `scripts/`. Esto permite actualizar el upstream en el futuro sin conflictos.

### 7.5 Validación que se hizo

Para verificar que la integración funciona sin esperar a descargar un dataset real, se construyó un **caso MRI sintético** con la misma estructura BraTS (4 modalidades de 240×240×155, "cerebro" elipsoidal con un "tumor" focal plantado).

Resultados medidos en este equipo (Windows 11, sin GPU):

| Variante | Voxels NCR/NET | Voxels ED | Voxels ET |
|----------|---------------:|----------:|----------:|
| Pesos aleatorios (control) | 1.146.239 | 132.772 | 3.882.512 |
| **Pesos pre-entrenados** | **623** | **60** | **1.075** |

El modelo entrenado detecta **exactamente** la región del tumor sintético plantado (~1.700 voxels totales), mientras que con pesos aleatorios la salida es ruido distribuido por todo el volumen. Esto confirma que (1) los pesos cargan correctamente, (2) el pre-procesamiento es correcto, y (3) el pipeline completo (sliding-window + post-procesamiento) produce salidas plausibles.

### 7.6 Datasets reales recomendados

Para correr la pista 2 sobre datos reales (no sintéticos), se usan datasets públicos compatibles con el formato esperado por el runner (4 archivos `_t1.nii.gz`, `_t1ce.nii.gz`, `_t2.nii.gz`, `_flair.nii.gz` por caso):

| Dataset | Tamaño | Acceso | Recomendación |
|---------|-------:|--------|---------------|
| **Medical Decathlon Task01_BrainTumour** | ~7 GB, 750 casos | Descarga directa | Recomendado para empezar; subconjunto de BraTS 2017 |
| **Kaggle BraTS 2021** | ~30 GB, 1.251 casos | Requiere cuenta Kaggle + API key | Mismo dataset con el que se entrenó el modelo |
| **Synapse BraTS oficial** | ~30 GB | Requiere aprobación manual | Sólo si se necesita el dataset original con la última versión |

Detalles de uso, links y conversión de formatos en [brats21.md sección 7](brats21.md#7-datasets-recomendados-para-datos-reales).

### 7.7 Métrica de evaluación

Cuando el caso de entrada incluye **ground truth** (segmentación experta), se puede computar el **Dice score** por sub-región:

\[ \text{Dice}(A, B) = \frac{2 \cdot |A \cap B|}{|A| + |B|} \]

con A = predicción binaria, B = ground truth, evaluado por separado para WT, TC y ET. El paper reporta Dice promedio ~0.88 sobre el set de validación oficial.

### 7.8 Conclusión sobre la extensión

La pista 2 **complementa** la pista 1 demostrando cómo aplicar la recomendación de la cátedra de usar modelos pre-entrenados. No reemplaza el trabajo clásico de la pista 1 (que sigue siendo el aprendizaje principal de la materia) sino que muestra que:

1. Para problemas con dominio bien definido (tumor cerebral en MRI multimodal) **existen** modelos pre-entrenados de altísima calidad.
2. Integrarlos en un proyecto propio es **factible** con esfuerzo razonable, aunque requiere resolver incompatibilidades de versiones y supuestos sobre el entorno (GPU, OS).
3. El resultado supera ampliamente lo que podríamos lograr con técnicas clásicas en MRI 3D multimodal (problema donde las técnicas clásicas tienen rendimiento muy limitado por la complejidad del dominio).
