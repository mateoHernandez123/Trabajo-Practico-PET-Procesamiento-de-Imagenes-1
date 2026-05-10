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

| Métrica        | Fórmula                    | Órganos                         | Tumores                      |
| -------------- | -------------------------- | ------------------------------- | ---------------------------- |
| **Compacidad** | \( 4\pi A / P^2 \)         | Alta (> 0.40): forma redondeada | Variable: bordes irregulares |
| **Solidez**    | \( A / A\_{convex_hull} \) | Alta (> 0.65): contorno suave   | Variable: más concavidades   |
| **Área**       | Conteo de píxeles          | Grande (> 350 px)               | Menor                        |

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

| Archivo              | Contenido                                                           |
| -------------------- | ------------------------------------------------------------------- |
| `raw.png`            | Máscara directa de la segmentación (antes de cualquier morfología)  |
| `eroded.png`         | Después de la erosión (regiones separadas, cerebro chico eliminado) |
| `dilated.png`        | Después de la dilatación (bordes recuperados, expansión neta)       |
| `closed.png`         | Después del cierre (huecos internos sellados)                       |
| `area_filtered.png`  | Después del filtro por área (componentes < 15 px descartados)       |
| `shape_filtered.png` | Máscara final (órganos descartados por forma)                       |

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

---

# Análisis longitudinal de tumores PET (`longitudinal_pet_analysis.py`)

## Objetivo

> Evaluar la **evolución temporal** de tumores cerebrales comparando estudios PET del mismo paciente en distintos momentos, determinando si un tumor está creciendo, respondiendo al tratamiento o se mantiene estable.

El análisis longitudinal extiende el pipeline de segmentación del TP1: en lugar de analizar **una sola imagen**, se comparan **múltiples estudios** del mismo paciente a lo largo del tiempo. Esto permite pasar de la detección puntual a un **seguimiento clínico** con métricas cuantitativas.

---

## 7. Registro espacial (alineación entre timepoints)

### ¿Qué se hizo?

Antes de comparar dos estudios PET del mismo paciente, las imágenes deben estar **alineadas espacialmente**. Aunque se trate del mismo paciente, la posición de la cabeza dentro del scanner varía entre sesiones: puede haber traslación, rotación leve o diferencias de escala.

Se implementa un **registro rígido 2D** basado en **correlación de fase** (`cv2.phaseCorrelate`): se estima el desplazamiento traslacional entre la imagen baseline (T0) y cada follow-up, y se aplica una transformación afín para alinearlas.

```python
(dx, dy), response = cv2.phaseCorrelate(fixed_f, moving_f)
M = np.float32([[1, 0, dx], [0, 1, dy]])
registered = cv2.warpAffine(moving, M, (cols, rows))
```

Las máscaras de segmentación se re-muestrean con **interpolación nearest-neighbor** para no crear valores intermedios en la máscara binaria.

### ¿Por qué correlación de fase?

- Opera en el dominio de frecuencia: es robusta a cambios de intensidad entre sesiones (distinto scanner, distinta calibración).
- Estima traslación subpíxel con alta precisión.
- Para imágenes PET 2D con cambios pequeños entre sesiones, un modelo traslacional es suficiente.
- Para datos 3D clínicos reales, se recomienda SimpleITK con transformaciones rígidas/deformables, pero el principio es el mismo.

### Salida

Cada timepoint posterior al baseline se transforma al espacio del T0. Las imágenes y máscaras registradas se usan en todos los cálculos posteriores.

---

## 8. Segmentación tumoral — métodos disponibles

### ¿Qué se hizo?

Se ofrecen dos backends de segmentación intercambiables:

### Método clásico (`--method classical`)

K-Means sobre intensidades de píxel dentro de la máscara cerebral, seleccionando el cluster más brillante (tumor en PET cerebral), seguido de morfología (erosión + dilatación + cierre + filtro por área):

1. **Gaussiano (5×5, σ=1.0):** suavizado.
2. **Máscara cerebral:** Otsu + morfología + componente conexa más grande.
3. **K-Means (K=4):** cluster más brillante = tumor.
4. **Erosión** (kernel 3×3, 1 iter) + **Dilatación** (kernel 5×5, 2 iter) + **Cierre** (kernel 5×5, 2 iter).
5. **Filtro por área** (≥ 30 px).

### Método nnU-Net (`--method nnunet`)

Utiliza el modelo pre-entrenado **JuST_BrainPET** (Task169_BrainTumorPET) del framework [nnU-Net v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). Es una red neuronal 3D U-Net que fue entrenada específicamente para segmentar tumores en PET cerebrales con trazador 18F-FET.

- **Arquitectura:** nnU-Net 3d_fullres auto-configurado.
- **Patch size:** 64×192×192 mm.
- **Performance reportada:** F1=92%, Sensibilidad=93%, PPV=95%.
- **Entrada:** volúmenes NIfTI (.nii.gz) de PET estático FET.
- **Paper:** Lohmann et al., _Automated Brain Tumor Detection and Segmentation for Treatment Response Assessment Using Amino Acid PET_, J Nucl Med, 2023.

### ¿Por qué dos métodos?

El método clásico funciona con imágenes 2D (PNG/JPG) sin dependencias pesadas. El método nnU-Net requiere PyTorch + GPU + datos en NIfTI 3D, pero ofrece segmentación state-of-the-art. El pipeline longitudinal es agnóstico al backend: acepta máscaras de cualquier origen.

---

## 9. Métricas longitudinales

### ¿Qué se calculó?

Para cada par de timepoints consecutivos se calculan métricas que cuantifican la evolución tumoral:

| Métrica                      | Fórmula                                       | Interpretación clínica                                                |
| ---------------------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| **Área tumoral**             | Suma de píxeles del tumor                     | Tamaño de la lesión en cada momento                                   |
| **Cambio absoluto**          | Área(t) - Área(t-1)                           | Cuánto creció o se redujo en píxeles                                  |
| **Cambio porcentual**        | (Área(t) - Área(t-1)) / Área(t-1) × 100       | Velocidad relativa de cambio                                          |
| **Cambio vs baseline**       | (Área(t) - Área(T0)) / Área(T0) × 100         | Evolución acumulada desde el diagnóstico                              |
| **Dice Similarity**          | 2\|A∩B\| / (\|A\|+\|B\|)                      | Similitud espacial entre máscaras; 1.0 = idénticas, 0.0 = sin overlap |
| **Jaccard (IoU)**            | \|A∩B\| / \|A∪B\|                             | Overlap real; más estricto que Dice                                   |
| **Desplazamiento centroide** | √((cx₁-cx₂)² + (cy₁-cy₂)²)                    | ¿El tumor se movió? Posible efecto masa o migración                   |
| **Nuevas regiones**          | Píxeles en máscara(t) pero no en máscara(t-1) | Aparición de nuevas áreas tumorales                                   |
| **Regiones desaparecidas**   | Píxeles en máscara(t-1) pero no en máscara(t) | Regiones que respondieron al tratamiento                              |
| **Intensidad media**         | Media de gris en la región tumoral            | Proxy de actividad metabólica (captación PET)                         |

### ¿Por qué estas métricas?

- **Área + cambio porcentual:** son la base de los criterios clínicos de respuesta (RECIST).
- **Dice + Jaccard:** miden cuánto cambió la _distribución espacial_ del tumor, no solo su tamaño. Un tumor puede mantener el mismo volumen pero moverse o cambiar de forma.
- **Centroide:** si el tumor se desplaza significativamente, puede indicar efecto masa (desplazamiento de estructuras cerebrales) o aparición de nuevos focos.
- **Nuevas/desaparecidas:** permiten distinguir entre reducción uniforme y aparición de nuevas lesiones satélite.
- **Intensidad:** en PET, un tumor que mantiene su tamaño pero reduce su intensidad media puede estar respondiendo al tratamiento (menor actividad metabólica).

---

## 10. Clasificación RECIST

### ¿Qué se hizo?

Se implementa una versión simplificada de los criterios [RECIST](https://recist.eortc.org/) (Response Evaluation Criteria in Solid Tumors), adaptada a mediciones 2D de área tumoral:

| Clasificación                | Criterio (cambio vs baseline) | Interpretación                                      |
| ---------------------------- | ----------------------------- | --------------------------------------------------- |
| **CR** (Complete Response)   | Reducción > 90%               | El tumor prácticamente desapareció                  |
| **PR** (Partial Response)    | Reducción > 30%               | Buena respuesta al tratamiento                      |
| **SD** (Stable Disease)      | Entre -30% y +20%             | Sin cambio significativo                            |
| **PD** (Progressive Disease) | Aumento > 20%                 | El tumor está creciendo; el tratamiento no funciona |

### ¿Por qué RECIST?

RECIST es el estándar clínico internacional para evaluar respuesta tumoral en ensayos oncológicos. Aunque la versión clínica real usa medidas unidimensionales sobre CT/MRI, adaptar los umbrales a mediciones de área en PET permite presentar resultados en un formato que los médicos reconocen y comprenden. Los umbrales (-30% y +20%) provienen directamente de las guías RECIST 1.1.

---

## 11. Visualizaciones longitudinales

### ¿Qué se generó?

El pipeline produce 5 tipos de visualizaciones que permiten evaluar la evolución tumoral:

| Visualización             | Archivo                       | Descripción                                                                                                                                               |
| ------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Comparación temporal**  | `comparacion_temporal.png`    | Grilla 2×N: fila superior = imagen PET con overlay verde de la segmentación, fila inferior = máscara binaria con cambio porcentual y clasificación RECIST |
| **Timeline de volumen**   | `timeline_volumen.png`        | Gráfico de línea con el área tumoral en cada fecha; marca el pico tumoral con línea punteada roja                                                         |
| **Heatmap de cambios**    | `heatmaps_cambio_resumen.png` | Mapa de cambios entre timepoints consecutivos: **rojo** = crecimiento, **verde** = reducción, **amarillo/cyan** = estable                                 |
| **Dashboard**             | `dashboard_metricas.png`      | Panel con 4 gráficos: área por timepoint, cambio % vs baseline con umbrales RECIST, curva de Dice, tabla resumen                                          |
| **Heatmaps individuales** | `heatmaps_cambio/*.png`       | Un heatmap por cada par de timepoints consecutivos                                                                                                        |

### ¿Por qué estas visualizaciones?

- **Comparación temporal:** permite evaluar visualmente si el overlay de segmentación tiene sentido clínico en cada momento.
- **Timeline:** la curva de volumen es el indicador más intuitivo de respuesta al tratamiento. Un oncólogo puede ver inmediatamente si hay tendencia a la reducción.
- **Heatmaps:** muestran **dónde** cambió el tumor, no solo cuánto. Un tumor puede reducirse en un lado y crecer en otro (cambio de forma, no de tamaño).
- **Dashboard:** condensa todas las métricas en una sola imagen para presentaciones o reportes.

---

## 12. Reporte clínico

### ¿Qué se generó?

Un reporte de texto (`reporte.txt`) con:

- Identificación del paciente y período de análisis.
- Evolución tumoral por timepoint: fecha, área, cambio porcentual, clasificación RECIST.
- Resumen: área baseline vs final, cambio total, pico tumoral, evaluación clínica.
- Conclusión automática: _"El tumor muestra BUENA RESPUESTA al tratamiento"_ / _"muestra PROGRESIÓN"_ / _"se mantiene ESTABLE"_.

Además se genera un CSV (`metricas_longitudinales.csv`) con todas las métricas numéricas para análisis posterior.

### ¿Por qué un reporte textual?

En contexto clínico, el radiólogo o oncólogo necesita un resumen legible, no solo gráficos. El reporte combina datos cuantitativos (áreas, porcentajes, Dice) con interpretación cualitativa (RECIST, conclusión), similar a un informe radiológico real.

---

## Resultados del demo

El modo `--generate-demo` ejecuta el pipeline completo sobre un escenario clínico simulado con 4 timepoints de un cerebro PET sintético:

| Timepoint      | Fecha      | Escenario             | Área     | Cambio vs baseline | RECIST   |
| -------------- | ---------- | --------------------- | -------- | ------------------ | -------- |
| T0_baseline    | 2025-01-15 | Tumor detectado       | 1,162 px | —                  | Baseline |
| T1_crecimiento | 2025-04-15 | Crece sin tratamiento | 1,620 px | +39.4%             | PD       |
| T2_tratamiento | 2025-07-14 | Inicio tratamiento    | 1,162 px | 0.0%               | SD       |
| T3_respuesta   | 2025-10-12 | Buena respuesta       | 459 px   | -60.5%             | PR       |

**Evaluación final:** Respuesta Parcial (PR) — reducción del 60.5% respecto al baseline. El tumor muestra buena respuesta al tratamiento.

**Métricas de similitud entre timepoints consecutivos:**

| Transición | Dice  | Jaccard | Nuevas regiones | Desaparecidas |
| ---------- | ----- | ------- | --------------- | ------------- |
| T0 → T1    | 0.835 | 0.717   | 458 px          | 0 px          |
| T1 → T2    | 0.835 | 0.717   | 0 px            | 458 px        |
| T2 → T3    | 0.566 | 0.395   | 0 px            | 703 px        |

Se observa que el Dice cae significativamente en T2→T3 (0.566), lo cual es consistente con una reducción tumoral marcada: la máscara del tumor en T3 es mucho más pequeña que la de T2, por lo que el overlap se reduce.

---

## Comparación: segmentación puntual vs análisis longitudinal

| Aspecto           | Segmentación puntual (TP1)        | Análisis longitudinal                   |
| ----------------- | --------------------------------- | --------------------------------------- |
| **Entrada**       | Una imagen                        | Múltiples imágenes del mismo paciente   |
| **Pregunta**      | "¿Dónde está el tumor?"           | "¿Cómo evoluciona el tumor?"            |
| **Salida**        | Máscara + features                | Timeline + métricas + RECIST + heatmaps |
| **Registro**      | No necesario                      | Obligatorio (alinear timepoints)        |
| **Valor clínico** | Detección                         | Seguimiento y evaluación de tratamiento |
| **Segmentación**  | Clásica (K-Means, Region Growing) | Clásica o Deep Learning (nnU-Net)       |

---

## Herramientas y frameworks relacionados

El pipeline longitudinal se inspira en herramientas y estándares de investigación médica:

| Herramienta                                                                                                                            | Relación con el proyecto                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)                                                                                          | Framework de segmentación biomédica state-of-the-art; se integra como backend opcional (JuST_BrainPET, Task169) |
| [JuST_BrainPET](https://nmmitools.org/2024/08/12/just_brainpet-juelich-segmentation-tool-for-amino-acid-pet-brain-tumor-segmentation/) | Modelo pre-entrenado específico para PET cerebral con 18F-FET                                                   |
| [RECIST 1.1](https://recist.eortc.org/)                                                                                                | Criterios clínicos de evaluación de respuesta tumoral; se implementa una versión simplificada                   |
| [SimpleITK](https://simpleitk.readthedocs.io/)                                                                                         | Librería de registro de imágenes médicas; recomendada para datos 3D NIfTI                                       |
| [Yale-Brain-Mets-Longitudinal](https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/)                          | Dataset longitudinal de 11,892 estudios MRI de 1,430 pacientes con metástasis cerebrales                        |
| [BraTS 2024](https://brats.readthedocs.io/)                                                                                            | Challenge de segmentación de tumores cerebrales; dataset referente en el campo                                  |
