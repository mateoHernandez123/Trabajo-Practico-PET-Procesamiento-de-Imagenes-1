# Respuestas y justificaciones de la consigna

## Consigna

> A partir de la imagen de cada proyecto, extraer el o los objetos de interés presentes en la escena con el objetivo de caracterizarlos.

La imagen elegida es un **PET de cuerpo completo** (Tomografía por Emisión de Positrones). En este tipo de imágenes, las zonas de alta actividad metabólica (tumores, inflamaciones) aparecen como regiones **oscuras** sobre un fondo claro. El objetivo es aislar y medir esas lesiones.

---

## 1. Pre-procesamiento

> *Pre procesar la imagen con los elementos que sean necesarios para generar una extracción lo más limpia posible.*

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

> *Obtener los bordes.*

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

> *Obtener bounding box.*

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

> *Obtener features (área, ejes, centroide, etc.)*

### ¿Qué se calculó?

| Feature | Cómo se calcula | Interpretación |
|---------|-----------------|----------------|
| **Área (px)** | Conteo de píxeles del componente conexo | Tamaño de la lesión |
| **Perímetro (px)** | `cv2.arcLength()` sobre el contorno cerrado | Irregularidad del borde |
| **Centroide (x, y)** | Centro de masa geométrico del componente | Ubicación espacial |
| **BBox (x, y, w, h)** | Rectángulo contenedor del componente | Extensión espacial |
| **Eje mayor (px)** | Eje mayor de la elipse ajustada (`cv2.fitEllipse`) | Dimensión principal |
| **Eje menor (px)** | Eje menor de la elipse ajustada | Dimensión secundaria |
| **Orientación (°)** | Ángulo de la elipse ajustada | Dirección del eje mayor |
| **Excentricidad** | \( e = \sqrt{1 - (b/a)^2} \), donde a=semieje mayor, b=semieje menor | 0 = circular, →1 = elongado |
| **Compacidad** | \( C = 4\pi \cdot A / P^2 \) | 1 = círculo perfecto, <1 = irregular |
| **Intensidad media** | Promedio de valores de gris dentro del componente | Nivel de captación metabólica |

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

## 5. Máscara binaria

> *Generar máscara binaria.*

### ¿Qué se hizo?

La máscara binaria es el resultado de la segmentación después de un post-procesamiento morfológico:

1. **Segmentación** (Region Growing o K-Means) → máscara cruda con ruido
2. **Apertura morfológica** (kernel elíptico 3×3, 1 iteración) → elimina conexiones espurias y protuberancias finas
3. **Cierre morfológico** (kernel elíptico 3×3, 2 iteraciones) → rellena huecos internos
4. **Filtrado por área** (≥ 15 px) → descarta componentes demasiado pequeños para ser lesiones

### Resultado

- **Blanco (255):** píxeles que pertenecen a un objeto de interés (lesión)
- **Negro (0):** fondo (tejido normal + fondo de imagen)

### ¿Por qué apertura antes que cierre?

La apertura primero elimina píxeles sueltos que podrían crear "puentes" espurios entre componentes separados. Luego el cierre rellena huecos internos de las lesiones reales sin reconectar componentes que ya se separaron.

### Salida

`resultados/<método>/mask_binary.png`

---

## 6. Recorte del objeto original

> *Generar a partir de la máscara un recorte de la imagen original que sólo contenga el objeto.*

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

| Aspecto | Detalle |
|---------|---------|
| **Enfoque** | Umbral por percentil + BFS desde semillas |
| **Ventaja** | Control fino sobre la tolerancia de crecimiento |
| **Limitación** | Sensible a la elección de percentil y tolerancia; puede sub/sobre-segmentar |
| **Cuándo usar** | Cuando se conoce aproximadamente el rango de intensidad de las lesiones |

### K-Means

| Aspecto | Detalle |
|---------|---------|
| **Enfoque** | Clustering no supervisado en el espacio de intensidades |
| **Ventaja** | No requiere umbrales manuales; separa automáticamente niveles de captación |
| **Limitación** | El número K es un hiperparámetro; puede agrupar tejidos distintos |
| **Cuándo usar** | Como exploración inicial cuando no se tienen umbrales de referencia |

### Filtro anatómico (heurístico)

El flag `--filter-anatomy` descarta componentes por ubicación y tamaño:

- **Área > 500 px:** probablemente órganos (cerebro, hígado, riñones)
- **Centroide en el 30% superior:** captación cerebral (normal en PET)
- **Centroide debajo del 93%:** vejiga (acumula trazador)

> **Limitación:** es una heurística basada en posición relativa. Un enfoque riguroso requeriría atlas anatómico o delimitación manual de ROIs.
