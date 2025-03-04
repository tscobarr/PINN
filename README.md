# README: Análisis Comparativo de Planteamientos en Redes Neuronales Informadas por la Física (PINN)

## **Introducción**

Las ecuaciones diferenciales son fundamentales en la modelación de sistemas físicos, pero su resolución mediante métodos numéricos tradicionales puede ser costosa y compleja. Las Redes Neuronales Informadas por la Física (PINN) han emergido como una alternativa innovadora, integrando el conocimiento físico directamente en el proceso de aprendizaje de la red.

Este proyecto compara dos planteamientos distintos para el diseño y entrenamiento de PINNs, con el objetivo de determinar cuál ofrece mejor rendimiento en términos de precisión y eficiencia computacional.

## **Marco Teórico**

### 1. Redes Neuronales Artificiales
Las redes neuronales artificiales (RNA) son modelos computacionales inspirados en la estructura y funcionamiento del cerebro humano. Están compuestas por unidades llamadas neuronas artificiales, organizadas en capas que procesan la información de manera jerárquica. Su entrenamiento se basa en la propagación de información y el ajuste de pesos mediante algoritmos de optimización.

### 2. Redes Neuronales Informadas por la Física (PINN)
Las PINNs son una variante de redes neuronales diseñadas para resolver ecuaciones diferenciales parciales (PDEs). A diferencia de las redes tradicionales, integran información física en su función de pérdida para garantizar que las soluciones obtenidas cumplan con las ecuaciones diferenciales y sus condiciones de frontera.
El entrenamiento de una PINN implica definir una ecuación diferencial como parte de la función de pérdida, evaluar la red en múltiples puntos del dominio y optimizar los parámetros para minimizar el error en la predicción.

### 3. Métodos Implementados en PINNs
Se han desarrollado dos enfoques principales para la implementación de PINNs:
1. **Incorporación Directa de la Ecuación Diferencial**: Se integra la ecuación diferencial y las condiciones de frontera en la función de pérdida. La red neuronal se entrena minimizando esta función.
$$
\mathcal{L}(u_{NN}) = \frac{1}{N} \sum_{i=1}^{N} \left( -\Delta u_{NN}(x_i) + \alpha u_{NN}(x_i) - f(x_i) \right)^2
+ \lambda_1 \left( u_{NN}(0) - g(0) \right)^2
+ \lambda_2 \left( u_{NN}(1) - g(1) \right)^2
$$

En donde $\lambda_1$ y $\lambda_2$ son parámetros.


2. **Reformulación de la Solución**: La solución se reformula para satisfacer automáticamente las condiciones de frontera, permitiendo que la red solo aprenda una corrección sobre una solución base.
$$
u_{NN} = NN(x) \cdot x \cdot (x - \pi)
$$

Con función de pérdida

$$
\mathcal{L}(u_{NN}) = \frac{1}{N} \sum_{i=1}^{N} \left( -\Delta u_{NN}(x_i) + \alpha u_{NN}(x_i) - f(x_i) \right)^2
$$

3. **Error Relativo**: El error relativo es una métrica que mide la diferencia entre una solución aproximada y la solución exacta en relación con la magnitud de la solución exacta. Se utiliza para evaluar la precisión de modelos numéricos y su desempeño en la aproximación de soluciones a problemas matemáticos.

Matemáticamente, el error relativo se define como:

$$
\text{Error Relativo} = \frac{\|\text{Solución Exacta} - \text{Solución Aproximada} \|}{\|\text{Solución Exacta} \|}
$$

Donde:
- $ \|\text{Solución Exacta} - \text{Solución Aproximada} \| $ representa la diferencia entre ambas soluciones.
- $ \|\text{Solución Exacta} \| $ es la norma de la solución exacta utilizada como referencia.

El error relativo es útil cuando se comparan soluciones en diferentes escalas, ya que permite evaluar la precisión de una solución aproximada sin verse afectado por el tamaño absoluto de la solución exacta.

## **Planteamiento del Problema**
Los métodos tradicionales para resolver ecuaciones diferenciales requieren discretización y altos recursos computacionales. Las PINNs han surgido como una alternativa viable, integrando las ecuaciones diferenciales en el proceso de entrenamiento de redes neuronales. Sin embargo, existen múltiples estrategias para implementar las PINNs, y la elección del enfoque adecuado puede impactar significativamente la precisión y eficiencia computacional.

En este estudio, se comparan dos metodologías para entrenar PINNs:
1. **Incorporación Directa de la Ecuación Diferencial en la Función de Pérdida**.
2. **Reformulación de la Solución para Satisfacer Automáticamente las Condiciones de Frontera**.

La pregunta clave que este estudio busca responder es:  

**¿Cuál de estos dos enfoques proporciona una mejor aproximación en términos de precisión y eficiencia computacional al resolver el siguiente sistema de ecuaciones?**  

$$
\begin{cases}
-\Delta u + \alpha u = f, & x \in (0,1) \\
u = g, & x \in \{0,1\}
\end{cases}
$$


## **Objetivos**

### Objetivo General
Comparar dos planteamientos distintos para entrenar redes neuronales PINN y evaluar su desempeño en la resolución de ecuaciones diferenciales.

### Objetivos Específicos
- Implementar dos arquitecturas diferentes de PINNs.
- Resolver un conjunto de ecuaciones diferenciales mediante cada planteamiento.
- Evaluar la precisión de los resultados mediante el error cuadrático medio (L² error).
- Analizar el costo computacional de cada enfoque.
- Determinar cuál estrategia es más efectiva en términos de precisión y eficiencia computacional.

## **Metodología**  

**Herramientas y Tecnologías**  
Para la implementación de los modelos se utilizaron las siguientes herramientas:  

- **Lenguaje de programación**: Python.  
- **Bibliotecas**:  
  - TensorFlow y Keras para la construcción y entrenamiento de las redes neuronales.  
  - Optimizador Adam para el ajuste de pesos.  
- **Visualización**: Matplotlib para graficar los resultados y compararlos con la solución analítica.  

**Desarrollo e Implementación de Modelos**  
Se implementaron dos enfoques distintos para resolver la ecuación diferencial:  

- **Planteamiento 1**:  
  La ecuación diferencial y las condiciones de frontera se incorporan directamente en la función de pérdida.  

- **Planteamiento 2**:  
  La solución se reformula para que la red neuronal aprenda solo una corrección sobre una solución base.  

**Evaluación del Error**  
Se utilizó el **error L²** como métrica principal para evaluar la precisión de los modelos.

$$
\text{Error Relativo} =
\frac{\frac{1}{N} \sum (u_{NN}(x) - u(x))^2}
{\frac{1}{N} \sum (u(x))^2}
$$


**Experimentación y Optimización de Hiperparámetros**  
Se realizaron **30 experimentos** con diferentes configuraciones de hiperparámetros para analizar su impacto en la precisión del modelo.  

**Comparación y Selección del Mejor Modelo**  
Se seleccionó la arquitectura con **mejor precisión** y **menor costo computacional**.  

## **Funcionamiento del codigo**


### Planteamiento 1

#### 1. Configuración Inicial
- Se usa **Python** con **TensorFlow/Keras** para construir la red neuronal.
- Se establecen **parámetros clave**:
  - Número de neuronas y capas en la red.
  - Cantidad de puntos de muestreo (`nPts`).
  - Iteraciones de entrenamiento (`iterations`).
  - Factores de penalización en la función de pérdida (`\lambda_0, \lambda_1, \lambda_2`).

#### 2. Construcción del Modelo
- `makeModel1(neurons, nLayers, activation)`:  
  Crea la red neuronal `uModel1` con capas densas y activación `tanh`.

- `Loss1`:  
  - Define la **función de pérdida**, incorporando la ecuación diferencial y condiciones de frontera.  
  - Usa diferenciación automática (`tf.GradientTape`) para calcular derivadas de `u(x)`.

- `makeLossModel1()`:  
  - Construye un modelo auxiliar para minimizar la función de pérdida personalizada.

#### 3. Entrenamiento del Modelo
- Se usa **Adam** como optimizador.
- Se define `trickyLoss()` para permitir la optimización de la pérdida `Loss1`.
- `RelativeErrorCallback` calcula el error relativo en cada época.
- `lossModel1.fit()` entrena la red neuronal durante `500` iteraciones.

#### 4. Evaluación y Visualización
- `plotResults()`:  
  - Grafica la solución aproximada vs. la solución exacta.
  - Muestra la evolución de la pérdida (`loss`) y el error relativo (`relative error`).

#### 5. Ejecución Final
- Se entrena la red neuronal y se evalúa el desempeño comparando con la solución exacta.

### Planteamiento 2



#### 1. Configuración Inicial
- Se usa **Python** con **TensorFlow/Keras** para construir la red neuronal.  
- Se establecen **parámetros clave**:
  - Número de neuronas y capas en la red.
  - Cantidad de puntos de muestreo (`nPts`).
  - Iteraciones de entrenamiento (`iterations`).

#### 2. Construcción del Modelo
- `makeModel2(neurons, nLayers, activation)`:  
  - Se genera una red neuronal con capas densas y activación `tanh`.  
  - Se **garantiza que la solución cumple automáticamente las condiciones de frontera** multiplicando la salida por una función de contorno:  
    $$
    u_{NN}(x) = NN(x) \cdot (x - x_{\text{min}}) \cdot (x - x_{\text{max}})
    $$

- `Loss2`:  
  - Define la **función de pérdida**, incorporando la ecuación diferencial.  
  - Usa diferenciación automática (`tf.GradientTape`) para calcular derivadas de `u(x)`.  
  - **No se incluyen términos de penalización de frontera**, ya que la solución reformulada las satisface por construcción.  

- `makeLossModel2()`:  
  - Construye un modelo auxiliar para minimizar la función de pérdida personalizada.  

#### 3. Entrenamiento del Modelo
- Se usa **Adam** como optimizador.  
- Se define `trickyLoss()` para permitir la optimización de la pérdida `Loss2`.  
- `RelativeErrorCallback` calcula el error relativo en cada época.  
- `lossModel2.fit()` entrena la red neuronal durante `500` iteraciones.  

#### 4. Evaluación y Visualización
- `plotResults()`:  
  - Grafica la solución aproximada vs. la solución exacta.  
  - Muestra la evolución de la pérdida (`loss`) y el error relativo (`relative error`).  

#### 5. Ejecución Final
- Se entrena la red neuronal y se evalúa el desempeño comparando con la solución exacta.  


**Nota**: **tricky loss** en ambos casos es simplemente **return yTrue**, lo que significa que la optimización no se realiza directamente sobre la pérdida calculada en la ecuación diferencial. Esto permite el uso de otro **modelo auxiliar** que corrige el error del primerO.

## Conclusiones y Trabajo Futuro
(Se deben agregar las conclusiones y las posibles direcciones futuras de la investigación).
