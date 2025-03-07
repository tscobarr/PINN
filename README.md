# README: Análisis Comparativo de Planteamientos en Redes Neuronales Informadas por la Física (PINN)

## **Introducción**

Las ecuaciones diferenciales son fundamentales en la modelación de sistemas físicos, pero su resolución mediante métodos numéricos tradicionales puede ser costosa y compleja. Las Redes Neuronales Informadas por la Física (PINN) han emergido como una alternativa innovadora, integrando el conocimiento físico directamente en el proceso de aprendizaje de la red.

Este proyecto compara dos planteamientos distintos para el diseño y entrenamiento de PINN, con el objetivo de determinar cuál ofrece mejor rendimiento en términos de precisión y eficiencia computacional.

## **Marco teórico**

### 1. Redes neuronales artificiales

Las redes neuronales artificiales (RNA) son modelos computacionales inspirados en la estructura y funcionamiento del cerebro humano. Están compuestas por unidades llamadas neuronas artificiales, organizadas en capas que procesan la información de manera jerárquica. Su entrenamiento se basa en la propagación de información y el ajuste de pesos mediante algoritmos de optimización.

### 2. Redes neuronales informadas por la física (PINN)

Las PINNs son una variante de redes neuronales diseñadas para resolver ecuaciones diferenciales parciales. A diferencia de las redes tradicionales, integran información física en su función de pérdida para garantizar que las soluciones obtenidas cumplan con las ecuaciones diferenciales y sus condiciones de frontera.
El entrenamiento de una PINN implica definir una ecuación diferencial como parte de la función de pérdida, evaluar la red en múltiples puntos del dominio y optimizar los parámetros para minimizar el error en la predicción.

### 3. Métodos implementados en PINN

Se han desarrollado dos enfoques principales para la implementación de PINN:

1. **Incorporación directa de la ecuación diferencial**: Se integra la ecuación diferencial y las condiciones de frontera en la función de pérdida. La red neuronal se entrena minimizando esta función.

$$
\mathcal{L}(u_{NN}) = \frac{1}{N} \sum_{i=1}^{N} \left[\lambda_0 \left( -\Delta u_{NN}(x_i) + \alpha u_{NN} (x_i) - f(x_i) \right)^2 + \lambda_1 \left( u_{NN}(0) - g(0) \right)^2 + \lambda_2 \left( u_{NN}(\pi) - g(\pi) \right)^2\right]
$$

En donde $\lambda_0$, $\lambda_1$ y $\lambda_2$ son los parámetros de peso del modelo.

2. **Reformulación de la solución**: La solución se reformula para satisfacer automáticamente las condiciones de frontera, permitiendo que la red solo aprenda una corrección sobre una solución base.
   
$$
u_{NN} = NN(x) \cdot x \cdot (x - \pi)
$$

Con función de pérdida

$$
\mathcal{L}(u_{NN}) = \frac{1}{N} \sum_{i=1}^{N} \left[ -\Delta u_{NN}(x_i) + \alpha u_{NN}(x_i) - f(x_i) \right]^2
$$

3. **Error relativo**: El error relativo es una métrica que mide la diferencia entre una solución aproximada y la solución exacta en relación con la magnitud de la solución exacta. Se utiliza para evaluar la precisión de modelos numéricos y su desempeño en la aproximación de soluciones a problemas matemáticos.

Matemáticamente, el error relativo se define como:

$$
\text{Error Relativo} = \frac{\|\text{Solución Exacta} - \text{Solución Aproximada} \|}{\|\text{Solución Exacta} \|}
$$

Donde:
- $\|\text{Solución Exacta} - \text{Solución Aproximada} \|$ representa la diferencia entre ambas soluciones.
- $\|\text{Solución Exacta} \|$ es la norma de la solución exacta utilizada como referencia.

El error relativo es útil cuando se comparan soluciones en diferentes escalas, ya que permite evaluar la precisión de una solución aproximada sin verse afectado por el tamaño absoluto de la solución exacta.

## **Planteamiento del problema**
Los métodos tradicionales para resolver ecuaciones diferenciales requieren discretización y altos recursos computacionales. Las PINNs han surgido como una alternativa viable, integrando las ecuaciones diferenciales en el proceso de entrenamiento de redes neuronales. Sin embargo, existen múltiples estrategias para implementar las PINNs, y la elección del enfoque adecuado puede impactar significativamente la precisión y eficiencia computacional.

En este estudio, se comparan dos metodologías para entrenar PINN:
1. **Incorporación directa de la ecuación diferencial en la función de pérdida**.
2. **Reformulación de la solución para satisfacer automáticamente las condiciones de frontera**.

La pregunta clave que este estudio busca responder es:  

**¿Cuál de estos dos enfoques proporciona una mejor aproximación en términos de precisión y eficiencia computacional al resolver el siguiente problema de valores de frontera?**  

$$\begin{cases}
  -\Delta u + \alpha u = f, & x \in (0,\pi) \\
  u = g, & x \in \{ 0, \pi \}
\end{cases}$$

donde $u(x)$ es la solución, $\Delta$ es el operador Laplaciano, $\alpha$ es un parámetro, $f(x)$ es una función fuente y $g(x)$ son las condiciones de frontera.

## **Objetivos**

### Objetivo general

Comparar dos planteamientos distintos para entrenar redes neuronales informadas por la física y evaluar su desempeño en la resolución del problema con valores de frontera propuesto con la finalidad de determinar cuál enfoque ofrece mejor precisión y eficiencia computacional.

### Objetivos específicos
- Implementar dos arquitecturas diferentes de PINN.
- Resolver un conjunto de ecuaciones diferenciales mediante cada planteamiento.
- Evaluar la precisión de los resultados mediante el error cuadrático medio (L² error).
- Analizar el costo computacional de cada enfoque.
- Determinar cuál estrategia es más efectiva en términos de precisión y eficiencia computacional.

## **Metodología**  

**Herramientas y tecnologías** 

Para la implementación de los modelos se utilizaron las siguientes herramientas:  

- **Lenguaje de programación**: Python.  
- **Bibliotecas**:  
  - TensorFlow y Keras para la construcción y entrenamiento de las redes neuronales.  
  - Optimizador Adam para el ajuste de pesos.  
- **Visualización**: Matplotlib para graficar los resultados y compararlos con la solución analítica.  

**Desarrollo e implementación de modelos**

Se implementaron dos enfoques distintos para resolver la ecuación diferencial:  

- **Planteamiento 1**:  
  La ecuación diferencial y las condiciones de frontera se incorporan directamente en la función de pérdida.  

- **Planteamiento 2**:  
  La solución se reformula para que la red neuronal aprenda solo una corrección sobre una solución base.  

**Evaluación del error**  
Se utilizó el **error L²** como métrica principal para evaluar la precisión de los modelos.

$$
\text{Error Relativo} =
\frac{\frac{1}{N} \sum (u_{NN}(x) - u(x))^2}
{\frac{1}{N} \sum (u(x))^2}
$$

**Optimización de hiperparámetros**

Para analizar el impacto de los hiperparámetros en el desempeño de las redes neuronales PINN, se realizaron **45 experimentos** utilizando una estrategia de búsqueda aleatoria (*random search*).

Los siguientes hiperparámetros fueron seleccionados para su variación:

- **Número de neuronas por capa**: entre **4** y **10**.
- **Número de capas ocultas**: entre **4** y **5**.
- **Número de puntos de muestreo**: entre **500** y **3000**.
- **Pesos de penalización** para la ecuación diferencial (**$\lambda_0$**) y las condiciones de contorno (**$\lambda_1$**, **$\lambda_2$**): entre **1** y **10**.

Además, se realizó un experimento con una configuración base que empleaba los siguientes valores predeterminados:

| Hiperparámetro | Valor |
|----------------|-------|
| Número de neuronas | 10 |
| Número de capas | 5 |
| Puntos de muestreo | 2000 |
| $\lambda_0$ | 5 |
| $\lambda_1$ | 7 |
| $\lambda_2$ | 7 |

Cada configuración se entrenó durante **400 épocas** utilizando el optimizador **Adam** con una tasa de aprendizaje fija de **$1\times10^{-3}$**.

Los resultados de los experimentos fueron evaluados en términos del **error relativo** y el **costo computacional**, medido como el porcentaje de uso de CPU durante la ejecución. Esta evaluación permite comparar la precisión y la eficiencia computacional de cada configuración de hiperparámetros.

**Comparación y selección del mejor modelo**  
Se seleccionó la arquitectura con **mejor precisión** y **menor costo computacional**.  

## **Funcionamiento del codigo**

### Planteamiento 1

#### 1. Configuración jnicial
- Se usa **Python** con **TensorFlow/Keras** para construir la red neuronal.
- Se establecen **parámetros clave**:
  - Número de neuronas y capas en la red.
  - Cantidad de puntos de muestreo (`nPts`).
  - Iteraciones de entrenamiento (`iterations`).
  - Factores de penalización en la función de pérdida (`\lambda_0, \lambda_1, \lambda_2`).

#### 2. Construcción del modelo
- `makeModel1(neurons, nLayers, activation)`:  
  Crea la red neuronal `uModel1` con capas densas y activación `tanh`.

- `Loss1`:  
  - Define la **función de pérdida**, incorporando la ecuación diferencial y condiciones de frontera.  
  - Usa diferenciación automática (`tf.GradientTape`) para calcular derivadas de `u(x)`.

- `makeLossModel1()`:  
  - Construye un modelo auxiliar para minimizar la función de pérdida personalizada.

#### 3. Entrenamiento del modelo
- Se usa **Adam** como optimizador.
- Se define `trickyLoss()` para permitir la optimización de la pérdida `Loss1`.
- `RelativeErrorCallback` calcula el error relativo en cada época.
- `lossModel1.fit()` entrena la red neuronal durante `500` iteraciones.

#### 4. Evaluación y visualización
- `plotResults()`:  
  - Grafica la solución aproximada vs. la solución exacta.
  - Muestra la evolución de la pérdida (`loss`) y el error relativo (`relative error`).

#### 5. Ejecución final
- Se entrena la red neuronal y se evalúa el desempeño comparando con la solución exacta.

### Planteamiento 2

#### 1. Configuración inicial
- Se usa **Python** con **TensorFlow/Keras** para construir la red neuronal.  
- Se establecen **parámetros clave**:
  - Número de neuronas y capas en la red.
  - Cantidad de puntos de muestreo (`nPts`).
  - Iteraciones de entrenamiento (`iterations`).

#### 2. Construcción del modelo
- `makeModel2(neurons, nLayers, activation)`:  
  - Se genera una red neuronal con capas densas y activación `tanh`.  
  - Se **garantiza que la solución cumple automáticamente las condiciones de frontera** multiplicando la salida por una función de contorno:
      
    $$u_{NN}(x) = NN(x) \cdot x \cdot (x - \pi)$$
    
- `Loss2`:  
  - Define la **función de pérdida**, incorporando la ecuación diferencial.  
  - Usa diferenciación automática (`tf.GradientTape`) para calcular derivadas de `u(x)`.  
  - **No se incluyen términos de penalización de frontera**, ya que la solución reformulada las satisface por construcción.  

- `makeLossModel2()`:  
  - Construye un modelo auxiliar para minimizar la función de pérdida personalizada.  

#### 3. Entrenamiento del modelo
- Se usa **Adam** como optimizador.  
- Se define `trickyLoss()` para permitir la optimización de la pérdida `Loss2`.  
- `RelativeErrorCallback` calcula el error relativo en cada época.  
- `lossModel2.fit()` entrena la red neuronal durante `500` iteraciones.  

#### 4. Evaluación y visualización
- `plotResults()`:  
  - Grafica la solución aproximada vs. la solución exacta.  
  - Muestra la evolución de la pérdida (`loss`) y el error relativo (`relative error`).  

#### 5. Ejecución final
- Se entrena la red neuronal y se evalúa el desempeño comparando con la solución exacta.  

**Nota**: **tricky loss** en ambos casos es simplemente **return yTrue**, lo que significa que la optimización no se realiza directamente sobre la pérdida calculada en la ecuación diferencial. Esto permite el uso de otro **modelo auxiliar** que corrige el error del primero.
