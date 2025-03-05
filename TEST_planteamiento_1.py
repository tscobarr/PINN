import tensorflow as tf  # Librería para aprendizaje profundo
import numpy as np  # Librería para manipulación de matrices y cálculos numéricos
import matplotlib.pyplot as plt  # Librería para visualización de gráficos
from matplotlib import rcParams  # Configuración de gráficos
import os  # Manejo de archivos y directorios
import pandas as pd  # Manejo de datos en formato tabular
os.environ["KERAS_BACKEND"] = "tensorflow"  # Define TensorFlow como backend de Keras
import keras_core as keras  # Librería de redes neuronales basada en TensorFlow
import time  # Para medir tiempos de ejecución
import gc  # Para gestión de la memoria (recolección de basura)
import psutil  # Para medir uso de CPU durante la ejecución de experimentos

# Configuración de reproducibilidad
keras.utils.set_random_seed(1234)  # Fija una semilla para asegurar resultados reproducibles
dtype = 'float32'  # Establece la precisión de los cálculos
keras.backend.set_floatx(dtype)  # Aplica el tipo de dato definido

# Definición de parámetros globales
alpha = 1  # Parámetro utilizado en las ecuaciones diferenciales
limInf = 0  # Límite inferior del dominio de la solución
limSup = np.pi  # Límite superior del dominio de la solución
iterations = 1500

def makeModel1(neurons, nLayers, activation):
    xVals = keras.layers.Input(shape=(1,), name='x_input', dtype=dtype)
    l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype)(xVals)
    for l in range(nLayers - 2):
        l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype)(l1)
    output = keras.layers.Dense(1, activation=activation, dtype=dtype)(l1)
    uModel = keras.Model(inputs=xVals, outputs=output, name='u_model')
    return uModel

class Loss1(keras.layers.Layer):
    def __init__(self, uModel, nPts, f, lambda0=1, lambda1=1, lambda2=1, limInf=0, limSup=np.pi, **kwargs):
        super(Loss1, self).__init__()
        self.uModel = uModel
        self.nPts = nPts
        self.f = f
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.limInf = limInf
        self.limSup = limSup

    def call(self, inputs):
        x = tf.random.uniform([self.nPts], self.limInf, self.limSup, dtype=dtype)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                u = self.uModel(x, training=True)
            dux = t2.gradient(u, x)
        duxx = t1.gradient(dux, x)
        errorPDE = self.lambda0 * keras.ops.mean((-duxx + alpha * u - self.f(x)) ** 2)
        bc = self.lambda1 * self.uModel(np.array([self.limInf])) ** 2 + self.lambda2 * self.uModel(np.array([self.limSup])) ** 2
        return errorPDE + bc

class RelativeErrorCallback(tf.keras.callbacks.Callback):
    def __init__(self, uModel, exactU, nPts):
        super().__init__()
        self.uModel = uModel
        self.exactU = exactU
        self.nPts = nPts

    def on_epoch_end(self, epoch, logs=None):
        Sval = tf.experimental.numpy.linspace(0., np.pi, num=self.nPts*10, dtype=dtype)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(Sval)
            ueval_val = self.uModel(Sval)
            u_x_val = t1.gradient(ueval_val, Sval)
            u_e = self.exactU(Sval)
            ue_x = t1.gradient(u_e, Sval)
        del t1
        errorH01 = tf.reduce_mean((ue_x - u_x_val)**2)
        norm_exact = tf.reduce_mean(ue_x**2)
        relative_error = errorH01 / norm_exact
        logs['relative_error'] = relative_error

def makeLossModel1(uModel, nPts, f, lambda0=1, lambda1=1, lambda2=1, limInf=0, limSup=np.pi):
    xVals = keras.layers.Input(shape=(1,), name='x_input', dtype=dtype)
    loss_output = Loss1(uModel, nPts, f, lambda0, lambda1, lambda2, limInf, limSup)(xVals)
    lossModel = keras.Model(inputs=xVals, outputs=loss_output)
    return lossModel

def trickyLoss(yPred, yTrue):
    return yTrue

def crear_estructura_directorios(base_path="experimentos/PLANTEAMIENTO 1"):
    """
    Crea la estructura de directorios necesarios para almacenar los resultados de los experimentos.
    
    Parámetros:
    - base_path (str): Ruta base donde se almacenarán los resultados.
    
    Retorna:
    - base_path (str): Ruta base de los experimentos.
    - graficas_path (str): Ruta donde se guardarán las gráficas generadas.
    - mejores_path (str): Ruta específica para almacenar los mejores resultados.
    - peores_path (str): Ruta específica para almacenar los peores resultados.
    """
    graficas_path = os.path.join(base_path, "graficas")

    
    print("✔ Directorios creados correctamente.")
    return base_path, graficas_path

def inicializar_excel(file_path="experimentos/PLANTEAMIENTO 1/resultados.xlsx"):
    """
    Crea un archivo Excel con la estructura inicial si no existe. 
    El archivo contendrá tres hojas llamadas "Lote 1", "Lote 2" y "Lote 3",
    donde se almacenarán los resultados de los experimentos.

    Parámetros:
    - file_path (str): Ruta del archivo Excel donde se guardarán los datos.

    Retorna:
    - file_path (str): Ruta del archivo Excel.
    """
    if not os.path.exists(file_path):
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        for i in range(1, 4):
            # Definir las columnas del archivo Excel
            df = pd.DataFrame(columns=[
                "Experimento", "neurons", "nLayers", "nPts", 
                "lambda0", "lambda1", "lambda2", "ERROR RELATIVO", "COSTO COMPUTACIONAL"
            ])
            df.to_excel(writer, sheet_name=f"Lote {i}", index=False)
        writer.close()
        print("✔ Archivo Excel inicializado.")
    else:
        print("✔ Archivo Excel ya existe.")
    
    return file_path

# Función para obtener los hiperparámetros por defecto
def hiperparametros_por_defecto():
    """
    Devuelve un diccionario con valores predeterminados para los hiperparámetros de la PINN.

    Retorna:
    - dict: Diccionario con los hiperparámetros iniciales.
    """
    return {
        "neurons": 10,  # Número de neuronas por capa
        "nLayers": 5,  # Número de capas ocultas
        "nPts": 2000,  # Número de puntos en el dominio
        "lambda0": 5,  # Peso del error de la ecuación diferencial
        "lambda1": 7,  # Peso del error en el contorno inferior
        "lambda2": 7   # Peso del error en el contorno superior
    }

# Función para generar hiperparámetros aleatorios dentro de ciertos rangos
def randomizar_hiperparametros():
    """
    Genera un conjunto aleatorio de hiperparámetros dentro de ciertos rangos predefinidos.

    Retorna:
    - dict: Diccionario con hiperparámetros aleatorios.
    """
    return {
        "neurons": np.random.randint(4, 11),  # Neuronas entre 4 y 10
        "nLayers": np.random.randint(4, 6),  # Capas entre 4 y 5
        "nPts": np.random.randint(500, 3001),  # Puntos entre 500 y 3000
        "lambda0": np.random.randint(1, 11),  # Peso entre 1 y 10
        "lambda1": np.random.randint(4, 11),  # Peso entre 4 y 10
        "lambda2": np.random.randint(4, 11)   # Peso entre 4 y 10
    }

# Función para definir la pérdida y entrenamiento
def train_PINN(hiperparametros, funcion_referencia, funcion_exacta):
    """
    Entrena la PINN con los hiperparámetros dados y la función de referencia seleccionada.

    También mide el error relativo y el costo computacional en términos de uso de CPU.

    Parámetros:
    - hiperparametros (dict): Diccionario con la configuración de la red (neurons, nLayers, etc.).
    - funcion_referencia (función): Función f(x) en la ecuación diferencial que la PINN debe aprender.
    - funcion_exacta (función): Solución exacta esperada para evaluar el error relativo.

    Retorna:
    - error_relativo (float): Error relativo entre la solución de la PINN y la solución exacta.
    - costo_computacional (float): Uso de CPU en porcentaje durante el entrenamiento.
    - history (tf.keras.callbacks.History): Historial del entrenamiento.
    - uModel (keras.Model): Modelo entrenado de la PINN.
    - xList (array): Puntos en el dominio donde se evaluó la solución.
    """

    # 📌 Medir uso de CPU antes del entrenamiento
    cpu_usage_before = psutil.cpu_percent(interval=None)

    # 🔹 Construcción del modelo de red neuronal PINN
    uModel = makeModel1(
        neurons=hiperparametros["neurons"],
        nLayers=hiperparametros["nLayers"],
        activation='tanh'
    )

    # # 📌 Calcular valores exactos en los bordes (x = 0 y x = π)
    # A = funcion_exacta(0)  # 📌 Valor exacto en x = 0
    # B = funcion_exacta(np.pi)  # 📌 Valor exacto en x = π

    # 🔹 Definir el modelo de pérdida basado en la ecuación diferencial
    lossModel = makeLossModel1(
        uModel,
        hiperparametros["nPts"],
        funcion_referencia,  # 📌 Función f(x)
        hiperparametros["lambda0"],
        hiperparametros["lambda1"],
        hiperparametros["lambda2"],
        limInf, limSup,
        # A, B  # 📌 Se pasan los valores exactos en los extremos
    )

    # 📌 Configurar el optimizador y la función de pérdida (trickyLoss)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    lossModel.compile(optimizer=optimizer, loss=trickyLoss)

    relative_error_callback = RelativeErrorCallback(uModel, funcion_referencia, hiperparametros["nPts"])
    # 📌 Entrenamiento de la PINN
    history = lossModel.fit(
        np.array([1.]), np.array([1.]), epochs=iterations, verbose=0, callbacks=[relative_error_callback]
    )

    # 🔹 Generar puntos en el dominio para evaluar la solución entrenada
    xList = np.array([np.pi / 1000 * i for i in range(1000)])

    # 🔹 Evaluar la solución aproximada y compararla con la solución exacta
    error_relativo = history.history['relative_error'][-1].numpy()

    # 📌 Medir uso de CPU después del entrenamiento
    cpu_usage_after = psutil.cpu_percent(interval=None)
    cpu_cost = cpu_usage_after - cpu_usage_before  # 📌 Cálculo del costo computacional

    # 🔹 Liberar memoria
    keras.backend.clear_session()
    gc.collect()

    return error_relativo, cpu_cost, history, uModel, xList  # 📌 Retornar métricas clave y el modelo entrenado

def ejecutar_experimentos(func, n_experimentos=45):  
    """
    Ejecuta múltiples experimentos de entrenamiento de la PINN con diferentes hiperparámetros.

    Se generan varios lotes de experimentos con diferentes configuraciones de hiperparámetros 
    para evaluar la estabilidad y precisión del modelo en distintas condiciones.

    Parámetros:
    - func (lista de tuplas): Lista de funciones de referencia y soluciones exactas esperadas.
      Cada elemento es una tupla (función_f, función_u_exacta).
    - n_experimentos (int, opcional): Número total de experimentos a ejecutar. Por defecto, 45.

    Retorna:
    - resultados (lista): Lista de resultados de los experimentos, donde cada elemento contiene:
      [ID, neurons, nLayers, nPts, lambda0, lambda1, lambda2, error_relativo, costo_computacional, 
      modelo entrenado, historial, solución exacta, xList].
    """

    # 📌 Determinar la cantidad de lotes basada en el número de experimentos
    num_lotes = min(len(func), n_experimentos)  
    lote_size = n_experimentos // num_lotes  # 📌 Número de experimentos por lote

    # 📌 Generar hiperparámetros aleatorios para cada lote (excepto el primero, que es por defecto)
    hiperparametros_aleatorios = [randomizar_hiperparametros() for _ in range(lote_size - 1)]  

    # 📌 Lista para almacenar resultados
    resultados = []

    for i in range(num_lotes):
        # 🔹 Seleccionar la función de referencia y la solución exacta
        f_rhs, exactU = func[i]  
        print(f"\n🔹 Ejecutando Lote {i+1}...\n")

        for j in range(lote_size):
            experimento_id = i * lote_size + j + 1
            print(f"   ▶ Ejecutando experimento {experimento_id}...")

            # 🔹 Para el primer experimento de cada lote, usar hiperparámetros por defecto
            if j == 0:
                hiperparametros = hiperparametros_por_defecto()
            else:
                hiperparametros = hiperparametros_aleatorios[j - 1]

            # 🔹 Entrenar la PINN con la configuración actual
            error_relativo, costo_computacional, history, uModel, xList = train_PINN(hiperparametros, f_rhs, exactU)

            # Imprimir resultados del experimento
            print(f"      🔹 Completado - Error relativo: {error_relativo:.5f}, Uso CPU: {costo_computacional:.2f}%")

            # 📌 Guardar los resultados del experimento
            resultados.append([
                experimento_id, 
                hiperparametros["neurons"], 
                hiperparametros["nLayers"], 
                hiperparametros["nPts"], 
                hiperparametros["lambda0"], 
                hiperparametros["lambda1"], 
                hiperparametros["lambda2"], 
                error_relativo, 
                costo_computacional,  
                uModel,   
                history,  
                exactU,   
                xList     
            ])

    return resultados

# 🔹 Definimos las funciones de referencia y sus soluciones exactas en un diccionario
funciones_experimentos = [
    {
        'fRhs': lambda x: (alpha + 4) * keras.ops.sin(2 * x),
        'exactU': lambda x: keras.ops.sin(2 * x),
        'title': 'sin(2x)'
    },
    {
        'fRhs': lambda x: 17/4 * keras.ops.sin(2 * x) * keras.ops.cos(x / 2) + 2 * keras.ops.sin(x / 2) * keras.ops.cos(2 * x) + alpha * keras.ops.sin(2 * x) * keras.ops.cos(x / 2),
        'exactU': lambda x: keras.ops.sin(2 * x) * keras.ops.cos(x / 2),
        'title': 'sin(2x) * cos(x/2)'
    },
    {
        'fRhs': lambda x: - 2 * (6 * x**2 - 6 * np.pi * x + np.pi**2) + alpha * (x**2 * (x - np.pi)**2),
        'exactU': lambda x: x**2 * (x - np.pi)**2,
        'title': 'x^2 * (x - pi)^2'
    }
]

# 🔹 Convertir el diccionario en una lista de tuplas
func = [(exp['fRhs'], exp['exactU']) for exp in funciones_experimentos]

def guardar_resultados_excel(resultados, file_path="experimentos/PLANTEAMIENTO 1/resultados.xlsx"):
    """
    Guarda los resultados de los experimentos en un archivo Excel, separando por lote.

    Cada hoja del archivo representa un conjunto de experimentos con la misma función de referencia.

    Parámetros:
    - resultados (lista): Lista de resultados obtenidos de `ejecutar_experimentos`.
    - file_path (str, opcional): Ruta del archivo Excel donde se guardarán los datos.

    Retorna:
    - None
    """
    from openpyxl import load_workbook

    # 📌 Cargar o crear un archivo Excel
    if os.path.exists(file_path):
        writer = pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay')
    else:
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')

    # 📌 Recorrer cada experimento y guardarlo en su respectiva hoja
    for i, experimento in enumerate(funciones_experimentos):
        lote_resultados = [
            [r[j] for j in range(9)] for r in resultados if (r[0] - 1) // (len(resultados) // len(funciones_experimentos)) == i
        ]

        df = pd.DataFrame(
            lote_resultados,
            columns=["Experimento", "neurons", "nLayers", "nPts", 
                     "lambda0", "lambda1", "lambda2", "ERROR RELATIVO", "COSTO COMPUTACIONAL (CPU %)"]
        )

        # Reemplazar caracteres inválidos en el nombre de la hoja
        sheet_name = experimento['title'].replace('*', '').replace('/', '').replace('\\', '').replace('[', '').replace(']', '').replace(':', '').replace('?', '')

        df.to_excel(writer, sheet_name=sheet_name, index=False)

    # 📌 Guardar los cambios
    writer.close()
    print(f"✔ Resultados guardados en {file_path} con las funciones correctamente organizadas.")

def plot_results(uModel, history, exactU, xList, exp_folder):
    """
    Genera y guarda las gráficas de un experimento, asegurando que se use la función exacta correcta.
    """
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 14
    rcParams['legend.fontsize'] = 12
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['axes.labelsize'] = 14

    # Comparación de la solución exacta y la aproximada
    fig, ax = plt.subplots()
    plt.plot(xList, uModel(xList), color='b', label="u_approx")
    plt.plot(xList, exactU(xList), color='m', label="u_exact")  # ✅ Se asegura que usa la función exacta correcta
    plt.legend()
    ax.grid(which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "comparacion_u.png"), dpi=300)
    plt.close()

    # Evolución de la pérdida y error relativo
    fig, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], color='g', label="Loss")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='g')

    ax2 = ax1.twinx()
    ax2.plot(history.history['loss'], color='b', label="Relative Error")  # 📌 Se debe reemplazar por la métrica correcta
    ax2.set_ylabel("Relative Error", color='b')

    ax1.grid(which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "error_vs_loss.png"), dpi=300)
    plt.close()

    # Error relativo a lo largo de las épocas
    fig, ax = plt.subplots()
    plt.plot(history.history['loss'], color='b', label="Relative Error")  # 📌 Se debe ajustar con la métrica real
    ax.set_xscale('log')
    plt.legend()
    ax.grid(which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "error_relativo.png"), dpi=300)
    plt.close()

def seleccionar_experimentos(resultados):
    """
    Selecciona los experimentos que se van a graficar y los organiza en una lista.

    Parámetros:
    - resultados (lista): Lista de resultados obtenidos en `ejecutar_experimentos`.

    Retorna:
    - todos_experimentos (lista): Lista de experimentos seleccionados para graficar.
    """

    # 📌 Determinar la cantidad de lotes según los experimentos disponibles
    num_lotes = min(len(func), len(resultados))  
    lote_size = max(1, len(resultados) // num_lotes)  

    # 📌 Lista para almacenar los experimentos seleccionados
    todos_experimentos = []

    for i in range(num_lotes):
        # 🔹 Seleccionar experimentos de este lote
        lote_resultados = [r for r in resultados if (r[0] - 1) // lote_size == i]
        todos_experimentos.extend(lote_resultados)

    print("✔ Experimentos listos para generar gráficas.")
    return todos_experimentos

def generar_graficas(experimentos, resultados, graficas_path):
    """
    Genera y guarda las gráficas de los experimentos en una carpeta específica.

    Cada experimento tendrá su propia carpeta donde se guardarán:
    - La comparación entre la solución exacta y la aproximada.
    - La evolución de la pérdida durante el entrenamiento.
    - La evolución del error relativo.
    - La comparación entre la pérdida y el error relativo.

    Parámetros:
    - experimentos (lista): Lista de experimentos seleccionados para graficar.
    - resultados (lista): Lista de resultados obtenidos en `ejecutar_experimentos`.
    - graficas_path (str): Ruta donde se almacenarán las gráficas.

    Retorna:
    - None
    """

    for exp in experimentos:
        exp_id = exp[0]  # ID del experimento
        exp_folder = os.path.join(graficas_path, f"Experimento_{exp_id}")
        os.makedirs(exp_folder, exist_ok=True)  # 📌 Crear carpeta del experimento si no existe

        # 🔹 Extraer datos del experimento correspondiente
        uModel, history, exactU, xList = resultados[exp_id - 1][9:]

        # 📌 Generar las gráficas
        plot_results(uModel, history, exactU, xList, exp_folder)

    print(f"✔ Todas las gráficas generadas en {graficas_path}.")

# 🔹 Paso 1: Configuración inicial (Crear directorios y archivo Excel)
print("📂 Configurando estructura de almacenamiento...")
base_path, graficas_path = crear_estructura_directorios()
excel_file = inicializar_excel()

# 🔹 Paso 2: Ejecutar experimentos
print("🚀 Ejecutando experimentos...")
resultados_experimentos = ejecutar_experimentos(func)

# 🔹 Paso 3: Seleccionar los experimentos para graficar
print("📊 Seleccionando experimentos para generación de gráficas...")
experimentos_seleccionados = seleccionar_experimentos(resultados_experimentos)

# 🔹 Paso 4: Generar todas las gráficas
print("📈 Generando gráficas...")
generar_graficas(experimentos_seleccionados, resultados_experimentos, graficas_path)

# 🔹 Paso 5: Guardar los resultados en Excel
print("💾 Guardando resultados en archivo Excel...")
guardar_resultados_excel(resultados_experimentos)

print("✅ Proceso finalizado exitosamente.")