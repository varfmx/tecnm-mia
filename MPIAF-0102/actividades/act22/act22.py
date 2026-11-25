# =============================================================================
# Actividad 22: "Crea tu Primer Reconocedor de Dígitos"
# Tema 4.4. Redes neuronales y aprendizaje profundo
#
# Dependencias (en Colab ya vienen instaladas):  pip install tensorflow scikit-learn pandas matplotlib seaborn scipy
# MNIST se descarga solo la primera vez (~11 MB). En CPU el script completo tarda unos 25 minutos.
# =============================================================================
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.datasets import mnist

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
pd.set_option("display.width", 220)

SEMILLA = 42
SEMILLAS = (0, 1, 2)                                  # repeticiones de cada experimento (el entrenamiento es aleatorio)
RAPIDO = os.environ.get("ACT22_RAPIDO") == "1"        # modo de depuración: 1 época, 1 semilla
if RAPIDO:
    SEMILLAS = (0,)
AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
IMG = os.path.join(AQUI, "img")
os.makedirs(IMG, exist_ok=True)
sns.set_theme(style="whitegrid")
tf.keras.utils.set_random_seed(SEMILLA)


def epocas(n):
    return 1 if RAPIDO else n


# =============================================================================
# DATOS (código del tema)
# =============================================================================
def cargar_y_preparar_datos():
    """Carga y prepara el dataset MNIST de dígitos escritos a mano"""
    print("Cargando datos de dígitos escritos a mano...")
    (X_entrenamiento, y_entrenamiento), (X_prueba, y_prueba) = mnist.load_data()
    print(f"Datos de entrenamiento: {X_entrenamiento.shape[0]} imágenes")
    print(f"Datos de prueba: {X_prueba.shape[0]} imágenes")
    print(f"Tamaño de cada imagen: {X_entrenamiento.shape[1]}x{X_entrenamiento.shape[2]} píxeles")

    # Normalizar los píxeles (de 0-255 a 0-1)
    X_entrenamiento = X_entrenamiento.astype("float32") / 255
    X_prueba = X_prueba.astype("float32") / 255

    # Reformatear para la red neuronal (de 28x28 a 784 valores)
    X_entrenamiento = X_entrenamiento.reshape((60000, 784))
    X_prueba = X_prueba.reshape((10000, 784))
    return (X_entrenamiento, y_entrenamiento), (X_prueba, y_prueba)


(X_entrenamiento, y_entrenamiento), (X_prueba, y_prueba) = cargar_y_preparar_datos()
print("Dígitos por clase (entrenamiento):", np.bincount(y_entrenamiento).tolist())
print("Dígitos por clase (prueba):       ", np.bincount(y_prueba).tolist())
print(f"Píxeles: mínimo {X_entrenamiento.min():.1f}, máximo {X_entrenamiento.max():.1f}, "
      f"promedio {X_entrenamiento.mean():.3f}; {(X_entrenamiento == 0).mean():.1%} de los píxeles es fondo (0)")

fig, ejes = plt.subplots(4, 10, figsize=(11, 4.6))
for d in range(10):
    ids = np.where(y_entrenamiento == d)[0][:4]
    for f, i in enumerate(ids):
        ejes[f, d].imshow(X_entrenamiento[i].reshape(28, 28), cmap="gray")
        ejes[f, d].axis("off")
        if f == 0:
            ejes[f, d].set_title(f"Dígito {d}", fontsize=9)
plt.tight_layout()
plt.savefig(f"{IMG}/muestras.png", dpi=200)
plt.close()


# =============================================================================
# TAREA 1: EXPERIMENTAR CON ARQUITECTURAS
# =============================================================================
# --- 1a. Función base del tema, sin cambios -----------------------------------
def experimentar_con_arquitecturas_base():
    """Plantilla para experimentar con diferentes redes"""
    arquitecturas = [
        {"capas": [64], "nombre": "Red Pequeña"},
        {"capas": [128, 64], "nombre": "Red Mediana"},
        {"capas": [256, 128, 64], "nombre": "Red Grande"}
    ]
    resultados = []
    for arq in arquitecturas:
        print(f"\n Probando: {arq['nombre']}")
        # Construir modelo
        modelo = models.Sequential()
        modelo.add(layers.Dense(arq['capas'][0], activation='relu', input_shape=(784,)))
        for neuronas in arq['capas'][1:]:
            modelo.add(layers.Dense(neuronas, activation='relu'))
        modelo.add(layers.Dense(10, activation='softmax'))
        # Compilar y entrenar
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        historia = modelo.fit(X_entrenamiento, y_entrenamiento, epochs=epocas(3), batch_size=128,
                              validation_split=0.2, verbose=0)
        # Evaluar
        pérdida, precisión = modelo.evaluate(X_prueba, y_prueba, verbose=0)
        resultados.append({"arquitectura": arq['nombre'], "precisión": precisión})
        print(f"Precisión: {precisión:.1%}")
    return resultados


print("Función base del tema (3 épocas, una sola corrida, evaluada en prueba):")
resultados_base = experimentar_con_arquitecturas_base()


# --- 1b. Versión ampliada: varias semillas, métrica de validación -------------
def construir_mlp(capas, lr=1e-3, optimizador="adam", dropout=0.0, momentum=0.0):
    """Red densa: Dense(relu) por cada tamaño en `capas`, y una capa softmax de 10 salidas."""
    modelo = models.Sequential()
    modelo.add(layers.Input(shape=(784,)))
    for neuronas in capas:
        modelo.add(layers.Dense(neuronas, activation="relu"))
        if dropout > 0:
            modelo.add(layers.Dropout(dropout))
    modelo.add(layers.Dense(10, activation="softmax"))
    opt = {"adam": lambda: optimizers.Adam(lr),
           "sgd": lambda: optimizers.SGD(lr, momentum=momentum),
           "rmsprop": lambda: optimizers.RMSprop(lr)}[optimizador]()
    modelo.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def entrenar(modelo, n_epocas, batch_size=128, paciencia=None):
    """Entrena con 20 % de validación (los últimos 12 000 ejemplos de entrenamiento). La prueba no se toca."""
    cbs = []
    if paciencia:
        cbs.append(callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True))
    return modelo.fit(X_entrenamiento, y_entrenamiento, epochs=epocas(n_epocas), batch_size=batch_size,
                      validation_split=0.2, verbose=0, callbacks=cbs)


def correr(nombre, n_epocas=10, batch_size=128, **hp):
    """Entrena la misma configuración con cada semilla y resume la exactitud en validación."""
    val, ent, tiempos, params = [], [], [], 0
    for s in SEMILLAS:
        tf.keras.utils.set_random_seed(s)
        modelo = construir_mlp(**hp)
        t0 = time.time()
        h = entrenar(modelo, n_epocas, batch_size)
        tiempos.append(time.time() - t0)
        val.append(h.history["val_accuracy"][-1])
        ent.append(h.history["accuracy"][-1])
        params = modelo.count_params()
    return {"configuración": nombre, "parámetros": params, "entrenamiento": np.mean(ent),
            "validación": np.mean(val), "desv.": np.std(val), "tiempo (s)": np.mean(tiempos)}


def tabla(filas):
    df = pd.DataFrame(filas).set_index("configuración")
    return df.round({"entrenamiento": 4, "validación": 4, "desv.": 4, "tiempo (s)": 1})


EPOCAS_BASE = 10
print(f"\nArquitecturas (media de {len(SEMILLAS)} semillas, {EPOCAS_BASE} épocas, Adam 1e-3, lotes de 128; solo validación):")
arquitecturas = [
    ("Sin capas ocultas (softmax)", []),
    ("Red Pequeña [64]", [64]),
    ("Red Mediana [128, 64]", [128, 64]),
    ("Red Grande [256, 128, 64]", [256, 128, 64]),
    ("Ancha 1 capa [16]", [16]),
    ("Ancha 1 capa [32]", [32]),
    ("Ancha 1 capa [256]", [256]),
    ("Ancha 1 capa [1024]", [1024]),
    ("2 capas [128, 128]", [128, 128]),
    ("3 capas [128, 128, 128]", [128, 128, 128]),
    ("4 capas [128] x 4", [128] * 4),
    ("6 capas [128] x 6", [128] * 6),
    ("Embudo [512, 256, 128, 64]", [512, 256, 128, 64]),
]
res_arq = tabla([correr(n, EPOCAS_BASE, capas=c) for n, c in arquitecturas])
print(res_arq.to_string())
mejor_arq = res_arq["validación"].idxmax()
print(f"\nMejor arquitectura por exactitud de validación: {mejor_arq}")

fig, ejes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
orden = res_arq.sort_values("validación")
ejes[0].barh(orden.index, orden["validación"] * 100, xerr=orden["desv."] * 100, color="#4C78A8", capsize=3)
ejes[0].set_xlim(orden["validación"].min() * 100 - 1, orden["validación"].max() * 100 + 0.6)
ejes[0].set_xlabel("Exactitud en validación (%)")
ejes[0].set_title("Exactitud por arquitectura (media ± desv. de las semillas)")
ejes[1].scatter(res_arq["parámetros"], res_arq["validación"] * 100, color="#E45756", s=45)
for n, f in res_arq.iterrows():
    ejes[1].annotate(n.split("[")[0].strip()[:18] or "softmax", (f["parámetros"], f["validación"] * 100), fontsize=6.5,
                     xytext=(3, 3), textcoords="offset points")
ejes[1].set_xscale("log")
ejes[1].set_xlabel("Parámetros entrenables (escala logarítmica)")
ejes[1].set_ylabel("Exactitud en validación (%)")
ejes[1].set_title("Tamaño del modelo contra desempeño")
plt.tight_layout()
plt.savefig(f"{IMG}/arquitecturas.png", dpi=200)
plt.close()


# =============================================================================
# TAREA 2: OPTIMIZAR HIPERPARÁMETROS
# =============================================================================
# Se parte de la Red Mediana del tema y se mueve un hiperparámetro a la vez.
BASE = dict(capas=[128, 64], lr=1e-3, optimizador="adam", dropout=0.0, momentum=0.0)
print(f"\nHiperparámetros: un factor a la vez sobre la Red Mediana (media de {len(SEMILLAS)} semillas, {EPOCAS_BASE} épocas):")

print("\n--- Tasa de aprendizaje (Adam) ---")
res_lr = tabla([correr(f"lr = {lr:g}", EPOCAS_BASE, **{**BASE, "lr": lr}) for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)])
print(res_lr.to_string())

print("\n--- Optimizador ---")
opts = [("SGD lr=0.01", dict(optimizador="sgd", lr=0.01)),
        ("SGD lr=0.1", dict(optimizador="sgd", lr=0.1)),
        ("SGD lr=0.01 + momentum 0.9", dict(optimizador="sgd", lr=0.01, momentum=0.9)),
        ("RMSprop lr=0.001", dict(optimizador="rmsprop", lr=1e-3)),
        ("Adam lr=0.001", dict(optimizador="adam", lr=1e-3))]
res_opt = tabla([correr(n, EPOCAS_BASE, **{**BASE, **h}) for n, h in opts])
print(res_opt.to_string())

print("\n--- Tamaño de lote ---")
res_bs = tabla([correr(f"lote = {b}", EPOCAS_BASE, batch_size=b, **BASE) for b in (32, 64, 128, 256, 512)])
print(res_bs.to_string())

print("\n--- Dropout (con 256, 128 neuronas para que haya capacidad que regularizar) ---")
res_do = tabla([correr(f"dropout = {d}", EPOCAS_BASE, **{**BASE, "capas": [256, 128], "dropout": d}) for d in (0.0, 0.1, 0.2, 0.3, 0.5)])
print(res_do.to_string())

print("\n--- Número de épocas: curvas de una red grande sin regularizar, 30 épocas ---")
tf.keras.utils.set_random_seed(SEMILLA)
modelo_largo = construir_mlp([512, 256, 128, 64])
h_largo = entrenar(modelo_largo, 30).history
mejor_ep = int(np.argmin(h_largo["val_loss"])) + 1
print(f"La pérdida de validación es mínima en la época {mejor_ep} ({min(h_largo['val_loss']):.4f}); "
      f"al final llega a {h_largo['val_loss'][-1]:.4f} mientras la exactitud de entrenamiento sube a {h_largo['accuracy'][-1]:.4f} "
      f"(validación {h_largo['val_accuracy'][-1]:.4f})")

fig, ejes = plt.subplots(2, 2, figsize=(11, 7.5))
ax = ejes[0, 0]
ax.errorbar(np.arange(len(res_lr)), res_lr["validación"] * 100, yerr=res_lr["desv."] * 100, marker="o", color="#4C78A8", capsize=3)
ax.set_xticks(np.arange(len(res_lr)))
ax.set_xticklabels([i.replace("lr = ", "") for i in res_lr.index], fontsize=8)
ax.set_xlabel("Tasa de aprendizaje (Adam)")
ax.set_ylabel("Exactitud en validación (%)")
ax.set_title("Tasa de aprendizaje")
ax = ejes[0, 1]
ax.barh(res_opt.index, res_opt["validación"] * 100, xerr=res_opt["desv."] * 100, color="#72B7B2", capsize=3)
ax.set_xlim(res_opt["validación"].min() * 100 - 1, 100)
ax.set_xlabel("Exactitud en validación (%)")
ax.set_title("Optimizador")
ax.tick_params(axis="y", labelsize=8)
ax = ejes[1, 0]
ax.errorbar(np.arange(len(res_bs)), res_bs["validación"] * 100, yerr=res_bs["desv."] * 100, marker="o", color="#E45756", capsize=3, label="exactitud")
ax.set_xticks(np.arange(len(res_bs)))
ax.set_xticklabels([i.replace("lote = ", "") for i in res_bs.index])
ax.set_xlabel("Tamaño de lote")
ax.set_ylabel("Exactitud en validación (%)")
ax.set_title("Tamaño de lote (10 épocas)")
ax2 = ax.twinx()
ax2.plot(np.arange(len(res_bs)), res_bs["tiempo (s)"], marker="s", color="#999999", linestyle="--", label="tiempo")
ax2.set_ylabel("Tiempo de entrenamiento (s)")
ax2.grid(False)
ax = ejes[1, 1]
ax.errorbar(np.arange(len(res_do)), res_do["validación"] * 100, yerr=res_do["desv."] * 100, marker="o", color="#F58518", capsize=3, label="validación")
ax.plot(np.arange(len(res_do)), res_do["entrenamiento"] * 100, marker="o", color="#54A24B", linestyle="--", label="entrenamiento")
ax.set_xticks(np.arange(len(res_do)))
ax.set_xticklabels([i.replace("dropout = ", "") for i in res_do.index])
ax.set_xlabel("Tasa de dropout")
ax.set_ylabel("Exactitud (%)")
ax.set_title("Dropout")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG}/hiperparametros.png", dpi=200)
plt.close()

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(1, len(h_largo["loss"]) + 1)
ejes[0].plot(x, h_largo["loss"], "b-", label="Pérdida entrenamiento")
ejes[0].plot(x, h_largo["val_loss"], "r-", label="Pérdida validación")
ejes[0].axvline(mejor_ep, color="gray", linestyle=":", label=f"mínimo de validación (época {mejor_ep})")
ejes[0].set_xlabel("Época")
ejes[0].set_ylabel("Pérdida")
ejes[0].set_title("Pérdida")
ejes[0].legend(fontsize=8)
ejes[1].plot(x, np.array(h_largo["accuracy"]) * 100, "b-", label="Entrenamiento")
ejes[1].plot(x, np.array(h_largo["val_accuracy"]) * 100, "r-", label="Validación")
ejes[1].set_ylim(96, 100.2)
ejes[1].set_xlabel("Época")
ejes[1].set_ylabel("Exactitud (%)")
ejes[1].set_title("Exactitud")
ejes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG}/curvas.png", dpi=200)
plt.close()


# --- 2b. Búsqueda aleatoria combinando todo -------------------------------------
N_PRUEBAS = 3 if RAPIDO else 14
print(f"\n--- Búsqueda aleatoria: {N_PRUEBAS} combinaciones, 1 semilla, hasta 25 épocas con parada temprana (paciencia 3) ---")
rng = np.random.default_rng(SEMILLA)
busqueda = []
for i in range(N_PRUEBAS):
    n_capas = int(rng.integers(1, 5))
    ancho = int(rng.choice([64, 128, 256, 512]))
    capas = [max(32, ancho // (2 ** j)) for j in range(n_capas)]        # cada capa mide la mitad de la anterior
    lr = float(rng.choice([3e-4, 1e-3, 3e-3]))
    lote = int(rng.choice([64, 128, 256]))
    dropout = float(rng.choice([0.0, 0.1, 0.2, 0.3]))
    tf.keras.utils.set_random_seed(SEMILLA)
    m = construir_mlp(capas, lr=lr, dropout=dropout)
    t0 = time.time()
    h = entrenar(m, 25, lote, paciencia=3).history
    mejor = int(np.argmin(h["val_loss"]))
    busqueda.append({"capas": str(capas), "lr": lr, "lote": lote, "dropout": dropout, "parámetros": m.count_params(),
                     "época": mejor + 1, "val_exactitud": h["val_accuracy"][mejor], "val_pérdida": h["val_loss"][mejor],
                     "tiempo (s)": time.time() - t0})
    print(f"  {i + 1:2d}/{N_PRUEBAS}: capas={capas}, lr={lr:g}, lote={lote}, dropout={dropout} -> val {h['val_accuracy'][mejor]:.4f}")
df_busqueda = pd.DataFrame(busqueda).sort_values("val_exactitud", ascending=False).reset_index(drop=True)
print("\nResultados de la búsqueda (ordenados por exactitud de validación):")
print(df_busqueda.round({"val_exactitud": 4, "val_pérdida": 4, "tiempo (s)": 1}).to_string())
MEJOR_HP = df_busqueda.iloc[0]
MEJOR_CAPAS = eval(MEJOR_HP["capas"])
print(f"\nMejor combinación: capas={MEJOR_CAPAS}, lr={MEJOR_HP['lr']:g}, lote={int(MEJOR_HP['lote'])}, dropout={MEJOR_HP['dropout']}")

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
sc = ejes[0].scatter(df_busqueda["parámetros"], df_busqueda["val_exactitud"] * 100, c=np.log10(df_busqueda["lr"]), cmap="viridis", s=60)
ejes[0].set_xscale("log")
ejes[0].set_xlabel("Parámetros entrenables")
ejes[0].set_ylabel("Exactitud en validación (%)")
ejes[0].set_title("Búsqueda aleatoria (color: log10 de la tasa de aprendizaje)")
ejes[0].title.set_fontsize(9)
plt.colorbar(sc, ax=ejes[0])
sns.boxplot(data=df_busqueda.assign(exactitud=df_busqueda["val_exactitud"] * 100), x="dropout", y="exactitud", ax=ejes[1], color="#9ECAE9")
ejes[1].set_xlabel("Dropout")
ejes[1].set_ylabel("Exactitud en validación (%)")
ejes[1].set_title("Efecto del dropout en la búsqueda")
plt.tight_layout()
plt.savefig(f"{IMG}/busqueda.png", dpi=200)
plt.close()


# =============================================================================
# EVALUACIÓN FINAL EN PRUEBA (una sola vez por modelo)
# =============================================================================
def crear_cnn_para_imagenes():
    """Crea una Red Neuronal Convolucional para reconocer imágenes (código del tema)"""
    modelo = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


X_entrenamiento_cnn = X_entrenamiento.reshape(60000, 28, 28, 1)
X_prueba_cnn = X_prueba.reshape(10000, 28, 28, 1)


def final(nombre, construir, cnn=False, n_epocas=25, batch=128, paciencia=3):
    """Entrena con cada semilla (parada temprana en validación) y evalúa UNA vez en prueba."""
    accs, val, modelos = [], [], []
    X, Xp = (X_entrenamiento_cnn, X_prueba_cnn) if cnn else (X_entrenamiento, X_prueba)
    for s in SEMILLAS:
        tf.keras.utils.set_random_seed(s)
        m = construir()
        h = m.fit(X, y_entrenamiento, epochs=epocas(n_epocas), batch_size=batch, validation_split=0.2, verbose=0,
                  callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)])
        _, a = m.evaluate(Xp, y_prueba, verbose=0)
        accs.append(a)
        val.append(max(h.history["val_accuracy"]))
        modelos.append(m)
    return {"modelo": nombre, "parámetros": modelos[0].count_params(), "validación": np.mean(val),
            "prueba": np.mean(accs), "desv.": np.std(accs), "mínimo": np.min(accs), "máximo": np.max(accs)}, modelos


print(f"\nEvaluación final en el conjunto de prueba ({len(SEMILLAS)} semillas, parada temprana en validación):")
# La red mediana "tal cual" del tema se entrena 3 épocas sin parada temprana:
accs_med = []
for s in SEMILLAS:
    tf.keras.utils.set_random_seed(s)
    m = construir_mlp([128, 64])
    m.fit(X_entrenamiento, y_entrenamiento, epochs=epocas(3), batch_size=128, validation_split=0.2, verbose=0)
    accs_med.append(m.evaluate(X_prueba, y_prueba, verbose=0)[1])
fila_med = {"modelo": "Red Mediana del tema (3 épocas)", "parámetros": m.count_params(), "validación": np.nan,
            "prueba": np.mean(accs_med), "desv.": np.std(accs_med), "mínimo": np.min(accs_med), "máximo": np.max(accs_med)}

fila_mlp, mod_mlp = final("Mejor red densa de la búsqueda",
                          lambda: construir_mlp(MEJOR_CAPAS, lr=float(MEJOR_HP["lr"]), dropout=float(MEJOR_HP["dropout"])),
                          batch=int(MEJOR_HP["lote"]))
fila_cnn, mod_cnn = final("CNN del tema", crear_cnn_para_imagenes, cnn=True, n_epocas=15)
res_final = pd.DataFrame([fila_med, fila_mlp, fila_cnn]).set_index("modelo")
print(res_final.round(4).to_string())

# Modelos que se analizan en la Tarea 3 (primera semilla de cada uno)
modelo_mlp, modelo_cnn = mod_mlp[0], mod_cnn[0]
p_mlp = modelo_mlp.predict(X_prueba, verbose=0)
p_cnn = modelo_cnn.predict(X_prueba_cnn, verbose=0)
pred_mlp, pred_cnn = p_mlp.argmax(1), p_cnn.argmax(1)
print(f"\nModelos analizados (semilla {SEMILLAS[0]}): red densa {np.mean(pred_mlp == y_prueba):.4f}, CNN {np.mean(pred_cnn == y_prueba):.4f}")


# =============================================================================
# TAREA 3: ANALIZAR ERRORES
# =============================================================================
print("\n--- Métricas por dígito (red densa óptima, conjunto de prueba) ---")
print(classification_report(y_prueba, pred_mlp, digits=4))

cm = confusion_matrix(y_prueba, pred_mlp)
cm_cnn = confusion_matrix(y_prueba, pred_cnn)
fig, ejes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, c, titulo in ((ejes[0], cm, "Red densa óptima"), (ejes[1], cm_cnn, "CNN del tema")):
    anot = np.where(c == 0, "", c.astype(str))
    sns.heatmap(c, annot=anot, fmt="", cmap="Blues", cbar=False, ax=ax, linewidths=0.4, linecolor="white",
                vmax=60, xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel("Dígito predicho")
    ax.set_ylabel("Dígito real")
    ax.set_title(f"{titulo}: {int(c.sum() - np.trace(c))} errores de {int(c.sum())} (el color satura en 60)")
    ax.title.set_fontsize(10)
plt.tight_layout()
plt.savefig(f"{IMG}/matriz_confusion.png", dpi=200)
plt.close()

# Pares más confundidos
def pares_confundidos(c, n=10):
    filas = [(i, j, c[i, j]) for i in range(10) for j in range(10) if i != j and c[i, j] > 0]
    filas.sort(key=lambda t: -t[2])
    return pd.DataFrame(filas[:n], columns=["real", "predicho", "errores"])


print("--- Confusiones más frecuentes (real -> predicho) ---")
top_mlp = pares_confundidos(cm, 10)
print(top_mlp.to_string(index=False))
print("\n--- Lo mismo con la CNN ---")
print(pares_confundidos(cm_cnn, 6).to_string(index=False))

err_mlp = np.where(pred_mlp != y_prueba)[0]
err_cnn = np.where(pred_cnn != y_prueba)[0]
print(f"\nErrores: red densa {len(err_mlp)}, CNN {len(err_cnn)}; en común {len(set(err_mlp) & set(err_cnn))}; "
      f"la CNN corrige {len(set(err_mlp) - set(err_cnn))} de los {len(err_mlp)} errores de la red densa "
      f"y comete {len(set(err_cnn) - set(err_mlp))} nuevos")
por_digito = pd.DataFrame({"errores red densa": cm.sum(1) - np.diag(cm), "errores CNN": cm_cnn.sum(1) - np.diag(cm_cnn)})
por_digito.index.name = "dígito real"
print(por_digito.T.to_string())

# Galería de errores de las confusiones principales
fig, ejes = plt.subplots(4, 6, figsize=(11, 8))
for f, (_, fila) in enumerate(top_mlp.head(4).iterrows()):
    ids = np.where((y_prueba == fila["real"]) & (pred_mlp == fila["predicho"]))[0]
    ids = ids[np.argsort(-p_mlp[ids, fila["predicho"]])][:6]       # los más "seguros" de su error
    for c in range(6):
        ax = ejes[f, c]
        ax.axis("off")
        if c < len(ids):
            ax.imshow(X_prueba[ids[c]].reshape(28, 28), cmap="gray")
            ax.set_title(f"real {fila['real']} -> pred {fila['predicho']}\np = {p_mlp[ids[c], fila['predicho']]:.2f}", fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG}/errores.png", dpi=200)
plt.close()

# ¿Por qué? 1) Parecido entre dígitos medido con la imagen promedio de cada clase
protos = np.stack([X_entrenamiento[y_entrenamiento == d].mean(0) for d in range(10)])
protos_n = protos / np.linalg.norm(protos, axis=1, keepdims=True)
sim = protos_n @ protos_n.T
pares = [(i, j) for i in range(10) for j in range(i + 1, 10)]
sim_par = np.array([sim[i, j] for i, j in pares])
err_par_mlp = np.array([cm[i, j] + cm[j, i] for i, j in pares])
err_par_cnn = np.array([cm_cnn[i, j] + cm_cnn[j, i] for i, j in pares])
rho_mlp, p_rho_mlp = spearmanr(sim_par, err_par_mlp)
rho_cnn, p_rho_cnn = spearmanr(sim_par, err_par_cnn)
print(f"\nParecido entre dígitos (coseno de las imágenes promedio) contra errores por par (45 pares):")
print(f"  red densa: rho de Spearman = {rho_mlp:.2f} (p = {p_rho_mlp:.1e});  CNN: rho = {rho_cnn:.2f} (p = {p_rho_cnn:.1e})")
orden_sim = np.argsort(-sim_par)[:6]
print("  Pares más parecidos:", ", ".join(f"{pares[k][0]}-{pares[k][1]} ({sim_par[k]:.3f}, {err_par_mlp[k]} errores)" for k in orden_sim))
print("  Pares con más errores:", ", ".join(f"{pares[k][0]}-{pares[k][1]} ({err_par_mlp[k]})" for k in np.argsort(-err_par_mlp)[:6]))

fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6))
ejes[0].scatter(sim_par, err_par_mlp, color="#4C78A8", s=40)
for k in np.argsort(-err_par_mlp)[:7]:
    ejes[0].annotate(f"{pares[k][0]}-{pares[k][1]}", (sim_par[k], err_par_mlp[k]), xytext=(4, 3), textcoords="offset points", fontsize=8)
ejes[0].set_xlabel("Parecido entre las imágenes promedio de los dos dígitos (coseno)")
ejes[0].set_ylabel("Errores entre los dos dígitos (ambos sentidos)")
ejes[0].set_title(f"Red densa: rho de Spearman = {rho_mlp:.2f}")
sns.heatmap(sim, annot=True, fmt=".2f", cmap="Reds", cbar=False, ax=ejes[1], annot_kws={"fontsize": 6.5},
            xticklabels=range(10), yticklabels=range(10))
ejes[1].set_title("Parecido entre imágenes promedio")
plt.tight_layout()
plt.savefig(f"{IMG}/similitud.png", dpi=200)
plt.close()

# 2) Confianza de los errores y segunda opción
conf_mlp = p_mlp.max(1)
ok = pred_mlp == y_prueba
top2 = np.argsort(-p_mlp, axis=1)[:, :2]
top2_acc = np.mean([y_prueba[i] in top2[i] for i in range(len(y_prueba))])
print(f"\nConfianza (probabilidad máxima) de la red densa: aciertos {conf_mlp[ok].mean():.3f}, errores {conf_mlp[~ok].mean():.3f} (mediana {np.median(conf_mlp[~ok]):.3f})")
print(f"Errores con confianza > 0.9: {np.mean(conf_mlp[~ok] > 0.9):.1%}; exactitud top-2: {top2_acc:.4f} (top-1: {ok.mean():.4f})")
print("\nRevisión humana de los casos poco seguros (red densa):")
filas = []
for umbral in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
    aceptados = conf_mlp >= umbral
    filas.append({"umbral": umbral, "automático": aceptados.mean(), "exactitud de lo automático": ok[aceptados].mean(),
                  "errores que pasan": int((~ok & aceptados).sum()), "a revisión humana": 1 - aceptados.mean()})
print(pd.DataFrame(filas).round(4).to_string(index=False))

# 3) Impacto en un formulario: un campo de n dígitos sale mal si falla alguno (se supone independencia)
acc = ok.mean()
print("\nProbabilidad de que un campo de n dígitos tenga al menos un error (independencia):")
print("  " + ", ".join(f"n={n}: {1 - acc ** n:.1%}" for n in (4, 8, 10, 16)))

# 4) Mirada a los dígitos más difíciles: errores que cometen las dos redes
compartidos = np.array(sorted(set(err_mlp) & set(err_cnn)))
print(f"\nErrores comunes a las dos redes: {len(compartidos)}; etiquetas reales: {np.bincount(y_prueba[compartidos], minlength=10).tolist()}")
fig, ejes = plt.subplots(3, 8, figsize=(11, 4.8))
for k, ax in enumerate(ejes.ravel()):
    ax.axis("off")
    if k < len(compartidos):
        i = compartidos[k]
        ax.imshow(X_prueba[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"real {y_prueba[i]}, densa {pred_mlp[i]}, CNN {pred_cnn[i]}", fontsize=7)
plt.tight_layout()
plt.savefig(f"{IMG}/errores_comunes.png", dpi=200)
plt.close()

print("\nListo. Figuras guardadas en", IMG)
