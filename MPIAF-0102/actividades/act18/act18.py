# =============================================================================
# Actividad 18: Proyecto "Clasificador de Especies de Animales"
# Tema 4.2. Visión artificial
#
# Dependencias (en Colab anteponer "!"):  pip install -U tensorflow scikit-learn pandas matplotlib seaborn pillow
# El conjunto de datos se descarga solo la primera vez (unas 600 imágenes, ~80 MB).
# =============================================================================
import json
import os
import random
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import applications, callbacks, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEMILLA = 42
RAPIDO = os.environ.get("ACT18_RAPIDO") == "1"      # modo de depuración: 1 época, 2 pliegues
AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
IMG = os.path.join(AQUI, "img")
DATOS = os.path.join(AQUI, "datos")
DIR_ENTRENAMIENTO = os.path.join(DATOS, "entrenamiento")
DIR_PRUEBA = os.path.join(DATOS, "prueba")
os.makedirs(IMG, exist_ok=True)
sns.set_theme(style="whitegrid")

CLASES = ["cat", "dog", "elephant", "horse", "lion", "panda", "penguin", "sheep", "tiger", "zebra"]
OBJETIVO = (224, 224)
BATCH = 32
N_PRUEBA = 10            # imágenes por especie reservadas para la prueba final
HF = "https://huggingface.co"
REPO = "mertcobanov/animals"       # conjunto público de 90 especies, 60 imágenes por especie

random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.keras.utils.set_random_seed(SEMILLA)


# =============================================================================
# TAREA 1: PREPARACIÓN DE DATOS
# =============================================================================
def _listar(clase):
    with urllib.request.urlopen(f"{HF}/api/datasets/{REPO}/tree/main/animals/{clase}", timeout=60) as r:
        items = json.load(r)
    return sorted(i["path"] for i in items if i["type"] == "file" and i["path"].lower().endswith((".jpg", ".jpeg", ".png")))


def _bajar(par):
    ruta_remota, destino = par
    if os.path.exists(destino):
        return
    for intento in range(3):
        try:
            urllib.request.urlretrieve(f"{HF}/datasets/{REPO}/resolve/main/{ruta_remota}", destino)
            return
        except Exception:
            time.sleep(1 + intento)
    raise RuntimeError(f"No se pudo descargar {ruta_remota}")


def preparar_dataset():
    """Descarga las especies elegidas y las organiza en carpetas por clase:
    datos/entrenamiento/<clase>/*.jpg (para entrenar y validar) y datos/prueba/<clase>/*.jpg (reservadas)."""
    rng = random.Random(SEMILLA)
    tareas = []
    for clase in CLASES:
        rutas = _listar(clase)
        rng.shuffle(rutas)
        for i, ruta in enumerate(rutas):
            carpeta = DIR_PRUEBA if i < N_PRUEBA else DIR_ENTRENAMIENTO
            os.makedirs(os.path.join(carpeta, clase), exist_ok=True)
            tareas.append((ruta, os.path.join(carpeta, clase, os.path.basename(ruta))))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(_bajar, tareas))
    # Se descartan archivos ilegibles (hay imágenes corruptas en algunos conjuntos públicos)
    for _, destino in tareas:
        try:
            with Image.open(destino) as im:
                im.verify()
        except Exception:
            os.remove(destino)


def contar(carpeta):
    return {c: len(os.listdir(os.path.join(carpeta, c))) for c in CLASES}


# ---- Código base del tema ---------------------------------------------------
def crear_generadores_datos(ruta_base, objetivo=(224, 224), batch_size=32):
    """
    Crea generadores de datos para entrenamiento y validación
    """
    datagen_entrenamiento = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    datagen_validacion = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    generador_entrenamiento = datagen_entrenamiento.flow_from_directory(
        ruta_base,
        target_size=objetivo,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    generador_validacion = datagen_validacion.flow_from_directory(
        ruta_base,
        target_size=objetivo,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return generador_entrenamiento, generador_validacion


print("=" * 70)
print("TAREA 1: PREPARACIÓN DE DATOS")
print("=" * 70)
preparar_dataset()
tabla = pd.DataFrame({"entrenamiento+validación": contar(DIR_ENTRENAMIENTO), "prueba": contar(DIR_PRUEBA)})
tabla.loc["TOTAL"] = tabla.sum()
print("Imágenes por especie:")
print(tabla.to_string())

gen_entrenamiento, gen_validacion = crear_generadores_datos(DIR_ENTRENAMIENTO, OBJETIVO, BATCH)
print("\nÍndices de clase:", gen_entrenamiento.class_indices)
print(f"Lotes de entrenamiento: {len(gen_entrenamiento)} (tamaño de lote {BATCH}); lotes de validación: {len(gen_validacion)}")
gen_prueba = ImageDataGenerator(rescale=1./255).flow_from_directory(
    DIR_PRUEBA, target_size=OBJETIVO, batch_size=BATCH, class_mode="categorical", shuffle=False)
N_CLASES = len(CLASES)
NOMBRES = list(gen_prueba.class_indices)          # orden alfabético, igual que las demás particiones

# Muestras del conjunto de datos
fig, axs = plt.subplots(2, 5, figsize=(11, 4.8))
for ax, clase in zip(axs.ravel(), CLASES):
    archivo = sorted(os.listdir(os.path.join(DIR_ENTRENAMIENTO, clase)))[0]
    ax.imshow(Image.open(os.path.join(DIR_ENTRENAMIENTO, clase, archivo)).convert("RGB").resize(OBJETIVO))
    ax.set_title(clase)
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{IMG}/muestras.png", dpi=150)
plt.close()

# Aumento de datos: una misma foto con transformaciones aleatorias
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
archivo = os.path.join(DIR_ENTRENAMIENTO, "tiger", sorted(os.listdir(os.path.join(DIR_ENTRENAMIENTO, "tiger")))[3])
original = np.asarray(Image.open(archivo).convert("RGB").resize(OBJETIVO), dtype="float32")
fig, axs = plt.subplots(1, 6, figsize=(13, 2.5))
axs[0].imshow(original.astype("uint8"))
axs[0].set_title("original")
for i, ax in enumerate(axs[1:]):
    ax.imshow(aug.random_transform(original, seed=SEMILLA + i).astype("uint8"))
    ax.set_title(f"aumento {i + 1}")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{IMG}/aumento.png", dpi=150)
plt.close()


# =============================================================================
# TAREA 2: ARQUITECTURA DEL MODELO
# =============================================================================
def crear_cnn(filtros=(32, 64, 128), dropout=0.5, batchnorm=False, lr=1e-3, unidades=128):
    """CNN personalizada: bloques Conv-(BN)-ReLU-MaxPool, GlobalAveragePooling y capa densa."""
    capas = [layers.Input((*OBJETIVO, 3))]
    for f in filtros:
        capas.append(layers.Conv2D(f, 3, padding="same", use_bias=not batchnorm))
        if batchnorm:
            capas.append(layers.BatchNormalization())
        capas += [layers.Activation("relu"), layers.MaxPooling2D()]
    capas += [layers.GlobalAveragePooling2D(), layers.Dense(unidades, activation="relu"),
              layers.Dropout(dropout), layers.Dense(N_CLASES, activation="softmax")]
    m = models.Sequential(capas)
    m.compile(optimizer=optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def crear_transfer(nombre="mobilenet", lr=1e-3, descongelar=0):
    """Transfer learning: base preentrenada en ImageNet + cabeza nueva.
    Las imágenes llegan en [0, 1] (rescale=1/255 del código base) y se adaptan al preprocesamiento de cada base."""
    entrada = layers.Input((*OBJETIVO, 3))
    if nombre == "mobilenet":
        base = applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(*OBJETIVO, 3))
        x = layers.Rescaling(2.0, offset=-1.0)(entrada)                       # [0,1] -> [-1,1]
    else:
        base = applications.ResNet50(include_top=False, weights="imagenet", input_shape=(*OBJETIVO, 3))
        x = layers.Lambda(lambda t: applications.resnet50.preprocess_input(t * 255.0))(entrada)
    base.trainable = descongelar > 0
    if descongelar > 0:
        for capa in base.layers[:-descongelar]:
            capa.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(N_CLASES, activation="softmax")(x)
    m = models.Model(entrada, salida)
    m.compile(optimizer=optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def entrenar(construir, epocas, paciencia=6, gen_ent=None, gen_val=None):
    """Entrena con parada temprana. Devuelve (modelo, historial, segundos)."""
    tf.keras.utils.set_random_seed(SEMILLA)
    modelo = construir()
    cb = [callbacks.EarlyStopping(monitor="val_accuracy", patience=paciencia, restore_best_weights=True)]
    t0 = time.perf_counter()
    h = modelo.fit(gen_ent or gen_entrenamiento, validation_data=gen_val or gen_validacion,
                   epochs=1 if RAPIDO else epocas, callbacks=cb, verbose=0)
    return modelo, h.history, time.perf_counter() - t0


def exactitud_prueba(modelo):
    p = modelo.predict(gen_prueba, verbose=0)
    return float((p.argmax(1) == gen_prueba.classes).mean()), p


print("\n" + "=" * 70)
print("TAREA 2: ARQUITECTURA DEL MODELO")
print("=" * 70)
arquitecturas = {
    "CNN-A (2 bloques)": (lambda: crear_cnn(filtros=(32, 64), dropout=0.3), 30),
    "CNN-B (3 bloques)": (lambda: crear_cnn(filtros=(32, 64, 128), dropout=0.5), 30),
    "CNN-C (3 bloques + BatchNorm)": (lambda: crear_cnn(filtros=(32, 64, 128), dropout=0.5, batchnorm=True), 30),
    "MobileNetV2 (base congelada)": (lambda: crear_transfer("mobilenet"), 15),
    "ResNet50 (base congelada)": (lambda: crear_transfer("resnet"), 15),
}
resultados, historias, modelos_entrenados = [], {}, {}
for nombre, (construir, epocas) in arquitecturas.items():
    modelo, h, seg = entrenar(construir, epocas)
    acc_prueba, _ = exactitud_prueba(modelo)
    mejor = int(np.argmax(h["val_accuracy"]))
    resultados.append({"modelo": nombre, "parámetros": modelo.count_params(), "épocas": len(h["loss"]),
                       "acc. entrenamiento": h["accuracy"][mejor], "acc. validación": h["val_accuracy"][mejor],
                       "acc. prueba": acc_prueba, "tiempo (s)": seg})
    historias[nombre], modelos_entrenados[nombre] = h, modelo
    print(f"  {nombre}: val={h['val_accuracy'][mejor]:.3f}  prueba={acc_prueba:.3f}  ({seg:.0f} s)")

# Ajuste fino (fine-tuning) de MobileNetV2: se entrena la cabeza y luego se descongelan las últimas capas
t0 = time.perf_counter()
tf.keras.utils.set_random_seed(SEMILLA)
ft = crear_transfer("mobilenet", lr=1e-3)
cb = [callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)]
h1 = ft.fit(gen_entrenamiento, validation_data=gen_validacion, epochs=1 if RAPIDO else 8, verbose=0)
base_mb = next(c for c in ft.layers if isinstance(c, models.Model))   # la base MobileNetV2
base_mb.trainable = True
for capa in base_mb.layers[:-30]:
    capa.trainable = False
ft.compile(optimizer=optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
h2 = ft.fit(gen_entrenamiento, validation_data=gen_validacion, epochs=1 if RAPIDO else 12, callbacks=cb, verbose=0)
seg = time.perf_counter() - t0
hist_ft = {k: h1.history[k] + h2.history[k] for k in h1.history}
mejor = int(np.argmax(h2.history["val_accuracy"]))
acc_prueba, _ = exactitud_prueba(ft)
nombre_ft = "MobileNetV2 (ajuste fino, 30 capas)"
resultados.append({"modelo": nombre_ft, "parámetros": ft.count_params(), "épocas": len(hist_ft["loss"]),
                   "acc. entrenamiento": h2.history["accuracy"][mejor], "acc. validación": h2.history["val_accuracy"][mejor],
                   "acc. prueba": acc_prueba, "tiempo (s)": seg})
historias[nombre_ft], modelos_entrenados[nombre_ft] = hist_ft, ft
print(f"  {nombre_ft}: val={h2.history['val_accuracy'][mejor]:.3f}  prueba={acc_prueba:.3f}  ({seg:.0f} s)")

comparacion = pd.DataFrame(resultados)
print("\nComparación de arquitecturas (validación = 20 % de entrenamiento; prueba = 100 imágenes reservadas):")
print(comparacion.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for nombre, h in historias.items():
    axs[0].plot(h["accuracy"], label=nombre)
    axs[1].plot(h["val_accuracy"], label=nombre)
axs[0].set_title("Exactitud de entrenamiento")
axs[1].set_title("Exactitud de validación")
for ax in axs:
    ax.set_xlabel("época")
axs[0].set_ylabel("exactitud")
axs[1].legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig(f"{IMG}/curvas.png", dpi=200)
plt.close()


# =============================================================================
# TAREA 3: EVALUACIÓN Y OPTIMIZACIÓN
# =============================================================================
print("\n" + "=" * 70)
print("TAREA 3: EVALUACIÓN Y OPTIMIZACIÓN")
print("=" * 70)

# Lista de las imágenes de entrenamiento+validación para la validación cruzada
archivos = np.array([os.path.join(DIR_ENTRENAMIENTO, c, f) for c in CLASES for f in sorted(os.listdir(os.path.join(DIR_ENTRENAMIENTO, c)))])
etiquetas = np.array([os.path.basename(os.path.dirname(a)) for a in archivos])
y_todas = np.array([NOMBRES.index(e) for e in etiquetas])
N_PLIEGUES = 2 if RAPIDO else 5
pliegues = list(StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=SEMILLA).split(archivos, y_todas))

# ---- (a) Validación cruzada de la mejor CNN personalizada (con aumento de datos) ----
mejor_cnn = comparacion[comparacion["modelo"].str.startswith("CNN")].sort_values("acc. validación", ascending=False).iloc[0]["modelo"]
construir_cnn = arquitecturas[mejor_cnn][0]
aug_cv = dict(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
acc_cnn_cv = []
for i, (tr, va) in enumerate(pliegues):
    df_tr = pd.DataFrame({"archivo": archivos[tr], "clase": etiquetas[tr]})
    df_va = pd.DataFrame({"archivo": archivos[va], "clase": etiquetas[va]})
    g_tr = ImageDataGenerator(**aug_cv).flow_from_dataframe(df_tr, x_col="archivo", y_col="clase", classes=NOMBRES,
                                                            target_size=OBJETIVO, batch_size=BATCH, class_mode="categorical")
    g_va = ImageDataGenerator(rescale=1./255).flow_from_dataframe(df_va, x_col="archivo", y_col="clase", classes=NOMBRES,
                                                                  target_size=OBJETIVO, batch_size=BATCH, class_mode="categorical", shuffle=False)
    _, h, _ = entrenar(construir_cnn, 30, gen_ent=g_tr, gen_val=g_va)
    acc_cnn_cv.append(max(h["val_accuracy"]))
    print(f"  CNN, pliegue {i + 1}/{N_PLIEGUES}: exactitud = {acc_cnn_cv[-1]:.3f}")

# ---- (b) Validación cruzada y optimización de hiperparámetros con MobileNetV2 congelada ----
# Como la base no se entrena, sus características (1280 por imagen) se calculan una sola vez
# y solo se entrena la cabeza de clasificación en cada pliegue (sin fuga de información).
entrada = layers.Input((*OBJETIVO, 3))
extractor = models.Model(entrada, layers.GlobalAveragePooling2D()(
    applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(*OBJETIVO, 3))(layers.Rescaling(2.0, offset=-1.0)(entrada))))


def caracteristicas(rutas):
    X = np.stack([np.asarray(Image.open(r).convert("RGB").resize(OBJETIVO), dtype="float32") / 255.0 for r in rutas])
    return extractor.predict(X, batch_size=32, verbose=0)


X_todas = caracteristicas(archivos)
print(f"\nCaracterísticas MobileNetV2 de las imágenes de entrenamiento: {X_todas.shape}")


def crear_cabeza(unidades=128, dropout=0.3, lr=1e-3):
    capas = [layers.Input((X_todas.shape[1],))]
    if unidades:
        capas.append(layers.Dense(unidades, activation="relu"))
    capas += [layers.Dropout(dropout), layers.Dense(N_CLASES, activation="softmax")]
    m = models.Sequential(capas)
    m.compile(optimizer=optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def cv_cabeza(**hp):
    accs = []
    for tr, va in pliegues:
        tf.keras.utils.set_random_seed(SEMILLA)
        m = crear_cabeza(**hp)
        m.fit(X_todas[tr], y_todas[tr], epochs=1 if RAPIDO else 30, batch_size=32, verbose=0)
        accs.append(m.evaluate(X_todas[va], y_todas[va], verbose=0)[1])
    return np.array(accs)


base_cv = cv_cabeza()
print(f"  MobileNetV2 + cabeza por defecto (128 unidades, dropout 0.3, lr 1e-3): {base_cv.mean():.3f} +/- {base_cv.std():.3f}")

busqueda = []
for unidades in [0, 128, 256]:
    for dropout in [0.2, 0.5]:
        for lr in [1e-2, 1e-3, 1e-4]:
            a = cv_cabeza(unidades=unidades, dropout=dropout, lr=lr)
            busqueda.append({"unidades": unidades, "dropout": dropout, "lr": lr, "exactitud CV": a.mean(), "desv.": a.std()})
busqueda = pd.DataFrame(busqueda).sort_values("exactitud CV", ascending=False).reset_index(drop=True)
print("\nBúsqueda en rejilla de hiperparámetros (validación cruzada de 5 pliegues), mejores 6 y peor:")
print(busqueda.head(6).round(4).to_string())
print(busqueda.tail(1).round(4).to_string())
mejor_hp = {"unidades": int(busqueda.loc[0, "unidades"]), "dropout": float(busqueda.loc[0, "dropout"]),
            "lr": float(busqueda.loc[0, "lr"])}
cv_mejor = cv_cabeza(**mejor_hp)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
sns.heatmap(busqueda.pivot_table(index=["unidades", "dropout"], columns="lr", values="exactitud CV"),
            annot=True, fmt=".3f", cmap="YlGn", ax=ax, cbar=False)
ax.set_title("Exactitud de validación cruzada")
plt.tight_layout()
plt.savefig(f"{IMG}/hiperparametros.png", dpi=200)
plt.close()

cv_resumen = pd.DataFrame([
    {"modelo": f"{mejor_cnn}", "exactitud CV": np.mean(acc_cnn_cv), "desv.": np.std(acc_cnn_cv)},
    {"modelo": "MobileNetV2 + cabeza por defecto", "exactitud CV": base_cv.mean(), "desv.": base_cv.std()},
    {"modelo": "MobileNetV2 + cabeza optimizada", "exactitud CV": cv_mejor.mean(), "desv.": cv_mejor.std()},
])
print(f"\nResumen de validación cruzada ({N_PLIEGUES} pliegues, {len(archivos)} imágenes):")
print(cv_resumen.round(3).to_string(index=False))
print(f"Hiperparámetros elegidos: {mejor_hp}")

# ---- (c) Modelo final: cabeza optimizada entrenada con todas las imágenes de entrenamiento ----
tf.keras.utils.set_random_seed(SEMILLA)
cabeza_final = crear_cabeza(**mejor_hp)
cabeza_final.fit(X_todas, y_todas, epochs=1 if RAPIDO else 30, batch_size=32, verbose=0)
rutas_prueba = np.array(gen_prueba.filepaths)
y_prueba = gen_prueba.classes
prob = cabeza_final.predict(caracteristicas(rutas_prueba), verbose=0)
pred = prob.argmax(1)
acc_final = float((pred == y_prueba).mean())
print(f"\nModelo final (MobileNetV2 + cabeza optimizada), exactitud en prueba: {acc_final:.3f}")

# Comparación final en prueba, incluyendo los demás modelos
print("\nExactitud en el conjunto de prueba (100 imágenes nuevas):")
final_tabla = comparacion[["modelo", "acc. prueba"]].copy()
final_tabla.loc[len(final_tabla)] = ["MobileNetV2 + cabeza optimizada (modelo final)", acc_final]
print(final_tabla.round(3).to_string(index=False))

# ---- Matriz de confusión ----
print("\nReporte de clasificación (modelo final, conjunto de prueba):")
print(classification_report(y_prueba, pred, target_names=NOMBRES, digits=3, zero_division=0))
cm = confusion_matrix(y_prueba, pred)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=NOMBRES, yticklabels=NOMBRES, cbar=False, ax=ax)
ax.set_xlabel("predicho")
ax.set_ylabel("real")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{IMG}/matriz_confusion.png", dpi=200)
plt.close()
errores = [(NOMBRES[i], NOMBRES[j], int(cm[i, j])) for i in range(N_CLASES) for j in range(N_CLASES) if i != j and cm[i, j] > 0]
print("Confusiones (real -> predicho, número de imágenes):", sorted(errores, key=lambda e: -e[2]) or "ninguna")

# Imágenes mal clasificadas por el modelo final
mal = np.where(pred != y_prueba)[0]
print(f"\nImágenes mal clasificadas por el modelo final: {len(mal)} de {len(y_prueba)}")
if len(mal):
    fig, axs = plt.subplots(1, len(mal), figsize=(4.2 * len(mal), 3.6), squeeze=False)
    for ax, idx in zip(axs[0], mal):
        top = np.argsort(prob[idx])[::-1][:2]
        ax.imshow(Image.open(rutas_prueba[idx]).convert("RGB").resize(OBJETIVO))
        ax.set_title(f"real: {NOMBRES[y_prueba[idx]]}\n1.º {NOMBRES[top[0]]} ({prob[idx][top[0]]:.0%}), 2.º {NOMBRES[top[1]]} ({prob[idx][top[1]]:.0%})", fontsize=9)
        ax.axis("off")
        print(f"  {os.path.basename(rutas_prueba[idx])}: real={NOMBRES[y_prueba[idx]]}, predicho={NOMBRES[top[0]]} ({prob[idx][top[0]]:.1%}), "
              f"segunda opción={NOMBRES[top[1]]} ({prob[idx][top[1]]:.1%})")
    plt.tight_layout()
    plt.savefig(f"{IMG}/errores.png", dpi=200)
    plt.close()

# Matriz de confusión de la mejor CNN personalizada, para comparar
_, p_cnn = exactitud_prueba(modelos_entrenados[mejor_cnn])
cm_cnn = confusion_matrix(y_prueba, p_cnn.argmax(1))
fig, ax = plt.subplots(figsize=(6.2, 5.2))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Oranges", xticklabels=NOMBRES, yticklabels=NOMBRES, cbar=False, ax=ax)
ax.set_xlabel("predicho")
ax.set_ylabel("real")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{IMG}/matriz_confusion_cnn.png", dpi=200)
plt.close()
print(f"\nCNN personalizada ({mejor_cnn}), exactitud en prueba: {(p_cnn.argmax(1) == y_prueba).mean():.3f}")

# ---- Cinco pruebas con imágenes nuevas (de especies distintas) ----
rng = random.Random(SEMILLA)
elegidas = [rng.choice(np.where(y_prueba == NOMBRES.index(c))[0]) for c in rng.sample(CLASES, 5)]
filas = []
fig, axs = plt.subplots(1, 5, figsize=(13, 3.1))
for k, (ax, idx) in enumerate(zip(axs, elegidas)):
    real, predicho, conf = NOMBRES[y_prueba[idx]], NOMBRES[pred[idx]], prob[idx].max()
    filas.append({"prueba": k + 1, "animal real": real, "clasificación": predicho, "confianza": f"{conf:.1%}",
                  "¿correcta?": "sí" if real == predicho else "no"})
    ax.imshow(Image.open(rutas_prueba[idx]).convert("RGB").resize(OBJETIVO))
    ax.set_title(f"real: {real}\npredicho: {predicho} ({conf:.0%})", fontsize=9, color="#2f9e44" if real == predicho else "#e03131")
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{IMG}/pruebas.png", dpi=200)
plt.close()
print("\nCinco pruebas con imágenes nuevas:")
print(pd.DataFrame(filas).to_string(index=False))
