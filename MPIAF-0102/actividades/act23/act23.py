# =============================================================================
# Actividad 23: "Sistema de Clasificación de Emociones en Texto"
# Tema 4.4. Redes neuronales y aprendizaje profundo
#
# Dependencias (en Colab ya vienen instaladas):  pip install tensorflow scikit-learn pandas matplotlib seaborn scipy
# El conjunto de datos (dair-ai/emotion, ~1.3 MB) se descarga solo la primera vez.
# Este archivo también es el motor de la interfaz:  streamlit run app.py
# =============================================================================
import json
import os
import re
import time
import urllib.request
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

pd.set_option("display.width", 220)

SEMILLA = 42
SEMILLAS = (0, 1, 2)                                   # repeticiones de cada experimento (el entrenamiento es aleatorio)
RAPIDO = os.environ.get("ACT23_RAPIDO") == "1"         # modo de depuración: 1 época, 1 semilla
if RAPIDO:
    SEMILLAS = (0,)
AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
IMG = os.path.join(AQUI, "img")
DATOS = os.path.join(AQUI, "datos")
MODELO = os.path.join(AQUI, "modelo")
URL = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/split/{}-00000-of-00001.parquet"
EMOCIONES = ["tristeza", "alegría", "amor", "enojo", "miedo", "sorpresa"]     # etiquetas 0 a 5 del conjunto (sadness, joy, love, anger, fear, surprise)
MAX_FEATURES = 10000      # tamaño del vocabulario (el tema usa 10 000 en el ejemplo con IMDB)
MAXLEN = 40               # longitud de las secuencias (cubre ~95 % de los textos)


def epocas(n):
    return 1 if RAPIDO else n


# =============================================================================
# TAREA 1: PREPROCESAR TEXTO (palabras -> números)
# =============================================================================
def tokenizar(texto):
    """Minúsculas y palabras (letras y apóstrofos); se descarta puntuación y dígitos."""
    return re.findall(r"[a-z']+", texto.lower())


def construir_vocabulario(textos, max_features=MAX_FEATURES):
    """Palabra -> índice. 0 se reserva para el relleno y 1 para las palabras desconocidas (OOV)."""
    frecuencias = Counter(p for t in textos for p in tokenizar(t))
    return {p: i + 2 for i, (p, _) in enumerate(frecuencias.most_common(max_features - 2))}


def texto_a_indices(textos, vocab, maxlen=MAXLEN):
    """Convierte textos en una matriz de enteros de ancho fijo (relleno y truncado al inicio, como en el tema)."""
    secuencias = [[vocab.get(p, 1) for p in tokenizar(t)] for t in textos]
    return pad_sequences(secuencias, maxlen=maxlen)


def cargar_datos():
    os.makedirs(DATOS, exist_ok=True)
    partes = {}
    for nombre in ("train", "validation", "test"):
        ruta = os.path.join(DATOS, f"{nombre}.parquet")
        if not os.path.exists(ruta):
            urllib.request.urlretrieve(URL.format(nombre), ruta)
        partes[nombre] = pd.read_parquet(ruta)
    return partes["train"], partes["validation"], partes["test"]


# =============================================================================
# MODELO (la función "crear_rn_para_texto" que el tema usa sin definirla)
# =============================================================================
def crear_rn_para_texto(tipo="lstm", max_features=MAX_FEATURES, embedding=64, unidades=64, dropout=0.3,
                        bidireccional=False, lr=1e-3, clases=6):
    """Red para clasificar texto: Embedding -> capa recurrente -> Dropout -> softmax.
    tipo: "bolsa" (sin recurrencia: promedio de los embeddings), "rnn", "lstm" o "gru"."""
    modelo = models.Sequential()
    modelo.add(layers.Input(shape=(None,), dtype="int32"))
    modelo.add(layers.Embedding(max_features, embedding, mask_zero=True))
    if tipo == "bolsa":
        modelo.add(layers.GlobalAveragePooling1D())
    else:
        celda = {"rnn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}[tipo](unidades)
        modelo.add(layers.Bidirectional(celda) if bidireccional else celda)
    modelo.add(layers.Dropout(dropout))
    modelo.add(layers.Dense(clases, activation="softmax"))
    modelo.compile(optimizer=optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


# =============================================================================
# CLASIFICADOR PARA LA INTERFAZ (Tarea 3)
# =============================================================================
class ClasificadorEmociones:
    """Carga el modelo y el vocabulario guardados por este script y clasifica textos nuevos."""

    def __init__(self, carpeta=MODELO):
        self.modelo = models.load_model(os.path.join(carpeta, "emociones.keras"))
        with open(os.path.join(carpeta, "vocabulario.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.vocab, self.maxlen = cfg["vocab"], cfg["maxlen"]

    def probabilidades(self, textos):
        return self.modelo.predict(texto_a_indices(textos, self.vocab, self.maxlen), verbose=0)

    def clasificar(self, texto):
        """Devuelve (emoción, confianza, {emoción: probabilidad}, palabras desconocidas)."""
        p = self.probabilidades([texto])[0]
        i = int(p.argmax())
        desconocidas = [w for w in tokenizar(texto) if w not in self.vocab]
        return EMOCIONES[i], float(p[i]), {e: float(x) for e, x in zip(EMOCIONES, p)}, desconocidas


# =============================================================================
# ANÁLISIS ADICIONAL: ¿qué tan "natural" es el conjunto y qué pasa con las negaciones?
# (usa el modelo guardado; no vuelve a entrenar)
# =============================================================================
def analisis_extra():
    train, val, test = cargar_datos()
    tiene_feel = lambda t: bool(re.search(r"\b(feel|feeling|felt|feels)\b", t))
    negacion = lambda t: bool(re.search(r"\b(not|no|never|dont|didnt|doesnt|cant|wont|isnt|wasnt|nothing|nobody|without)\b", t))
    print(f"\nTextos que contienen feel/feeling/felt: entrenamiento {train.text.map(tiene_feel).mean():.1%}, prueba {test.text.map(tiene_feel).mean():.1%}")
    clasificador = ClasificadorEmociones()
    ok = clasificador.probabilidades(list(test.text)).argmax(1) == test.label.values
    neg = test.text.map(negacion).values
    print(f"Textos con negación: entrenamiento {train.text.map(negacion).mean():.1%}, prueba {neg.mean():.1%}")
    print(f"Exactitud en prueba con negación: {ok[neg].mean():.3f} (n={neg.sum()}); sin negación: {ok[~neg].mean():.3f} (n={(~neg).sum()})")
    print("\nFrases de clientes sin la estructura \"i feel ...\" (lo que probablemente llegaría en producción):")
    frases = ["i really love how easy the app is to use", "the package arrived and it was fine", "the delivery was two weeks late",
              "customer support never answered my emails", "great product, fast shipping", "i feel disappointed with the quality"]
    for t in frases:
        emo, c, _, desc = clasificador.clasificar(t)
        print(f'  "{t}" -> {emo} ({c:.2f}){"  [confianza baja]" if c < 0.6 else ""}')


# =============================================================================
# EXPERIMENTOS
# =============================================================================
def preparar(train, val, test, max_features=MAX_FEATURES, maxlen=MAXLEN):
    vocab = construir_vocabulario(train.text, max_features)
    return vocab, [texto_a_indices(d.text, vocab, maxlen) for d in (train, val, test)]


def correr(nombre, train, val, max_features=MAX_FEATURES, maxlen=MAXLEN, n_epocas=15, batch_size=64,
           pesos=False, **hp):
    """Entrena la misma configuración con cada semilla (parada temprana en validación); no toca la prueba."""
    _, (Xe, Xv, _) = preparar(train, val, val, max_features, maxlen)
    ye, yv = train.label.values, val.label.values
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(6), y=ye))) if pesos else None
    acc, f1, ep, tiempos, params = [], [], [], [], 0
    for s in SEMILLAS:
        tf.keras.utils.set_random_seed(s)
        m = crear_rn_para_texto(max_features=max_features, **hp)
        t0 = time.time()
        h = m.fit(Xe, ye, validation_data=(Xv, yv), epochs=epocas(n_epocas), batch_size=batch_size, verbose=0, class_weight=cw,
                  callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
        tiempos.append(time.time() - t0)
        pred = m.predict(Xv, verbose=0).argmax(1)
        acc.append(accuracy_score(yv, pred))
        f1.append(f1_score(yv, pred, average="macro"))
        ep.append(int(np.argmin(h.history["val_loss"])) + 1)
        params = m.count_params()
    return {"configuración": nombre, "parámetros": params, "exactitud": np.mean(acc), "F1 macro": np.mean(f1),
            "desv. F1": np.std(f1), "época": np.mean(ep), "tiempo (s)": np.mean(tiempos)}


def tabla(filas):
    df = pd.DataFrame(filas).set_index("configuración")
    return df.round({"exactitud": 4, "F1 macro": 4, "desv. F1": 4, "época": 1, "tiempo (s)": 1})


def principal():
    sns.set_theme(style="whitegrid")
    tf.keras.utils.set_random_seed(SEMILLA)
    train, val, test = cargar_datos()

    # ---------------------------------------------------------------------
    # Datos
    # ---------------------------------------------------------------------
    print(f"Textos: entrenamiento {len(train)}, validación {len(val)}, prueba {len(test)}")
    dist = pd.DataFrame({"emoción": [EMOCIONES[i] for i in range(6)],
                         "entrenamiento": train.label.value_counts().sort_index().values,
                         "validación": val.label.value_counts().sort_index().values,
                         "prueba": test.label.value_counts().sort_index().values})
    dist["% entrenamiento"] = (100 * dist["entrenamiento"] / len(train)).round(1)
    print(dist.to_string(index=False))
    largos = train.text.map(lambda t: len(tokenizar(t)))
    print(f"\nLongitud en palabras: mínimo {largos.min()}, mediana {largos.median():.0f}, promedio {largos.mean():.1f}, "
          f"percentil 95 {largos.quantile(.95):.0f}, máximo {largos.max()}")
    print("Ejemplos de entrenamiento:")
    for i in range(6):
        ej = train[train.label == i].iloc[0]
        print(f"  [{EMOCIONES[i]}] {ej.text}")

    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    sns.barplot(x="emoción", y="entrenamiento", data=dist, color="#4C78A8", ax=ejes[0])
    for i, v in enumerate(dist["% entrenamiento"]):
        ejes[0].text(i, dist["entrenamiento"][i] + 80, f"{v} %", ha="center", fontsize=8)
    ejes[0].set_title("Textos por emoción (entrenamiento)")
    ejes[0].set_ylabel("Textos")
    ejes[1].hist(largos, bins=range(0, 70, 2), color="#72B7B2")
    ejes[1].axvline(MAXLEN, color="#E45756", linestyle="--", label=f"maxlen = {MAXLEN}")
    ejes[1].set_title("Longitud de los textos")
    ejes[1].set_xlabel("Palabras")
    ejes[1].set_ylabel("Textos")
    ejes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{IMG}/datos.png", dpi=200)
    plt.close()

    # ---------------------------------------------------------------------
    # Tarea 1: representación numérica
    # ---------------------------------------------------------------------
    vocab_completo = construir_vocabulario(train.text, 10 ** 6)
    frec = Counter(p for t in train.text for p in tokenizar(t))
    total_tokens = sum(frec.values())
    print(f"\nVocabulario completo: {len(vocab_completo)} palabras distintas en {total_tokens} palabras de entrenamiento")
    print("Cobertura según el tamaño del vocabulario (proporción de palabras de validación que sí están en él):")
    tokens_val = [p for t in val.text for p in tokenizar(t)]
    filas = []
    for n in (500, 1000, 2500, 5000, 10000):
        v = construir_vocabulario(train.text, n)
        filas.append({"vocabulario": n, "cobertura en validación": np.mean([p in v for p in tokens_val])})
    print(pd.DataFrame(filas).round(4).to_string(index=False))

    vocab, (Xe, Xv, Xp) = preparar(train, val, test)
    print(f"\nVocabulario usado: {len(vocab)} entradas (0 = relleno, 1 = desconocida). Matrices: {Xe.shape}, {Xv.shape}, {Xp.shape}")
    ejemplo = "i didnt feel humiliated"
    print(f'\nEjemplo de conversión:\n  texto:    "{ejemplo}"\n  palabras: {tokenizar(ejemplo)}\n'
          f"  índices:  {[vocab.get(p, 1) for p in tokenizar(ejemplo)]}\n  relleno a {MAXLEN}: {texto_a_indices([ejemplo], vocab)[0].tolist()}")

    print("\n--- Efecto del preprocesamiento (LSTM de 64 unidades, media de las semillas, validación) ---")
    pre = [correr(f"vocabulario = {n}", train, val, max_features=n, tipo="lstm") for n in (1000, 3000, 10000)]
    pre += [correr(f"longitud = {m}", train, val, maxlen=m, tipo="lstm") for m in (10, 20, 66)]
    res_pre = tabla(pre)
    print(res_pre.to_string())

    # ---------------------------------------------------------------------
    # Tarea 2: arquitecturas recurrentes
    # ---------------------------------------------------------------------
    print("\n--- Línea base sin redes: TF-IDF (unigramas y bigramas) + regresión logística ---")
    tfidf = TfidfVectorizer(tokenizer=tokenizar, ngram_range=(1, 2), min_df=2, lowercase=False, token_pattern=None)
    Te = tfidf.fit_transform(train.text)
    lr = LogisticRegression(max_iter=2000, C=10).fit(Te, train.label)
    pv = lr.predict(tfidf.transform(val.text))
    f1_tfidf = f1_score(val.label, pv, average="macro")
    print(f"validación: exactitud {accuracy_score(val.label, pv):.4f}, F1 macro {f1_tfidf:.4f}")

    print(f"\n--- Arquitecturas (media de {len(SEMILLAS)} semillas, hasta 15 épocas, parada temprana; validación) ---")
    arqs = [("Bolsa de embeddings (sin recurrencia)", dict(tipo="bolsa")),
            ("SimpleRNN", dict(tipo="rnn")),
            ("LSTM", dict(tipo="lstm")),
            ("GRU", dict(tipo="gru")),
            ("LSTM bidireccional", dict(tipo="lstm", bidireccional=True)),
            ("GRU bidireccional", dict(tipo="gru", bidireccional=True))]
    res_arq = tabla([correr(n, train, val, **hp) for n, hp in arqs])
    print(res_arq.to_string())
    mejor_nombre = res_arq["F1 macro"].idxmax()
    mejor_hp = dict(arqs)[mejor_nombre]
    print(f"\nMejor por F1 macro en validación: {mejor_nombre}")

    print(f"\n--- Hiperparámetros: un factor a la vez sobre {mejor_nombre} ---")
    hps = [("embedding = 32", dict(embedding=32)), ("embedding = 128", dict(embedding=128)),
           ("unidades = 32", dict(unidades=32)), ("unidades = 128", dict(unidades=128)),
           ("dropout = 0.0", dict(dropout=0.0)), ("dropout = 0.5", dict(dropout=0.5)),
           ("lr = 0.0003", dict(lr=3e-4)), ("lr = 0.003", dict(lr=3e-3)),
           ("pesos por clase", dict(pesos=True))]
    res_hp = tabla([correr("base (embedding 64, unidades 64, dropout 0.3, lr 0.001)", train, val, **mejor_hp)]
                   + [correr(n, train, val, **{**mejor_hp, **h}) for n, h in hps])
    print(res_hp.to_string())
    mejor_cfg = res_hp["F1 macro"].idxmax()
    cfg_final = dict(mejor_hp)
    if mejor_cfg != res_hp.index[0]:
        cfg_final.update(dict(hps)[mejor_cfg])
    print(f"\nConfiguración elegida (mayor F1 macro en validación): {mejor_cfg}")

    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
    o = res_arq.sort_values("F1 macro")
    ejes[0].barh(o.index, o["F1 macro"] * 100, xerr=o["desv. F1"] * 100, color="#4C78A8", capsize=3)
    ejes[0].axvline(f1_tfidf * 100, color="#E45756", linestyle="--", label="TF-IDF + regresión logística")
    ejes[0].set_xlim(o["F1 macro"].min() * 100 - 4, 100)
    ejes[0].set_xlabel("F1 macro en validación (%)")
    ejes[0].set_title("Arquitecturas")
    ejes[0].legend(fontsize=8, loc="lower left")
    h = res_hp.sort_values("F1 macro")
    ejes[1].barh([i[:28] for i in h.index], h["F1 macro"] * 100, xerr=h["desv. F1"] * 100, color="#72B7B2", capsize=3)
    ejes[1].set_xlim(h["F1 macro"].min() * 100 - 3, h["F1 macro"].max() * 100 + 2)
    ejes[1].set_xlabel("F1 macro en validación (%)")
    ejes[1].set_title("Hiperparámetros")
    ejes[1].tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(f"{IMG}/arquitecturas.png", dpi=200)
    plt.close()

    # ---------------------------------------------------------------------
    # Evaluación final (prueba, una sola vez) y modelo guardado
    # ---------------------------------------------------------------------
    print(f"\n--- Evaluación final en prueba ({len(SEMILLAS)} semillas) ---")
    hp_final = {k: v for k, v in cfg_final.items() if k != "pesos"}
    pesos = cfg_final.get("pesos", False)
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(6), y=train.label.values))) if pesos else None
    ye, yv, yp = train.label.values, val.label.values, test.label.values
    finales, curvas = [], None
    for s in SEMILLAS:
        tf.keras.utils.set_random_seed(s)
        m = crear_rn_para_texto(**hp_final)
        hist = m.fit(Xe, ye, validation_data=(Xv, yv), epochs=epocas(15), batch_size=64, verbose=0, class_weight=cw,
                     callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
        p = m.predict(Xp, verbose=0)
        finales.append((m, p))
        if curvas is None:
            curvas = hist.history
    accs = [accuracy_score(yp, p.argmax(1)) for _, p in finales]
    f1s = [f1_score(yp, p.argmax(1), average="macro") for _, p in finales]
    # Referencias: bolsa de embeddings y TF-IDF con las mismas condiciones
    bolsa = []
    for s in SEMILLAS:
        tf.keras.utils.set_random_seed(s)
        mb = crear_rn_para_texto(tipo="bolsa")
        mb.fit(Xe, ye, validation_data=(Xv, yv), epochs=epocas(15), batch_size=64, verbose=0,
               callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
        bolsa.append(f1_score(yp, mb.predict(Xp, verbose=0).argmax(1), average="macro"))
    pt = lr.predict(tfidf.transform(test.text))
    res_final = pd.DataFrame([
        {"modelo": "TF-IDF + regresión logística", "exactitud": accuracy_score(yp, pt), "F1 macro": f1_score(yp, pt, average="macro"), "desv. F1": 0.0},
        {"modelo": "Bolsa de embeddings", "exactitud": np.nan, "F1 macro": np.mean(bolsa), "desv. F1": np.std(bolsa)},
        {"modelo": f"{mejor_nombre} (elegido)", "exactitud": np.mean(accs), "F1 macro": np.mean(f1s), "desv. F1": np.std(f1s)}]).set_index("modelo")
    print(res_final.round(4).to_string())

    modelo, probas = finales[0]
    pred = probas.argmax(1)
    print(f"\nModelo analizado y guardado (semilla {SEMILLAS[0]}): exactitud {accs[0]:.4f}, F1 macro {f1s[0]:.4f}")
    os.makedirs(MODELO, exist_ok=True)
    modelo.save(os.path.join(MODELO, "emociones.keras"))
    with open(os.path.join(MODELO, "vocabulario.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "maxlen": MAXLEN, "emociones": EMOCIONES}, f, ensure_ascii=False)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    ejes[0].plot(curvas["loss"], "b-", label="Entrenamiento")
    ejes[0].plot(curvas["val_loss"], "r-", label="Validación")
    ejes[0].set_xlabel("Época")
    ejes[0].set_ylabel("Pérdida")
    ejes[0].set_title("Pérdida")
    ejes[0].legend()
    ejes[1].plot(np.array(curvas["accuracy"]) * 100, "b-", label="Entrenamiento")
    ejes[1].plot(np.array(curvas["val_accuracy"]) * 100, "r-", label="Validación")
    ejes[1].set_xlabel("Época")
    ejes[1].set_ylabel("Exactitud (%)")
    ejes[1].set_title("Exactitud")
    ejes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{IMG}/curvas.png", dpi=200)
    plt.close()

    # ---------------------------------------------------------------------
    # Análisis de errores
    # ---------------------------------------------------------------------
    print("\n--- Métricas por emoción (prueba) ---")
    print(classification_report(yp, pred, target_names=EMOCIONES, digits=4))
    cm = confusion_matrix(yp, pred)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ejes[0], xticklabels=EMOCIONES, yticklabels=EMOCIONES)
    ejes[0].set_title("Matriz de confusión (conteos)")
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ejes[1], xticklabels=EMOCIONES, yticklabels=EMOCIONES)
    ejes[1].set_title("Matriz de confusión (proporción de cada emoción real)")
    for ax in ejes:
        ax.set_xlabel("Predicha")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"{IMG}/matriz_confusion.png", dpi=200)
    plt.close()

    filas = sorted([(EMOCIONES[i], EMOCIONES[j], cm[i, j]) for i in range(6) for j in range(6) if i != j], key=lambda t: -t[2])[:8]
    print("Confusiones más frecuentes (real -> predicha):")
    print(pd.DataFrame(filas, columns=["real", "predicha", "errores"]).to_string(index=False))

    conf = probas.max(1)
    ok = pred == yp
    print(f"\nConfianza media: aciertos {conf[ok].mean():.3f}, errores {conf[~ok].mean():.3f}")
    print(f"Exactitud top-2: {np.mean([yp[i] in np.argsort(-probas[i])[:2] for i in range(len(yp))]):.4f}")
    L = test.text.map(lambda t: len(tokenizar(t))).values
    cortos, medios, largos_ = L < 10, (L >= 10) & (L < 25), L >= 25
    print("Exactitud según la longitud del texto: " + ", ".join(
        f"{n} {ok[m].mean():.3f} (n={m.sum()})" for n, m in (("< 10 palabras", cortos), ("10 a 24", medios), (">= 25", largos_))))
    oov = test.text.map(lambda t: np.mean([p not in vocab for p in tokenizar(t)])).values
    print(f"Proporción de palabras desconocidas: en aciertos {oov[ok].mean():.3f}, en errores {oov[~ok].mean():.3f}")

    print("\nErrores con mayor confianza:")
    idx = np.where(~ok)[0]
    for i in idx[np.argsort(-conf[idx])][:8]:
        print(f"  real {EMOCIONES[yp[i]]:9s} pred {EMOCIONES[pred[i]]:9s} (p={conf[i]:.2f}): {test.text.iloc[i]}")

    # ---------------------------------------------------------------------
    # ¿Entiende el contexto? Frases de contraste hechas a mano (cualitativo)
    # ---------------------------------------------------------------------
    sondas = ["i feel happy", "i do not feel happy", "i am not afraid anymore", "i am afraid",
              "i feel loved by everyone", "i feel so angry at him", "i was surprised by the gift",
              "i feel sad and alone", "i dont feel sad at all", "i feel nothing but joy"]
    tf.keras.utils.set_random_seed(SEMILLAS[0])
    mb_modelo = crear_rn_para_texto(tipo="bolsa")
    mb_modelo.fit(Xe, ye, validation_data=(Xv, yv), epochs=epocas(15), batch_size=64, verbose=0,
                  callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
    Xs = texto_a_indices(sondas, vocab)
    p_rnn, p_bolsa, p_tfidf = modelo.predict(Xs, verbose=0), mb_modelo.predict(Xs, verbose=0), lr.predict_proba(tfidf.transform(sondas))
    print("\nFrases de contraste (cualitativo): emoción predicha (confianza)")
    filas = []
    for k, t in enumerate(sondas):
        filas.append({"frase": t,
                      "RNN": f"{EMOCIONES[p_rnn[k].argmax()]} ({p_rnn[k].max():.2f})",
                      "bolsa": f"{EMOCIONES[p_bolsa[k].argmax()]} ({p_bolsa[k].max():.2f})",
                      "TF-IDF": f"{EMOCIONES[p_tfidf[k].argmax()]} ({p_tfidf[k].max():.2f})"})
    print(pd.DataFrame(filas).to_string(index=False))

    # ---------------------------------------------------------------------
    # Tarea 3: prueba de la interfaz (motor)
    # ---------------------------------------------------------------------
    clasificador = ClasificadorEmociones()
    print("\nPrueba del clasificador guardado (el mismo que usa la interfaz):")
    for t in ["i feel so thrilled and grateful for this team", "i am terrified of what the doctor will say",
              "this delay makes me furious", "xyzzy qwerty"]:
        emo, c, _, desc = clasificador.clasificar(t)
        print(f'  "{t}" -> {emo} ({c:.2f}); palabras desconocidas: {desc}')
    t0 = time.time()
    clasificador.probabilidades(list(test.text[:200]))
    print(f"Tiempo para clasificar 200 textos: {time.time() - t0:.2f} s")
    print("\nListo. Figuras guardadas en", IMG)


if __name__ == "__main__" or "__file__" not in globals():
    os.makedirs(IMG, exist_ok=True)
    principal()
    analisis_extra()
