# =============================================================================
# Actividad 16: Proyecto "Analizador de Sentimientos para Reseñas de Productos"
# Tema 4.1. Procesamiento del lenguaje natural
#
# Dependencias (en Colab anteponer "!" a las dos líneas):
#   pip install -U spacy nltk gensim scikit-learn pandas matplotlib seaborn
#   python -m spacy download es_core_news_sm
# =============================================================================
import os
import random
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)

SEMILLA = 42
AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
IMG = os.path.join(AQUI, "img")
os.makedirs(IMG, exist_ok=True)
sns.set_theme(style="whitegrid")
CLASES = ["negativo", "neutral", "positivo"]


# =============================================================================
# DATOS: código base (5 reseñas) + ampliación simulada
# =============================================================================
def cargar_datos():
    # Simulación de datos de reseñas (código base del tema)
    datos = {
        'reseña': [
            "Excelente producto, cumple con todas las expectativas",
            "Mala calidad, no lo recomiendo",
            "Regular, podría ser mejor",
            "Increíble, superó mis expectativas",
            "Pésimo servicio al cliente"
        ],
        'sentimiento': ['positivo', 'negativo', 'neutral', 'positivo', 'negativo']
    }
    return pd.DataFrame(datos)


SUJETOS = ["El producto", "La licuadora", "El celular", "Los audífonos",
           "La laptop", "El envío", "La atención al cliente", "La calidad",
           "El empaque", "El vendedor", "La entrega", "El material",
           "La batería", "La cafetera", "El servicio", "La pantalla"]

POS = ["es excelente", "me encantó", "superó mis expectativas",
       "funciona perfecto", "llegó antes de lo esperado",
       "es de muy buena calidad", "lo recomiendo totalmente",
       "vale cada peso", "está increíble", "cumple con todo lo que prometen",
       "es justo lo que buscaba", "es fantástico"]
POS_NEG = ["no está nada mal", "no tuve ningún problema",
           "no me arrepiento de comprarlo", "nunca me ha fallado",
           "no tengo nada que reprochar", "no podría estar más contento"]
NEG = ["es pésimo", "llegó roto", "es de muy mala calidad",
       "es una pérdida de dinero", "se descompuso a la semana",
       "es una decepción total", "nunca llegó a tiempo", "es horrible",
       "me dejó muy molesto", "es un fraude"]
NEG_NEG = ["no funciona", "no lo recomiendo", "no me gustó",
           "no cumple con lo prometido", "no vale lo que cuesta",
           "no está nada bien", "no es nada bueno", "no volvería a comprarlo"]
NEU = ["es regular", "está más o menos", "cumple, pero nada especial",
       "es normal, ni bueno ni malo", "podría ser mejor",
       "es aceptable, sin más", "hace lo básico", "es lo que esperaba, nada más",
       "es pasable", "está bien sin ser extraordinario"]

EMO = {"positivo": [":)", ":D", "<3", ";)", "xD", "😀", "👍"],
       "negativo": [":(", ":'(", ">:(", "😡", "👎", "😞"],
       "neutral": [":/", ":|", "😐", "🤷"]}

INFORMAL = {"que": "q", "porque": "xq", "bien": "bn", "también": "tmb",
            "muy": "muuuy", "excelente": "excelenteee", "bueno": "buenooo",
            "está": "ta", "mucho": "muxo", "para": "pa", "gracias": "grax"}


def _informalizar(texto, rng, p=0.35):
    """Introduce lenguaje informal (abreviaturas, letras repetidas)."""
    palabras = texto.split()
    for i, w in enumerate(palabras):
        base = w.lower().strip(",.")
        if base in INFORMAL and rng.random() < p * 2:
            palabras[i] = INFORMAL[base]
    return " ".join(palabras)


def _reseña(clase, rng):
    sujeto = rng.choice(SUJETOS)
    otro = rng.choice([s for s in SUJETOS if s != sujeto])
    tipo = rng.random()
    if clase == "positivo":
        base = rng.choice(POS if tipo < 0.7 else POS_NEG)
        txt = f"{sujeto} {base}"
        if tipo > 0.88:   # opinión mixta con polaridad dominante positiva
            txt += f", aunque {otro.lower()} {rng.choice(['llegó tarde', 'es regular', 'no es perfecto'])}"
    elif clase == "negativo":
        base = rng.choice(NEG if tipo < 0.5 else NEG_NEG)
        txt = f"{sujeto} {base}"
        if tipo > 0.88:   # opinión mixta con polaridad dominante negativa
            txt = f"{otro} {rng.choice(['llegó rápido', 'es bonito', 'se ve bien'])}, pero {sujeto.lower()} {base}"
    else:
        txt = f"{sujeto} {rng.choice(NEU)}"
        if tipo > 0.75:   # mezcla equilibrada de aspectos
            txt = f"{sujeto} {rng.choice(POS[:3])}, pero {otro.lower()} {rng.choice(NEG[:3])}"
    if rng.random() < 0.55:
        txt = _informalizar(txt, rng)
    if rng.random() < 0.45:
        emo_clase = clase if rng.random() < 0.85 else rng.choice(CLASES)
        txt += " " + rng.choice(EMO[emo_clase])
    return txt + rng.choice(["", ".", "!", "!!", "..."])


def ampliar_datos(df_base, n_por_clase=60, semilla=SEMILLA):
    """Amplía el código base con reseñas simuladas (misma distribución de estilo)."""
    rng = random.Random(semilla)
    filas = list(df_base.itertuples(index=False, name=None))
    vistos = {f[0] for f in filas}
    for clase in CLASES:
        n = n_por_clase - (df_base["sentimiento"] == clase).sum()
        while n > 0:
            txt = _reseña(clase, rng)
            if txt not in vistos:
                vistos.add(txt)
                filas.append((txt, clase))
                n -= 1
    return pd.DataFrame(filas, columns=["reseña", "sentimiento"])


print("=" * 70)
print("DATOS")
print("=" * 70)
df_base = cargar_datos()
print("Datos cargados (código base):")
print(df_base.head())
df = ampliar_datos(df_base)
print(f"\nDataset ampliado: {len(df)} reseñas")
print(df["sentimiento"].value_counts().sort_index())
print("\nMuestra aleatoria del dataset ampliado:")
pd.set_option("display.max_colwidth", 80, "display.width", 140)
print(df.iloc[5:].sample(8, random_state=SEMILLA).to_string())


# =============================================================================
# TAREA 1: PREPROCESAMIENTO DE TEXTO
# =============================================================================
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

NEGADORES = {"no", "nunca", "jamás", "tampoco", "ni", "sin", "nada", "ningún",
             "ninguno", "ninguna"}
# Palabras que SÍ aportan polaridad o intensidad: se retiran de la lista de stopwords
CONSERVAR = NEGADORES | {"muy", "más", "poco", "mucho", "pero", "aunque", "bien", "mal"}
STOP = set(stopwords.words("spanish")) - CONSERVAR
ROMPEN_ALCANCE = {"pero", "aunque", "sino"}   # cierran el alcance de una negación

EMOTICONOS = [  # (patrón, token). El orden importa: primero los más largos
    (r"(:'\(|>:\(|:\(|:-\(|😡|👎|😞|😢)", " emoneg "),
    (r"(:\)|:-\)|:d|;\)|xd|<3|😀|👍|😊|❤️)", " emopos "),
    (r"(:/|:\||:-/|😐|🤷)", " emoneu "),
]
ABREVIATURAS = {"q": "que", "xq": "porque", "pq": "porque", "porq": "porque",
                "k": "que", "bn": "bien", "tmb": "también", "tb": "también",
                "x": "por", "pa": "para", "ta": "está", "toy": "estoy",
                "muxo": "mucho", "grax": "gracias", "sip": "sí", "nop": "no"}


# Correcciones puntuales a lemas erróneos de es_core_news_sm, y formas que no se lematizan
LEMA_FIX = {"recomeir": "recomendar", "cafetero": "cafetera", "buen": "bueno"}
CONSERVAR_FORMA = {"muy", "más"}   # "muy" se lematiza como "mucho": se conserva tal cual


def normalizar(texto):
    """Limpieza y normalización: minúsculas, emoticonos, jerga, letras repetidas."""
    t = texto.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)           # URLs
    for patron, token in EMOTICONOS:                       # emoticonos -> token
        t = re.sub(patron, token, t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)                     # buenooo -> bueno
    t = re.sub(r"[^\wáéíóúüñ\s.,;:!?¡¿]", " ", t)          # símbolos raros
    t = re.sub(r"\d+", " ", t)
    # Separa la puntuación para que sirva como límite del alcance de negaciones
    t = re.sub(r"([.,;:!?¡¿])", r" \1 ", t)
    return " ".join(ABREVIATURAS.get(w, w) for w in t.split())


def expandir_negaciones(lemas, ventana=3):
    """no bueno -> no no_bueno. El alcance termina en puntuación o conjunción adversativa."""
    salida, restante = [], 0
    for lema, es_punt in lemas:
        if es_punt or lema in ROMPEN_ALCANCE:
            restante = 0
            if not es_punt:
                salida.append(lema)
            continue
        if lema in NEGADORES:
            restante = ventana
            salida.append(lema)
        elif restante > 0:
            salida.append("no_" + lema)
            restante -= 1
        else:
            salida.append(lema)
    return salida


def preprocesar(textos, modo="completo"):
    """
    modo="basico"   -> pipeline del tema: lematizar + quitar stopwords (todas) + solo alfabéticos
    modo="completo" -> normalización + emoticonos + jerga + expansión de negaciones
    """
    if modo == "basico":
        docs = nlp.pipe([t.lower() for t in textos])
        sw = set(stopwords.words("spanish"))
        return [" ".join(tk.lemma_.lower() for tk in d
                         if tk.is_alpha and tk.lemma_.lower() not in sw) for d in docs]
    docs = nlp.pipe([normalizar(t) for t in textos])
    resultado = []
    for d in docs:
        lemas = []
        for tk in d:
            if tk.is_space:
                continue
            if tk.text.startswith("emo") or tk.text in CONSERVAR_FORMA:
                txt = tk.text
            else:
                txt = LEMA_FIX.get(tk.lemma_.lower(), tk.lemma_.lower())
            if tk.is_punct:
                lemas.append((txt, True))
            elif txt in STOP or not txt.isalpha():
                continue
            else:
                lemas.append((txt, False))
        resultado.append(" ".join(expandir_negaciones(lemas)))
    return resultado


print("\n" + "=" * 70)
print("TAREA 1: PREPROCESAMIENTO")
print("=" * 70)
ejemplos = [
    "Excelente producto, cumple con todas las expectativas",
    "No me gustó, xq llegó roto y no funciona nada bn :(",
    "Muuuy bueno el celular!!! tmb llegó rápido :D",
    "Regular, no es nada bueno ni malo :/",
    "El envío no está nada mal, no tuve ningún problema <3",
]
for e, b, c in zip(ejemplos, preprocesar(ejemplos, "basico"), preprocesar(ejemplos, "completo")):
    print(f"Original : {e}\n  Básico  : {b}\n  Completo: {c}\n")

df["basico"] = preprocesar(df["reseña"], "basico")
df["limpio"] = preprocesar(df["reseña"], "completo")
y = df["sentimiento"].to_numpy()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEMILLA)
METRICAS = {"accuracy": "accuracy", "precision": "precision_macro",
            "recall": "recall_macro", "f1": "f1_macro"}


def evaluar(modelo, X, nombre):
    r = cross_validate(modelo, X, y, cv=cv, scoring=METRICAS, n_jobs=1)
    fila = {"modelo": nombre}
    for m in METRICAS:
        fila[m] = r[f"test_{m}"].mean()
        fila[m + "_sd"] = r[f"test_{m}"].std()
    return fila


def tfidf_lr(**kw):
    return Pipeline([("tfidf", TfidfVectorizer(**kw)),
                     ("clf", LogisticRegression(max_iter=1000, random_state=SEMILLA))])


print("Efecto del preprocesamiento (TF-IDF + Regresión Logística, CV 5x3):")
filas = [evaluar(tfidf_lr(), df["reseña"], "Texto crudo"),
         evaluar(tfidf_lr(), df["basico"], "Pipeline básico (tema 4.1)"),
         evaluar(tfidf_lr(), df["limpio"], "Pipeline completo (+emoticonos, jerga, negaciones)")]
ablacion = pd.DataFrame(filas)
print(ablacion[["modelo", "accuracy", "f1", "f1_sd"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(ablacion["modelo"], ablacion["f1"], xerr=ablacion["f1_sd"], color=["#adb5bd", "#74c0fc", "#2f9e44"])
ax.set_xlabel("F1 macro (validación cruzada)")
ax.set_xlim(0, 1)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{IMG}/preprocesamiento.png", dpi=200)
plt.close()


# =============================================================================
# TAREA 2: EXTRACCIÓN DE CARACTERÍSTICAS (TF-IDF, BoW, Word2Vec)
# =============================================================================
print("\n" + "=" * 70)
print("TAREA 2: EXTRACCIÓN DE CARACTERÍSTICAS")
print("=" * 70)


class Word2VecMedio(BaseEstimator, TransformerMixin):
    """Entrena Word2Vec solo con el pliegue de entrenamiento y promedia los vectores."""

    def __init__(self, dim=50, ventana=3, epocas=100, semilla=SEMILLA):
        self.dim, self.ventana, self.epocas, self.semilla = dim, ventana, epocas, semilla

    def fit(self, X, y=None):
        corpus = [t.split() for t in X]
        self.w2v_ = Word2Vec(corpus, vector_size=self.dim, window=self.ventana,
                             min_count=1, epochs=self.epocas, seed=self.semilla,
                             workers=1, sg=1)
        return self

    def transform(self, X):
        kv = self.w2v_.wv
        return np.array([np.mean([kv[w] for w in t.split() if w in kv] or [np.zeros(self.dim)], axis=0)
                         for t in X])


# Vistazo a lo aprendido por Word2Vec (entrenado con todo el corpus, solo para inspección)
w2v = Word2Vec([t.split() for t in df["limpio"]], vector_size=50, window=3,
               min_count=2, epochs=100, seed=SEMILLA, workers=1, sg=1)
print(f"Vocabulario Word2Vec: {len(w2v.wv)} palabras, vectores de 50 dimensiones")
for palabra in ["excelente", "pésimo", "regular", "emopos"]:
    if palabra in w2v.wv:
        vecinos = ", ".join(f"{p} ({s:.2f})" for p, s in w2v.wv.most_similar(palabra, topn=4))
        print(f"  Más similares a '{palabra}': {vecinos}")

lr = lambda: LogisticRegression(max_iter=1000, random_state=SEMILLA)
esquemas = {
    "Bolsa de palabras (BoW)": Pipeline([("v", CountVectorizer()), ("clf", lr())]),
    "TF-IDF unigramas": Pipeline([("v", TfidfVectorizer()), ("clf", lr())]),
    "TF-IDF uni+bigramas": Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2))), ("clf", lr())]),
    "Word2Vec (promedio, 50d)": Pipeline([("v", Word2VecMedio()), ("sc", StandardScaler()), ("clf", lr())]),
}
filas = [evaluar(m, df["limpio"], n) for n, m in esquemas.items()]
comp_vec = pd.DataFrame(filas)
print("\nComparación de vectorizaciones (Regresión Logística, CV 5x3):")
print(comp_vec[["modelo", "accuracy", "precision", "recall", "f1", "f1_sd"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(comp_vec["modelo"], comp_vec["f1"], xerr=comp_vec["f1_sd"],
        color=["#adb5bd", "#2f9e44", "#2b8a3e", "#e8590c"])
ax.set_xlabel("F1 macro (validación cruzada)")
ax.set_xlim(0, 1)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{IMG}/vectorizacion.png", dpi=200)
plt.close()

# --- Selección de características (chi cuadrada) -------------------------------
vec = TfidfVectorizer(ngram_range=(1, 2))
X_all = vec.fit_transform(df["limpio"])
terminos = np.array(vec.get_feature_names_out())
print(f"\nCaracterísticas TF-IDF (uni+bigramas): {X_all.shape[1]}")
chi, _ = chi2(X_all, y)
top = pd.DataFrame({"término": terminos, "chi2": chi}).sort_values("chi2", ascending=False)
print("Los 15 términos con mayor chi2:")
print(top.head(15).round(2).to_string(index=False))

filas = []
for k in [10, 25, 50, 100, 200, "todas"]:
    pipe = Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2))),
                     ("sel", SelectKBest(chi2, k="all" if k == "todas" else k)),
                     ("clf", lr())])
    r = evaluar(pipe, df["limpio"], str(k))
    filas.append({"k": k, "f1": r["f1"], "f1_sd": r["f1_sd"]})
sel = pd.DataFrame(filas)
print("\nF1 macro según el número k de características seleccionadas:")
print(sel.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
t15 = top.head(12).iloc[::-1]
axs[0].barh(t15["término"], t15["chi2"], color="#1971c2")
axs[0].set_xlabel("estadístico $\\chi^2$")
axs[0].set_title("Términos más relevantes")
axs[1].errorbar(sel["k"].astype(str), sel["f1"], yerr=sel["f1_sd"], marker="o", color="#2f9e44", capsize=3)
axs[1].set_xlabel("número de características (k)")
axs[1].set_ylabel("F1 macro (CV)")
axs[1].set_title("Efecto de la selección")
plt.tight_layout()
plt.savefig(f"{IMG}/seleccion.png", dpi=200)
plt.close()


# =============================================================================
# TAREA 3: MODELADO Y EVALUACIÓN
# =============================================================================
print("\n" + "=" * 70)
print("TAREA 3: MODELADO Y EVALUACIÓN")
print("=" * 70)

tfidf = lambda: TfidfVectorizer(ngram_range=(1, 2))
modelos = {
    "TF-IDF + Naive Bayes": Pipeline([("v", tfidf()), ("clf", MultinomialNB(alpha=0.5))]),
    "TF-IDF + Reg. Logística": Pipeline([("v", tfidf()), ("clf", lr())]),
    "TF-IDF + SVM lineal": Pipeline([("v", tfidf()), ("clf", LinearSVC(random_state=SEMILLA))]),
    "TF-IDF + Random Forest": Pipeline([("v", tfidf()), ("clf", RandomForestClassifier(n_estimators=200, random_state=SEMILLA))]),
    "Word2Vec + Reg. Logística": Pipeline([("v", Word2VecMedio()), ("sc", StandardScaler()), ("clf", lr())]),
    "Word2Vec + Random Forest": Pipeline([("v", Word2VecMedio()), ("clf", RandomForestClassifier(n_estimators=200, random_state=SEMILLA))]),
}
filas = [evaluar(m, df["limpio"], n) for n, m in modelos.items()]
resumen = pd.DataFrame(filas).sort_values("f1", ascending=False).reset_index(drop=True)
print("Validación cruzada estratificada 5 pliegues x 3 repeticiones (15 ajustes por modelo):")
print(resumen[["modelo", "accuracy", "precision", "recall", "f1", "f1_sd"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.barh(resumen["modelo"], resumen["f1"], xerr=resumen["f1_sd"],
        color=["#e8590c" if "Word2Vec" in n else "#2f9e44" for n in resumen["modelo"]], capsize=3)
ax.set_xlabel("F1 macro (media ± desviación en 15 ajustes)")
ax.set_xlim(0, 1)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{IMG}/modelos.png", dpi=200)
plt.close()

# --- Mejor modelo: hold-out 80/20 y análisis de errores --------------------------
mejor = resumen.loc[0, "modelo"]
print(f"\nMejor modelo por F1 macro: {mejor}")
X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
    df["limpio"], y, df.index, test_size=0.2, stratify=y, random_state=SEMILLA)
final = modelos[mejor].fit(X_tr, y_tr)
pred = final.predict(X_te)
print(f"\nReporte de clasificación en el conjunto de prueba ({len(y_te)} reseñas):")
print(classification_report(y_te, pred, labels=CLASES, digits=3))

fig, ax = plt.subplots(figsize=(4.6, 4))
ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=CLASES), display_labels=CLASES).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.grid(False)
plt.tight_layout()
plt.savefig(f"{IMG}/matriz_confusion.png", dpi=200)
plt.close()

errores = df.loc[i_te][pred != y_te][["reseña", "sentimiento"]].copy()
errores["predicho"] = pred[pred != y_te]
print(f"Reseñas mal clasificadas: {len(errores)} de {len(y_te)}")
if len(errores):
    print(errores.to_string())

# --- Palabras más influyentes (coeficientes de la Regresión Logística) -------------
lr_final = Pipeline([("v", tfidf()), ("clf", lr())]).fit(df["limpio"], y)
nombres = np.array(lr_final["v"].get_feature_names_out())
coefs = lr_final["clf"].coef_
print("\nTérminos más influyentes por clase (Regresión Logística):")
for i, c in enumerate(lr_final["clf"].classes_):
    print(f"  {c:9s}: " + ", ".join(nombres[np.argsort(coefs[i])[::-1][:8]]))

# --- Comparación con el código base original (Random Forest sobre TF-IDF crudo) -----
base = Pipeline([("v", TfidfVectorizer()), ("clf", RandomForestClassifier(random_state=SEMILLA))])
r_base = evaluar(base, df["reseña"], "Código base")
print(f"\nCódigo base (TF-IDF crudo + Random Forest): F1 macro = {r_base['f1']:.3f} +/- {r_base['f1_sd']:.3f}")
print(f"Mejor modelo del proyecto ({mejor}):        F1 macro = {resumen.loc[0, 'f1']:.3f} +/- {resumen.loc[0, 'f1_sd']:.3f}")

# --- Predicciones sobre reseñas nuevas ------------------------------------------------
nuevas = ["Me encantó, llegó rápido y funciona perfecto :D",
          "No lo recomiendo, llegó roto y no funciona :(",
          "Es normal, hace lo básico",
          "No está nada mal, nunca me ha fallado",
          "xq nadie me dijo q era tan malo, pésimo!!!"]
print("\nPredicciones sobre reseñas nuevas:")
for t, p in zip(nuevas, final.predict(preprocesar(nuevas))):
    print(f"  [{p:8s}] {t}")
