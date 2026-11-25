# =============================================================================
# Actividad 25: "Sistema de Recomendación Basado en Reglas de Asociación"
# Tema 4.5. Minería de datos
#
# Dependencias (en Colab anteponer "!"):  pip install mlxtend scikit-learn pandas numpy matplotlib seaborn
# Los datos (Goodbooks-10k: calificaciones de libros, ~24 MB) se descargan solos la primera vez.
# Tarda unos 5 minutos.
# =============================================================================
import io
import os
import re
import time
import urllib.request
import warnings
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)

SEMILLA = 42
AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
IMG = os.path.join(AQUI, "img")
DATOS = os.path.join(AQUI, "datos")
os.makedirs(IMG, exist_ok=True)
os.makedirs(DATOS, exist_ok=True)
sns.set_theme(style="whitegrid")

URL_CALIFICACIONES = "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/ratings.zip"
URL_LIBROS = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
N_LIBROS = 300          # catálogo: los libros que más usuarios calificaron con 4 o más estrellas
UMBRAL = 4              # una "compra" es un libro que el usuario calificó con 4 o 5 estrellas
CARRITO = 3             # libros en el carrito actual al evaluar
MIN_OCULTOS = 8         # los usuarios de prueba tienen al menos 8 libros en el catálogo (3 en el carrito y 5 ocultos o más)
SEMILLAS = (0, 1, 2)    # repeticiones del sorteo de carritos


# =============================================================================
# DATOS
# =============================================================================
def cargar_datos():
    ruta_cal, ruta_lib = os.path.join(DATOS, "ratings.csv"), os.path.join(DATOS, "books.csv")
    if not os.path.exists(ruta_cal):
        with urllib.request.urlopen(URL_CALIFICACIONES, timeout=300) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extract("ratings.csv", DATOS)
    if not os.path.exists(ruta_lib):
        urllib.request.urlretrieve(URL_LIBROS, ruta_lib)
    return pd.read_csv(ruta_cal), pd.read_csv(ruta_lib)


def titulo_corto(t):
    """Quita la serie entre paréntesis: 'Catching Fire (The Hunger Games, #2)' -> 'Catching Fire'."""
    return re.sub(r"\s*\(.*\)\s*$", "", t).strip()


calificaciones, libros = cargar_datos()
print(f"Calificaciones: {len(calificaciones)}, usuarios: {calificaciones.user_id.nunique()}, libros: {calificaciones.book_id.nunique()}")
print("Distribución de calificaciones (proporción):", calificaciones.rating.value_counts(normalize=True).sort_index().round(3).to_dict())
gustados = calificaciones[calificaciones.rating >= UMBRAL]
print(f"Libros que gustaron (calificación >= {UMBRAL}): {len(gustados)} ({len(gustados) / len(calificaciones):.1%} de las calificaciones)")
por_usuario = gustados.groupby("user_id").size()
print(f"Libros que gustaron por usuario: mínimo {por_usuario.min()}, mediana {por_usuario.median():.0f}, máximo {por_usuario.max()}")
popularidad_total = gustados.book_id.value_counts()
print(f"Popularidad: el libro más popular gustó a {popularidad_total.iloc[0]} usuarios; el 50 % de las 'compras' corresponde al "
      f"{(popularidad_total.cumsum() / popularidad_total.sum() < 0.5).sum() / len(popularidad_total):.1%} de los libros")

# Catálogo y canastas ("compras" de cada usuario dentro del catálogo)
catalogo = popularidad_total.index[:N_LIBROS].values
columna = {b: i for i, b in enumerate(catalogo)}
info = libros.set_index("book_id").loc[catalogo]
titulos = [titulo_corto(t) for t in info.title]
autores = list(info.authors)
cobertura_catalogo = gustados.book_id.isin(catalogo).mean()
en_catalogo = gustados[gustados.book_id.isin(catalogo)]
tam = en_catalogo.groupby("user_id").size()
print(f"\nCatálogo: los {N_LIBROS} libros más gustados, que concentran el {cobertura_catalogo:.1%} de las 'compras'")
print(f"Usuarios con al menos un libro del catálogo: {len(tam)}; libros por canasta: mediana {tam.median():.0f}, máximo {tam.max()}")

usuarios = np.sort(tam.index.values)
M = np.zeros((len(usuarios), N_LIBROS), dtype=bool)
fila = {u: i for i, u in enumerate(usuarios)}
M[en_catalogo.user_id.map(fila).values, en_catalogo.book_id.map(columna).values] = True
rng = np.random.default_rng(SEMILLA)
es_prueba = np.zeros(len(usuarios), dtype=bool)
es_prueba[rng.choice(len(usuarios), size=int(0.2 * len(usuarios)), replace=False)] = True
M_ent, M_prueba = M[~es_prueba], M[es_prueba]
print(f"Usuarios de entrenamiento: {len(M_ent)}, de prueba: {len(M_prueba)}; densidad de la matriz: {M_ent.mean():.1%}")
soporte_libro = M_ent.mean(axis=0)
print("\nLos 10 libros más populares del catálogo (% de las canastas de entrenamiento):")
for i in range(10):
    print(f"  {soporte_libro[i]:.1%}  {titulos[i]} ({autores[i]})")

fig, ejes = plt.subplots(1, 3, figsize=(14, 3.8))
ejes[0].bar(sorted(calificaciones.rating.unique()), calificaciones.rating.value_counts().sort_index().values / 1e6, color="#4C78A8")
ejes[0].set_xlabel("Calificación")
ejes[0].set_ylabel("Millones")
ejes[0].set_title("Calificaciones")
ejes[1].hist(tam.values, bins=40, color="#72B7B2")
ejes[1].set_xlabel(f"Libros del catálogo por canasta")
ejes[1].set_ylabel("Usuarios")
ejes[1].set_title("Tamaño de las canastas")
ejes[2].plot(np.arange(1, len(popularidad_total) + 1), popularidad_total.values, color="#E45756")
ejes[2].axvline(N_LIBROS, color="gray", linestyle="--", label=f"catálogo ({N_LIBROS} libros)")
ejes[2].set_xscale("log")
ejes[2].set_yscale("log")
ejes[2].set_xlabel("Libros ordenados por popularidad")
ejes[2].set_ylabel("Usuarios a los que gustó")
ejes[2].set_title("Cola larga de popularidad")
ejes[2].legend()
plt.tight_layout()
plt.savefig(f"{IMG}/datos.png", dpi=200)
plt.close()


# =============================================================================
# TAREA 1: ANÁLISIS DE COMPRAS (¿qué libros se compran juntos?)
# =============================================================================
def minar_reglas(matriz, soporte_minimo, confianza_minima, max_len=3):
    """Apriori + reglas con un solo libro como consecuente. Devuelve (reglas, itemsets, segundos)."""
    t0 = time.time()
    df = pd.DataFrame(matriz, columns=range(matriz.shape[1]))
    itemsets = apriori(df, min_support=soporte_minimo, use_colnames=True, max_len=max_len, low_memory=True)
    reglas = association_rules(itemsets, num_itemsets=len(df), metric="confidence", min_threshold=confianza_minima)
    reglas = reglas[reglas.consequents.map(len) == 1].reset_index(drop=True)
    reglas["ants"] = reglas.antecedents.map(lambda s: tuple(sorted(s)))
    reglas["cons"] = reglas.consequents.map(lambda s: next(iter(s)))
    return reglas, itemsets, time.time() - t0


def nombre_regla(r):
    return " + ".join(titulos[a] for a in r.ants) + " → " + titulos[r.cons]


SOPORTE, CONFIANZA = 0.03, 0.30
reglas, itemsets, seg = minar_reglas(M_ent, SOPORTE, CONFIANZA)
print(f"Apriori (soporte >= {SOPORTE:.0%}, confianza >= {CONFIANZA:.0%}, a lo más 3 libros): {len(itemsets)} conjuntos frecuentes, "
      f"{len(reglas)} reglas con un libro de consecuente, {seg:.1f} s")
print(f"Reglas con un libro de antecedente: {(reglas.ants.map(len) == 1).sum()}; con dos: {(reglas.ants.map(len) == 2).sum()}")
print(f"Lift: mediana {reglas.lift.median():.2f}, máximo {reglas.lift.max():.2f}; reglas con lift > 3: {(reglas.lift > 3).sum()}")

print("\nLas 15 reglas con mayor lift:")
top = reglas.sort_values("lift", ascending=False).head(15)
print(pd.DataFrame({"regla": [nombre_regla(r) for _, r in top.iterrows()], "soporte": top.support.values,
                    "confianza": top.confidence.values, "lift": top.lift.values}).round(3).to_string(index=False))
print("\nLas 10 reglas con mayor confianza (con lift):")
top_c = reglas.sort_values("confidence", ascending=False).head(10)
print(pd.DataFrame({"regla": [nombre_regla(r) for _, r in top_c.iterrows()], "soporte": top_c.support.values,
                    "confianza": top_c.confidence.values, "lift": top_c.lift.values}).round(3).to_string(index=False))

# ¿Los libros que se compran juntos son del mismo autor?
def mismo_autor(r):
    return any(autores[r.cons] == autores[a] for a in r.ants)


reglas["mismo_autor"] = [mismo_autor(r) for _, r in reglas.iterrows()]
print("\nReglas cuyo consecuente es del mismo autor que algún antecedente, según el lift de la regla:")
reglas["tramo_lift"] = pd.cut(reglas.lift, [0, 1.5, 2, 3, 5, 100], labels=["<=1.5", "1.5-2", "2-3", "3-5", ">5"])
print(reglas.groupby("tramo_lift", observed=True).agg(reglas=("lift", "size"), mismo_autor=("mismo_autor", "mean")).round(3).to_string())
print(f"Total: {reglas.mismo_autor.mean():.1%} de las reglas son del mismo autor (azar: {np.mean([autores[i] == autores[j] for i in range(N_LIBROS) for j in range(N_LIBROS) if i != j]):.2%} de los pares de libros)")

print("\n--- Sensibilidad a los umbrales (soporte, confianza) ---")
filas = []
for sop in (0.02, 0.03, 0.05, 0.10):
    for conf in (0.2, 0.4):
        for ml in (2, 3):
            r_, it_, s_ = minar_reglas(M_ent, sop, conf, ml)
            filas.append({"soporte": sop, "confianza": conf, "max_libros": ml, "conjuntos": len(it_), "reglas": len(r_), "segundos": s_})
sens_reglas = pd.DataFrame(filas)
print(sens_reglas.round({"segundos": 1}).to_string(index=False))

fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
sc = ejes[0].scatter(reglas.support, reglas.confidence, c=reglas.lift, cmap="viridis", s=14, alpha=0.7)
plt.colorbar(sc, ax=ejes[0], label="lift")
ejes[0].set_xlabel("Soporte")
ejes[0].set_ylabel("Confianza")
ejes[0].set_title(f"{len(reglas)} reglas descubiertas")
top10 = reglas.sort_values("lift", ascending=False).head(12)
ejes[1].barh([nombre_regla(r)[:70] for _, r in top10.iterrows()][::-1], top10.lift.values[::-1], color="#4C78A8")
ejes[1].set_xlabel("Lift")
ejes[1].set_title("Reglas de mayor lift")
ejes[1].tick_params(axis="y", labelsize=7)
plt.tight_layout()
plt.savefig(f"{IMG}/reglas.png", dpi=200)
plt.close()


# =============================================================================
# TAREA 2: SISTEMA DE RECOMENDACIÓN BASADO EN EL CARRITO
# =============================================================================
class RecomendadorReglas:
    """Recomienda libros a partir del carrito: cada regla cuyo antecedente está completo en el carrito vota por su consecuente.
    puntaje: "confianza" (la mayor confianza), "lift" (el mayor lift) o "suma" (suma de las confianzas de todas las reglas activas).
    Los libros sin ninguna regla activa se ordenan por popularidad (así siempre se entregan k libros)."""

    def __init__(self, reglas, soporte_libro, puntaje="confianza"):
        self.a0 = np.array([r[0] for r in reglas.ants])
        self.a1 = np.array([r[1] if len(r) > 1 else r[0] for r in reglas.ants])
        self.cons = reglas.cons.values
        self.valor = {"confianza": reglas.confidence.values, "lift": reglas.lift.values, "suma": reglas.confidence.values}[puntaje]
        self.suma = puntaje == "suma"
        self.desempate = 1e-9 * soporte_libro / soporte_libro.max()

    def puntajes(self, carritos):
        """carritos: matriz booleana (usuarios x libros). Devuelve (puntajes, usuarios con alguna regla activa)."""
        p = np.zeros(carritos.shape, dtype=float)
        for a0, a1, c, v in zip(self.a0, self.a1, self.cons, self.valor):
            activa = carritos[:, a0] & carritos[:, a1] & ~carritos[:, c]
            if self.suma:
                p[activa, c] += v
            else:
                p[activa, c] = np.maximum(p[activa, c], v)
        return p, (p > 0).any(axis=1)

    def recomendar_matriz(self, carritos, k=10):
        p, cubiertos = self.puntajes(carritos)
        p = p + self.desempate
        p[carritos] = -1
        return np.argsort(-p, axis=1)[:, :k], cubiertos

    def recomendar(self, carrito_titulos, k=5):
        """Interfaz para un carrito con títulos; devuelve el libro, el puntaje y la regla que lo explica."""
        carrito = np.zeros((1, N_LIBROS), dtype=bool)
        for t in carrito_titulos:
            carrito[0, titulos.index(t)] = True
        recs, _ = self.recomendar_matriz(carrito, k)
        salida = []
        for c in recs[0]:
            activas = [i for i in range(len(self.cons)) if self.cons[i] == c and carrito[0, self.a0[i]] and carrito[0, self.a1[i]]]
            if activas:
                i = max(activas, key=lambda j: self.valor[j])
                motivo = f"por {titulos[self.a0[i]]}" + (f" + {titulos[self.a1[i]]}" if self.a1[i] != self.a0[i] else "") + f" (confianza {reglas_conf[(self.a0[i], self.a1[i], c)]:.0%})"
            else:
                motivo = "popular (ninguna regla aplica)"
            salida.append((titulos[c], motivo))
        return salida


reglas_conf = {(r.ants[0], r.ants[-1], r.cons): r.confidence for _, r in reglas.iterrows()}
recomendador = RecomendadorReglas(reglas, soporte_libro, "confianza")
print("Ejemplos de recomendación a partir del carrito:")
buscar = lambda texto: next(t for t in titulos if t.startswith(texto))
ejemplos = [["The Hunger Games", "Catching Fire"],
            [buscar("Harry Potter and the Sorcerer"), buscar("Harry Potter and the Chamber")],
            [buscar("The Fellowship of the Ring")],
            ["Gone Girl"],
            [buscar("Pride and Prejudice"), buscar("Jane Eyre")]]
for carrito in ejemplos:
    print(f"\n  Carrito: {', '.join(carrito)}")
    for t, motivo in recomendador.recomendar(carrito, 5):
        print(f"    - {t}   [{motivo}]")


# =============================================================================
# TAREA 3: EVALUACIÓN DE LAS RECOMENDACIONES
# =============================================================================
def preparar_evaluacion(M_prueba, semilla, tam_carrito=CARRITO):
    """Usuarios de prueba con al menos MIN_OCULTOS libros: se sortean tam_carrito para el carrito y el resto queda oculto."""
    r = np.random.default_rng(semilla)
    ok = M_prueba.sum(1) >= MIN_OCULTOS
    X = M_prueba[ok]
    claves = np.where(X, r.random(X.shape), np.inf)
    elegidos = np.argsort(claves, axis=1)[:, :tam_carrito]
    carrito = np.zeros_like(X)
    np.put_along_axis(carrito, elegidos, True, axis=1)
    return carrito, X & ~carrito


def metricas(recs, ocultos, cubiertos=None):
    """Hit rate, precisión, recall y NDCG de la lista (usuarios x k) frente a los libros ocultos."""
    k = recs.shape[1]
    acierto = np.take_along_axis(ocultos, recs, axis=1)
    descuento = 1 / np.log2(np.arange(2, k + 2))
    n_ocultos = ocultos.sum(1)
    ideal = np.array([descuento[:min(n, k)].sum() for n in n_ocultos])
    por_usuario = {"hit": acierto.any(1).astype(float), "precisión": acierto.mean(1), "recall": acierto.sum(1) / n_ocultos,
                   "NDCG": (acierto * descuento).sum(1) / ideal}
    res = {m: v.mean() for m, v in por_usuario.items()}
    res["cobertura del catálogo"] = len(np.unique(recs)) / N_LIBROS
    res["popularidad media"] = soporte_libro[recs].mean()
    if cubiertos is not None:
        res["carritos con regla"] = cubiertos.mean()
    return res, por_usuario


def recomendar_aleatorio(carritos, k, semilla):
    p = np.random.default_rng(semilla).random(carritos.shape)
    p[carritos] = -1
    return np.argsort(-p, axis=1)[:, :k]


def recomendar_populares(carritos, k):
    p = np.tile(soporte_libro, (len(carritos), 1))
    p[carritos] = -1
    return np.argsort(-p, axis=1)[:, :k]


SIM = (M_ent.T.astype(float) @ M_ent.astype(float))
norma = np.sqrt(np.diag(SIM))
SIM = SIM / np.outer(norma, norma)
np.fill_diagonal(SIM, 0)


def recomendar_knn(carritos, k):
    p = carritos.astype(float) @ SIM + 1e-9 * soporte_libro
    p[carritos] = -1
    return np.argsort(-p, axis=1)[:, :k]


def evaluar_metodos(metodos, k, semillas=SEMILLAS):
    """Promedia las métricas de cada método sobre varios sorteos de carritos."""
    filas, detalle = {}, {}
    for s in semillas:
        carrito, ocultos = preparar_evaluacion(M_prueba, s)
        for nombre, f in metodos.items():
            salida = f(carrito, k, s)
            recs, cub = salida if isinstance(salida, tuple) else (salida, None)
            m, por_u = metricas(recs, ocultos, cub)
            filas.setdefault(nombre, []).append(m)
            detalle.setdefault(nombre, []).append(por_u)
    tabla = pd.DataFrame({n: pd.DataFrame(v).mean() for n, v in filas.items()}).T
    desv = pd.DataFrame({n: pd.DataFrame(v)["hit"].std() for n, v in filas.items()}, index=["desv. hit"]).T
    return tabla.join(desv), detalle


carrito0, ocultos0 = preparar_evaluacion(M_prueba, SEMILLAS[0])
print(f"Evaluación: {len(carrito0)} usuarios de prueba con al menos {MIN_OCULTOS} libros del catálogo; carrito de {CARRITO} libros "
      f"sorteados y {ocultos0.sum(1).mean():.1f} libros ocultos en promedio ({len(SEMILLAS)} sorteos de carritos)")

rec_conf = RecomendadorReglas(reglas, soporte_libro, "confianza")
rec_lift = RecomendadorReglas(reglas, soporte_libro, "lift")
rec_suma = RecomendadorReglas(reglas, soporte_libro, "suma")
reglas2, _, _ = minar_reglas(M_ent, SOPORTE, CONFIANZA, max_len=2)
rec_pares = RecomendadorReglas(reglas2, soporte_libro, "suma")
metodos = {"Aleatorio": lambda c, k, s: recomendar_aleatorio(c, k, s),
           "Más populares": lambda c, k, s: recomendar_populares(c, k),
           "Libros similares (coseno)": lambda c, k, s: recomendar_knn(c, k),
           "Reglas: mayor confianza": lambda c, k, s: rec_conf.recomendar_matriz(c, k),
           "Reglas: mayor lift": lambda c, k, s: rec_lift.recomendar_matriz(c, k),
           "Reglas: suma de confianzas": lambda c, k, s: rec_suma.recomendar_matriz(c, k),
           "Reglas de a pares (suma)": lambda c, k, s: rec_pares.recomendar_matriz(c, k)}
res = {}
for k in (5, 10):
    t0 = time.time()
    res[k], det = evaluar_metodos(metodos, k)
    print(f"\n--- Listas de {k} libros (media de {len(SEMILLAS)} sorteos de carritos; {time.time() - t0:.0f} s) ---")
    print(res[k].round(4).to_string())
    if k == 10:
        detalle10 = det

print("\n--- ¿La diferencia contra \"Más populares\" es real? Hit rate@10, intervalo de confianza del 95 % (bootstrap sobre usuarios, primer sorteo) ---")
rng_b = np.random.default_rng(SEMILLA)
pop_hit = detalle10["Más populares"][0]["hit"]
for nombre in ("Libros similares (coseno)", "Reglas: mayor confianza", "Reglas: mayor lift", "Reglas: suma de confianzas", "Reglas de a pares (suma)"):
    d = detalle10[nombre][0]["hit"] - pop_hit
    idx = rng_b.integers(0, len(d), size=(2000, len(d)))
    medias = d[idx].mean(axis=1)
    print(f"  {nombre}: diferencia {d.mean():+.4f} (IC 95 %: {np.percentile(medias, 2.5):+.4f} a {np.percentile(medias, 97.5):+.4f}); "
          f"gana en {np.mean(d > 0):.1%} de los usuarios, pierde en {np.mean(d < 0):.1%}")

d = detalle10["Reglas: mayor confianza"][0]["hit"] - detalle10["Libros similares (coseno)"][0]["hit"]
idx = rng_b.integers(0, len(d), size=(2000, len(d)))
medias = d[idx].mean(axis=1)
dp = detalle10["Reglas: mayor confianza"][0]["precisión"] - detalle10["Libros similares (coseno)"][0]["precisión"]
mp = dp[idx].mean(axis=1)
print(f"  Reglas (mayor confianza) contra Libros similares: hit {d.mean():+.4f} (IC 95 %: {np.percentile(medias, 2.5):+.4f} a {np.percentile(medias, 97.5):+.4f}); "
      f"precisión@10 {dp.mean():+.4f} (IC 95 %: {np.percentile(mp, 2.5):+.4f} a {np.percentile(mp, 97.5):+.4f})")

print("\n--- Efecto del tamaño del carrito (hit rate@10 y precisión@10, un sorteo) ---")
filas = []
for n in (1, 2, 3, 4, 5):
    carrito, ocultos = preparar_evaluacion(M_prueba, SEMILLAS[0], tam_carrito=n)
    fila_ = {"libros en el carrito": n}
    for nombre, f in (("Más populares", lambda c: recomendar_populares(c, 10)), ("Libros similares", lambda c: recomendar_knn(c, 10)),
                      ("Reglas (suma)", lambda c: rec_suma.recomendar_matriz(c, 10))):
        salida = f(carrito)
        recs, cub = salida if isinstance(salida, tuple) else (salida, None)
        m, _ = metricas(recs, ocultos, cub)
        fila_[f"hit {nombre}"] = m["hit"]
        fila_[f"precisión {nombre}"] = m["precisión"]
        if cub is not None:
            fila_["carritos con regla"] = cub.mean()
    filas.append(fila_)
por_tam = pd.DataFrame(filas).set_index("libros en el carrito")
print(por_tam.round(3).to_string())

print("\n--- Sensibilidad de las reglas a los umbrales (suma de confianzas, hit rate@10 y precisión@10, un sorteo) ---")
carrito, ocultos = preparar_evaluacion(M_prueba, SEMILLAS[0])
filas = []
for sop in (0.02, 0.03, 0.05, 0.10):
    for conf in (0.2, 0.3, 0.5):
        r_, _, _ = minar_reglas(M_ent, sop, conf)
        if len(r_) == 0:
            continue
        recs, cub = RecomendadorReglas(r_, soporte_libro, "suma").recomendar_matriz(carrito, 10)
        m, _ = metricas(recs, ocultos, cub)
        filas.append({"soporte": sop, "confianza": conf, "reglas": len(r_), "hit@10": m["hit"], "precisión@10": m["precisión"],
                      "carritos con regla": m["carritos con regla"], "cobertura": m["cobertura del catálogo"]})
sens = pd.DataFrame(filas)
print(sens.round(3).to_string(index=False))

print("\n--- ¿Qué recomienda cada método? ---")
recs_pop = recomendar_populares(carrito, 10)
recs_reg, cub_reg = rec_suma.recomendar_matriz(carrito, 10)
mismo = lambda recs: np.mean([any(autores[r] == autores[c] for c in np.where(carrito[u])[0]) for u in range(len(recs)) for r in recs[u]])
print(f"Libros recomendados del mismo autor que algún libro del carrito: reglas {mismo(recs_reg):.1%}, populares {mismo(recs_pop):.1%}")
_, por_u_reg = metricas(recs_reg, ocultos, cub_reg)
_, por_u_pop = metricas(recs_pop, ocultos)
print(f"Hit rate@10 con regla activa: reglas {por_u_reg['hit'][cub_reg].mean():.3f}, populares {por_u_pop['hit'][cub_reg].mean():.3f} "
      f"(carritos cubiertos: {cub_reg.mean():.1%}); sin regla activa: reglas {por_u_reg['hit'][~cub_reg].mean():.3f}, populares {por_u_pop['hit'][~cub_reg].mean():.3f}")

fig, ejes = plt.subplots(1, 3, figsize=(15, 4.4))
t10 = res[10].sort_values("hit")
ejes[0].barh(t10.index, t10.hit * 100, xerr=t10["desv. hit"] * 100, color="#4C78A8", capsize=3)
ejes[0].set_xlabel("Hit rate@10 (%)")
ejes[0].set_title("Usuarios con al menos un acierto en 10 libros")
ejes[0].tick_params(axis="y", labelsize=8)
ejes[1].plot(por_tam.index, por_tam["hit Más populares"] * 100, "o-", label="Más populares")
ejes[1].plot(por_tam.index, por_tam["hit Libros similares"] * 100, "s-", label="Libros similares")
ejes[1].plot(por_tam.index, por_tam["hit Reglas (suma)"] * 100, "^-", label="Reglas (suma)")
ejes[1].set_xlabel("Libros en el carrito")
ejes[1].set_ylabel("Hit rate@10 (%)")
ejes[1].set_title("Efecto del tamaño del carrito")
ejes[1].legend(fontsize=8)
pivote = sens.pivot(index="soporte", columns="confianza", values="hit@10")
sns.heatmap(pivote * 100, annot=True, fmt=".1f", cmap="Blues", ax=ejes[2], cbar_kws={"label": "hit rate@10 (%)"})
ejes[2].set_title("Reglas: soporte y confianza mínimos")
plt.tight_layout()
plt.savefig(f"{IMG}/evaluacion.png", dpi=200)
plt.close()
print("\nListo. Figuras guardadas en", IMG)
