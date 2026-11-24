# =============================================================================
# Actividad 21: "Sistema de Recomendación de Películas"
# Tema 4.3. Aprendizaje automático
#
# Dependencias (en Colab ya vienen instaladas):  pip install scikit-learn pandas matplotlib seaborn scipy
# El conjunto de datos (MovieLens ml-latest-small, ~1 MB) se descarga solo la primera vez.
# =============================================================================
import io
import os
import urllib.request
import warnings
import zipfile
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

SEMILLA = 42
AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
IMG = os.path.join(AQUI, "img")
DATOS = os.path.join(AQUI, "datos")
os.makedirs(IMG, exist_ok=True)
os.makedirs(DATOS, exist_ok=True)
sns.set_theme(style="whitegrid")

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
N = 10              # tamaño de la lista de recomendaciones
UMBRAL = 4.0        # una película es "relevante" si el usuario la calificó con 4 o más
SUAVIZADO = 5       # pseudo-calificaciones para suavizar promedios con pocas calificaciones
K_ELEGIDO = None    # se decide más abajo con la inercia y la silueta


# =============================================================================
# DATOS
# =============================================================================
def cargar_datos():
    carpeta = os.path.join(DATOS, "ml-latest-small")
    if not os.path.exists(os.path.join(carpeta, "ratings.csv")):
        with urllib.request.urlopen(URL, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATOS)
    ratings = pd.read_csv(os.path.join(carpeta, "ratings.csv"))
    peliculas = pd.read_csv(os.path.join(carpeta, "movies.csv"))
    return ratings, peliculas


def dividir(ratings, semilla, fraccion=0.2):
    """Aparta al azar el 20 % de las calificaciones de CADA usuario como prueba."""
    rng = np.random.default_rng(semilla)
    elegidos = []
    for _, g in ratings.groupby("userId"):
        n = max(1, int(round(fraccion * len(g))))
        elegidos.extend(rng.choice(g.index.values, size=n, replace=False))
    prueba = ratings.loc[elegidos]
    return ratings.drop(prueba.index), prueba


# =============================================================================
# TAREA 1: AGRUPAR USUARIOS
# =============================================================================
def matriz_usuarios(ent, usuarios, peliculas_ids):
    """Matriz usuario-película con calificaciones centradas en el promedio de cada usuario
    (así se compara el GUSTO y no si el usuario es generoso o estricto). Lo no calificado vale 0."""
    u = pd.Categorical(ent.userId, categories=usuarios).codes
    m = pd.Categorical(ent.movieId, categories=peliculas_ids).codes
    medias = ent.groupby("userId").rating.mean().reindex(usuarios).values
    centrado = ent.rating.values - medias[u]
    return sparse.csr_matrix((centrado, (u, m)), shape=(len(usuarios), len(peliculas_ids))), medias


def embedding(R, componentes=20):
    """Reduce la matriz (610 x ~9000, casi vacía) con SVD truncado y normaliza cada usuario."""
    svd = TruncatedSVD(n_components=componentes, random_state=SEMILLA)
    Z = svd.fit_transform(R)
    return normalize(Z), svd.explained_variance_ratio_.sum()


def agrupar(Z, k):
    return KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(Z)


# =============================================================================
# TAREA 2: SISTEMA DE RECOMENDACIÓN
# =============================================================================
def promedio_suavizado(suma, cuenta, previo, m=SUAVIZADO):
    """Promedio con m calificaciones 'virtuales' iguales al valor previo: evita que una película
    con una sola calificación de 5 estrellas encabece la lista."""
    return (suma + m * previo) / (cuenta + m)


def puntajes_por_grupo(ent, grupo_de_usuario, k, peliculas_ids):
    """Puntaje de cada película en cada grupo: promedio suavizado de las calificaciones de sus miembros,
    con el promedio global de la película como valor previo."""
    mu = ent.rating.mean()
    glob = ent.groupby("movieId").rating.agg(["sum", "count"]).reindex(peliculas_ids).fillna(0)
    prev_peli = promedio_suavizado(glob["sum"].values, glob["count"].values, mu)
    idx = pd.Categorical(ent.movieId, categories=peliculas_ids).codes
    g = grupo_de_usuario[ent.userId.values]
    S = np.zeros((k, len(peliculas_ids)))
    C = np.zeros((k, len(peliculas_ids)))
    L = np.zeros((k, len(peliculas_ids)))
    np.add.at(S, (g, idx), ent.rating.values)
    np.add.at(C, (g, idx), 1)
    np.add.at(L, (g, idx), (ent.rating.values >= UMBRAL).astype(float))
    return promedio_suavizado(S, C, prev_peli[None, :]), C, prev_peli, L


def recomendar(puntaje, vistas, n=N):
    p = puntaje.copy()
    p[list(vistas)] = -np.inf
    return np.argsort(-p)[:n]


# =============================================================================
# TAREA 3: EVALUACIÓN
# =============================================================================
def dcg(rel):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))


def evaluar_listas(listas, prueba_relevantes, n=N):
    """listas: dict usuario -> lista de índices de película. Devuelve precisión, recall, NDCG y acierto@n."""
    P, R, ND, H = [], [], [], []
    for u, lista in listas.items():
        rel = prueba_relevantes.get(u, set())
        if not rel:
            continue
        aciertos = [1 if m in rel else 0 for m in lista[:n]]
        P.append(sum(aciertos) / n)
        R.append(sum(aciertos) / len(rel))
        ideal = dcg([1] * min(len(rel), n))
        ND.append(dcg(aciertos) / ideal)
        H.append(1.0 if sum(aciertos) else 0.0)
    return {"precisión@10": np.mean(P), "recall@10": np.mean(R), "NDCG@10": np.mean(ND), "acierto@10": np.mean(H)}


def experimento(ratings, semilla, k, detalle=False):
    """Divide, agrupa, recomienda y evalúa. Todo el ajuste usa SOLO el conjunto de entrenamiento."""
    ent, prueba = dividir(ratings, semilla)
    usuarios = np.sort(ratings.userId.unique())
    peliculas_ids = np.sort(ratings.movieId.unique())
    R, medias = matriz_usuarios(ent, usuarios, peliculas_ids)
    Z, var = embedding(R)
    grupos = agrupar(Z, k)
    grupo_de_usuario = pd.Series(grupos, index=usuarios).reindex(np.arange(usuarios.max() + 1)).fillna(-1).astype(int).values

    pos = {m: i for i, m in enumerate(peliculas_ids)}
    vistas = ent.groupby("userId").movieId.apply(lambda s: {pos[m] for m in s}).to_dict()
    relev = prueba[prueba.rating >= UMBRAL].groupby("userId").movieId.apply(lambda s: {pos[m] for m in s}).to_dict()

    punt_grupo, cuentas, prev_peli, gustadas = puntajes_por_grupo(ent, grupo_de_usuario, k, peliculas_ids)
    cuenta_global = ent.groupby("movieId").size().reindex(peliculas_ids).fillna(0).values
    rng = np.random.default_rng(semilla)

    N_A, N_B = f"Grupo: mejor valoradas (k={k})", f"Grupo: más gustadas (k={k})"
    listas = {"Aleatorio": {}, "Más vistas": {}, "Mejor valoradas (global)": {}, N_A: {}, N_B: {}}
    for u in usuarios:
        v = vistas[u]
        listas["Aleatorio"][u] = rng.permutation([i for i in range(len(peliculas_ids)) if i not in v])[:N]
        listas["Más vistas"][u] = recomendar(cuenta_global.astype(float), v)
        listas["Mejor valoradas (global)"][u] = recomendar(prev_peli, v)
        listas[N_A][u] = recomendar(punt_grupo[grupo_de_usuario[u]], v)
        listas[N_B][u] = recomendar(gustadas[grupo_de_usuario[u]], v)

    resultados = {nombre: evaluar_listas(l, relev) for nombre, l in listas.items()}
    # cobertura y popularidad de lo recomendado
    for nombre, l in listas.items():
        todas = np.concatenate([np.asarray(x) for x in l.values()])
        resultados[nombre]["cobertura"] = len(set(todas)) / len(peliculas_ids)
        resultados[nombre]["popularidad media"] = np.log10(1 + cuenta_global[todas]).mean()

    # predicción de calificaciones (RMSE) sobre todas las calificaciones de prueba
    mu = ent.rating.mean()
    pi = pd.Categorical(prueba.movieId, categories=peliculas_ids).codes
    gu = grupo_de_usuario[prueba.userId.values]
    real = prueba.rating.values
    pred_global = np.full(len(prueba), mu)
    pred_peli = prev_peli[pi]
    pred_grupo = punt_grupo[gu, pi]
    rmse = {"Promedio global": np.sqrt(np.mean((real - pred_global) ** 2)),
            "Promedio de la película": np.sqrt(np.mean((real - pred_peli) ** 2)),
            f"Promedio de la película en el grupo (k={k})": np.sqrt(np.mean((real - pred_grupo) ** 2))}

    if not detalle:
        return resultados, rmse
    return dict(resultados=resultados, rmse=rmse, ent=ent, prueba=prueba, usuarios=usuarios, peliculas_ids=peliculas_ids,
                R=R, Z=Z, var=var, grupos=grupos, punt_grupo=punt_grupo, cuentas=cuentas, vistas=vistas, relev=relev,
                pos=pos, gustadas=gustadas, listas=listas, cuenta_global=cuenta_global)


# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__" or "__file__" not in globals():
    ratings, peliculas = cargar_datos()
    titulo = peliculas.set_index("movieId").title
    generos = peliculas.set_index("movieId").genres.str.split("|")

    print("=" * 70 + "\nDATOS\n" + "=" * 70)
    print(f"Calificaciones: {len(ratings)}, usuarios: {ratings.userId.nunique()}, películas calificadas: {ratings.movieId.nunique()}"
          f" (de {len(peliculas)} en el catálogo)")
    dens = len(ratings) / (ratings.userId.nunique() * ratings.movieId.nunique())
    print(f"Densidad de la matriz usuario-película: {dens:.2%} (el {1 - dens:.2%} de las celdas está vacío)")
    por_usuario = ratings.groupby("userId").size()
    por_peli = ratings.groupby("movieId").size()
    print(f"Calificaciones por usuario: mínimo {por_usuario.min()}, mediana {por_usuario.median():.0f}, máximo {por_usuario.max()}")
    print(f"Calificaciones por película: mediana {por_peli.median():.0f}, máximo {por_peli.max()}; "
          f"{(por_peli == 1).mean():.1%} de las películas tienen una sola calificación")
    print("Distribución de calificaciones:")
    print(ratings.rating.value_counts().sort_index().to_string())
    print(f"Promedio: {ratings.rating.mean():.2f}; fracción >= {UMBRAL}: {(ratings.rating >= UMBRAL).mean():.1%}")

    fig, ejes = plt.subplots(1, 3, figsize=(14, 3.8))
    ratings.rating.value_counts().sort_index().plot.bar(ax=ejes[0], color="#264653")
    ejes[0].set_title("Calificaciones")
    ejes[0].set_xlabel("Estrellas")
    ejes[1].hist(por_usuario, bins=40, color="#2a9d8f")
    ejes[1].set_xscale("log")
    ejes[1].set_title("Calificaciones por usuario")
    ejes[1].set_xlabel("Número de calificaciones (escala log)")
    ejes[2].hist(por_peli, bins=40, color="#e76f51")
    ejes[2].set_xscale("log")
    ejes[2].set_yscale("log")
    ejes[2].set_title("Calificaciones por película")
    ejes[2].set_xlabel("Número de calificaciones (escala log)")
    plt.tight_layout()
    plt.savefig(f"{IMG}/datos.png", dpi=200)
    plt.close()

    # ---- Partición y elección de k -------------------------------------------------
    print("\n" + "=" * 70 + "\nTAREA 1: AGRUPAR USUARIOS\n" + "=" * 70)
    ent, prueba = dividir(ratings, SEMILLA)
    print(f"Entrenamiento: {len(ent)} calificaciones; prueba: {len(prueba)} "
          f"({(prueba.rating >= UMBRAL).sum()} relevantes, con calificación >= {UMBRAL})")
    usuarios = np.sort(ratings.userId.unique())
    peliculas_ids = np.sort(ratings.movieId.unique())
    R, medias = matriz_usuarios(ent, usuarios, peliculas_ids)
    Z, var = embedding(R)
    print(f"Matriz de usuarios: {R.shape}; el SVD con 20 componentes conserva {var:.1%} de la varianza")

    ks = list(range(2, 13))
    inercias, siluetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(Z)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(Z, km.labels_))
    tab_k = pd.DataFrame({"k": ks, "inercia": inercias, "silueta": siluetas}).set_index("k")
    print("\nElección del número de grupos:")
    print(tab_k.round(3).to_string())
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    ejes[0].plot(ks, inercias, "bo-")
    ejes[0].set_xlabel("Número de grupos (k)")
    ejes[0].set_ylabel("Inercia")
    ejes[0].set_title("Método del codo")
    ejes[1].plot(ks, siluetas, "go-")
    ejes[1].set_xlabel("Número de grupos (k)")
    ejes[1].set_ylabel("Coeficiente de silueta")
    ejes[1].set_title("Silueta")
    plt.tight_layout()
    plt.savefig(f"{IMG}/eleccion_k.png", dpi=200)
    plt.close()

    # La silueta es baja y el codo no es claro, así que k se elige por desempeño en VALIDACIÓN:
    # se aparta otro 20 % del entrenamiento (sin tocar la prueba) y se prueba cada k con 3 particiones.
    print("\nElección de k por validación (otro 20 % del entrenamiento, 3 particiones; la prueba no se usa):")
    val = {}
    for k in [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60]:
        rs = [experimento(ent, s, k)[0][f"Grupo: más gustadas (k={k})"] for s in [100, 101, 102]]
        val[k] = {"precisión@10": np.mean([r["precisión@10"] for r in rs]), "NDCG@10": np.mean([r["NDCG@10"] for r in rs])}
    val = pd.DataFrame(val).T
    val.index.name = "k"
    print(val.round(4).to_string())
    K_ELEGIDO = int(val["NDCG@10"].idxmax())
    print(f"\nk elegido (mayor NDCG@10 en validación): {K_ELEGIDO}")

    # ---- Resultados con la partición principal -------------------------------------
    d = experimento(ratings, SEMILLA, K_ELEGIDO, detalle=True)
    grupos, punt, cuentas = d["grupos"], d["punt_grupo"], d["cuentas"]
    print("Tamaño de los grupos:", {int(a): b for a, b in sorted(Counter(grupos).items())})

    # perfil de cada grupo
    print("\nPerfil de los grupos (entrenamiento):")
    gen_global = Counter(g for m in ent.movieId for g in generos.get(m, []))
    tot = sum(gen_global.values())
    gu = pd.Series(grupos, index=usuarios)
    ent_g = ent.assign(grupo=gu.reindex(ent.userId).values)
    perfil_filas, mat_genero = [], {}
    for c in range(K_ELEGIDO):
        sub = ent_g[ent_g.grupo == c]
        gen_c = Counter(g for m in sub.movieId for g in generos.get(m, []))
        tot_c = sum(gen_c.values())
        lift = {g: (gen_c[g] / tot_c) / (gen_global[g] / tot) for g in gen_global if gen_global[g] > 5000 and g != "(no genres listed)"}
        mat_genero[c] = lift
        sobre = sorted(lift, key=lift.get, reverse=True)[:3]
        top_p = (sub.groupby("movieId").rating.agg(["mean", "count"]).query("count >= 5")
                 .sort_values("mean", ascending=False).head(3).index)
        perfil_filas.append({"grupo": c, "usuarios": int((grupos == c).sum()), "calif./usuario": len(sub) / (grupos == c).sum(),
                             "promedio": sub.rating.mean(), "géneros que sobresalen": ", ".join(sobre),
                             "mejor valoradas": "; ".join(titulo[m] for m in top_p)})
    perfil = pd.DataFrame(perfil_filas).set_index("grupo")
    print(perfil.drop(columns="mejor valoradas").round(2).to_string())
    print("\nPelículas mejor valoradas en cada grupo (al menos 5 calificaciones del grupo):")
    for c, t in perfil["mejor valoradas"].items():
        print(f"  Grupo {c}: " + t.replace("; ", "\n           "))

    mg = pd.DataFrame(mat_genero).T
    plt.figure(figsize=(11, 4.5))
    sns.heatmap(mg, annot=True, fmt=".1f", cmap="RdBu_r", center=1, cbar_kws={"label": "Sobrerrepresentación vs. global"})
    plt.xlabel("")
    plt.ylabel("Grupo")
    plt.title("Géneros de las películas calificadas por cada grupo (1.0 = igual que el total)")
    plt.tight_layout()
    plt.savefig(f"{IMG}/generos_grupos.png", dpi=200)
    plt.close()

    xy = PCA(n_components=2, random_state=SEMILLA).fit_transform(d["Z"])
    plt.figure(figsize=(7, 5.5))
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=grupos, palette="tab10", s=28, alpha=0.8)
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title(f"Usuarios agrupados con K-Means (k={K_ELEGIDO}), proyección en 2 dimensiones")
    plt.legend(title="Grupo", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{IMG}/grupos.png", dpi=200)
    plt.close()

    # ---- Recomendaciones de ejemplo ------------------------------------------------
    print("\n" + "=" * 70 + "\nTAREA 2: RECOMENDACIONES\n" + "=" * 70)
    ejemplos = [int(usuarios[i]) for i in [0, 100, 300]]
    for u in ejemplos:
        c = int(gu[u])
        propias = ent[ent.userId == u].sort_values("rating", ascending=False).head(4)
        print(f"\nUsuario {u} (grupo {c}, {len(ent[ent.userId == u])} calificaciones en entrenamiento)")
        print("  Mejor calificadas por el usuario:", "; ".join(f"{titulo[m]} ({r:.1f})" for m, r in zip(propias.movieId, propias.rating)))
        rec = d["listas"][f"Grupo: más gustadas (k={K_ELEGIDO})"][u][:5]
        rel = d["relev"].get(u, set())
        for i in rec:
            m = d["peliculas_ids"][i]
            marca = "ACIERTO (el usuario la calificó >= 4 en prueba)" if i in rel else ""
            print(f"  -> {titulo[m]}  [{int(d["gustadas"][c, i])} de {int((grupos == c).sum())} usuarios del grupo la calificaron >= 4] {marca}")

    # ---- Evaluación ----------------------------------------------------------------
    print("\n" + "=" * 70 + "\nTAREA 3: EVALUACIÓN\n" + "=" * 70)
    tabla = pd.DataFrame(d["resultados"]).T
    print(f"\nMétricas con listas de {N} películas (partición principal; {len(d['relev'])} usuarios con películas relevantes en prueba):")
    print(tabla.round(4).to_string())
    print("\nPredicción de calificaciones de prueba (RMSE, menor es mejor):")
    print(pd.Series(d["rmse"]).round(4).to_string())

    # repetición con 5 particiones distintas
    semillas = [0, 1, 2, 3, 4]
    reps, reps_rmse = [], []
    for s in semillas:
        r, rm = experimento(ratings, s, K_ELEGIDO)
        reps.append(pd.DataFrame(r).T)
        reps_rmse.append(pd.Series(rm))
    media = sum(reps) / len(reps)
    desv = pd.concat(reps).groupby(level=0).std().reindex(media.index)
    print(f"\nPromedio de 5 particiones distintas (semillas {semillas}), media (desviación):")
    resumen = media[["precisión@10", "recall@10", "NDCG@10", "acierto@10"]].round(3).astype(str) + " (" + desv[["precisión@10", "recall@10", "NDCG@10", "acierto@10"]].round(3).astype(str) + ")"
    print(resumen.to_string())
    print("\nRMSE promedio de las 5 particiones:")
    print(pd.concat(reps_rmse, axis=1).mean(axis=1).round(4).to_string())

    ax = media[["precisión@10", "recall@10", "NDCG@10", "acierto@10"]].plot.bar(yerr=desv[["precisión@10", "recall@10", "NDCG@10", "acierto@10"]],
                                                                               figsize=(10, 4.5), rot=10, width=0.8, capsize=2)
    ax.set_title("Métricas de recomendación (promedio de 5 particiones, barras = desviación)")
    ax.set_ylabel("Valor")
    plt.tight_layout()
    plt.savefig(f"{IMG}/metricas.png", dpi=200)
    plt.close()

    # ---- Sensibilidad al número de grupos ------------------------------------------
    print("\nSensibilidad al número de grupos (promedio de 5 particiones; NO se usó para elegir k):")
    filas = []
    for k in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60]:
        rs = [experimento(ratings, s, k)[0][f"Grupo: más gustadas (k={k})"] for s in semillas]
        filas.append({"k": k, "precisión@10": np.mean([r["precisión@10"] for r in rs]), "NDCG@10": np.mean([r["NDCG@10"] for r in rs]),
                      "cobertura": np.mean([r["cobertura"] for r in rs])})
    sens = pd.DataFrame(filas).set_index("k")
    print(sens.round(4).to_string())
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(sens.index, sens["precisión@10"], "o-", color="#e76f51", label="Precisión@10")
    ax1.plot(sens.index, sens["NDCG@10"], "s-", color="#264653", label="NDCG@10")
    ax1.set_xlabel("Número de grupos (k = 1: un solo grupo)")
    ax1.set_ylabel("Valor")
    ax2 = ax1.twinx()
    ax2.plot(sens.index, sens["cobertura"], "^--", color="#2a9d8f", label="Cobertura")
    ax2.set_ylabel("Cobertura del catálogo")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    plt.title("Efecto del número de grupos en la calidad de las recomendaciones")
    plt.tight_layout()
    plt.savefig(f"{IMG}/sensibilidad_k.png", dpi=200)
    plt.close()

    print("\nListo. Figuras guardadas en", IMG)
