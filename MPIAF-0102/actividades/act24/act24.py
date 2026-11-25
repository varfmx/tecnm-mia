# =============================================================================
# Actividad 24: "Detective de Datos: Descubriendo Patrones Ocultos"
# Tema 4.5. Minería de datos
#
# Dependencias (en Colab anteponer "!"):  pip install mlxtend scikit-learn pandas numpy matplotlib seaborn
# No requiere descargas: los datos del supermercado se simulan (con semilla fija).
# Tarda unos 2 minutos.
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
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (adjusted_rand_score, average_precision_score, calinski_harabasz_score,
                             davies_bouldin_score, precision_recall_curve, roc_auc_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

SEMILLA = 42
N_TRANSACCIONES = 5000
AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
IMG = os.path.join(AQUI, "img")
os.makedirs(IMG, exist_ok=True)
sns.set_theme(style="whitegrid")


# =============================================================================
# CÓDIGO BASE DEL TEMA (sin cambios, salvo que devuelve el mismo resultado)
# =============================================================================
def proyecto_mineria_datos():
    """Plantilla para el proyecto de minería de datos"""
    print("Proyecto: Detective de Datos - Supermercado ABC")

    # Simular datos más realistas
    np.random.seed(42)
    n_transacciones = 5000

    # Patrones de compra predefinidos (reglas que queremos descubrir)
    patrones_conocidos = [
        {'antecedentes': ['Pan', 'Leche'], 'consecuentes': ['Mantequilla']},
        {'antecedentes': ['Cerveza', 'Papas'], 'consecuentes': ['Dulces']},
        {'antecedentes': ['Carne', 'Verduras'], 'consecuentes': ['Salsa']}
    ]

    print("\n Patrones que deberíamos descubrir:")
    for i, patron in enumerate(patrones_conocidos):
        print(f"{i+1}. {patron['antecedentes']} → {patron['consecuentes']}")

    # Tu código aquí para descubrir estos patrones automáticamente
    print("\n Tu tarea: Implementa el código para descubrir estos patrones")
    return patrones_conocidos


# Inicia tu proyecto aquí
patrones_objetivo = proyecto_mineria_datos()


# =============================================================================
# DATOS: SIMULACIÓN DEL SUPERMERCADO
# =============================================================================
PRODUCTOS = ['Leche', 'Pan', 'Mantequilla', 'Queso', 'Huevos', 'Cereal', 'Yogur', 'Fruta', 'Verduras', 'Carne',
             'Pescado', 'Arroz', 'Pasta', 'Salsa', 'Refresco', 'Cerveza', 'Papas', 'Dulces', 'Café', 'Té']
PRECIOS = {'Leche': 28, 'Pan': 35, 'Mantequilla': 45, 'Queso': 90, 'Huevos': 55, 'Cereal': 60, 'Yogur': 22,
           'Fruta': 40, 'Verduras': 35, 'Carne': 140, 'Pescado': 160, 'Arroz': 30, 'Pasta': 20, 'Salsa': 25,
           'Refresco': 25, 'Cerveza': 24, 'Papas': 28, 'Dulces': 18, 'Café': 95, 'Té': 45}
CATEGORIAS = {'frescos': ['Fruta', 'Verduras', 'Carne', 'Pescado'],
              'lácteos': ['Leche', 'Mantequilla', 'Queso', 'Yogur', 'Huevos'],
              'abarrotes': ['Pan', 'Cereal', 'Arroz', 'Pasta', 'Salsa'],
              'bebidas': ['Refresco', 'Cerveza', 'Café', 'Té'],
              'snacks': ['Papas', 'Dulces']}
CATEGORIA_DE = {p: c for c, ps in CATEGORIAS.items() for p in ps}

# Patrones plantados: con probabilidad "activación" el ticket incluye todos los antecedentes; cuando están todos,
# el consecuente se agrega con probabilidad "confianza" (además de su probabilidad normal de compra).
PATRONES = [{"antecedentes": ["Pan", "Leche"], "consecuentes": ["Mantequilla"], "activacion": 0.04, "confianza": 0.85},
            {"antecedentes": ["Cerveza", "Papas"], "consecuentes": ["Dulces"], "activacion": 0.035, "confianza": 0.80},
            {"antecedentes": ["Carne", "Verduras"], "consecuentes": ["Salsa"], "activacion": 0.04, "confianza": 0.75}]

# Segmentos reales de clientes (en la vida real no se conocen; aquí sirven para validar el clustering)
SEGMENTOS = [
    {"nombre": "Familias", "peso": 0.30, "visitas": 11, "tamano": 6.5, "hora": (17, 3), "qmax": 3,
     "afin": {"Leche": 3.0, "Pan": 2.5, "Cereal": 3.0, "Yogur": 2.5, "Huevos": 2.5, "Fruta": 1.8, "Verduras": 1.8,
              "Pasta": 2.0, "Arroz": 2.0, "Carne": 1.3, "Refresco": 1.5, "Cerveza": 0.2, "Café": 0.6}},
    {"nombre": "Jóvenes", "peso": 0.30, "visitas": 9, "tamano": 3.5, "hora": (20, 2.5), "qmax": 3,
     "afin": {"Cerveza": 4.0, "Papas": 4.0, "Dulces": 3.5, "Refresco": 3.0, "Café": 2.0, "Pasta": 1.3,
              "Verduras": 0.3, "Fruta": 0.4, "Pescado": 0.2, "Leche": 0.6, "Yogur": 0.5}},
    {"nombre": "Saludables", "peso": 0.25, "visitas": 8, "tamano": 5, "hora": (12, 3), "qmax": 3,
     "afin": {"Fruta": 4.0, "Verduras": 4.0, "Pescado": 4.0, "Yogur": 2.5, "Queso": 3.0, "Té": 4.0, "Café": 1.2,
              "Carne": 0.4, "Refresco": 0.1, "Cerveza": 0.2, "Papas": 0.15, "Dulces": 0.15}},
    {"nombre": "Abasto", "peso": 0.15, "visitas": 2.5, "tamano": 11, "hora": (11, 2), "qmax": 4,
     "afin": {"Arroz": 3.5, "Pasta": 3.0, "Carne": 3.0, "Huevos": 2.5, "Leche": 2.0, "Cereal": 2.0, "Salsa": 2.5,
              "Refresco": 2.0, "Cerveza": 1.2}},
]
TIPOS_ANOMALIA = ["cantidades excesivas", "madrugada con monto alto", "descuento anómalo"]


def generar_datos(semilla=SEMILLA, n_transacciones=N_TRANSACCIONES, n_clientes=600, tasa_anomalias=0.02):
    """Simula clientes, tickets de compra y fraudes. Devuelve (tickets, clientes)."""
    rng = np.random.default_rng(semilla)
    seg = rng.choice(len(SEGMENTOS), size=n_clientes, p=[s["peso"] for s in SEGMENTOS])
    ritmo = np.array([SEGMENTOS[s]["visitas"] for s in seg]) * rng.lognormal(0, 0.3, n_clientes)
    clientes = pd.DataFrame({"cliente": np.arange(n_clientes), "segmento_real": [SEGMENTOS[s]["nombre"] for s in seg]})
    duenos = rng.choice(n_clientes, size=n_transacciones, p=ritmo / ritmo.sum())
    ind = {p: i for i, p in enumerate(PRODUCTOS)}
    filas = []
    for t, c in enumerate(duenos):
        s = SEGMENTOS[seg[c]]
        w = np.array([s["afin"].get(p, 1.0) for p in PRODUCTOS])
        n = int(np.clip(rng.poisson(s["tamano"]), 2, 20))
        items = set(PRODUCTOS[i] for i in rng.choice(len(PRODUCTOS), size=n, replace=False, p=w / w.sum()))
        for pat in PATRONES:
            if rng.random() < pat["activacion"]:
                items |= set(pat["antecedentes"])
            if set(pat["antecedentes"]) <= items and rng.random() < pat["confianza"]:
                items |= set(pat["consecuentes"])
        items = sorted(items, key=ind.get)
        cant = {p: int(min(s["qmax"], 1 + rng.poisson(0.4))) for p in items}
        hora = rng.normal(*s["hora"])
        while not 7 <= hora <= 22:                     # truncada entre 7 y 22 h (sin acumular en los extremos)
            hora = rng.normal(*s["hora"])
        hora = float(hora)
        desc = int(rng.choice([0, 5, 10, 15], p=[0.6, 0.2, 0.15, 0.05]))
        filas.append({"ticket": t, "cliente": int(c), "dia": int(rng.integers(0, 90)), "hora": hora,
                      "productos": items, "cantidades": cant, "descuento_pct": desc, "es_anomalia": 0, "tipo_anomalia": ""})
    tickets = pd.DataFrame(filas)

    # Fraudes: se transforman tickets elegidos al azar
    n_anom = int(round(tasa_anomalias * n_transacciones))
    elegidos = rng.choice(n_transacciones, size=n_anom, replace=False)
    for k, i in enumerate(elegidos):
        tipo = k % 3
        fila = tickets.loc[i].copy()
        if tipo == 0:      # un producto o dos en cantidades enormes
            prods = list(rng.choice(PRODUCTOS, size=int(rng.integers(1, 3)), replace=False))
            fila["productos"], fila["cantidades"] = sorted(prods, key=ind.get), {p: int(rng.integers(12, 31)) for p in prods}
        elif tipo == 1:    # de madrugada y con cantidades altas
            prods = list(rng.choice(PRODUCTOS, size=int(rng.integers(2, 5)), replace=False))
            fila["productos"], fila["cantidades"] = sorted(prods, key=ind.get), {p: int(rng.integers(4, 9)) for p in prods}
            fila["hora"] = float(rng.uniform(1, 4.5))
        else:              # descuento fuera de lo normal
            fila["descuento_pct"] = int(rng.integers(40, 81))
        fila["es_anomalia"], fila["tipo_anomalia"] = 1, TIPOS_ANOMALIA[tipo]
        tickets.loc[i] = fila

    tickets["n_lineas"] = tickets.productos.map(len)
    tickets["n_unidades"] = tickets.cantidades.map(lambda d: sum(d.values()))
    tickets["cantidad_max"] = tickets.cantidades.map(lambda d: max(d.values()))
    tickets["monto"] = [sum(PRECIOS[p] * q for p, q in d.items()) * (1 - ds / 100)
                        for d, ds in zip(tickets.cantidades, tickets.descuento_pct)]
    tickets["segmento_real"] = clientes.segmento_real.values[tickets.cliente.values]
    return tickets, clientes


tickets, clientes = generar_datos()
print("Ejemplos de tickets:")
for _, f in tickets.head(5).iterrows():
    print(f"Ticket {f.ticket}: cliente {f.cliente}, día {f.dia}, {f.hora:.1f} h, ${f.monto:.0f}: {', '.join(f.productos)}")
print(f"\nTickets: {len(tickets)}, clientes: {tickets.cliente.nunique()}, productos distintos: {len(PRODUCTOS)}")
print(f"Productos por ticket: mínimo {tickets.n_lineas.min()}, mediana {tickets.n_lineas.median():.0f}, máximo {tickets.n_lineas.max()}")
print(f"Monto por ticket: mediana ${tickets.monto.median():.0f}, promedio ${tickets.monto.mean():.0f}, máximo ${tickets.monto.max():.0f}")
print(f"Tickets fraudulentos inyectados: {tickets.es_anomalia.sum()} ({tickets.es_anomalia.mean():.1%})")
print("\nPopularidad de los productos (% de tickets que lo incluyen):")
pop = pd.Series({p: np.mean([p in x for x in tickets.productos]) for p in PRODUCTOS}).sort_values(ascending=False)
print((pop * 100).round(1).to_string())

fig, ejes = plt.subplots(1, 3, figsize=(14, 3.8))
ejes[0].barh(pop.index[::-1], pop.values[::-1] * 100, color="#4C78A8")
ejes[0].set_xlabel("% de tickets")
ejes[0].set_title("Popularidad de los productos")
ejes[0].tick_params(axis="y", labelsize=8)
ejes[1].hist(tickets.n_lineas, bins=range(1, 24), color="#72B7B2")
ejes[1].set_xlabel("Productos distintos por ticket")
ejes[1].set_title("Tamaño del ticket")
ejes[2].hist(tickets.monto.clip(upper=2500), bins=40, color="#E45756")
ejes[2].set_xlabel("Monto ($, recortado en 2 500)")
ejes[2].set_title("Monto del ticket")
plt.tight_layout()
plt.savefig(f"{IMG}/datos.png", dpi=200)
plt.close()


# =============================================================================
# TAREA 1: ANÁLISIS DE CANASTA DE COMPRA (reglas de asociación)
# =============================================================================
def codificar(listas):
    """Transacciones -> matriz one-hot (una columna por producto)."""
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(listas).transform(listas), columns=te.columns_)


def descubrir_reglas(onehot, soporte_minimo=0.05, confianza_minima=0.5, algoritmo="apriori"):
    """Itemsets frecuentes + reglas de asociación (soporte, confianza, lift, leverage y convicción)."""
    f = apriori if algoritmo == "apriori" else fpgrowth
    itemsets = f(onehot, min_support=soporte_minimo, use_colnames=True)
    if len(itemsets) < 2:
        return pd.DataFrame(), itemsets
    reglas = association_rules(itemsets, num_itemsets=len(onehot), metric="confidence", min_threshold=confianza_minima)
    return reglas, itemsets


def texto(regla):
    return f"{', '.join(sorted(regla['antecedents']))} → {', '.join(sorted(regla['consequents']))}"


def buscar(reglas, antecedentes, consecuentes):
    """Fila de la regla pedida (o None si no está)."""
    if len(reglas) == 0:
        return None
    m = reglas[(reglas.antecedents == frozenset(antecedentes)) & (reglas.consequents == frozenset(consecuentes))]
    return m.iloc[0] if len(m) else None


onehot = codificar(list(tickets.productos))
SOPORTE, CONFIANZA = 0.05, 0.2           # valores por omisión del tema
reglas, itemsets = descubrir_reglas(onehot, SOPORTE, CONFIANZA)
print(f"Con soporte >= {SOPORTE:.0%} y confianza >= {CONFIANZA:.0%} (valores del tema): {len(itemsets)} conjuntos frecuentes y {len(reglas)} reglas")

print("\nTop 5 por confianza (como ordena el tema):")
for i, (_, r) in enumerate(reglas.sort_values("confidence", ascending=False).head(5).iterrows(), 1):
    print(f"{i}. {texto(r)}   confianza {r.confidence:.1%} | soporte {r.support:.1%} | lift {r.lift:.2f}")
print("\nTop 5 por lift:")
for i, (_, r) in enumerate(reglas.sort_values("lift", ascending=False).head(5).iterrows(), 1):
    print(f"{i}. {texto(r)}   confianza {r.confidence:.1%} | soporte {r.support:.1%} | lift {r.lift:.2f}")

print("\n--- ¿Se descubrieron los patrones objetivo? ---")
reglas = reglas.assign(rango_confianza=reglas.confidence.rank(ascending=False, method="min").astype(int),
                       rango_lift=reglas.lift.rank(ascending=False, method="min").astype(int))
filas = []
for p in patrones_objetivo:
    r = buscar(reglas, p["antecedentes"], p["consecuentes"])
    filas.append({"patrón": f"{', '.join(p['antecedentes'])} → {', '.join(p['consecuentes'])}",
                  "encontrada": "sí" if r is not None else "no",
                  "soporte": r.support if r is not None else np.nan, "confianza": r.confidence if r is not None else np.nan,
                  "lift": r.lift if r is not None else np.nan,
                  "rango por confianza": r.rango_confianza if r is not None else np.nan,
                  "rango por lift": r.rango_lift if r is not None else np.nan})
print(pd.DataFrame(filas).round(3).to_string(index=False))
print(f"(de {len(reglas)} reglas en total)")

print("\n--- Las 12 reglas con mayor lift ---")
top = reglas.sort_values("lift", ascending=False).head(12)
objetivo_itemsets = [frozenset(p["antecedentes"] + p["consecuentes"]) for p in patrones_objetivo]
top_tabla = pd.DataFrame({"regla": [texto(r) for _, r in top.iterrows()], "soporte": top.support.values,
                          "confianza": top.confidence.values, "lift": top.lift.values,
                          "¿patrón plantado?": ["sí" if (r.antecedents | r.consequents) in objetivo_itemsets else "no" for _, r in top.iterrows()]})
print(top_tabla.round(3).to_string(index=False))
print("\nReglas que comparten el conjunto de cada patrón objetivo (el mismo conjunto genera varias reglas):")
for its in objetivo_itemsets:
    n = sum((r.antecedents | r.consequents) == its for _, r in reglas.iterrows())
    print(f"  {{{', '.join(sorted(its))}}}: {n} reglas")

# --- Sensibilidad a los umbrales -------------------------------------------------
print("\n--- Sensibilidad al soporte mínimo (confianza mínima 0.5; tiempos: Apriori y FP-Growth) ---")
filas = []
for sop in (0.03, 0.05, 0.10, 0.15, 0.20):
    t0 = time.time()
    r_a, its_a = descubrir_reglas(onehot, sop, 0.5, "apriori")
    t_a = time.time() - t0
    t0 = time.time()
    r_f, its_f = descubrir_reglas(onehot, sop, 0.5, "fpgrowth")
    t_f = time.time() - t0
    encontrados = sum(buscar(r_a, p["antecedentes"], p["consecuentes"]) is not None for p in patrones_objetivo)
    iguales = {frozenset(x) for x in its_a.itemsets} == {frozenset(x) for x in its_f.itemsets}
    filas.append({"soporte mínimo": sop, "conjuntos": len(its_a), "reglas": len(r_a), "objetivo encontrados": f"{encontrados} de 3",
                  "Apriori = FP-Growth": "sí" if iguales else "no", "Apriori (s)": t_a, "FP-Growth (s)": t_f})
res_sop = pd.DataFrame(filas)
print(res_sop.round({"Apriori (s)": 3, "FP-Growth (s)": 3}).to_string(index=False))

print("\n--- Sensibilidad a la confianza mínima (soporte mínimo 0.05) ---")
filas = []
for conf in (0.2, 0.4, 0.5, 0.6, 0.7, 0.8):
    r_c, _ = descubrir_reglas(onehot, 0.05, conf)
    encontrados = sum(buscar(r_c, p["antecedentes"], p["consecuentes"]) is not None for p in patrones_objetivo)
    filas.append({"confianza mínima": conf, "reglas": len(r_c), "objetivo encontrados": f"{encontrados} de 3",
                  "reglas con lift > 1.5": int((r_c.lift > 1.5).sum())})
print(pd.DataFrame(filas).to_string(index=False))

# --- Validación en datos no vistos --------------------------------------------------
rng_v = np.random.default_rng(SEMILLA)
mascara = rng_v.random(len(onehot)) < 0.7
r_ent, _ = descubrir_reglas(onehot[mascara].reset_index(drop=True), 0.05, 0.5)
r_val, _ = descubrir_reglas(onehot[~mascara].reset_index(drop=True), 0.02, 0.2)
unidas = r_ent.merge(r_val[["antecedents", "consequents", "confidence", "lift"]], on=["antecedents", "consequents"],
                     suffixes=("", "_val"), how="left")
print(f"\n--- Validación: reglas minadas con el 70 % de los tickets y medidas en el 30 % restante ---")
print(f"Reglas del 70 %: {len(r_ent)}; reaparecen en el 30 %: {unidas.confidence_val.notna().sum()}")
print(f"Correlación de la confianza (70 % vs 30 %): {unidas[['confidence', 'confidence_val']].corr().iloc[0, 1]:.3f}; del lift: {unidas[['lift', 'lift_val']].corr().iloc[0, 1]:.3f}")
filas = []
for p in patrones_objetivo:
    a, b = buscar(r_ent, p["antecedentes"], p["consecuentes"]), buscar(r_val, p["antecedentes"], p["consecuentes"])
    filas.append({"patrón": f"{', '.join(p['antecedentes'])} → {', '.join(p['consecuentes'])}",
                  "confianza (70 %)": a.confidence, "confianza (30 %)": b.confidence, "lift (70 %)": a.lift, "lift (30 %)": b.lift})
print(pd.DataFrame(filas).round(3).to_string(index=False))

# --- Valor de negocio: tickets donde falta el consecuente ---------------------------------
print("\n--- Oportunidad de promoción cruzada ---")
filas = []
for p in patrones_objetivo:
    con_ants = onehot[p["antecedentes"]].all(axis=1)
    sin_cons = con_ants & ~onehot[p["consecuentes"][0]]
    filas.append({"patrón": f"{', '.join(p['antecedentes'])} → {', '.join(p['consecuentes'])}",
                  "tickets con los antecedentes": int(con_ants.sum()), "de ellos, sin el consecuente": int(sin_cons.sum()),
                  "% sin consecuente": 100 * sin_cons.sum() / con_ants.sum(),
                  "ingreso por ticket si lo compran ($)": PRECIOS[p["consecuentes"][0]]})
print(pd.DataFrame(filas).round(1).to_string(index=False))

fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8))
objetivo_mascara = np.array([(r.antecedents | r.consequents) in objetivo_itemsets for _, r in reglas.iterrows()])
sc = ejes[0].scatter(reglas.support[~objetivo_mascara], reglas.confidence[~objetivo_mascara], c=reglas.lift[~objetivo_mascara],
                     cmap="viridis", s=50, alpha=0.8, vmin=0.8, vmax=max(2.5, reglas.lift.max()))
ejes[0].scatter(reglas.support[objetivo_mascara], reglas.confidence[objetivo_mascara], c=reglas.lift[objetivo_mascara], cmap="viridis",
                s=110, edgecolor="red", linewidth=1.6, vmin=0.8, vmax=max(2.5, reglas.lift.max()), label="reglas de los 3 conjuntos objetivo")
plt.colorbar(sc, ax=ejes[0], label="lift")
ejes[0].set_xlabel("Soporte")
ejes[0].set_ylabel("Confianza")
ejes[0].set_title("Reglas descubiertas")
ejes[0].legend(fontsize=8, loc="lower right")
barras = pd.concat([top_tabla.head(9)[["regla", "lift"]].assign(objetivo=False),
                    pd.DataFrame([{"regla": texto(buscar(reglas, p["antecedentes"], p["consecuentes"])),
                                   "lift": buscar(reglas, p["antecedentes"], p["consecuentes"]).lift, "objetivo": True}
                                  for p in patrones_objetivo])]).sort_values("lift")
ejes[1].barh(barras.regla, barras.lift, color=["#E45756" if o else "#9ECAE9" for o in barras.objetivo])
ejes[1].axvline(1, color="gray", linestyle="--")
ejes[1].set_xlabel("Lift")
ejes[1].set_title("9 reglas de mayor lift y los 3 patrones objetivo (rojo)")
ejes[1].tick_params(axis="y", labelsize=7)
plt.tight_layout()
plt.savefig(f"{IMG}/reglas.png", dpi=200)
plt.close()


# =============================================================================
# TAREA 2: SEGMENTACIÓN DE CLIENTES
# =============================================================================
def caracteristicas_clientes(tk):
    """Un renglón por cliente: frecuencia, gasto, ticket, recencia y composición de lo que compra."""
    ultimo = tk.dia.max()
    filas = []
    for c, g in tk.groupby("cliente"):
        unidades = {k: 0 for k in CATEGORIAS}
        for d in g.cantidades:
            for p, q in d.items():
                unidades[CATEGORIA_DE[p]] += q
        total = sum(unidades.values())
        filas.append({"cliente": c, "frecuencia": len(g), "gasto_total": g.monto.sum(), "ticket_mediano": g.monto.median(),
                      "unidades_por_ticket": g.n_unidades.mean(), "recencia": ultimo - g.dia.max(),
                      **{f"pct_{k}": v / total for k, v in unidades.items() if k != "abarrotes"}})
    return pd.DataFrame(filas).set_index("cliente")


feat = caracteristicas_clientes(tickets)
print(f"Clientes con al menos un ticket: {len(feat)}")
print("\nCaracterísticas por cliente (estadísticos):")
print(feat.describe().loc[["mean", "std", "min", "50%", "max"]].round(2).to_string())

X_log = feat.copy()
for col in ("frecuencia", "gasto_total", "ticket_mediano", "unidades_por_ticket", "recencia"):
    X_log[col] = np.log1p(X_log[col])                         # variables sesgadas a la derecha
X_esc = StandardScaler().fit_transform(X_log)
real = clientes.set_index("cliente").segmento_real.reindex(feat.index).values

print("\n--- Elección del número de grupos con K-Means ---")
filas = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X_esc)
    filas.append({"k": k, "inercia": km.inertia_, "silueta": silhouette_score(X_esc, km.labels_),
                  "Calinski-Harabasz": calinski_harabasz_score(X_esc, km.labels_), "Davies-Bouldin": davies_bouldin_score(X_esc, km.labels_)})
res_k = pd.DataFrame(filas).set_index("k")
res_k["mejora de inercia (%)"] = 100 * (res_k.inercia.shift(1) - res_k.inercia) / res_k.inercia.shift(1)
print(res_k.round(3).to_string())
K_SILUETA = int(res_k.silueta.idxmax())
K = int(res_k.index[res_k["mejora de inercia (%)"] >= 10].max())     # último k que todavía reduce la inercia 10 % o más
print(f"\nMayor silueta: k = {K_SILUETA}. Último k con mejora de inercia >= 10 % (codo): k = {K}. Se usa k = {K}.")

print("\nARI contra los segmentos reales con K-Means: " + ", ".join(
    f"k={k}: {adjusted_rand_score(real, KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(X_esc)):.3f}" for k in range(2, 7)))
print("\n--- Comparación de algoritmos (k = %d) ---" % K)
modelos = {"K-Means": KMeans(n_clusters=K, n_init=10, random_state=SEMILLA),
           "Aglomerativo (Ward)": AgglomerativeClustering(n_clusters=K),
           "Mezcla gaussiana": GaussianMixture(n_components=K, n_init=5, random_state=SEMILLA)}
etiquetas = {n: m.fit_predict(X_esc) for n, m in modelos.items()}
filas = []
for n, e in etiquetas.items():
    filas.append({"algoritmo": n, "grupos": len(set(e)), "silueta": silhouette_score(X_esc, e),
                  "ARI contra segmentos reales": adjusted_rand_score(real, e)})
print(pd.DataFrame(filas).round(3).to_string(index=False))

print("\n--- DBSCAN: efecto de eps (min_samples = 5; el tema usa eps = 0.5) ---")
filas = []
for eps in (0.5, 0.8, 1.0, 1.3, 1.6, 2.0):
    e = DBSCAN(eps=eps, min_samples=5).fit_predict(X_esc)
    n_g = len(set(e)) - (1 if -1 in e else 0)
    ok = e != -1
    filas.append({"eps": eps, "grupos": n_g, "ruido (%)": 100 * np.mean(e == -1),
                  "silueta (sin ruido)": silhouette_score(X_esc[ok], e[ok]) if n_g > 1 else np.nan,
                  "ARI contra segmentos reales": adjusted_rand_score(real, e)})
print(pd.DataFrame(filas).round(3).to_string(index=False))

print("\n--- Estabilidad de K-Means: 10 inicializaciones y 10 submuestras del 80 % ---")
base = etiquetas["K-Means"]
ari_ini = [adjusted_rand_score(base, KMeans(n_clusters=K, n_init=1, random_state=s).fit_predict(X_esc)) for s in range(10)]
rng_s = np.random.default_rng(SEMILLA)
ari_sub = []
for _ in range(10):
    idx = rng_s.choice(len(X_esc), size=int(0.8 * len(X_esc)), replace=False)
    ari_sub.append(adjusted_rand_score(base[idx], KMeans(n_clusters=K, n_init=10, random_state=int(rng_s.integers(1e6))).fit_predict(X_esc[idx])))
print(f"ARI entre inicializaciones: promedio {np.mean(ari_ini):.3f} (mínimo {np.min(ari_ini):.3f}); entre submuestras: promedio {np.mean(ari_sub):.3f} (mínimo {np.min(ari_sub):.3f})")

feat["grupo"] = base
feat["segmento_real"] = real
print("\n--- Perfil de los grupos de K-Means (promedios en unidades originales) ---")
perfil = feat.groupby("grupo").agg(clientes=("frecuencia", "size"), frecuencia=("frecuencia", "mean"), gasto_total=("gasto_total", "mean"),
                                   ticket_mediano=("ticket_mediano", "mean"), unidades_por_ticket=("unidades_por_ticket", "mean"),
                                   recencia=("recencia", "mean"), pct_frescos=("pct_frescos", "mean"), pct_lácteos=("pct_lácteos", "mean"),
                                   pct_bebidas=("pct_bebidas", "mean"), pct_snacks=("pct_snacks", "mean"))
print(perfil.round(2).to_string())
print("\nGrupos encontrados contra segmentos reales (número de clientes):")
print(pd.crosstab(feat.grupo, feat.segmento_real).to_string())

pca = PCA(n_components=2, random_state=SEMILLA).fit(X_esc)
Z = pca.transform(X_esc)
print(f"\nPCA a 2 dimensiones: conserva {pca.explained_variance_ratio_.sum():.1%} de la varianza")

fig, ejes = plt.subplots(1, 3, figsize=(15, 4.4))
ejes[0].plot(res_k.index, res_k.inercia, "o-", color="#4C78A8")
ejes[0].set_xlabel("k")
ejes[0].set_ylabel("Inercia")
ejes[0].set_title("Método del codo")
ejes[1].plot(res_k.index, res_k.silueta, "o-", color="#E45756")
ejes[1].set_xlabel("k")
ejes[1].set_ylabel("Silueta")
ejes[1].set_title("Coeficiente de silueta")
for g in range(K):
    ejes[2].scatter(Z[base == g, 0], Z[base == g, 1], s=14, alpha=0.7, label=f"grupo {g}")
ejes[2].set_xlabel("Componente 1")
ejes[2].set_ylabel("Componente 2")
ejes[2].set_title("Clientes (PCA) coloreados por grupo")
ejes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG}/clustering.png", dpi=200)
plt.close()

zs = pd.DataFrame(X_esc, columns=feat.columns[:-2], index=feat.index).groupby(base).mean()
plt.figure(figsize=(9, 3.6))
sns.heatmap(zs, annot=True, fmt=".1f", cmap="RdBu_r", center=0, cbar_kws={"label": "desviaciones estándar"})
plt.xlabel("")
plt.ylabel("Grupo")
plt.title("Perfil de cada grupo (variables estandarizadas)")
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG}/perfil_grupos.png", dpi=200)
plt.close()

print("\n--- Reglas de asociación dentro de cada grupo (soporte >= 0.08, confianza >= 0.5; 4 reglas simples de mayor lift) ---")
grupo_de_cliente = feat.grupo
tk_g = tickets.assign(grupo=tickets.cliente.map(grupo_de_cliente))
for g in range(K):
    sub = tk_g[tk_g.grupo == g]
    r_g, _ = descubrir_reglas(codificar(list(sub.productos)), 0.08, 0.5)
    simples = r_g[(r_g.antecedents.map(len) <= 2) & (r_g.consequents.map(len) == 1)] if len(r_g) else r_g
    print(f"Grupo {g} ({len(sub)} tickets): {len(r_g)} reglas en total, {len(simples)} con a lo más 2 antecedentes y 1 consecuente")
    for _, r in simples.sort_values("lift", ascending=False).head(4).iterrows():
        print(f"    {texto(r)}   confianza {r.confidence:.1%} | soporte {r.support:.1%} | lift {r.lift:.2f}")


# =============================================================================
# TAREA 3: DETECCIÓN DE ANOMALÍAS (transacciones sospechosas)
# =============================================================================
def caracteristicas_tickets(tk, con_cliente=True):
    mediana_cliente = tk.groupby("cliente").monto.transform("median")
    X = pd.DataFrame({"log_monto": np.log1p(tk.monto), "n_lineas": tk.n_lineas, "cantidad_max": tk.cantidad_max,
                      "descuento_pct": tk.descuento_pct, "hora_sen": np.sin(2 * np.pi * tk.hora / 24),
                      "hora_cos": np.cos(2 * np.pi * tk.hora / 24)})
    if con_cliente:
        X["monto_vs_cliente"] = np.log(tk.monto / mediana_cliente)
    return X


def puntajes(X, tasa, semilla=SEMILLA):
    """Puntaje de anomalía (mayor = más raro) de los tres algoritmos del tema."""
    Xs = StandardScaler().fit_transform(X)
    isof = IsolationForest(contamination=tasa, random_state=semilla).fit(Xs)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=tasa).fit(Xs)
    svm = OneClassSVM(nu=tasa, gamma="scale").fit(Xs)
    return {"Isolation Forest": -isof.score_samples(Xs), "Local Outlier Factor": -lof.negative_outlier_factor_,
            "One-Class SVM": -svm.decision_function(Xs)}


def evaluar(y, p, n_marcar):
    """Métricas cuando se marcan como sospechosos los n_marcar tickets de mayor puntaje."""
    marca = np.zeros(len(p), dtype=bool)
    marca[np.argsort(-p)[:n_marcar]] = True
    tp = int((marca & (y == 1)).sum())
    prec, rec = tp / max(1, marca.sum()), tp / y.sum()
    return {"exactitud": np.mean(marca == (y == 1)), "precisión": prec, "recall": rec,
            "F1": 2 * prec * rec / max(1e-12, prec + rec), "AUC": roc_auc_score(y, p), "AP": average_precision_score(y, p)}


y = tickets.es_anomalia.values
tasa = y.mean()
n_anom = int(y.sum())
print(f"Anomalías reales: {n_anom} de {len(y)} ({tasa:.1%})")
print(f"Exactitud de marcar TODO como normal: {np.mean(y == 0):.1%} (por eso la exactitud engaña en este problema)")

X_t = caracteristicas_tickets(tickets)
print("\nCaracterísticas por ticket: " + ", ".join(X_t.columns))
print("\nPromedios de las características, tickets normales contra fraudulentos:")
print(X_t.groupby(y).mean().rename(index={0: "normal", 1: "fraude"}).round(2).to_string())

print("\n--- Comparación de algoritmos (se marcan los 100 tickets de mayor puntaje; cantidad de anomalías conocida) ---")
P = puntajes(X_t, tasa)
base_monto = {"Regla: monto alto": tickets.monto.values}
filas = []
for nombre, p in {**P, **base_monto}.items():
    filas.append({"algoritmo": nombre, **evaluar(y, p, n_anom)})
res_an = pd.DataFrame(filas).set_index("algoritmo")
print(res_an.round(3).to_string())

print("\n--- LOF: efecto del número de vecinos (el tema usa 20) ---")
filas = []
Xs_t = StandardScaler().fit_transform(X_t)
for nv in (20, 50, 100, 200, 400):
    p = -LocalOutlierFactor(n_neighbors=nv, contamination=tasa).fit(Xs_t).negative_outlier_factor_
    filas.append({"vecinos": nv, **{k: v for k, v in evaluar(y, p, n_anom).items() if k in ("precisión", "recall", "AUC", "AP")}})
print(pd.DataFrame(filas).round(3).to_string(index=False))

print("\n--- Recall por tipo de fraude (con los 100 tickets de mayor puntaje) ---")
filas = []
for nombre, p in {**P, **base_monto}.items():
    marca = np.zeros(len(p), dtype=bool)
    marca[np.argsort(-p)[:n_anom]] = True
    filas.append({"algoritmo": nombre, **{t: np.mean(marca[tickets.tipo_anomalia == t]) for t in TIPOS_ANOMALIA}})
print(pd.DataFrame(filas).set_index("algoritmo").round(2).to_string())

print("\n--- ¿Qué tan bien se elige cuántos tickets marcar? (Isolation Forest, puntaje fijo) ---")
filas = []
for n in (25, 50, 100, 150, 250, 500):
    m = evaluar(y, P["Isolation Forest"], n)
    filas.append({"tickets marcados": n, "% de los tickets": 100 * n / len(y), "precisión": m["precisión"], "recall": m["recall"], "F1": m["F1"]})
print(pd.DataFrame(filas).round(3).to_string(index=False))

print("\n--- Efecto de la característica relativa al cliente (monto_vs_cliente), Isolation Forest ---")
P_sin = puntajes(caracteristicas_tickets(tickets, con_cliente=False), tasa)
filas = []
for nombre, pp in (("con monto_vs_cliente", P["Isolation Forest"]), ("sin monto_vs_cliente", P_sin["Isolation Forest"])):
    marca = np.zeros(len(pp), dtype=bool)
    marca[np.argsort(-pp)[:n_anom]] = True
    fp = marca & (y == 0)
    filas.append({"versión": nombre, **{k: v for k, v in evaluar(y, pp, n_anom).items() if k in ("precisión", "recall", "AP")},
                  "falsas alarmas": int(fp.sum()), **{f"FP {s['nombre']}": int((fp & (tickets.segmento_real == s['nombre']).values).sum()) for s in SEGMENTOS}})
print(pd.DataFrame(filas).set_index("versión").round(3).to_string())

fig, ejes = plt.subplots(1, 3, figsize=(15, 4.3))
for nombre, p in P.items():
    pr, rc, _ = precision_recall_curve(y, p)
    ejes[0].plot(rc, pr, label=f"{nombre} (AP {average_precision_score(y, p):.2f})")
pr, rc, _ = precision_recall_curve(y, tickets.monto.values)
ejes[0].plot(rc, pr, "k--", label=f"Regla: monto alto (AP {average_precision_score(y, tickets.monto.values):.2f})")
ejes[0].axhline(tasa, color="gray", linestyle=":", label="al azar")
ejes[0].set_xlabel("Recall")
ejes[0].set_ylabel("Precisión")
ejes[0].set_title("Curvas precisión-recall")
ejes[0].legend(fontsize=7)
pi = P["Isolation Forest"]
ejes[1].hist(pi[y == 0], bins=40, alpha=0.7, label="normales", color="#4C78A8", density=True)
ejes[1].hist(pi[y == 1], bins=25, alpha=0.7, label="fraudes", color="#E45756", density=True)
ejes[1].set_xlabel("Puntaje de anomalía (Isolation Forest)")
ejes[1].set_ylabel("Densidad")
ejes[1].set_title("Separación de los puntajes")
ejes[1].legend()
marca = np.zeros(len(pi), dtype=bool)
marca[np.argsort(-pi)[:n_anom]] = True
ejes[2].scatter(tickets.hora[~marca & (y == 0)], tickets.monto[~marca & (y == 0)].clip(upper=3500), s=6, alpha=0.3, color="#9ECAE9", label="normal")
ejes[2].scatter(tickets.hora[marca & (y == 1)], tickets.monto[marca & (y == 1)].clip(upper=3500), s=22, color="#E45756", label="fraude detectado")
ejes[2].scatter(tickets.hora[~marca & (y == 1)], tickets.monto[~marca & (y == 1)].clip(upper=3500), s=22, marker="x", color="black", label="fraude no detectado")
ejes[2].scatter(tickets.hora[marca & (y == 0)], tickets.monto[marca & (y == 0)].clip(upper=3500), s=22, marker="s", facecolors="none", edgecolors="orange", label="falsa alarma")
ejes[2].set_xlabel("Hora del día")
ejes[2].set_ylabel("Monto ($, recortado en 3 500)")
ejes[2].set_title("Isolation Forest sobre hora y monto")
ejes[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{IMG}/anomalias.png", dpi=200)
plt.close()


# =============================================================================
# ROBUSTEZ: REPETIR TODO CON 10 SIMULACIONES DISTINTAS
# =============================================================================
print("\n--- Robustez: 10 simulaciones con semillas distintas ---")
filas = []
for s in range(10):
    tk, cl = generar_datos(semilla=100 + s)
    oh = codificar(list(tk.productos))
    rg, _ = descubrir_reglas(oh, 0.05, 0.5)
    ps = [buscar(rg, p["antecedentes"], p["consecuentes"]) for p in patrones_objetivo]
    if len(rg):
        rg = rg.assign(rango=rg.lift.rank(ascending=False, method="min"))
    ft = caracteristicas_clientes(tk)
    Xl = ft.copy()
    for col in ("frecuencia", "gasto_total", "ticket_mediano", "unidades_por_ticket", "recencia"):
        Xl[col] = np.log1p(Xl[col])
    Xe = StandardScaler().fit_transform(Xl)
    verdad = cl.set_index("cliente").segmento_real.reindex(ft.index).values
    ari = adjusted_rand_score(verdad, KMeans(n_clusters=4, n_init=10, random_state=SEMILLA).fit_predict(Xe))
    yy = tk.es_anomalia.values
    ap = average_precision_score(yy, puntajes(caracteristicas_tickets(tk), yy.mean())["Isolation Forest"])
    filas.append({"simulación": s, "patrones encontrados": sum(p is not None for p in ps),
                  "lift mínimo de los 3 patrones": min(p.lift for p in ps if p is not None) if any(p is not None for p in ps) else np.nan,
                  "ARI K-Means (k=4)": ari, "AP Isolation Forest": ap})
rob = pd.DataFrame(filas).set_index("simulación")
print(rob.round(3).to_string())
print("\nPromedio y desviación estándar:")
print(pd.DataFrame({"promedio": rob.mean(), "desv.": rob.std()}).round(3).to_string())
print("\nListo. Figuras guardadas en", IMG)
