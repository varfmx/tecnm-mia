# =============================================================================
# Actividad 20: Proyecto de pruebas estadísticas paramétricas y no paramétricas
# Tema 3.2: ANOVA, Friedman y pruebas post-hoc (Tukey y Nemenyi) para comparar cuatro clasificadores
# Ejecutar: python act20.py   (las figuras se guardan en img/)
# =============================================================================

# ---- Importaciones y función auxiliar ----------------------------------------
import os

import matplotlib

# En Colab o Jupyter no existe __file__; ahí las figuras también se muestran en pantalla
EN_NOTEBOOK = '__file__' not in globals()
if not EN_NOTEBOOK:
    matplotlib.use('Agg')  # Como script, genera las figuras como archivos PNG sin abrir ventanas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option('display.width', 120)
ALFA = 0.05

# Carpeta donde se guardan las figuras que se incluyen en el reporte
BASE = os.getcwd() if EN_NOTEBOOK else os.path.dirname(os.path.abspath(__file__))
CARPETA_IMG = os.path.join(BASE, 'img')
os.makedirs(CARPETA_IMG, exist_ok=True)


def guardar(nombre):
    """Guarda la figura actual en la carpeta img/ y la muestra si se ejecuta en un notebook."""
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_IMG, nombre), dpi=150)
    if EN_NOTEBOOK:
        plt.show()
    plt.close()


# ---- Datos -------------------------------------------------------------------
# AUC de cuatro variantes del clasificador C4.5 en 14 conjuntos de datos
# (tabla del enunciado, Tabla 6 de Demsar, 2006). Cada fila es un conjunto de datos.
clasificadores = ['C4.5', 'C4.5+m', 'C4.5+cf', 'C4.5+m+cf']
datos = pd.DataFrame(
    [[0.763, 0.768, 0.771, 0.798],   # adult (sample)
     [0.599, 0.591, 0.590, 0.569],   # breast cancer
     [0.954, 0.971, 0.968, 0.967],   # breast cancer wisconsin
     [0.628, 0.661, 0.654, 0.657],   # cmc
     [0.882, 0.888, 0.886, 0.898],   # ionosphere
     [0.936, 0.931, 0.916, 0.931],   # iris
     [0.661, 0.668, 0.609, 0.685],   # liver disorders
     [0.583, 0.583, 0.563, 0.625],   # lung cancer
     [0.775, 0.838, 0.866, 0.875],   # lymphography
     [1.000, 1.000, 1.000, 1.000],   # mushroom
     [0.940, 0.962, 0.965, 0.962],   # primary tumor
     [0.619, 0.666, 0.614, 0.669],   # rheum
     [0.972, 0.981, 0.975, 0.975],   # voting
     [0.957, 0.978, 0.946, 0.970]],  # wine
    index=['adult', 'breast cancer', 'breast cancer wisconsin', 'cmc', 'ionosphere', 'iris',
           'liver disorders', 'lung cancer', 'lymphography', 'mushroom', 'primary tumor',
           'rheum', 'voting', 'wine'],
    columns=clasificadores)
X = datos.to_numpy()
b, k = X.shape  # b conjuntos de datos (bloques), k clasificadores (tratamientos)

print("=== Datos ===")
print(f"Conjuntos de datos (b) = {b}; clasificadores (k) = {k}")
print(pd.DataFrame({'media': datos.mean(), 'desv. est.': datos.std(ddof=1)}).round(4).to_string())

# ---- Tarea 1a: ANOVA de un factor (diseño completamente al azar) -------------
# H0: mu_1 = mu_2 = mu_3 = mu_4     H1: al menos dos medias son distintas
# Trata las 4 columnas como muestras independientes (Walpole et al., sección 13.3).
print(f"\n=== Tarea 1a: ANOVA de un factor (alfa = {ALFA}) ===")
gran_media = X.mean()
sst = ((X - gran_media) ** 2).sum()                      # variabilidad total
ssa = b * ((X.mean(axis=0) - gran_media) ** 2).sum()     # entre tratamientos
sse1 = sst - ssa                                         # dentro de tratamientos (error)
gl_a, gl_e1 = k - 1, k * (b - 1)
f1 = (ssa / gl_a) / (sse1 / gl_e1)
p1 = stats.f.sf(f1, gl_a, gl_e1)
f_critico1 = stats.f.ppf(1 - ALFA, gl_a, gl_e1)
tabla1 = pd.DataFrame({'SC': [ssa, sse1, sst], 'gl': [gl_a, gl_e1, b * k - 1],
                       'CM': [ssa / gl_a, sse1 / gl_e1, np.nan], 'F': [f1, np.nan, np.nan],
                       'p': [p1, np.nan, np.nan]},
                      index=['Entre clasificadores', 'Error', 'Total'])
print(tabla1.round(6).to_string())
print(f"F crítico F(0.05; {gl_a}, {gl_e1}) = {f_critico1:.4f}")
print(f"Verificación con scipy: F = {stats.f_oneway(*X.T).statistic:.4f}, p = {stats.f_oneway(*X.T).pvalue:.4f}")
print(f"Decisión: {'se rechaza H0' if p1 <= ALFA else 'no se rechaza H0'}")

# ---- Tarea 1b: ANOVA con bloques (conjuntos de datos como bloques) -----------
# Los 4 clasificadores se evaluaron en los mismos 14 conjuntos de datos, así que las
# observaciones están emparejadas: cada conjunto de datos es un bloque
# (Walpole et al., sección 13.9, diseño por bloques completamente aleatorios).
print(f"\n=== Tarea 1b: ANOVA por bloques (alfa = {ALFA}) ===")
ssb = k * ((X.mean(axis=1) - gran_media) ** 2).sum()     # entre conjuntos de datos
sse2 = sst - ssa - ssb                                   # error tras quitar los bloques
gl_b, gl_e2 = b - 1, (k - 1) * (b - 1)
cme2 = sse2 / gl_e2
f2 = (ssa / gl_a) / cme2
p2 = stats.f.sf(f2, gl_a, gl_e2)
f_critico2 = stats.f.ppf(1 - ALFA, gl_a, gl_e2)
tabla2 = pd.DataFrame({'SC': [ssa, ssb, sse2, sst], 'gl': [gl_a, gl_b, gl_e2, b * k - 1],
                       'CM': [ssa / gl_a, ssb / gl_b, cme2, np.nan], 'F': [f2, np.nan, np.nan, np.nan],
                       'p': [p2, np.nan, np.nan, np.nan]},
                      index=['Entre clasificadores', 'Entre conjuntos de datos', 'Error', 'Total'])
print(tabla2.round(6).to_string())
print(f"F crítico F(0.05; {gl_a}, {gl_e2}) = {f_critico2:.4f}")
print(f"Proporción de la variabilidad total debida a los conjuntos de datos: {ssb / sst:.3f}")
print(f"Decisión: {'se rechaza H0' if p2 <= ALFA else 'no se rechaza H0'}")

residuos = X - X.mean(axis=0) - X.mean(axis=1, keepdims=True) + gran_media
w, p_sw = stats.shapiro(residuos.ravel())
print(f"Shapiro-Wilk de los residuos del modelo con bloques: W = {w:.4f}, p = {p_sw:.4f}")

# Figura: los valores brutos y los valores sin el efecto del conjunto de datos
centrados = X - X.mean(axis=1, keepdims=True)
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
ejes[0].boxplot(X, tick_labels=clasificadores)
ejes[0].set_title('AUC observada (se solapa casi todo)')
ejes[0].set_ylabel('AUC')
ejes[1].boxplot(centrados, tick_labels=clasificadores)
ejes[1].axhline(0, color='tab:red', linestyle='--', linewidth=1)
ejes[1].set_title('AUC menos la media de su conjunto de datos')
ejes[1].set_ylabel('Diferencia respecto a la media del conjunto')
guardar('comparacion_clasificadores.png')

# ---- Tarea 2: Prueba de Friedman ---------------------------------------------
# H0: los k clasificadores son equivalentes (rangos promedio iguales)
# H1: al menos uno difiere. Los rangos son los de la tabla (1 = mejor AUC; empates con
# rango promedio), tal como aparecen en la Tabla 6 de Demsar (2006).
print(f"\n=== Tarea 2: prueba de Friedman (alfa = {ALFA}) ===")
rangos = pd.DataFrame(
    [[4, 3, 2, 1], [1, 2, 3, 4], [4, 1, 2, 3], [4, 1, 3, 2], [4, 2, 3, 1], [1, 2.5, 4, 2.5],
     [3, 2, 4, 1], [2.5, 2.5, 4, 1], [4, 3, 2, 1], [2.5, 2.5, 2.5, 2.5], [4, 2.5, 1, 2.5],
     [3, 2, 4, 1], [4, 1, 2, 3], [3, 1, 4, 2]],
    index=datos.index, columns=clasificadores)
R = rangos.mean()  # rango promedio de cada clasificador
print("Rangos promedio:", R.round(3).to_dict())
chi2_f = 12 * b / (k * (k + 1)) * ((R ** 2).sum() - k * (k + 1) ** 2 / 4)
p_chi = stats.chi2.sf(chi2_f, k - 1)
print(f"Friedman: chi2_F = {chi2_f:.4f}, gl = {k - 1}, p = {p_chi:.4f}")
# Corrección de Iman y Davenport (la que recomienda Demsar), menos conservadora
f_f = (b - 1) * chi2_f / (b * (k - 1) - chi2_f)
gl1, gl2 = k - 1, (k - 1) * (b - 1)
p_ff = stats.f.sf(f_f, gl1, gl2)
print(f"Iman-Davenport: F_F = {f_f:.4f}, gl = ({gl1}, {gl2}), p = {p_ff:.4f}, "
      f"F crítico = {stats.f.ppf(1 - ALFA, gl1, gl2):.4f}")
fr = stats.friedmanchisquare(*rangos.to_numpy().T)  # verificación (con corrección por empates)
print(f"Verificación con scipy (corrige empates): chi2 = {fr.statistic:.4f}, p = {fr.pvalue:.4f}")
print(f"Decisión: {'se rechaza H0' if p_ff <= ALFA else 'no se rechaza H0'}")

# ---- Tarea 3: Pruebas post-hoc: Tukey (paramétrica) y Nemenyi (no paramétrica) -
print("\n=== Tarea 3: pruebas post-hoc ===")
# Tukey: diferencia mínima significativa con el error cuadrático medio del modelo con bloques
q = stats.studentized_range.ppf(1 - ALFA, k, gl_e2)
hsd = q * np.sqrt(cme2 / b)
print(f"Tukey: q(0.05; {k}, {gl_e2}) = {q:.4f}; diferencia mínima significativa = {hsd:.4f}")
# Nemenyi: diferencia crítica en rangos promedio, CD = q_alfa * sqrt(k(k+1) / (6 b)),
# con q_alfa = (rango studentizado con gl infinitos) / raiz(2)
q_nem = stats.studentized_range.ppf(1 - ALFA, k, np.inf) / np.sqrt(2)
cd = q_nem * np.sqrt(k * (k + 1) / (6 * b))
print(f"Nemenyi: q_alfa = {q_nem:.4f}; diferencia crítica CD = {cd:.4f}")
medias = datos.mean()
ee_rangos = np.sqrt(k * (k + 1) / (6 * b))
filas = []
for i in range(k):
    for j in range(i + 1, k):
        d_media = medias.iloc[j] - medias.iloc[i]
        d_rango = R.iloc[i] - R.iloc[j]  # positivo si j tiene mejor (menor) rango
        p_nem = stats.studentized_range.sf(abs(d_rango) / ee_rangos * np.sqrt(2), k, np.inf)
        filas.append((f"{clasificadores[j]} vs {clasificadores[i]}", d_media, abs(d_media) > hsd,
                      d_rango, p_nem, abs(d_rango) > cd))
tabla = pd.DataFrame(filas, columns=['comparación', 'dif. AUC', 'Tukey sig.',
                                     'dif. rangos', 'p Nemenyi', 'Nemenyi sig.'])
print(tabla.round(4).to_string(index=False))

# Figura: rangos promedio con la diferencia crítica de Nemenyi (1 = mejor)
fig, eje = plt.subplots(figsize=(8, 2.6))
eje.hlines(0, 1, 4, color='black')
for pos, (nombre, r) in enumerate(R.sort_values().items()):
    arriba = pos % 2 == 0
    eje.plot(r, 0, 'o', color='tab:blue')
    eje.annotate(f"{nombre} ({r:.2f})", (r, 0), xytext=(r, 0.45 if arriba else -0.45),
                 ha='center', va='bottom' if arriba else 'top',
                 arrowprops=dict(arrowstyle='-', color='gray'))
r_min, r_max = R.min(), R.max()
if r_max - r_min < cd:  # todos dentro de la diferencia crítica: una sola barra los une
    eje.hlines(-0.12, r_min, r_max, color='tab:red', linewidth=4)
eje.hlines(1.15, 1, 1 + cd, color='black', linewidth=2)
eje.text(1 + cd / 2, 1.25, f'CD = {cd:.2f}', ha='center')
eje.set_xlim(0.8, 4.2)
eje.set_ylim(-1.1, 1.6)
eje.invert_xaxis()
eje.set_yticks([])
eje.set_xlabel('Rango promedio (1 = mejor)')
for lado in ('left', 'right', 'top'):
    eje.spines[lado].set_visible(False)
guardar('diferencia_critica_nemenyi.png')
