# =============================================================================
# Actividad 19: Práctica guiada con datos simulados
# Tema 3.1: prueba t de Student y Mann-Whitney, valor p, decisión sobre H0 y errores tipo I y II
# Ejecutar: python act19.py   (las figuras se guardan en img/)
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
from sklearn.datasets import load_iris

pd.set_option('display.width', 120)
SEMILLA = 42
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


# ---- Tarea 1: Conjunto de datos y exploración --------------------------------
# Conjunto Iris (150 flores, incluido en scikit-learn). Pregunta: ¿el ancho del sépalo
# es distinto en Iris versicolor y en Iris virginica? (50 flores por especie).
iris = load_iris(as_frame=True)
df = iris.frame
df['especie'] = df['target'].map(dict(enumerate(iris.target_names)))
ancho = 'sepal width (cm)'
versicolor = df.loc[df['especie'] == 'versicolor', ancho].to_numpy()
virginica = df.loc[df['especie'] == 'virginica', ancho].to_numpy()

print("=== Tarea 1: exploración de los datos ===")
resumen = pd.DataFrame({
    'versicolor': pd.Series(versicolor).describe(),
    'virginica': pd.Series(virginica).describe(),
}).loc[['count', 'mean', 'std', 'min', '50%', 'max']].round(3)
print(resumen)

fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
ejes[0].boxplot([versicolor, virginica], tick_labels=['versicolor', 'virginica'])
ejes[0].set_ylabel('Ancho del sépalo (cm)')
ejes[0].set_title('Diagrama de caja')
for datos, color, nombre in [(versicolor, 'tab:blue', 'versicolor'), (virginica, 'tab:orange', 'virginica')]:
    ejes[1].hist(datos, bins=np.arange(2.0, 3.9, 0.2), alpha=0.6, color=color, label=nombre)
ejes[1].set_xlabel('Ancho del sépalo (cm)')
ejes[1].set_ylabel('Frecuencia')
ejes[1].set_title('Histogramas')
ejes[1].legend()
guardar('exploracion.png')

# ---- Tarea 2: Verificación de supuestos --------------------------------------
# La t de Student pide normalidad en cada grupo (Shapiro-Wilk) y, en su versión clásica,
# varianzas iguales (Levene). Si no se cumplen, se usa Mann-Whitney.
print("\n=== Tarea 2: supuestos ===")
for nombre, datos in [('versicolor', versicolor), ('virginica', virginica)]:
    w, p = stats.shapiro(datos)
    print(f"Shapiro-Wilk {nombre}: W = {w:.4f}, p = {p:.4f}")
estad_lev, p_lev = stats.levene(versicolor, virginica)
print(f"Levene (varianzas iguales): estadístico = {estad_lev:.4f}, p = {p_lev:.4f}")

# ---- Tarea 3: Pruebas de hipótesis -------------------------------------------
# H0: mu_versicolor = mu_virginica     H1: mu_versicolor != mu_virginica (bilateral)
print(f"\n=== Tarea 3: pruebas de hipótesis (alfa = {ALFA}) ===")
dif = versicolor.mean() - virginica.mean()
gl_clasica = len(versicolor) + len(virginica) - 2
t_student = stats.ttest_ind(versicolor, virginica, equal_var=True)
t_welch = stats.ttest_ind(versicolor, virginica, equal_var=False)
ic = t_student.confidence_interval(0.95)
sp = np.sqrt((versicolor.var(ddof=1) + virginica.var(ddof=1)) / 2)  # desviación combinada (n iguales)
d_cohen = dif / sp
u = stats.mannwhitneyu(versicolor, virginica, alternative='two-sided')

print(f"Diferencia de medias (versicolor - virginica): {dif:.3f} cm")
print(f"t de Student:  t = {t_student.statistic:.4f}, gl = {gl_clasica}, p = {t_student.pvalue:.5f}")
print(f"IC 95% de la diferencia: [{ic.low:.3f}, {ic.high:.3f}] cm")
print(f"t de Welch:    t = {t_welch.statistic:.4f}, p = {t_welch.pvalue:.5f}")
print(f"d de Cohen = {d_cohen:.3f}")
print(f"Mann-Whitney:  U = {u.statistic:.1f}, p = {u.pvalue:.5f}")
for nombre, p in [('t de Student', t_student.pvalue), ('Mann-Whitney', u.pvalue)]:
    decision = 'se rechaza H0' if p <= ALFA else 'no se rechaza H0'
    print(f"{nombre}: p = {p:.5f} {'<=' if p <= ALFA else '>'} {ALFA} -> {decision}")

# ---- Tarea 4: Riesgo de error tipo I y tipo II (simulación) ------------------
# Se repite la prueba 10 000 veces con datos simulados donde se conoce la realidad.
# Con H0 verdadera, la proporción de rechazos estima alfa (error tipo I).
# Con H1 verdadera (diferencia observada), la proporción de no rechazos estima beta.
print("\n=== Tarea 4: simulación de errores ===")
rng = np.random.default_rng(SEMILLA)
N_SIM = 10_000
media_comun = np.mean(np.concatenate([versicolor, virginica]))


def simular(n, diferencia):
    """Devuelve los valores p de N_SIM pruebas t con muestras normales de tamaño n."""
    g1 = rng.normal(media_comun - diferencia / 2, sp, size=(N_SIM, n))
    g2 = rng.normal(media_comun + diferencia / 2, sp, size=(N_SIM, n))
    return stats.ttest_ind(g1, g2, axis=1).pvalue


p_h0 = simular(50, 0.0)
filas = [('H0 verdadera', 50, 0.0, np.mean(p_h0 <= ALFA), 'error tipo I (alfa)')]
p_h1_50 = simular(50, abs(dif))
p_h1_10 = simular(10, abs(dif))
filas.append(('H1 verdadera', 50, abs(dif), np.mean(p_h1_50 > ALFA), 'error tipo II (beta)'))
filas.append(('H1 verdadera', 10, abs(dif), np.mean(p_h1_10 > ALFA), 'error tipo II (beta)'))
tabla = pd.DataFrame(filas, columns=['realidad', 'n por grupo', 'diferencia', 'proporción', 'estima'])
print(tabla.round(4).to_string(index=False))

fig, ejes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for eje, p, titulo in [(ejes[0], p_h0, 'H0 verdadera, n = 50'),
                       (ejes[1], p_h1_50, 'H1 verdadera, n = 50'),
                       (ejes[2], p_h1_10, 'H1 verdadera, n = 10')]:
    eje.hist(p, bins=20, range=(0, 1), color='tab:gray', edgecolor='white')
    eje.axvline(ALFA, color='tab:red', linestyle='--', label=f'alfa = {ALFA}')
    eje.set_title(titulo)
    eje.set_xlabel('valor p')
ejes[0].set_ylabel('Frecuencia (de 10 000)')
ejes[0].legend()
guardar('simulacion_errores.png')
