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

pd.set_option('display.width', 100)

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


# ===============================================================
# a) Distribución normal: medias 0, 5, 10 y desviaciones 1, 2, 3
# ===============================================================

N_MUESTRAS = 1000
PARAMETROS_NORMAL = [(0, 1), (5, 2), (10, 3)]  # (media, desviación estándar)
COLORES = ['tab:blue', 'tab:orange', 'tab:green']

np.random.seed(9)
muestras_normal = {(mu, sigma): np.random.normal(mu, sigma, N_MUESTRAS)
                   for mu, sigma in PARAMETROS_NORMAL}

filas = []
for (mu, sigma), muestra in muestras_normal.items():
    filas.append({
        'mu': mu, 'sigma': sigma,
        'Media muestral': muestra.mean(),
        'Desv. muestral': muestra.std(ddof=1),
        # La altura del pico de la densidad normal es 1 / (sigma * raíz de 2 pi)
        'Altura del pico': stats.norm.pdf(mu, mu, sigma),
        'Intervalo 95%': f'[{mu - 1.96 * sigma:.2f}, {mu + 1.96 * sigma:.2f}]',
    })
tabla_normal = pd.DataFrame(filas)

print("=== a) Distribución normal: 1000 muestras por combinación ===")
print(tabla_normal.round(3).to_string(index=False))

# Muestras (histogramas) y densidades teóricas superpuestas en una misma gráfica
x = np.linspace(-5, 22, 500)
plt.figure(figsize=(9, 5))
for ((mu, sigma), muestra), color in zip(muestras_normal.items(), COLORES):
    plt.hist(muestra, bins=40, density=True, alpha=0.3, color=color)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), color=color, linewidth=2.2,
             label=f'N($\\mu$ = {mu}, $\\sigma$ = {sigma})')
plt.title('Distribuciones normales: muestras y densidades teóricas')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
guardar('a_normales.png')

# Extensión: efecto de cada parámetro por separado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for mu, color in zip([0, 5, 10], COLORES):
    ax1.plot(x, stats.norm.pdf(x, mu, 1), color=color, linewidth=2, label=f'$\\mu$ = {mu}')
ax1.set_title('Cambio de la media ($\\sigma$ = 1 fija)')
for sigma, color in zip([1, 2, 3], COLORES):
    ax2.plot(x, stats.norm.pdf(x, 5, sigma), color=color, linewidth=2, label=f'$\\sigma$ = {sigma}')
ax2.set_title('Cambio de la desviación estándar ($\\mu$ = 5 fija)')
for ax in (ax1, ax2):
    ax.set_xlabel('x')
    ax.set_ylabel('Densidad')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.5)
guardar('a_normales_parametros.png')

# ===============================================================
# b) Distribución de Poisson (lambda = 1, 5, 10) y comparación con la binomial
# ===============================================================

LAMBDAS = [1, 5, 10]

np.random.seed(9)
muestras_poisson = {lam: np.random.poisson(lam, N_MUESTRAS) for lam in LAMBDAS}

print("\n=== b) Distribución de Poisson: 1000 muestras por valor de lambda ===")
filas = []
for lam, muestra in muestras_poisson.items():
    k_max = muestra.max()
    frecuencias = np.bincount(muestra, minlength=k_max + 1) / N_MUESTRAS
    teorica = stats.poisson.pmf(np.arange(k_max + 1), lam)
    filas.append({'lambda': lam, 'Media muestral': muestra.mean(),
                  'Varianza muestral': muestra.var(ddof=1), 'Moda': np.argmax(frecuencias),
                  'Máx. |empírica - teórica|': np.abs(frecuencias - teorica).max()})
print(pd.DataFrame(filas).round(3).to_string(index=False))

# Frecuencias simuladas contra la función de masa teórica
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (lam, muestra), color in zip(axes, muestras_poisson.items(), COLORES):
    k = np.arange(0, 23)
    frecuencias = np.bincount(muestra, minlength=len(k))[:len(k)] / N_MUESTRAS
    ax.bar(k, frecuencias, color=color, alpha=0.6, edgecolor='black', label='Simulación')
    ax.plot(k, stats.poisson.pmf(k, lam), 'ko', markersize=4, label='P(X = k) teórica')
    ax.set_title(f'Poisson, $\\lambda$ = {lam}')
    ax.set_xlabel('k (número de eventos)')
    ax.set_ylabel('Probabilidad')
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
guardar('b_poisson.png')


def distancia_variacion_total(p, q):
    """Mitad de la suma de diferencias absolutas entre dos funciones de masa."""
    return 0.5 * np.abs(p - q).sum()


# Binomiales con la misma media que cada Poisson: n * p = lambda
TAMANOS_N = [20, 100]
k = np.arange(0, 41)
filas = []
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, lam in zip(axes, LAMBDAS):
    pmf_poisson = stats.poisson.pmf(k, lam)
    ax.bar(k, pmf_poisson, color='lightgray', edgecolor='black', label=f'Poisson ($\\lambda$ = {lam})')
    for n, estilo in zip(TAMANOS_N, ['s--', 'o-']):
        p = lam / n
        pmf_binomial = stats.binom.pmf(k, n, p)
        ax.plot(k, pmf_binomial, estilo, markersize=3.5, linewidth=1,
                label=f'Binomial (n = {n}, p = {p:g})')
        filas.append({'lambda': lam, 'n': n, 'p': p,
                      'Var. binomial': n * p * (1 - p), 'Var. Poisson': lam,
                      'Dist. variación total': distancia_variacion_total(pmf_binomial, pmf_poisson)})
    ax.set_xlim(-0.5, lam + 4 * np.sqrt(lam) + 1)
    ax.set_title(f'$\\lambda$ = n$\\cdot$p = {lam}')
    ax.set_xlabel('k')
    ax.set_ylabel('Probabilidad')
    ax.legend(fontsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
guardar('b_poisson_binomial.png')

print("\n=== b) Comparación con binomiales de la misma media (n * p = lambda) ===")
print(pd.DataFrame(filas).round(4).to_string(index=False))

# ===============================================================
# c) Ajuste de una normal a la temperatura del Weather Dataset
# ===============================================================

# Temperatura (°F) de la versión numérica del Weather Dataset (Witten et al., 2017)
temperatura = np.array([85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71], dtype=float)

# Ajuste por máxima verosimilitud: la media muestral y la desviación con divisor n
mu_ajuste, sigma_ajuste = stats.norm.fit(temperatura)
shapiro = stats.shapiro(temperatura)

print("\n=== c) Ajuste de una normal a la temperatura (Weather Dataset) ===")
print(f"n = {len(temperatura)}, mínimo = {temperatura.min():g}, máximo = {temperatura.max():g}")
print(f"Media = {temperatura.mean():.3f}, mediana = {np.median(temperatura):.1f}")
print(f"Ajuste normal (máxima verosimilitud): mu = {mu_ajuste:.3f}, sigma = {sigma_ajuste:.3f}")
print(f"Desviación estándar muestral (n - 1): {temperatura.std(ddof=1):.3f}")
print(f"Asimetría: {stats.skew(temperatura):.3f}   curtosis (exceso): {stats.kurtosis(temperatura):.3f}")
print(f"Prueba de Shapiro-Wilk: W = {shapiro.statistic:.4f}, p = {shapiro.pvalue:.4f}")

# Frecuencias observadas por intervalo contra las esperadas por la normal ajustada
bordes = np.arange(62.5, 92.5, 5)  # de 62.5 a 87.5 °F: cubre de 64 a 85 °F
observadas = np.histogram(temperatura, bins=bordes)[0]
esperadas = len(temperatura) * np.diff(stats.norm.cdf(bordes, mu_ajuste, sigma_ajuste))
tabla_intervalos = pd.DataFrame({
    'Intervalo (°F)': [f'[{a:g}, {b:g})' for a, b in zip(bordes[:-1], bordes[1:])],
    'Observadas': observadas,
    'Esperadas (normal)': esperadas.round(2),
})
print("\nFrecuencias por intervalo:")
print(tabla_intervalos.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.linspace(55, 95, 300)
ax1.hist(temperatura, bins=bordes, density=True, color='skyblue', edgecolor='black',
         label='Histograma (14 días)')
ax1.plot(x, stats.norm.pdf(x, mu_ajuste, sigma_ajuste), color='red', linewidth=2,
         label=f'Normal ajustada ($\\mu$ = {mu_ajuste:.1f}, $\\sigma$ = {sigma_ajuste:.1f})')
ax1.plot(x, stats.gaussian_kde(temperatura)(x), color='navy', linestyle='--', linewidth=1.5,
         label='Densidad estimada (KDE)')
ax1.plot(temperatura, np.zeros_like(temperatura), '|', color='black', markersize=12)
ax1.set_title('Temperatura del Weather Dataset')
ax1.set_xlabel('Temperatura (°F)')
ax1.set_ylabel('Densidad')
ax1.legend(fontsize=8)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

stats.probplot(temperatura, dist='norm', plot=ax2)
ax2.get_lines()[0].set(markerfacecolor='skyblue', markeredgecolor='navy', markersize=6)
ax2.get_lines()[1].set(color='red', linewidth=2)
ax2.set_title('Gráfico Q-Q contra la distribución normal')
ax2.set_xlabel('Cuantiles teóricos')
ax2.set_ylabel('Cuantiles de la muestra (°F)')
ax2.grid(linestyle='--', alpha=0.5)
guardar('c_ajuste_temperatura.png')

print(f"\nFiguras guardadas en {os.path.basename(CARPETA_IMG)}/")
