# =============================================================================
# Actividad 10: Explorando distribuciones discretas
# Tema 2.2: Bernoulli, binomial, Poisson, geométrica e hipergeométrica
# Ejecutar: python act10.py   (las figuras se guardan en img/)
# =============================================================================

# ---- Importaciones y función auxiliar ----------------------------------------
import math
import os
import random

import matplotlib

# En Colab o Jupyter no existe __file__; ahí las figuras también se muestran en pantalla
EN_NOTEBOOK = '__file__' not in globals()
if not EN_NOTEBOOK:
    matplotlib.use('Agg')  # Como script, genera las figuras como archivos PNG sin abrir ventanas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option('display.width', 110)

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


# ---- Tarea 1: Simulación individual ------------------------------------------
# Cada distribución se simula con dos conjuntos distintos de parámetros.
N_SIM = 10000
SEMILLA = 10
COLORES = ['tab:blue', 'tab:orange']

# Parámetros (hipergeométrica: N = total de bolas, K = negras, n = extracciones)
DISTRIBUCIONES = {
    'Bernoulli':      [dict(p=0.3), dict(p=0.7)],
    'Binomial':       [dict(n=10, p=0.5), dict(n=20, p=0.2)],
    'Poisson':        [dict(lam=2), dict(lam=7)],
    'Geométrica':     [dict(p=0.5), dict(p=0.2)],
    'Hipergeométrica': [dict(N=50, K=10, n=10), dict(N=50, K=25, n=10)],
}


def etiqueta(par):
    """Texto con los parámetros, por ejemplo 'n = 10, p = 0.5'."""
    nombres = {'lam': '$\\lambda$'}
    return ', '.join(f'{nombres.get(k, k)} = {v:g}' for k, v in par.items())


def simular(dist, par, tam, rng):
    """Genera `tam` valores de la distribución con los parámetros `par`."""
    if dist == 'Bernoulli':
        return rng.binomial(1, par['p'], tam)
    if dist == 'Binomial':
        return rng.binomial(par['n'], par['p'], tam)
    if dist == 'Poisson':
        return rng.poisson(par['lam'], tam)
    if dist == 'Geométrica':
        return rng.geometric(par['p'], tam)  # número de pruebas hasta el primer éxito
    if dist == 'Hipergeométrica':
        return rng.hypergeometric(par['K'], par['N'] - par['K'], par['n'], tam)


def pmf_scipy(dist, par, k):
    """Función de masa de probabilidad calculada con scipy.stats."""
    if dist == 'Bernoulli':
        return stats.bernoulli.pmf(k, par['p'])
    if dist == 'Binomial':
        return stats.binom.pmf(k, par['n'], par['p'])
    if dist == 'Poisson':
        return stats.poisson.pmf(k, par['lam'])
    if dist == 'Geométrica':
        return stats.geom.pmf(k, par['p'])
    if dist == 'Hipergeométrica':
        return stats.hypergeom.pmf(k, par['N'], par['K'], par['n'])


def pmf_manual(dist, par, k):
    """Función de masa calculada a mano con las fórmulas del documento fuente (un valor k)."""
    if dist == 'Bernoulli':
        return par['p'] ** k * (1 - par['p']) ** (1 - k) if k in (0, 1) else 0.0
    if dist == 'Binomial':
        n, p = par['n'], par['p']
        return math.comb(n, k) * p ** k * (1 - p) ** (n - k) if 0 <= k <= n else 0.0
    if dist == 'Poisson':
        lam = par['lam']
        return math.exp(-lam) * lam ** k / math.factorial(k)
    if dist == 'Geométrica':
        return (1 - par['p']) ** (k - 1) * par['p'] if k >= 1 else 0.0
    if dist == 'Hipergeométrica':
        N, K, n = par['N'], par['K'], par['n']
        if k < max(0, n - (N - K)) or k > min(n, K):
            return 0.0
        return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def momentos_teoricos(dist, par):
    """Media y varianza teóricas de cada distribución."""
    if dist == 'Bernoulli':
        p = par['p']
        return p, p * (1 - p)
    if dist == 'Binomial':
        n, p = par['n'], par['p']
        return n * p, n * p * (1 - p)
    if dist == 'Poisson':
        return par['lam'], par['lam']
    if dist == 'Geométrica':
        p = par['p']
        return 1 / p, (1 - p) / p ** 2
    if dist == 'Hipergeométrica':
        N, K, n = par['N'], par['K'], par['n']
        return n * K / N, n * (K / N) * (1 - K / N) * (N - n) / (N - 1)


rng = np.random.default_rng(SEMILLA)
muestras = {(d, i): simular(d, par, N_SIM, rng)
            for d, pares in DISTRIBUCIONES.items() for i, par in enumerate(pares)}

# Simulación "desde cero" con random.random(), como en el documento fuente
random.seed(SEMILLA)


def bernoulli_manual(p):
    return 1 if random.random() < p else 0


def geometrica_manual(p):
    pruebas = 1
    while random.random() >= p:
        pruebas += 1
    return pruebas


def poisson_manual(lam):
    # Método de Knuth: multiplicar uniformes hasta que el producto baje de e^(-lambda)
    L, k, prod = math.exp(-lam), 0, 1.0
    while prod > L:
        k += 1
        prod *= random.random()
    return k - 1


manuales = {
    'Bernoulli (p = 0.3)': [bernoulli_manual(0.3) for _ in range(N_SIM)],
    'Geométrica (p = 0.2)': [geometrica_manual(0.2) for _ in range(N_SIM)],
    'Poisson ($\\lambda$ = 7)': [poisson_manual(7) for _ in range(N_SIM)],
}
print(f"=== Tarea 1: simulación con {N_SIM} valores por caso (semilla {SEMILLA}) ===")
print("\nSimulación desde cero con random.random() (media muestral vs teórica):")
for nombre, valores in manuales.items():
    print(f"  {nombre.replace('$\\lambda$', 'lambda')}: media = {np.mean(valores):.3f}")


# ---- Tarea 2: Visualización --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (dist, pares) in zip(axes.flat, DISTRIBUCIONES.items()):
    k_max = max(muestras[(dist, i)].max() for i in range(len(pares)))
    if dist == 'Geométrica':
        k_max = min(k_max, 20)  # la cola es larga; se muestran los primeros valores
    ks = np.arange(0, k_max + 1)
    ancho = 0.4
    for i, (par, color) in enumerate(zip(pares, COLORES)):
        frec = np.bincount(muestras[(dist, i)], minlength=k_max + 1)[:k_max + 1] / N_SIM
        ax.bar(ks + (i - 0.5) * ancho, frec, ancho, color=color, alpha=0.7,
               edgecolor='black', linewidth=0.4, label=f'Simulación: {etiqueta(par)}')
        ax.plot(ks + (i - 0.5) * ancho, pmf_scipy(dist, par, ks), 'k.', markersize=5)
    if dist == 'Bernoulli':
        ax.set_xticks([0, 1])  # solo toma los valores 0 y 1
    ax.set_title(dist)
    ax.set_xlabel('k')
    ax.set_ylabel('Frecuencia relativa')
    ax.legend(fontsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
axes.flat[-1].axis('off')
axes.flat[-1].text(0.05, 0.5, 'Barras: frecuencias simuladas\nPuntos negros: P(X = k) teórica',
                   fontsize=11, va='center')
guardar('t2_comparacion_distribuciones.png')

# Cómo cambia la forma al variar un solo parámetro (probabilidades teóricas)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
k = np.arange(0, 31)
for p in [0.1, 0.5, 0.9]:
    axes[0].plot(k, stats.binom.pmf(k, 30, p), 'o-', markersize=3, label=f'n = 30, p = {p}')
axes[0].set_title('Binomial: efecto de p')
for lam in [1, 5, 15]:
    axes[1].plot(k, stats.poisson.pmf(k, lam), 'o-', markersize=3, label=f'$\\lambda$ = {lam}')
axes[1].set_title('Poisson: efecto de $\\lambda$')
for p in [0.2, 0.5, 0.8]:
    axes[2].plot(k[1:], stats.geom.pmf(k[1:], p), 'o-', markersize=3, label=f'p = {p}')
axes[2].set_title('Geométrica: efecto de p')
for ax in axes:
    ax.set_xlabel('k')
    ax.set_ylabel('P(X = k)')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.5)
guardar('t2_efecto_parametros.png')


# ---- Tarea 3: Comparación teórica-empírica ------------------------------------
# a) Media y varianza
filas = []
for (dist, i), muestra in muestras.items():
    par = DISTRIBUCIONES[dist][i]
    media_t, var_t = momentos_teoricos(dist, par)
    filas.append({'Distribución': dist, 'Parámetros': etiqueta(par).replace('$\\lambda$', 'lambda'),
                  'Media teórica': media_t, 'Media simulada': muestra.mean(),
                  'Var. teórica': var_t, 'Var. simulada': muestra.var(ddof=1)})
tabla_momentos = pd.DataFrame(filas)
print("\n=== Tarea 3a: media y varianza, teóricas vs simuladas ===")
print(tabla_momentos.round(3).to_string(index=False))

# b) Probabilidades puntuales: fórmula manual, scipy y simulación
casos = [('Bernoulli', 0, 1), ('Binomial', 0, 5), ('Binomial', 1, 4), ('Poisson', 0, 2),
         ('Poisson', 1, 7), ('Geométrica', 0, 2), ('Geométrica', 1, 5),
         ('Hipergeométrica', 0, 1), ('Hipergeométrica', 1, 5)]
filas = []
for dist, i, kk in casos:
    par = DISTRIBUCIONES[dist][i]
    manual = pmf_manual(dist, par, kk)
    assert abs(manual - pmf_scipy(dist, par, kk)) < 1e-12  # la fórmula coincide con scipy
    empirica = np.mean(muestras[(dist, i)] == kk)
    # Error estándar de una proporción: raíz de p(1 - p) / N
    ee = math.sqrt(manual * (1 - manual) / N_SIM)
    filas.append({'Distribución': dist, 'Parámetros': etiqueta(par).replace('$\\lambda$', 'lambda'),
                  'k': kk, 'P teórica': manual, 'Frec. simulada': empirica,
                  'Diferencia': empirica - manual, 'Error estándar': ee})
tabla_puntual = pd.DataFrame(filas)
print("\n=== Tarea 3b: P(X = k) con la fórmula, y frecuencia simulada ===")
print(tabla_puntual.round(4).to_string(index=False))

# c) Prueba chi-cuadrada de bondad de ajuste para cada simulación
filas = []
for (dist, i), muestra in muestras.items():
    par = DISTRIBUCIONES[dist][i]
    ks = np.arange(0, muestra.max() + 1)
    obs = np.bincount(muestra, minlength=len(ks)).astype(float)
    esp = pmf_scipy(dist, par, ks) * N_SIM
    # Se agrupan las categorías con frecuencia esperada menor que 5 (regla usual)
    grupos_o, grupos_e, acum_o, acum_e = [], [], 0.0, 0.0
    for o, e in zip(obs, esp):
        acum_o += o
        acum_e += e
        if acum_e >= 5:
            grupos_o.append(acum_o)
            grupos_e.append(acum_e)
            acum_o = acum_e = 0.0
    grupos_o[-1] += acum_o
    grupos_e[-1] += acum_e
    grupos_e = np.array(grupos_e) * (sum(grupos_o) / sum(grupos_e))  # iguala los totales
    chi2 = ((np.array(grupos_o) - grupos_e) ** 2 / grupos_e).sum()
    filas.append({'Distribución': dist, 'Parámetros': etiqueta(par).replace('$\\lambda$', 'lambda'),
                  'Categorías': len(grupos_o), 'chi2': chi2,
                  'p-valor': stats.chi2.sf(chi2, len(grupos_o) - 1)})
tabla_chi = pd.DataFrame(filas)
print("\n=== Tarea 3c: prueba chi-cuadrada de bondad de ajuste ===")
print(tabla_chi.round(4).to_string(index=False))

# d) Ejemplo del documento fuente: lotería de California (6 de 53)
ks = np.arange(0, 7)
teorica_loteria = np.array([pmf_manual('Hipergeométrica', dict(N=53, K=6, n=6), k) for k in ks])
sim_loteria = rng.hypergeometric(6, 47, 6, 200000)
frec_loteria = np.bincount(sim_loteria, minlength=7) / len(sim_loteria)
print("\n=== Tarea 3d: lotería de California (hipergeométrica N = 53, K = 6, n = 6) ===")
print(pd.DataFrame({'k': ks, 'P teórica': teorica_loteria,
                    'Frec. (200000 sorteos)': frec_loteria}).to_string(
    index=False, float_format=lambda v: f'{v:.6f}'))

# e) Convergencia: el error máximo baja al aumentar el número de simulaciones
tamanos = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]
rng_conv = np.random.default_rng(SEMILLA)
errores = {}
for dist in ['Binomial', 'Poisson', 'Geométrica']:
    par = DISTRIBUCIONES[dist][0]
    errores[dist] = []
    for tam in tamanos:
        m = simular(dist, par, tam, rng_conv)
        ks = np.arange(0, m.max() + 1)
        errores[dist].append(np.abs(np.bincount(m, minlength=len(ks)) / tam
                                    - pmf_scipy(dist, par, ks)).max())
plt.figure(figsize=(8, 4.5))
for dist, e in errores.items():
    plt.loglog(tamanos, e, 'o-', label=f'{dist} ({etiqueta(DISTRIBUCIONES[dist][0])})')
plt.loglog(tamanos, 1 / np.sqrt(tamanos), 'k--', alpha=0.5, label='$1/\\sqrt{N}$ (referencia)')
plt.title('Convergencia de la simulación a la distribución teórica')
plt.xlabel('Número de simulaciones N')
plt.ylabel('Máx. |frecuencia - P(X = k)|')
plt.legend(fontsize=8)
plt.grid(which='both', linestyle='--', alpha=0.5)
guardar('t3_convergencia.png')
print("\n=== Tarea 3e: error máximo según el número de simulaciones ===")
print(pd.DataFrame(errores, index=tamanos).rename_axis('N').round(4).to_string())

print(f"\nFiguras guardadas en {os.path.basename(CARPETA_IMG)}/")
