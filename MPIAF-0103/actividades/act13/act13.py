# =============================================================================
# Actividad 13: Construcción de un MiniModelo Preventivo
# Tema 2.3: regresión lineal bayesiana con PyMC y Posterior Predictive Check (PPC)
# Ejecutar: python act13.py   (las figuras se guardan en img/)
# En Colab, antes de ejecutar: !pip install -q pymc arviz
# =============================================================================

# ---- Importaciones y función auxiliar ----------------------------------------
import os

import matplotlib

# En Colab o Jupyter no existe __file__; ahí las figuras también se muestran en pantalla
EN_NOTEBOOK = '__file__' not in globals()
if not EN_NOTEBOOK:
    matplotlib.use('Agg')  # Como script, genera las figuras como archivos PNG sin abrir ventanas

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats

pd.set_option('display.width', 120)
SEMILLA = 42

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


def muestras(idata, nombre):
    """Muestras posteriores de una variable como arreglo 1D (cadenas y pasos juntos)."""
    return idata.posterior[nombre].values.reshape(-1)


# ---- Tarea 1: Simulación de datos --------------------------------------------
# Relación lineal entre horas de estudio y índice de aprobación (0 a 100) con ruido normal.
# Los valores verdaderos se conocen, así que se puede comprobar si el modelo los recupera.
ALFA_REAL, BETA_REAL, SIGMA_REAL = 35.0, 5.0, 6.0
n = 100
rng = np.random.default_rng(SEMILLA)
horas = rng.uniform(1, 10, size=n)
indice = ALFA_REAL + BETA_REAL * horas + rng.normal(0, SIGMA_REAL, size=n)
datos = pd.DataFrame({'horas': horas, 'indice': indice})

print("=== Tarea 1: datos simulados ===")
print(f"Modelo real: indice = {ALFA_REAL} + {BETA_REAL} * horas + Normal(0, {SIGMA_REAL})")
print(datos.describe().round(2).to_string())
print(f"Correlación de Pearson: {datos['horas'].corr(datos['indice']):.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(datos['horas'], datos['indice'], color='steelblue', edgecolor='black', alpha=0.7,
           label='Datos simulados')
xs = np.linspace(0, 11, 100)
ax.plot(xs, ALFA_REAL + BETA_REAL * xs, 'k--', linewidth=1.5, label='Recta real')
ax.set_xlabel('Horas de estudio')
ax.set_ylabel('Índice de aprobación')
ax.set_title('Relación entre horas de estudio e índice de aprobación')
ax.legend()
ax.grid(linestyle='--', alpha=0.5)
guardar('datos_simulados.png')


# ---- Tarea 2: Modelo bayesiano de regresión lineal ---------------------------
# indice_i ~ Normal(alfa + beta * horas_i, sigma)
# Priors débilmente informativos, a la escala del problema (índice de 0 a 100)
def modelo_normal(x, y):
    with pm.Model() as modelo:
        alfa = pm.Normal('alfa', mu=50, sigma=30)      # índice esperado con 0 horas
        beta = pm.Normal('beta', mu=0, sigma=10)       # puntos ganados por hora de estudio
        sigma = pm.HalfNormal('sigma', sigma=10)       # desviación del ruido
        mu = alfa + beta * x
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    return modelo


modelo = modelo_normal(datos['horas'].to_numpy(), datos['indice'].to_numpy())
with modelo:
    idata = pm.sample(2000, tune=1000, chains=4, random_seed=SEMILLA, progressbar=False)

print("\n=== Tarea 2: resumen de la distribución posterior ===")
resumen = az.summary(idata, var_names=['alfa', 'beta', 'sigma'], round_to=3)
print(resumen.to_string())
n_div = int(idata.sample_stats['diverging'].values.sum())
print(f"Divergencias: {n_div}")
reales = {'alfa': ALFA_REAL, 'beta': BETA_REAL, 'sigma': SIGMA_REAL}
for v, real in reales.items():
    m = muestras(idata, v)
    lo, hi = np.percentile(m, [3, 97])
    print(f"{v:<5} real = {real:<5} intervalo 94 %: [{lo:.2f}, {hi:.2f}]  contiene el real: {lo <= real <= hi}")

# Trazas por cadena y distribuciones posteriores
fig, ejes = plt.subplots(3, 2, figsize=(11, 8))
for fila, v in enumerate(('alfa', 'beta', 'sigma')):
    arr = idata.posterior[v].values  # (cadenas, pasos)
    for c in range(arr.shape[0]):
        ejes[fila, 0].plot(arr[c], linewidth=0.5, alpha=0.7)
    ejes[fila, 0].set_ylabel(v)
    ejes[fila, 1].hist(arr.reshape(-1), bins=50, density=True, color='skyblue', edgecolor='white')
    ejes[fila, 1].axvline(reales[v], color='k', linestyle='--', label='Valor real')
    ejes[fila, 1].set_xlabel(v)
    ejes[fila, 1].grid(linestyle='--', alpha=0.5)
ejes[0, 0].set_title('Trazas (4 cadenas)')
ejes[0, 1].set_title('Distribución posterior')
ejes[0, 1].legend()
ejes[2, 0].set_xlabel('Paso')
guardar('posterior.png')

# Recta de regresión con incertidumbre: 100 rectas posteriores y banda del 94 %
alfa_m, beta_m = muestras(idata, 'alfa'), muestras(idata, 'beta')
rectas = alfa_m[:, None] + beta_m[:, None] * xs[None, :]
banda = np.percentile(rectas, [3, 97], axis=0)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(datos['horas'], datos['indice'], color='steelblue', edgecolor='black', alpha=0.7,
           label='Datos', zorder=3)
for i in rng.choice(len(alfa_m), 100, replace=False):
    ax.plot(xs, rectas[i], color='gray', alpha=0.15, linewidth=0.8)
ax.fill_between(xs, banda[0], banda[1], color='orange', alpha=0.3, label='Banda de credibilidad 94 %')
ax.plot(xs, rectas.mean(axis=0), color='darkorange', linewidth=2, label='Recta posterior media')
ax.plot(xs, ALFA_REAL + BETA_REAL * xs, 'k--', linewidth=1.5, label='Recta real')
ax.set_xlabel('Horas de estudio')
ax.set_ylabel('Índice de aprobación')
ax.set_title('Regresión lineal bayesiana')
ax.legend()
ax.grid(linestyle='--', alpha=0.5)
guardar('regresion_posterior.png')


# ---- Tarea 3: Posterior Predictive Check (PPC) -------------------------------
# Se simulan conjuntos de datos nuevos con parámetros tomados de la posterior. Si el
# modelo es adecuado, los datos observados deben verse como uno más de los simulados.
def ppc(modelo, idata):
    with modelo:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=SEMILLA,
                                       progressbar=False)
    return idata.posterior_predictive['y_obs'].values.reshape(-1, n)  # (simulaciones, n)


def valor_p(sim, observado):
    """Valor p bayesiano: proporción de simulaciones con el estadístico >= al observado.
    Valores cerca de 0 o de 1 indican que el modelo no reproduce esa característica."""
    return float(np.mean(sim >= observado))


# Matriz que convierte un conjunto de datos y en los residuos de la recta de mínimos cuadrados
# contra las horas (x fijo). Permite medir la forma del ruido alrededor de la recta.
X_DISENO = np.column_stack([np.ones(n), datos['horas'].to_numpy()])
M_RESIDUOS = np.eye(n) - X_DISENO @ np.linalg.solve(X_DISENO.T @ X_DISENO, X_DISENO.T)


def curtosis_res(a, axis=None):
    """Curtosis en exceso de los residuos de una recta MCO ajustada a cada conjunto de datos.
    Vale cerca de 0 si el ruido es normal y es alta si hay valores atípicos."""
    return stats.kurtosis(np.asarray(a) @ M_RESIDUOS.T, axis=-1)


ESTADISTICOS = {'media': np.mean, 'desv. estándar': np.std, 'mínimo': np.min, 'máximo': np.max,
                'curtosis de residuos': curtosis_res}


def estadisticos_ppc(sim, y):
    """Valor observado, media simulada y valor p de cada estadístico."""
    filas = {}
    for nombre, f in ESTADISTICOS.items():
        s = f(sim, axis=1)
        filas[nombre] = {'observado': f(y), 'simulado (media)': s.mean(), 'valor p': valor_p(s, f(y))}
    return pd.DataFrame(filas).T


def graficar_ppc(sim, x, y, nombre_archivo, titulo):
    """Tres paneles: densidades simuladas contra la observada, curtosis de residuos y banda predictiva."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    # (1) Densidad de 100 conjuntos simulados contra la observada
    mallas = np.linspace(y.min() - 30, y.max() + 30, 200)  # acotada: la t simula valores extremos
    for i in np.random.default_rng(SEMILLA).choice(len(sim), 100, replace=False):
        ax1.plot(mallas, stats.gaussian_kde(sim[i])(mallas), color='tab:blue', alpha=0.15)
    ax1.plot(mallas, stats.gaussian_kde(y)(mallas), color='k', linewidth=2.5, label='Observado')
    ax1.plot([], [], color='tab:blue', label='Simulados (PPC)')
    ax1.set_xlabel('Índice de aprobación')
    ax1.set_ylabel('Densidad')
    ax1.set_title('Densidades observada y simuladas')
    ax1.legend()
    ax1.grid(linestyle='--', alpha=0.5)
    # (2) Curtosis de los residuos: colas del ruido alrededor de la recta
    s_k = curtosis_res(sim)
    obs_k = float(curtosis_res(y))
    tope = max(np.percentile(s_k, 99), obs_k) * 1.1
    ax2.hist(s_k[s_k <= tope], bins=50, density=True, color='tab:green', alpha=0.6, label='Simulado')
    ax2.axvline(obs_k, color='k', linewidth=2.5, linestyle='--', label='Observado')
    ax2.set_xlabel('Curtosis de los residuos')
    ax2.set_ylabel('Densidad')
    ax2.set_title('Colas del ruido (valores > percentil 99 omitidos)')
    ax2.legend()
    ax2.grid(linestyle='--', alpha=0.5)
    # (3) Banda predictiva del 94 % contra los datos
    orden = np.argsort(x)
    lo, hi = np.percentile(sim, [3, 97], axis=0)
    ax3.fill_between(x[orden], lo[orden], hi[orden], color='orange', alpha=0.3,
                     label='Intervalo predictivo 94 %')
    ax3.scatter(x, y, color='steelblue', edgecolor='black', alpha=0.7, label='Datos')
    ax3.set_ylim(y.min() - 30, y.max() + 30)
    ax3.set_xlabel('Horas de estudio')
    ax3.set_title('Banda predictiva')
    ax3.legend()
    ax3.grid(linestyle='--', alpha=0.5)
    fig.suptitle(titulo)
    guardar(nombre_archivo)


x_obs, y_obs = datos['horas'].to_numpy(), datos['indice'].to_numpy()
sim = ppc(modelo, idata)
tabla_ppc = estadisticos_ppc(sim, y_obs)
cobertura = float(np.mean((y_obs >= np.percentile(sim, 3, axis=0)) & (y_obs <= np.percentile(sim, 97, axis=0))))

print("\n=== Tarea 3: Posterior Predictive Check ===")
print(f"Simulaciones: {sim.shape[0]} conjuntos de {sim.shape[1]} datos")
print(tabla_ppc.round(3).to_string())
print(f"Cobertura del intervalo predictivo 94 %: {cobertura:.0%} de los datos observados")
graficar_ppc(sim, x_obs, y_obs, 'ppc_normal.png', 'PPC del modelo normal')


# ---- Extensión: datos atípicos y modelo robusto ------------------------------
# Se contaminan 6 estudiantes que estudiaron mucho pero obtuvieron un índice muy bajo.
# El modelo normal no los espera; el PPC debe delatarlo y la t de Student corregirlo.
idx_atipicos = np.argsort(x_obs)[-30:][np.random.default_rng(SEMILLA).choice(30, 6, replace=False)]
y_atip = y_obs.copy()
y_atip[idx_atipicos] = rng.uniform(5, 25, size=6)


def modelo_t(x, y):
    with pm.Model() as modelo:
        alfa = pm.Normal('alfa', mu=50, sigma=30)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=10)
        nu = pm.Exponential('nu', 1 / 30)              # grados de libertad de la t
        mu = alfa + beta * x
        pm.StudentT('y_obs', mu=mu, sigma=sigma, nu=nu, observed=y)
    return modelo


modelo_n2 = modelo_normal(x_obs, y_atip)
modelo_t2 = modelo_t(x_obs, y_atip)
with modelo_n2:
    idata_n2 = pm.sample(2000, tune=1000, chains=4, random_seed=SEMILLA, progressbar=False)
with modelo_t2:
    idata_t2 = pm.sample(2000, tune=1000, chains=4, random_seed=SEMILLA, progressbar=False)
sim_n2, sim_t2 = ppc(modelo_n2, idata_n2), ppc(modelo_t2, idata_t2)

print("\n=== Extensión: datos con 6 atípicos ===")
print(f"Atípicos en las posiciones de horas: {np.round(np.sort(x_obs[idx_atipicos]), 2)}")
for nombre, idt in (('Normal', idata_n2), ('t de Student', idata_t2)):
    print(f"\n{nombre}:")
    cols = ['alfa', 'beta', 'sigma'] + (['nu'] if nombre != 'Normal' else [])
    print(az.summary(idt, var_names=cols, round_to=3)[['mean', 'sd', 'r_hat']].to_string())
print("\nPPC del modelo normal con atípicos:")
print(estadisticos_ppc(sim_n2, y_atip).round(3).to_string())
print("\nPPC del modelo t de Student con atípicos:")
print(estadisticos_ppc(sim_t2, y_atip).round(3).to_string())
graficar_ppc(sim_n2, x_obs, y_atip, 'ppc_normal_atipicos.png', 'PPC del modelo normal con atípicos')
graficar_ppc(sim_t2, x_obs, y_atip, 'ppc_t_atipicos.png', 'PPC del modelo t de Student con atípicos')

# Efecto de los atípicos sobre la pendiente
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(muestras(idata_n2, 'beta'), bins=50, density=True, alpha=0.6, label='Normal', color='tab:red')
ax.hist(muestras(idata_t2, 'beta'), bins=50, density=True, alpha=0.6, label='t de Student',
        color='tab:blue')
ax.axvline(BETA_REAL, color='k', linestyle='--', label='Valor real')
ax.set_xlabel('beta (puntos por hora de estudio)')
ax.set_ylabel('Densidad')
ax.set_title('Pendiente posterior con datos atípicos')
ax.legend()
ax.grid(linestyle='--', alpha=0.5)
guardar('beta_atipicos.png')
