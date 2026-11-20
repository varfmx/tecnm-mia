# =============================================================================
# Actividad 12: Ajuste de una distribución a datos reales
# Tema 2.2: histograma con curva de densidad, ajuste de modelos discretos y continuos
# Ejecutar: python act12.py   (las figuras se guardan en img/)
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
import seaborn as sns
from scipy import stats

pd.set_option('display.width', 120)

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


# ---- Tarea 1: Selección de datos ---------------------------------------------
# Conjunto de datos del Titanic (891 pasajeros).
# Variable 1: edad. Se mide en años (continua), pero la edad en años cumplidos es un
# conteo, así que se puede modelar con distribuciones discretas y continuas.
# Variable 2: número de hermanos o cónyuges a bordo (sibsp), un conteo.
titanic = sns.load_dataset('titanic')
edad = titanic['age'].dropna().to_numpy()   # 714 pasajeros con edad registrada
edad_cumplida = np.floor(edad).astype(int)  # años cumplidos
hermanos = titanic['sibsp'].to_numpy()      # 891 pasajeros


def resumen(x):
    """Medidas descriptivas; var/media cerca de 1 es lo esperado en una Poisson."""
    return pd.Series({
        'n': len(x), 'media': x.mean(), 'varianza': x.var(ddof=1),
        'var/media': x.var(ddof=1) / x.mean(), 'mínimo': x.min(), 'máximo': x.max(),
        'asimetría': stats.skew(x), 'curtosis': stats.kurtosis(x),
    })


print("=== Tarea 1: selección de datos (Titanic) ===")
print(f"Pasajeros: {len(titanic)}; con edad registrada: {len(edad)}; sin edad: {len(titanic) - len(edad)}")
print(pd.DataFrame({'edad (años cumplidos)': resumen(edad_cumplida),
                    'hermanos/cónyuges': resumen(hermanos)}).round(3).to_string())


# ---- Tarea 2: Visualización inicial ------------------------------------------
# Histograma con curva de densidad (KDE) de la edad y gráfico de barras de sibsp
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.linspace(0, 85, 400)
ax1.hist(edad, bins=np.arange(0, 85, 5), density=True, color='skyblue', edgecolor='black',
         alpha=0.7, label='Histograma (5 años)')
ax1.plot(x, stats.gaussian_kde(edad)(x), color='navy', linewidth=2, label='Curva de densidad (KDE)')
ax1.set_title('Edad de los pasajeros del Titanic')
ax1.set_xlabel('Edad (años)')
ax1.set_ylabel('Densidad')
ax1.legend()
ax1.grid(linestyle='--', alpha=0.5)

valores_h = np.arange(0, hermanos.max() + 1)
frec_h = np.array([(hermanos == k).mean() for k in valores_h])
ax2.bar(valores_h, frec_h, color='skyblue', edgecolor='black', alpha=0.7, label='Frecuencia relativa')
ax2.set_title('Hermanos o cónyuges a bordo')
ax2.set_xlabel('Número (k)')
ax2.set_ylabel('Proporción de pasajeros')
ax2.legend()
ax2.grid(linestyle='--', alpha=0.5)
guardar('visualizacion_inicial.png')


# ---- Tarea 3: Ajuste de modelos ----------------------------------------------
# Los parámetros se estiman por máxima verosimilitud. Las distribuciones del tema:
#   discretas: Poisson y binomial (y geométrica para los conteos)
#   continuas: normal, exponencial y gamma
# En la exponencial y la gamma, scipy usa la escala (1 / lambda del documento fuente).
def ajustar_edad():
    """Ajusta los cinco modelos a la edad. Devuelve {nombre: (distribución, tipo, parámetros)}."""
    media = edad_cumplida.mean()
    n_bin = int(edad_cumplida.max())                  # número de pruebas de la binomial
    mu, sigma = stats.norm.fit(edad)
    _, escala_exp = stats.expon.fit(edad, floc=0)     # escala = media
    alfa, _, escala_gam = stats.gamma.fit(edad, floc=0)
    return {
        'Poisson':     (stats.poisson(media), 'discreta', 1),
        'Binomial':    (stats.binom(n_bin, media / n_bin), 'discreta', 1),  # n fijo
        'Normal':      (stats.norm(mu, sigma), 'continua', 2),
        'Exponencial': (stats.expon(scale=escala_exp), 'continua', 1),
        'Gamma':       (stats.gamma(alfa, scale=escala_gam), 'continua', 2),
    }, dict(media=media, n_bin=n_bin, mu=mu, sigma=sigma, escala_exp=escala_exp,
            alfa=alfa, escala_gam=escala_gam)


def probabilidades(dist, tipo, valores, inicio):
    """Probabilidad que el modelo asigna a cada valor entero k.
    Discreta: pmf(k). Continua: área bajo la densidad en [k + inicio, k + inicio + 1)
    (inicio = 0 para años cumplidos; -0.5 para conteos, con corrección por continuidad)."""
    if tipo == 'discreta':
        return dist.pmf(valores)
    return dist.cdf(valores + inicio + 1) - dist.cdf(valores + inicio)


modelos_edad, par_edad = ajustar_edad()

print("\n=== Tarea 3: parámetros ajustados a la edad (máxima verosimilitud) ===")
print(f"Poisson:     lambda = {par_edad['media']:.3f}")
print(f"Binomial:    n = {par_edad['n_bin']}, p = {par_edad['media'] / par_edad['n_bin']:.4f}")
print(f"Normal:      mu = {par_edad['mu']:.3f}, sigma = {par_edad['sigma']:.3f}")
print(f"Exponencial: lambda = {1 / par_edad['escala_exp']:.4f}  (media 1/lambda = {par_edad['escala_exp']:.3f})")
print(f"Gamma:       alpha = {par_edad['alfa']:.3f}, lambda = {1 / par_edad['escala_gam']:.4f}")

# Histograma de años cumplidos con las curvas de los modelos. Con barras de 1 año, la
# altura del histograma es la proporción por año, comparable con pmf y densidad.
ks_edad = np.arange(0, 81)
obs_edad = np.array([(edad_cumplida == k).mean() for k in ks_edad])
colores = {'Poisson': 'tab:orange', 'Binomial': 'tab:green', 'Normal': 'tab:red',
           'Exponencial': 'tab:purple', 'Gamma': 'tab:blue'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, tipo in zip((ax1, ax2), ('continua', 'discreta')):
    ax.bar(ks_edad, obs_edad, width=1, align='edge', color='lightgray', edgecolor='gray',
           alpha=0.8, label='Observado (1 año)')
    for nombre, (dist, t, _) in modelos_edad.items():
        if t != tipo:
            continue
        p = probabilidades(dist, t, ks_edad, 0)
        estilo = '-' if t == 'continua' else 'o--'
        ax.plot(ks_edad + 0.5, p, estilo, color=colores[nombre], linewidth=2, markersize=3,
                label=nombre)
    ax.set_title(f'Modelos {tipo}s ajustados a la edad')
    ax.set_xlabel('Edad (años cumplidos)')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.5)
ax1.set_ylabel('Probabilidad por año')
guardar('ajuste_edad.png')


# ---- Tarea 4: Evaluación -----------------------------------------------------
def evaluar(modelos, valores, inicio, observado, n):
    """Error cuadrático medio entre las proporciones observadas y las del modelo, más
    verosimilitud y AIC sobre la misma escala (probabilidad de cada valor entero)."""
    filas = {}
    for nombre, (dist, tipo, k_par) in modelos.items():
        p = probabilidades(dist, tipo, valores, inicio)
        err = observado - p
        # Verosimilitud multinomial de los datos agrupados por valor
        loglik = float(np.sum(n * observado[observado > 0] * np.log(np.maximum(p[observado > 0], 1e-300))))
        filas[nombre] = {'tipo': tipo, 'MSE': np.mean(err ** 2), 'RMSE': np.sqrt(np.mean(err ** 2)),
                         'error máx.': np.abs(err).max(), 'masa en rango': p.sum(),
                         'logL': loglik, 'AIC': 2 * k_par - 2 * loglik}
    return pd.DataFrame(filas).T


tabla_edad = evaluar(modelos_edad, ks_edad, 0, obs_edad, len(edad_cumplida))
tabla_edad['MSE x 1e4'] = tabla_edad['MSE'].astype(float) * 1e4
print("\n=== Tarea 4: evaluación del ajuste a la edad (MSE sobre proporciones por año) ===")
print(tabla_edad[['tipo', 'MSE x 1e4', 'RMSE', 'error máx.', 'masa en rango', 'logL', 'AIC']]
      .sort_values('MSE x 1e4').astype({'MSE x 1e4': float, 'RMSE': float, 'error máx.': float,
                                        'masa en rango': float, 'logL': float, 'AIC': float})
      .round(4).to_string())

# Comparación visual adicional: Q-Q de los dos mejores continuos y función acumulada
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
probs = (np.arange(1, len(edad) + 1) - 0.5) / len(edad)
orden = np.sort(edad)
for ax, nombre in zip((ax1, ax2), ('Normal', 'Gamma')):
    teo = modelos_edad[nombre][0].ppf(probs)
    ax.plot(teo, orden, '.', color=colores[nombre], markersize=4)
    ax.plot([0, 80], [0, 80], 'k--', linewidth=1)
    ax.set_title(f'Gráfico Q-Q: {nombre}')
    ax.set_xlabel('Cuantiles del modelo')
    ax.set_ylabel('Cuantiles de la edad')
    ax.grid(linestyle='--', alpha=0.5)
ax3.step(orden, probs, color='black', linewidth=3, alpha=0.4, label='Empírica')
xs = np.linspace(0, 80, 400)
for nombre, (dist, tipo, _) in modelos_edad.items():
    cdf = dist.cdf(np.floor(xs)) if tipo == 'discreta' else dist.cdf(xs)
    ax3.plot(xs, cdf, color=colores[nombre], linewidth=1.5, label=nombre)
ax3.set_title('Función acumulada')
ax3.set_xlabel('Edad (años)')
ax3.legend(fontsize=8)
ax3.grid(linestyle='--', alpha=0.5)
guardar('qq_acumulada_edad.png')

# Prueba de Kolmogorov-Smirnov para los modelos continuos (edad sin redondear)
print("\nPrueba de Kolmogorov-Smirnov (modelos continuos, edad original):")
for nombre in ('Normal', 'Exponencial', 'Gamma'):
    ks = stats.kstest(edad, modelos_edad[nombre][0].cdf)
    print(f"  {nombre:<12} D = {ks.statistic:.4f}   p = {ks.pvalue:.4g}")


# ---- Segunda variable: conteo de hermanos o cónyuges ---------------------------
# Aquí los datos son conteos genuinos; se espera que un modelo discreto sea competitivo.
media_h = hermanos.mean()
modelos_h = {
    'Poisson':     (stats.poisson(media_h), 'discreta', 1),
    'Binomial':    (stats.binom(int(hermanos.max()), media_h / hermanos.max()), 'discreta', 1),
    # Geométrica con soporte 0, 1, 2, ...: P(k) = (1 - p)^k p, con p = 1 / (1 + media)
    'Geométrica':  (stats.geom(1 / (1 + media_h), loc=-1), 'discreta', 1),
    'Normal':      (stats.norm(media_h, hermanos.std()), 'continua', 2),
    'Exponencial': (stats.expon(scale=media_h), 'continua', 1),
}
obs_h = np.array([(hermanos == k).mean() for k in valores_h])
tabla_h = evaluar(modelos_h, valores_h, -0.5, obs_h, len(hermanos))
tabla_h['MSE x 1e4'] = tabla_h['MSE'].astype(float) * 1e4

print("\n=== Segunda variable: hermanos o cónyuges (k = 0..8) ===")
print(f"Geométrica: p = {1 / (1 + media_h):.4f}; Poisson: lambda = {media_h:.4f}")
prob_h = pd.DataFrame({'observada': obs_h}, index=valores_h)
for nombre, (dist, tipo, _) in modelos_h.items():
    prob_h[nombre] = probabilidades(dist, tipo, valores_h, -0.5)
print(prob_h.round(4).to_string())
print(tabla_h[['tipo', 'MSE x 1e4', 'RMSE', 'error máx.', 'masa en rango', 'logL', 'AIC']]
      .sort_values('MSE x 1e4').astype({'MSE x 1e4': float, 'RMSE': float, 'error máx.': float,
                                        'masa en rango': float, 'logL': float, 'AIC': float})
      .round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 4.5))
ancho = 0.8 / (len(modelos_h) + 1)
ax.bar(valores_h - 0.4, obs_h, width=ancho, align='edge', color='lightgray', edgecolor='gray',
       label='Observado')
col_h = {'Poisson': 'tab:orange', 'Binomial': 'tab:green', 'Geométrica': 'tab:blue',
         'Normal': 'tab:red', 'Exponencial': 'tab:purple'}
for i, nombre in enumerate(modelos_h):
    ax.bar(valores_h - 0.4 + ancho * (i + 1), prob_h[nombre], width=ancho, align='edge',
           color=col_h[nombre], label=nombre)
ax.set_xlabel('Hermanos o cónyuges a bordo (k)')
ax.set_ylabel('Probabilidad')
ax.set_title('Ajuste a los conteos de hermanos o cónyuges')
ax.legend()
ax.grid(linestyle='--', alpha=0.5, axis='y')
guardar('ajuste_hermanos.png')
