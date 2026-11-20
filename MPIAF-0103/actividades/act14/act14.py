# =============================================================================
# Actividad 14: Explorando las Distribuciones de Probabilidad
# Tema 2.3: clasificación, tabla comparativa y efecto de los parámetros (normal y Poisson)
# Ejecutar: python act14.py   (las figuras se guardan en img/)
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


# ---- Tareas 1 y 2: Clasificación y tabla comparativa --------------------------
# Distribuciones del documento fuente (Tema 2.3), con su tipo, uso principal y un ejemplo.
# Cada una se instancia en scipy.stats con parámetros de ejemplo para comprobar que su tipo
# (masa o densidad) y sus momentos coinciden con lo que dice la tabla.
CATALOGO = [
    # (nombre, tipo, uso principal, ejemplo, distribución de scipy con parámetros de ejemplo)
    ('Normal', 'continua', 'Variables continuas simétricas; efecto de muchas causas pequeñas',
     'Estatura de una población', stats.norm(170, 8)),
    ('Exponencial', 'continua', 'Tiempo entre eventos de un proceso de Poisson',
     'Minutos entre llegadas a una ventanilla', stats.expon(scale=5)),
    ('Gamma', 'continua', 'Tiempos de espera y valores positivos asimétricos',
     'Tiempo hasta recibir 3 llamadas', stats.gamma(3, scale=2)),
    ('Uniforme', 'continua', 'Variables con rango finito conocido y sin valores preferidos',
     'Hora de llegada dentro de una hora', stats.uniform(0, 60)),
    ('Poisson', 'discreta', 'Número de eventos en un intervalo fijo de tiempo o espacio',
     'Correos recibidos por hora', stats.poisson(4)),
    ('Binomial', 'discreta', 'Número de éxitos en n ensayos independientes de dos resultados',
     'Alumnos que aprueban de un grupo de 20', stats.binom(20, 0.7)),
    ('Binomial negativa', 'discreta', 'Número de ensayos hasta alcanzar un número fijo de éxitos',
     'Intentos hasta lograr 3 ventas', stats.nbinom(3, 0.4, loc=3)),  # scipy cuenta fracasos; loc=3 cuenta ensayos
    ('Multinomial', 'discreta', 'Conteos de variables con más de dos categorías',
     'Resultados de 10 lanzamientos de un dado', stats.multinomial(10, [1 / 6] * 6)),
]

filas = []
for nombre, tipo, uso, ejemplo, dist in CATALOGO:
    if nombre == 'Multinomial':
        # Vectorial: se resume con la media y la varianza de una de sus categorías
        media, var = dist.mean()[0], dist.cov()[0, 0]
        funcion = 'masa'
    else:
        media, var = dist.mean(), dist.var()
        funcion = 'masa' if tipo == 'discreta' else 'densidad'
    filas.append({'distribución': nombre, 'tipo': tipo, 'función': funcion,
                  'media': media, 'varianza': var})

print("=== Tareas 1 y 2: clasificación de las distribuciones ===")
print(pd.DataFrame(filas).round(3).to_string(index=False))

# Comprobación de la clasificación: una discreta suma 1 sobre sus valores y una continua
# integra 1 bajo su densidad (con la regla del trapecio)
print("\nComprobación de la normalización:")
print(f"Poisson(4):  suma de la masa en k = 0..60 = {stats.poisson(4).pmf(np.arange(61)).sum():.6f}")
xs_n = np.linspace(100, 240, 20001)
print(f"Normal(170, 8): integral de la densidad = {np.trapezoid(stats.norm(170, 8).pdf(xs_n), xs_n):.6f}")


# ---- Tarea 3: Distribución normal y efecto de sus parámetros -----------------
# mu desplaza la curva; sigma controla su anchura (y, por tanto, su altura máxima)
x = np.linspace(-10, 10, 600)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for mu in (-3, 0, 3):
    ax1.plot(x, stats.norm(mu, 1).pdf(x), linewidth=2, label=fr'$\mu = {mu}$, $\sigma = 1$')
ax1.set_title(r'Normal: cambio de $\mu$ con $\sigma$ fija')
ax1.set_ylabel('Densidad $f(x)$')
for sigma in (0.5, 1, 2, 3):
    ax2.plot(x, stats.norm(0, sigma).pdf(x), linewidth=2, label=fr'$\mu = 0$, $\sigma = {sigma}$')
ax2.set_title(r'Normal: cambio de $\sigma$ con $\mu$ fija')
for ax in (ax1, ax2):
    ax.set_xlabel('$x$')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.5)
guardar('normal_parametros.png')

print("\n=== Tarea 3: normal, altura máxima y probabilidad en [mu - sigma, mu + sigma] ===")
for sigma in (0.5, 1, 2, 3):
    d = stats.norm(0, sigma)
    print(f"sigma = {sigma:<3}  f(mu) = {d.pdf(0):.3f}  P(|X - mu| < sigma) = {d.cdf(sigma) - d.cdf(-sigma):.4f}")


# ---- Tarea 3: Distribución Poisson y efecto de su parámetro ------------------
# lambda es a la vez la media y la varianza; al crecer, la masa se desplaza y se dispersa
# y la forma se vuelve simétrica, parecida a una normal
k = np.arange(0, 41)
lambdas = (1, 4, 10, 20)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ancho = 0.8 / len(lambdas)
for i, lam in enumerate(lambdas):
    ax1.bar(k - 0.4 + ancho * (i + 0.5), stats.poisson(lam).pmf(k), width=ancho,
            label=fr'$\lambda = {lam}$')
ax1.set_title(r'Poisson: cambio de $\lambda$')
ax1.set_xlabel('$k$ (número de eventos)')
ax1.set_ylabel('Masa $p(k)$')
ax1.set_xlim(-1, 35)
ax1.legend()
ax1.grid(linestyle='--', alpha=0.5)

# Aproximación normal: con lambda grande la Poisson se parece a Normal(lambda, sqrt(lambda))
lam = 20
ax2.bar(k, stats.poisson(lam).pmf(k), color='skyblue', edgecolor='black', alpha=0.8,
        label=fr'Poisson($\lambda = {lam}$)')
xc = np.linspace(0, 40, 400)
ax2.plot(xc, stats.norm(lam, np.sqrt(lam)).pdf(xc), color='tab:red', linewidth=2,
         label=fr'Normal($\mu = {lam}$, $\sigma = \sqrt{{{lam}}}$)')
ax2.set_title('Poisson con $\\lambda$ grande y su aproximación normal')
ax2.set_xlabel('$k$')
ax2.legend()
ax2.grid(linestyle='--', alpha=0.5)
guardar('poisson_parametros.png')

print("\n=== Tarea 3: Poisson, media, varianza y masa máxima ===")
for lam in lambdas:
    d = stats.poisson(lam)
    print(f"lambda = {lam:<3} media = {d.mean():.1f}  varianza = {d.var():.1f}  "
          f"masa máxima p(k) = {d.pmf(k).max():.3f}")
aprox = stats.norm(20, np.sqrt(20))
error = np.abs(stats.poisson(20).pmf(k) - (aprox.cdf(k + 0.5) - aprox.cdf(k - 0.5))).max()
print(f"Error máximo de la aproximación normal con lambda = 20 (corrección por continuidad): {error:.4f}")
