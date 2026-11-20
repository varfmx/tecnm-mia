# =============================================================================
# Actividad 11: Explorando distribuciones continuas
# Tema 2.2: Normal, exponencial y gamma
# Ejecutar: python act11.py   (las figuras se guardan en img/)
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


# ---- Tarea 1: Simulación con numpy -------------------------------------------
# En el documento fuente, la exponencial y la gamma usan lambda como TASA (la media de la
# exponencial es 1/lambda). numpy y scipy piden la ESCALA, que es 1 / lambda.
N_SIM = 10000
SEMILLA = 11

# Cada caso: (distribución, parámetros). Se varía un solo parámetro a la vez.
CASOS = {
    'Normal mu=-2, s=1':   ('Normal', dict(mu=-2, sigma=1)),
    'Normal mu=0, s=1':    ('Normal', dict(mu=0, sigma=1)),
    'Normal mu=2, s=1':    ('Normal', dict(mu=2, sigma=1)),
    'Normal mu=0, s=0.5':  ('Normal', dict(mu=0, sigma=0.5)),
    'Normal mu=0, s=2':    ('Normal', dict(mu=0, sigma=2)),
    'Exponencial l=0.5':   ('Exponencial', dict(lam=0.5)),
    'Exponencial l=1':     ('Exponencial', dict(lam=1)),
    'Exponencial l=2':     ('Exponencial', dict(lam=2)),
    'Gamma a=0.5, l=1':    ('Gamma', dict(alpha=0.5, lam=1)),
    'Gamma a=1, l=1':      ('Gamma', dict(alpha=1, lam=1)),
    'Gamma a=2, l=1':      ('Gamma', dict(alpha=2, lam=1)),
    'Gamma a=5, l=1':      ('Gamma', dict(alpha=5, lam=1)),
    'Gamma a=10, l=1':     ('Gamma', dict(alpha=10, lam=1)),
    'Gamma a=3, l=0.5':    ('Gamma', dict(alpha=3, lam=0.5)),
    'Gamma a=3, l=1':      ('Gamma', dict(alpha=3, lam=1)),
    'Gamma a=3, l=2':      ('Gamma', dict(alpha=3, lam=2)),
}


def teorica(dist, par):
    """Distribución teórica de scipy con los parámetros del documento fuente."""
    if dist == 'Normal':
        return stats.norm(par['mu'], par['sigma'])
    if dist == 'Exponencial':
        return stats.expon(scale=1 / par['lam'])
    if dist == 'Gamma':
        return stats.gamma(par['alpha'], scale=1 / par['lam'])


def simular(dist, par, tam, rng):
    """Genera `tam` valores con numpy (la escala es 1 / lambda)."""
    if dist == 'Normal':
        return rng.normal(par['mu'], par['sigma'], tam)
    if dist == 'Exponencial':
        return rng.exponential(1 / par['lam'], tam)
    if dist == 'Gamma':
        return rng.gamma(par['alpha'], 1 / par['lam'], tam)


rng = np.random.default_rng(SEMILLA)
muestras = {nombre: simular(d, p, N_SIM, rng) for nombre, (d, p) in CASOS.items()}

# Simulación "desde cero" con random.random(), como en el documento fuente
random.seed(SEMILLA)


def exponencial_manual(lam):
    # Transformada inversa: si U es uniforme(0, 1), -ln(1 - U) / lambda es exponencial
    return -math.log(1 - random.random()) / lam


def gamma_manual(alpha, lam):
    # Suma de alpha exponenciales independientes (alpha entero)
    return sum(exponencial_manual(lam) for _ in range(alpha))


manual_exp = np.array([exponencial_manual(2) for _ in range(N_SIM)])
manual_gamma = np.array([gamma_manual(3, 2) for _ in range(N_SIM)])

print(f"=== Tarea 1: {len(CASOS)} muestras de {N_SIM} valores (semilla {SEMILLA}) ===")
resumen = pd.DataFrame({nombre: pd.Series(m).describe() for nombre, m in muestras.items()}).T
print(resumen[['mean', 'std', 'min', '50%', 'max']].round(3).to_string())
print("\nSimulación desde cero con random.random() (transformada inversa y suma de exponenciales):")
print(f"  Exponencial (lambda = 2):          media = {manual_exp.mean():.3f}  (teórica 1/lambda = 0.500)")
print(f"  Gamma (alpha = 3, lambda = 2):     media = {manual_gamma.mean():.3f}  (teórica alpha/lambda = 1.500)")


# ---- Tarea 2: Visualización (gráficos de densidad) ---------------------------
COLORES = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']


def graficar(ax, nombres, titulo, x, etiquetas):
    """Histograma de densidad de cada muestra con su densidad teórica superpuesta."""
    for nombre, etiqueta, color in zip(nombres, etiquetas, COLORES):
        dist, par = CASOS[nombre]
        ax.hist(muestras[nombre], bins=60, density=True, alpha=0.25, color=color)
        ax.plot(x, teorica(dist, par).pdf(x), color=color, linewidth=2, label=etiqueta)
    ax.set_title(titulo)
    ax.set_xlabel('x')
    ax.set_ylabel('Densidad')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.5)


# Normal: cambio de la media y cambio de la desviación estándar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.linspace(-7, 7, 600)
graficar(ax1, ['Normal mu=-2, s=1', 'Normal mu=0, s=1', 'Normal mu=2, s=1'],
         'Normal: cambio de la media ($\\sigma$ = 1)', x,
         ['$\\mu$ = -2', '$\\mu$ = 0', '$\\mu$ = 2'])
graficar(ax2, ['Normal mu=0, s=0.5', 'Normal mu=0, s=1', 'Normal mu=0, s=2'],
         'Normal: cambio de la desviación estándar ($\\mu$ = 0)', x,
         ['$\\sigma$ = 0.5', '$\\sigma$ = 1', '$\\sigma$ = 2'])
ax2.set_ylim(0, 0.85)
guardar('normal_parametros.png')

# Exponencial: densidad y función de distribución acumulada (teórica y empírica)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.linspace(0, 8, 600)
nombres_exp = ['Exponencial l=0.5', 'Exponencial l=1', 'Exponencial l=2']
graficar(ax1, nombres_exp, 'Exponencial: densidad', x,
         ['$\\lambda$ = 0.5', '$\\lambda$ = 1', '$\\lambda$ = 2'])
ax1.set_xlim(0, 8)
ax1.set_ylim(0, 2.1)
for nombre, color, lam in zip(nombres_exp, COLORES, [0.5, 1, 2]):
    m = np.sort(muestras[nombre])
    ax2.step(m, np.arange(1, N_SIM + 1) / N_SIM, color=color, alpha=0.5, linewidth=3)
    ax2.plot(x, 1 - np.exp(-lam * x), '--', color=color, linewidth=1.5,
             label=f'$\\lambda$ = {lam}: $1 - e^{{-\\lambda x}}$')
ax2.set_xlim(0, 8)
ax2.set_title('Exponencial: función acumulada (línea gruesa: empírica)')
ax2.set_xlabel('x')
ax2.set_ylabel('P(X $\\leq$ x)')
ax2.legend(fontsize=8)
ax2.grid(linestyle='--', alpha=0.5)
guardar('exponencial.png')

# Gamma: efecto del parámetro de forma alpha y de la tasa lambda
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.linspace(0.01, 25, 800)
graficar(ax1, ['Gamma a=0.5, l=1', 'Gamma a=1, l=1', 'Gamma a=2, l=1', 'Gamma a=5, l=1',
               'Gamma a=10, l=1'],
         'Gamma: cambio del parámetro de forma ($\\lambda$ = 1)', x,
         ['$\\alpha$ = 0.5', '$\\alpha$ = 1', '$\\alpha$ = 2', '$\\alpha$ = 5', '$\\alpha$ = 10'])
ax1.set_xlim(0, 25)
ax1.set_ylim(0, 1.0)
x = np.linspace(0.01, 14, 600)
graficar(ax2, ['Gamma a=3, l=0.5', 'Gamma a=3, l=1', 'Gamma a=3, l=2'],
         'Gamma: cambio de la tasa ($\\alpha$ = 3)', x,
         ['$\\lambda$ = 0.5', '$\\lambda$ = 1', '$\\lambda$ = 2'])
ax2.set_xlim(0, 14)
guardar('gamma.png')


# ---- Tarea 3: Análisis comparativo -------------------------------------------
# a) Media, desviación estándar, asimetría y curtosis: teóricas vs de la muestra
filas = []
for nombre, (dist, par) in CASOS.items():
    m = muestras[nombre]
    media, var, asim, curt = teorica(dist, par).stats(moments='mvsk')
    filas.append({'Caso': nombre, 'Media t.': float(media), 'Media m.': m.mean(),
                  'Desv. t.': math.sqrt(var), 'Desv. m.': m.std(ddof=1),
                  'Asim. t.': float(asim), 'Asim. m.': stats.skew(m),
                  'Curt. t.': float(curt), 'Curt. m.': stats.kurtosis(m)})
tabla_momentos = pd.DataFrame(filas)
print("\n=== Tarea 3a: momentos teóricos (t.) y de la muestra (m.); curtosis en exceso ===")
print(tabla_momentos.round(3).to_string(index=False))

# b) Probabilidades de intervalos: integral de la densidad (cdf) vs proporción de la muestra
intervalos = [('Normal mu=0, s=1', -1.96, 1.96), ('Normal mu=0, s=2', -2, 2),
              ('Exponencial l=0.5', 0, 2), ('Exponencial l=2', 0, 0.5),
              ('Gamma a=3, l=1', 2, 4), ('Gamma a=10, l=1', 8, 12)]
filas = []
for nombre, a, b in intervalos:
    dist, par = CASOS[nombre]
    p_teo = teorica(dist, par).cdf(b) - teorica(dist, par).cdf(a)
    p_emp = np.mean((muestras[nombre] > a) & (muestras[nombre] <= b))
    filas.append({'Caso': nombre, 'Intervalo': f'({a:g}, {b:g}]', 'P teórica': p_teo,
                  'Proporción simulada': p_emp, 'Diferencia': p_emp - p_teo,
                  'Error estándar': math.sqrt(p_teo * (1 - p_teo) / N_SIM)})
tabla_prob = pd.DataFrame(filas)
print("\n=== Tarea 3b: P(a < X <= b), teórica y simulada ===")
print(tabla_prob.round(4).to_string(index=False))

# c) Prueba de Kolmogorov-Smirnov: ¿la muestra viene de la distribución teórica?
filas = []
for nombre, (dist, par) in CASOS.items():
    ks = stats.kstest(muestras[nombre], teorica(dist, par).cdf)
    filas.append({'Caso': nombre, 'D': ks.statistic, 'p-valor': ks.pvalue})
tabla_ks = pd.DataFrame(filas)
print("\n=== Tarea 3c: prueba de Kolmogorov-Smirnov contra la distribución teórica ===")
print(tabla_ks.round(4).to_string(index=False))
print(f"Rechazos al nivel 0.05: {(tabla_ks['p-valor'] < 0.05).sum()} de {len(tabla_ks)}")

# d) Relaciones entre distribuciones
print("\n=== Tarea 3d: relaciones entre las distribuciones ===")
# La gamma con alpha = 1 es la exponencial
ks_1 = stats.kstest(muestras['Gamma a=1, l=1'], stats.expon(scale=1).cdf)
print(f"Gamma(alpha=1, lambda=1) contra Exponencial(1): D = {ks_1.statistic:.4f}, p = {ks_1.pvalue:.4f}")
# Los simuladores del documento fuente producen las distribuciones correctas
ks_e = stats.kstest(manual_exp, stats.expon(scale=1 / 2).cdf)
ks_g = stats.kstest(manual_gamma, stats.gamma(3, scale=1 / 2).cdf)
print(f"Exponencial desde cero (lambda=2):        D = {ks_e.statistic:.4f}, p = {ks_e.pvalue:.4f}")
print(f"Gamma desde cero, suma de 3 exp. (l=2):   D = {ks_g.statistic:.4f}, p = {ks_g.pvalue:.4f}")
# La mediana de la exponencial es ln(2) / lambda
for lam in [0.5, 1, 2]:
    mediana_m = np.median(muestras[f'Exponencial l={lam:g}'])
    print(f"Mediana exponencial lambda={lam:g}: teórica ln2/lambda = {math.log(2) / lam:.3f}, "
          f"muestra = {mediana_m:.3f}")
# Falta de memoria: P(X > s + t | X > s) = P(X > t)
lam, s, t = 1, 1.0, 1.0
m = muestras['Exponencial l=1']
condicional = np.mean(m[m > s] > s + t)
print(f"Falta de memoria (lambda=1, s=1, t=1): P(X > 2 | X > 1) = {condicional:.4f}, "
      f"P(X > 1) = {np.mean(m > t):.4f}, teórica e^(-1) = {math.exp(-lam * t):.4f}")

# La asimetría de la gamma baja como 2 / raíz(alpha): para alpha grande se parece a una normal
alfas = [1, 5, 20, 100]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
z = np.linspace(-4, 4, 400)
filas = []
for ax, alpha, color in zip(axes, alfas, COLORES):
    muestra = rng.gamma(alpha, 1, N_SIM)
    estandar = (muestra - alpha) / math.sqrt(alpha)  # media y varianza de la gamma(alpha, 1): alpha
    ax.hist(estandar, bins=50, density=True, alpha=0.4, color=color)
    ax.plot(z, stats.norm.pdf(z), 'k--', linewidth=1.5, label='Normal estándar')
    ax.set_title(f'Gamma $\\alpha$ = {alpha}, estandarizada')
    ax.set_xlabel('z')
    ax.grid(linestyle='--', alpha=0.5)
    filas.append({'alpha': alpha, 'Asimetría teórica 2/raíz(alpha)': 2 / math.sqrt(alpha),
                  'Asimetría muestral': stats.skew(muestra)})
axes[0].set_ylabel('Densidad')
axes[0].legend(fontsize=8)
guardar('gamma_hacia_normal.png')
print("\nAsimetría de la gamma al aumentar alpha:")
print(pd.DataFrame(filas).round(3).to_string(index=False))


# ---- Tarea 4: Aplicación práctica --------------------------------------------
# Tres situaciones reales, cada una con una distribución distinta (valores supuestos)
rng_app = np.random.default_rng(SEMILLA)
N_APP = 100000
print("\n=== Tarea 4: aplicaciones prácticas (100000 simulaciones por situación) ===")

# 1) Normal: estatura de adultos (media 170 cm, desviación 7 cm)
estaturas = rng_app.normal(170, 7, N_APP)
est = stats.norm(170, 7)
print("1) Estatura ~ Normal(170, 7):")
print(f"   P(165 < X < 180): teórica = {est.cdf(180) - est.cdf(165):.4f}, "
      f"simulada = {np.mean((estaturas > 165) & (estaturas < 180)):.4f}")
print(f"   Estatura que supera el 95 % de la población: teórica = {est.ppf(0.95):.2f} cm, "
      f"simulada = {np.percentile(estaturas, 95):.2f} cm")

# 2) Exponencial: tiempo entre llegadas de clientes (3 por hora, lambda = 3 por hora)
esperas = rng_app.exponential(1 / 3, N_APP)  # horas
print("2) Tiempo entre llegadas ~ Exponencial(lambda = 3 por hora):")
print(f"   Tiempo medio: teórico = {60 / 3:.1f} min, simulado = {60 * esperas.mean():.2f} min")
print(f"   P(esperar más de 30 min): teórica = {math.exp(-3 * 0.5):.4f}, "
      f"simulada = {np.mean(esperas > 0.5):.4f}")

# 3) Gamma: tiempo hasta atender al tercer cliente (alpha = 3, lambda = 3 por hora)
tiempos3 = rng_app.gamma(3, 1 / 3, N_APP)  # horas
g3 = stats.gamma(3, scale=1 / 3)
print("3) Tiempo hasta la llegada del 3.er cliente ~ Gamma(alpha = 3, lambda = 3 por hora):")
print(f"   Tiempo medio: teórico = {60 * 3 / 3:.1f} min, simulado = {60 * tiempos3.mean():.2f} min")
print(f"   P(tardar menos de 1 hora): teórica = {g3.cdf(1):.4f}, "
      f"simulada = {np.mean(tiempos3 < 1):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, datos, dist, titulo, xlabel, color in zip(
        axes, [estaturas, 60 * esperas, 60 * tiempos3],
        [est, stats.expon(scale=20), stats.gamma(3, scale=20)],
        ['Estatura de adultos: Normal(170, 7)', 'Tiempo entre llegadas: Exponencial($\\lambda$ = 3/h)',
         'Tiempo hasta el 3.er cliente: Gamma(3, $\\lambda$ = 3/h)'],
        ['Estatura (cm)', 'Minutos', 'Minutos'], COLORES):
    ax.hist(datos, bins=60, density=True, alpha=0.4, color=color)
    xs = np.linspace(datos.min(), datos.max(), 500)
    ax.plot(xs, dist.pdf(xs), color='black', linewidth=1.8, label='Densidad teórica')
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Densidad')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.5)
guardar('aplicaciones.png')

print(f"\nFiguras guardadas en {os.path.basename(CARPETA_IMG)}/")
