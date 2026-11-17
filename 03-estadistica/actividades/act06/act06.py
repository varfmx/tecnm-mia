import os

import matplotlib

# En Colab o Jupyter no existe __file__; ahí las figuras también se muestran en pantalla
EN_NOTEBOOK = '__file__' not in globals()
if not EN_NOTEBOOK:
    matplotlib.use('Agg')  # Como script, genera las figuras como archivos PNG sin abrir ventanas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats
from scipy import stats

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


def resumen(datos):
    """Medidas de tendencia central y dispersión de una muestra."""
    datos = pd.Series(datos).dropna()
    return {'n': len(datos), 'Media': datos.mean(), 'Mediana': datos.median(),
            'Desv. est.': datos.std(), 'Mínimo': datos.min(), 'Máximo': datos.max()}


# ===============================================================
# Ejercicio 1: distribución empírica de la altura de 200 estudiantes
# ===============================================================

# Las alturas se simulan con una distribución normal de media 165 cm y
# desviación estándar 8 cm, redondeadas a 0.1 cm como si se midieran con cinta
np.random.seed(15)
alturas = np.round(np.random.normal(165, 8, 200), 1)

# Parámetros de la normal teórica estimados a partir de la muestra
mu, sigma = alturas.mean(), alturas.std(ddof=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Histograma de densidad (área total = 1) con la curva normal teórica encima
ax1.hist(alturas, bins=15, density=True, color='skyblue', edgecolor='black',
         label='Distribución empírica')
x = np.linspace(alturas.min() - 5, alturas.max() + 5, 200)
ax1.plot(x, stats.norm.pdf(x, mu, sigma), color='red', linewidth=2,
         label=f'Normal teórica\n($\\mu$ = {mu:.1f}, $\\sigma$ = {sigma:.1f})')
ax1.set_title('Altura de 200 estudiantes')
ax1.set_xlabel('Altura (cm)')
ax1.set_ylabel('Densidad')
ax1.legend(fontsize=8)
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# Gráfico Q-Q: si los puntos siguen la recta, la muestra se parece a la normal
stats.probplot(alturas, dist='norm', plot=ax2)
ax2.get_lines()[0].set(markerfacecolor='skyblue', markeredgecolor='navy', markersize=4)
ax2.get_lines()[1].set(color='red', linewidth=2)
ax2.set_title('Gráfico Q-Q contra la distribución normal')
ax2.set_xlabel('Cuantiles teóricos')
ax2.set_ylabel('Cuantiles de la muestra (cm)')
ax2.grid(linestyle='--', alpha=0.6)
guardar('ej1_altura.png')

# Proporción de datos a 1, 2 y 3 desviaciones estándar de la media
regla = pd.DataFrame({
    'Muestra (%)': [np.mean(np.abs(alturas - mu) <= k * sigma) * 100 for k in (1, 2, 3)],
    'Normal (%)': [(stats.norm.cdf(k) - stats.norm.cdf(-k)) * 100 for k in (1, 2, 3)],
}, index=['± 1 sigma', '± 2 sigma', '± 3 sigma']).round(1)

print("=== Ejercicio 1: altura de 200 estudiantes ===")
print(pd.Series(resumen(alturas)).round(2).to_string())
print(f"Asimetría: {stats.skew(alturas):.3f}")
print(f"Curtosis (exceso): {stats.kurtosis(alturas):.3f}")
print("\nDatos dentro de k desviaciones estándar de la media:")
print(regla)

# ===============================================================
# Ejercicio 2: sueldos de 100 empleados con 2 directivos
# ===============================================================

# 98 empleados ganan entre $15,000 y $30,000; 2 directivos ganan más de $150,000
np.random.seed(25)
sueldos_empleados = np.round(np.random.uniform(15_000, 30_000, 98), -1)
sueldos_directivos = np.array([158_000, 175_000])
sueldos = np.concatenate([sueldos_empleados, sueldos_directivos])

# Detección de valores atípicos con la regla de 1.5 veces el rango intercuartílico
caja = boxplot_stats(sueldos)[0]
atipicos = caja['fliers']

comparacion = pd.DataFrame({
    'Con directivos': resumen(sueldos),
    'Sin directivos': resumen(sueldos_empleados),
}).T
comparacion['Media recortada 5%'] = [stats.trim_mean(sueldos, 0.05),
                                     stats.trim_mean(sueldos_empleados, 0.05)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                               gridspec_kw={'width_ratios': [2, 1]})

ax1.hist(sueldos, bins=30, color='lightgreen', edgecolor='black')
ax1.axvline(sueldos.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Media: ${sueldos.mean():,.0f}')
ax1.axvline(np.median(sueldos), color='blue', linewidth=2,
            label=f'Mediana: ${np.median(sueldos):,.0f}')
ax1.set_title('Sueldos mensuales de 100 empleados')
ax1.set_xlabel('Sueldo mensual (pesos)')
ax1.set_ylabel('Número de empleados')
ax1.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
ax1.tick_params(axis='x', labelrotation=20)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.6)

ax2.boxplot(sueldos, tick_labels=['Todos'], patch_artist=True, showmeans=True,
            boxprops=dict(facecolor='lightgreen'),
            meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red'))
ax2.set_title('Diagrama de caja')
ax2.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
ax2.grid(axis='y', linestyle='--', alpha=0.6)
guardar('ej2_sueldos.png')

print("\n=== Ejercicio 2: sueldos de 100 empleados ===")
print(comparacion.round(0).astype(int).T)
print(f"\nLímite superior para atípicos (Q3 + 1.5 RIC): "
      f"${caja['q3'] + 1.5 * caja['iqr']:,.0f}")
print(f"Valores atípicos detectados: {[f'${v:,.0f}' for v in atipicos]}")
cambio_media = sueldos.mean() - sueldos_empleados.mean()
cambio_mediana = np.median(sueldos) - np.median(sueldos_empleados)
print(f"Efecto de los directivos sobre la media: +${cambio_media:,.0f} "
      f"({cambio_media / sueldos_empleados.mean():.1%})")
print(f"Efecto de los directivos sobre la mediana: +${cambio_mediana:,.0f} "
      f"({cambio_mediana / np.median(sueldos_empleados):.1%})")
print(f"Empleados que ganan menos que la media: {(sueldos < sueldos.mean()).sum()} de 100")

# ===============================================================
# Ejercicio 3: ingresos con 15% de respuestas faltantes codificadas como 0
# ===============================================================

# Ingresos reales de 200 participantes (en pesos)
np.random.seed(35)
ingresos_reales = np.round(np.random.normal(18_000, 4_000, 200), -1)

# El 15% no respondió y en la base de datos aparece como 0
encuesta = pd.DataFrame({'ingreso': ingresos_reales})
no_respondieron = np.random.choice(encuesta.index, size=int(0.15 * len(encuesta)), replace=False)
encuesta.loc[no_respondieron, 'ingreso'] = 0

# Tratamiento correcto: los ceros son datos faltantes, no ingresos de $0
encuesta['ingreso_na'] = encuesta['ingreso'].replace(0, np.nan)
print("\n=== Ejercicio 3: ingresos con respuestas faltantes ===")
print(f"Participantes: {len(encuesta)}")
print(f"Respuestas registradas como 0: {(encuesta['ingreso'] == 0).sum()}")
print(f"Datos faltantes tras convertir 0 a NaN: {encuesta['ingreso_na'].isna().sum()}")

# Imputación por la mediana de los ingresos observados
mediana_obs = encuesta['ingreso_na'].median()
encuesta['ingreso_imputado'] = encuesta['ingreso_na'].fillna(mediana_obs)

comparacion = pd.DataFrame({
    'Con ceros': resumen(encuesta['ingreso']),
    'Sin ceros (NaN)': resumen(encuesta['ingreso_na']),
    'Imputado (mediana)': resumen(encuesta['ingreso_imputado']),
    'Valores reales': resumen(ingresos_reales),
})
print()
print(comparacion.round(0).astype(int))
error = encuesta['ingreso'].mean() - ingresos_reales.mean()
print(f"\nError de la media con ceros respecto a la real: ${error:,.0f} "
      f"({error / ingresos_reales.mean():.1%})")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
bins = np.arange(0, 32_001, 2_000)
paneles = [
    (encuesta['ingreso'], 'Con ceros', 'salmon'),
    (encuesta['ingreso_na'].dropna(), 'Sin ceros (faltantes excluidos)', 'skyblue'),
    (encuesta['ingreso_imputado'], 'Imputado con la mediana', 'plum'),
]
for ax, (datos, titulo, color) in zip(axes, paneles):
    ax.hist(datos, bins=bins, color=color, edgecolor='black')
    ax.axvline(datos.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Media: ${datos.mean():,.0f}')
    ax.axvline(datos.median(), color='blue', linewidth=1.5,
               label=f'Mediana: ${datos.median():,.0f}')
    ax.set_title(titulo)
    ax.set_xlabel('Ingreso mensual (pesos)')
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    ax.tick_params(axis='x', labelrotation=30)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
axes[0].set_ylabel('Número de participantes')
guardar('ej3_ingresos.png')

print(f"\nFiguras guardadas en {os.path.basename(CARPETA_IMG)}/")
