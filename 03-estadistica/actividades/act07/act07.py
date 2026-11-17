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
# a) Simulación de una variable discreta: 1000 lanzamientos de un dado
# ===============================================================

N_LANZAMIENTOS = 1000
CARAS = np.arange(1, 7)
P_TEORICA = 1 / 6

np.random.seed(7)
lanzamientos = np.random.randint(1, 7, N_LANZAMIENTOS)  # el límite superior se excluye


def tabla_frecuencias(datos, valores, p_teorica):
    """Frecuencia absoluta y relativa de cada valor, comparada con la probabilidad teórica."""
    conteos = pd.Series(datos).value_counts().reindex(valores, fill_value=0)
    tabla = pd.DataFrame({'Frecuencia': conteos,
                          'Relativa': conteos / len(datos),
                          'Teórica': p_teorica})
    tabla['Diferencia'] = tabla['Relativa'] - tabla['Teórica']
    tabla.index.name = 'Cara'
    return tabla


tabla_dado = tabla_frecuencias(lanzamientos, CARAS, P_TEORICA)

# Media y varianza teóricas de un dado balanceado: E[X] = 3.5, Var[X] = 35/12
media_teorica = np.sum(CARAS * P_TEORICA)
var_teorica = np.sum((CARAS - media_teorica) ** 2 * P_TEORICA)

# Prueba chi-cuadrada de bondad de ajuste: ¿las frecuencias son compatibles con 1/6?
chi2 = stats.chisquare(tabla_dado['Frecuencia'], f_exp=[N_LANZAMIENTOS * P_TEORICA] * 6)

print("=== a) Variable discreta: 1000 lanzamientos de un dado ===")
print(tabla_dado.round(4))
print(f"\nSuma de frecuencias relativas: {tabla_dado['Relativa'].sum():.4f}")
print(f"Máxima diferencia absoluta con 1/6: {tabla_dado['Diferencia'].abs().max():.4f}")
print(f"Media muestral: {lanzamientos.mean():.4f}   media teórica: {media_teorica:.4f}")
print(f"Varianza muestral: {lanzamientos.var(ddof=1):.4f}   varianza teórica: {var_teorica:.4f}")
print(f"Prueba chi-cuadrada: estadístico = {chi2.statistic:.3f}, p = {chi2.pvalue:.3f}")

# Diagrama de barras de las frecuencias relativas
plt.figure(figsize=(7, 4.5))
barras = plt.bar(CARAS, tabla_dado['Relativa'], color='skyblue', edgecolor='black',
                 label='Frecuencia relativa observada')
plt.axhline(P_TEORICA, color='red', linestyle='--', linewidth=2,
            label=f'Probabilidad teórica (1/6 = {P_TEORICA:.4f})')
for barra, frecuencia in zip(barras, tabla_dado['Frecuencia']):
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.003,
             str(frecuencia), ha='center', fontsize=9)
plt.title('1000 lanzamientos de un dado balanceado')
plt.xlabel('Cara del dado')
plt.ylabel('Frecuencia relativa')
plt.xticks(CARAS)
plt.ylim(0, 0.22)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
guardar('a_dado_barras.png')

# ===============================================================
# b) Simulación de una variable continua: 500 muestras de N(100, 15)
# ===============================================================

MU, SIGMA, N_MUESTRAS = 100, 15, 500

np.random.seed(7)
muestras = np.random.normal(MU, SIGMA, N_MUESTRAS)

media_muestral = muestras.mean()
desv_muestral = muestras.std(ddof=1)
error_estandar = SIGMA / np.sqrt(N_MUESTRAS)

comparacion = pd.DataFrame({
    'Poblacional': [MU, SIGMA],
    'Muestral': [media_muestral, desv_muestral],
}, index=['Media', 'Desv. estándar'])
comparacion['Diferencia'] = comparacion['Muestral'] - comparacion['Poblacional']
comparacion['Error relativo (%)'] = comparacion['Diferencia'] / comparacion['Poblacional'] * 100

# Proporción de muestras a 1, 2 y 3 desviaciones estándar de la media poblacional
intervalos = pd.DataFrame({
    'Muestra (%)': [np.mean(np.abs(muestras - MU) <= k * SIGMA) * 100 for k in (1, 2, 3)],
    'Teórica (%)': [(stats.norm.cdf(k) - stats.norm.cdf(-k)) * 100 for k in (1, 2, 3)],
}, index=['[85, 115]', '[70, 130]', '[55, 145]'])

print("\n=== b) Variable continua: 500 muestras de N(100, 15) ===")
print(comparacion.round(3))
print(f"\nError estándar de la media (sigma / raíz de n): {error_estandar:.3f}")
print(f"Diferencia de la media en errores estándar: "
      f"{(media_muestral - MU) / error_estandar:.2f}")
print(f"Mínimo: {muestras.min():.2f}   Máximo: {muestras.max():.2f}")
print("\nProporción de muestras en cada intervalo:")
print(intervalos.round(1))

# Histograma de densidad con la curva teórica superpuesta
plt.figure(figsize=(8, 5))
plt.hist(muestras, bins=25, density=True, color='lightgreen', edgecolor='black',
         label='Histograma de las 500 muestras')
x = np.linspace(MU - 4 * SIGMA, MU + 4 * SIGMA, 300)
plt.plot(x, stats.norm.pdf(x, MU, SIGMA), color='red', linewidth=2,
         label='Densidad teórica N(100, 15)')
plt.axvline(MU, color='red', linestyle='--', linewidth=1, label=f'$\\mu$ = {MU}')
plt.axvline(media_muestral, color='navy', linestyle='-', linewidth=1.5,
            label=f'$\\bar{{x}}$ = {media_muestral:.2f}')
plt.title('Distribución normal: muestra vs. densidad teórica')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.legend(fontsize=8)
plt.grid(axis='y', linestyle='--', alpha=0.6)
guardar('b_normal_histograma.png')

# ===============================================================
# Extensión: convergencia al aumentar el número de observaciones
# ===============================================================

n_valores = np.arange(1, N_LANZAMIENTOS + 1)
# Frecuencia relativa acumulada de cada cara después de n lanzamientos
frecuencia_acumulada = {cara: np.cumsum(lanzamientos == cara) / n_valores for cara in CARAS}
media_acumulada = np.cumsum(muestras) / np.arange(1, N_MUESTRAS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for cara, serie in frecuencia_acumulada.items():
    ax1.plot(n_valores, serie, linewidth=1, label=f'Cara {cara}')
ax1.axhline(P_TEORICA, color='black', linestyle='--', linewidth=1.5, label='1/6')
ax1.set_xscale('log')
ax1.set_ylim(0, 0.5)
ax1.set_title('Dado: frecuencia relativa acumulada')
ax1.set_xlabel('Número de lanzamientos (escala logarítmica)')
ax1.set_ylabel('Frecuencia relativa')
ax1.legend(fontsize=7, ncol=2)
ax1.grid(linestyle='--', alpha=0.5)

n_normal = np.arange(1, N_MUESTRAS + 1)
ax2.plot(n_normal, media_acumulada, color='navy', linewidth=1.2, label='Media muestral acumulada')
ax2.axhline(MU, color='red', linestyle='--', linewidth=1.5, label=f'$\\mu$ = {MU}')
ax2.fill_between(n_normal, MU - 2 * SIGMA / np.sqrt(n_normal), MU + 2 * SIGMA / np.sqrt(n_normal),
                 color='red', alpha=0.12, label='$\\mu \\pm 2\\sigma/\\sqrt{n}$')
ax2.set_xscale('log')
ax2.set_ylim(70, 130)
ax2.set_title('Normal: media muestral acumulada')
ax2.set_xlabel('Número de muestras (escala logarítmica)')
ax2.set_ylabel('Media')
ax2.legend(fontsize=8)
ax2.grid(linestyle='--', alpha=0.5)
guardar('convergencia.png')

print("\n=== Extensión: convergencia ===")
for n in (10, 100, 1000):
    desviacion = max(abs(frecuencia_acumulada[c][n - 1] - P_TEORICA) for c in CARAS)
    print(f"Dado, n = {n:>4}: máxima |frecuencia relativa - 1/6| = {desviacion:.4f}")
for n in (10, 100, 500):
    print(f"Normal, n = {n:>3}: media acumulada = {media_acumulada[n - 1]:.2f}")

# ===============================================================
# c) Análisis comparativo: variables discretas y continuas en datos reales
# ===============================================================


def clasificar_variable(serie):
    """Clasifica una variable como discreta o continua según su tipo de dato."""
    # Texto, categorías y booleanos toman un conjunto finito de estados
    if pd.api.types.is_bool_dtype(serie) or not pd.api.types.is_numeric_dtype(serie):
        return 'Discreta (categórica)'
    if pd.api.types.is_integer_dtype(serie):
        return 'Discreta (numérica)'
    return 'Continua'


def resumen_variables(df, nombre):
    """Tabla con el tipo de dato, el número de valores distintos y la clasificación."""
    return pd.DataFrame({
        'Conjunto': nombre,
        'dtype': df.dtypes.astype(str),
        'Valores distintos': df.nunique(),
        'Tipo': [clasificar_variable(df[c]) for c in df.columns],
    })


# Weather Dataset, versión nominal (Tabla 1.1 del documento fuente)
weather_nominal = pd.DataFrame({
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast',
                'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool',
                    'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal',
                 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'windy': [False, True, False, False, False, True, True,
              False, False, False, True, True, False, True],
    'play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes',
             'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'],
})

# Weather Dataset, versión con atributos numéricos (Witten et al., 2017):
# temperatura en grados Fahrenheit y humedad relativa en porcentaje
weather_numerico = weather_nominal.copy()
weather_numerico['temperature'] = [85.0, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71]
weather_numerico['humidity'] = [85.0, 90, 86, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 91]

# Iris Dataset: cuatro medidas en centímetros y la especie
iris = load_iris(as_frame=True)
iris_df = iris.data.copy()
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

clasificacion = pd.concat([
    resumen_variables(weather_nominal, 'Weather'),
    resumen_variables(weather_numerico[['temperature', 'humidity']], 'Weather num.'),
    resumen_variables(iris_df, 'Iris'),
])

print("\n=== c) Clasificación de variables en conjuntos de datos reales ===")
print(clasificacion.to_string())

# Las variables continuas se registran con una resolución finita
print("\nResolución de medición de las variables continuas:")
for nombre, serie in [('Weather temperature (°F)', weather_numerico['temperature']),
                      ('Weather humidity (%)', weather_numerico['humidity']),
                      ('Iris sepal length (cm)', iris_df['sepal length (cm)'])]:
    paso = np.diff(np.sort(serie.unique())).min()
    print(f"  {nombre:<26} rango [{serie.min():g}, {serie.max():g}], "
          f"diferencia mínima entre valores: {paso:.1f}")

# Comparación gráfica: variable discreta (outlook) vs. continua (temperature)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
weather_nominal['outlook'].value_counts().reindex(['sunny', 'overcast', 'rainy']).plot.bar(
    ax=ax1, color='skyblue', edgecolor='black', rot=0)
ax1.set_title('Discreta: outlook (Weather)')
ax1.set_xlabel('Valor')
ax1.set_ylabel('Frecuencia')
ax2.hist(iris_df['sepal length (cm)'], bins=15, color='lightgreen', edgecolor='black')
ax2.set_title('Continua: sepal length (Iris)')
ax2.set_xlabel('Longitud del sépalo (cm)')
ax2.set_ylabel('Frecuencia')
guardar('c_discreta_continua.png')

print(f"\nFiguras guardadas en {os.path.basename(CARPETA_IMG)}/")
