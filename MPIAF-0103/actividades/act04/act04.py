import math
import os
import statistics
from collections import Counter

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
# Ejercicio 1: cálculo de la media
# ===============================================================

data = [12, 15, 10, 18, 20]

# Calcular la media
media = sum(data) / len(data)

print("=== Ejercicio 1: media ===")
print(f"Datos: {data}")
print(f"Media con la fórmula (suma / n): {sum(data)} / {len(data)} = {media}")
print(f"Media con np.mean(): {np.mean(data)}")

# ===============================================================
# Ejercicio 2: cálculo de la mediana
# ===============================================================

data = [7, 10, 8, 12, 15]

# Calcular la mediana: ordenar y tomar el valor central (n es impar)
ordenados = sorted(data)
mediana = ordenados[len(ordenados) // 2]

print("\n=== Ejercicio 2: mediana ===")
print(f"Datos: {data}")
print(f"Datos ordenados: {ordenados}")
print(f"Mediana (valor central): {mediana}")
print(f"Mediana con np.median(): {np.median(data)}")

# ===============================================================
# Ejercicio 3: cálculo de la moda
# ===============================================================

data = [5, 7, 5, 10, 5]

# Calcular la moda: multimode devuelve todas las modas si hay más de una
frecuencias = Counter(data)
modas = statistics.multimode(data)

print("\n=== Ejercicio 3: moda ===")
print(f"Datos: {data}")
print(f"Frecuencias: {dict(frecuencias)}")
print(f"Moda(s): {modas}, aparece {frecuencias[modas[0]]} veces")
moda_scipy = stats.mode(data)
print(f"Moda con stats.mode(): {moda_scipy.mode}, frecuencia {moda_scipy.count}")

# ===============================================================
# Ejercicio 4: cálculo de la varianza
# ===============================================================

data = [5, 7, 9, 6, 10]

# Calcular la varianza
media = sum(data) / len(data)
suma_cuadrados = sum((x - media) ** 2 for x in data)
varianza_muestral = suma_cuadrados / (len(data) - 1)
varianza_poblacional = suma_cuadrados / len(data)

print("\n=== Ejercicio 4: varianza ===")
print(f"Datos: {data}")
print(f"Media: {media}")
print(f"Desviaciones al cuadrado: {[round((x - media) ** 2, 2) for x in data]}")
print(f"Suma de cuadrados: {suma_cuadrados:.2f}")
print(f"Varianza muestral (n - 1): {varianza_muestral:.2f}")
print(f"Varianza poblacional (n):  {varianza_poblacional:.2f}")
print(f"np.var(data, ddof=1): {np.var(data, ddof=1):.2f}   np.var(data): {np.var(data):.2f}")

# ===============================================================
# Ejercicio 5: cálculo de la desviación estándar
# ===============================================================

data = [15, 17, 19, 13, 20]

# Calcular la desviación estándar (raíz cuadrada de la varianza)
media = sum(data) / len(data)
varianza = sum((x - media) ** 2 for x in data) / (len(data) - 1)
desviacion_estandar = math.sqrt(varianza)

print("\n=== Ejercicio 5: desviación estándar ===")
print(f"Datos: {data}")
print(f"Media: {media}, varianza muestral: {varianza:.2f}")
print(f"Desviación estándar muestral (n - 1): {desviacion_estandar:.4f}")
print(f"Desviación estándar poblacional (n):  {np.std(data):.4f}")

# ===============================================================
# Ejercicio 6: cálculo de percentiles
# ===============================================================

data = [8, 14, 10, 12, 18]

# Calcular percentiles
percentiles = np.percentile(data, [25, 50, 75])

print("\n=== Ejercicio 6: percentiles ===")
print(f"Datos: {data}, ordenados: {sorted(data)}")
print(f"Percentiles 25, 50, 75: {percentiles}")
print(f"Rango intercuartílico (Q3 - Q1): {percentiles[2] - percentiles[0]}")

# ===============================================================
# Ejercicio 7: histograma de la longitud del sépalo (Iris)
# ===============================================================

# Cargar el conjunto de datos Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
sepalo = df['sepal length (cm)']

# Código base: histograma para la longitud del sépalo
plt.figure(figsize=(7, 4.5))
plt.hist(sepalo, bins=10, alpha=0.5, color='blue', edgecolor='black')
# Extensión: media y mediana
plt.axvline(sepalo.mean(), color='red', linestyle='--', label=f'Media: {sepalo.mean():.2f}')
plt.axvline(sepalo.median(), color='green', label=f'Mediana: {sepalo.median():.2f}')
plt.title("Histograma de Longitud de Sépalo")
plt.xlabel("Longitud de Sépalo (cm)")
plt.ylabel("Frecuencia")
plt.legend()
guardar('ej7_histograma_iris.png')

# Extensión: histograma por especie para ver de dónde viene la forma general
plt.figure(figsize=(7, 4.5))
for especie, color in zip(iris.target_names, ['tab:blue', 'tab:orange', 'tab:green']):
    plt.hist(sepalo[df['species'] == especie], bins=np.arange(4.2, 8.2, 0.2),
             alpha=0.6, color=color, edgecolor='black', label=especie)
plt.title("Longitud de Sépalo por especie")
plt.xlabel("Longitud de Sépalo (cm)")
plt.ylabel("Frecuencia")
plt.legend()
guardar('ej7_histograma_especies.png')

print("\n=== Ejercicio 7: longitud del sépalo (Iris) ===")
print(sepalo.describe().round(2).to_string())
print(f"Moda(s): {statistics.multimode(sepalo)}")
print(f"Asimetría: {stats.skew(sepalo):.3f}")

# ===============================================================
# Ejercicio 8: boxplot de la longitud del sépalo por especie (Iris)
# ===============================================================

# Código base: gráfico de cajas
plt.figure(figsize=(7, 4.5))
sns.boxplot(x=iris.target_names[iris.target], y=df['sepal length (cm)'])
plt.title("Gráfico de Cajas - Longitud del Sépalo")
guardar('ej8_boxplot_iris.png')

# Extensión: los cuartiles que dibuja cada caja
por_especie = sepalo.groupby(df['species']).describe()
por_especie['RIC'] = por_especie['75%'] - por_especie['25%']

print("\n=== Ejercicio 8: longitud del sépalo por especie ===")
print(por_especie[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'RIC']].round(2))

# ===============================================================
# Caso de estudio: Titanic
# ===============================================================

# Paso 1: cargar y examinar los datos
titanic = sns.load_dataset('titanic')

print("\n=== Caso Titanic - Paso 1: datos ===")
print(titanic.head())
print(f"\nDimensiones: {titanic.shape}")
print("Valores faltantes por columna:")
print(titanic.isna().sum()[lambda s: s > 0].to_string())

edad = titanic['age']  # pandas ignora los 177 valores faltantes en los cálculos

# Paso 2: medidas de tendencia central
print("\n=== Caso Titanic - Paso 2: tendencia central ===")
print(f"Media de la edad: {edad.mean():.2f} años")
print(f"Mediana de la edad: {edad.median():.2f} años")
print(f"Moda de la clase (pclass): {titanic['pclass'].mode()[0]}")
print(titanic['pclass'].value_counts().sort_index().to_string())

# Paso 3: medidas de dispersión (pandas usa n - 1 por omisión)
print("\n=== Caso Titanic - Paso 3: dispersión ===")
print(f"Varianza de la edad: {edad.var():.2f} años^2")
print(f"Desviación estándar de la edad: {edad.std():.2f} años")

# Paso 4: percentiles
print("\n=== Caso Titanic - Paso 4: percentiles ===")
print(edad.quantile([0.25, 0.50, 0.75]).to_string())

# Paso 5: histograma de la edad y boxplot por supervivencia
plt.figure(figsize=(7, 4.5))
plt.hist(edad.dropna(), bins=30, color='steelblue', edgecolor='black')
plt.axvline(edad.mean(), color='red', linestyle='--', label=f'Media: {edad.mean():.1f}')
plt.axvline(edad.median(), color='green', label=f'Mediana: {edad.median():.1f}')
plt.title('Distribución de la edad de los pasajeros')
plt.xlabel('Edad (años)')
plt.ylabel('Número de pasajeros')
plt.legend()
guardar('titanic_histograma_edad.png')

plt.figure(figsize=(7, 4.5))
sns.boxplot(data=titanic, x='survived', y='age', hue='survived',
            palette=['salmon', 'mediumseagreen'], legend=False)
plt.xticks([0, 1], ['No sobrevivió', 'Sobrevivió'])
plt.title('Edad de los pasajeros según supervivencia')
plt.xlabel('')
plt.ylabel('Edad (años)')
guardar('titanic_boxplot_edad.png')

print("\n=== Caso Titanic - Paso 5: edad por supervivencia ===")
print(titanic.groupby('survived')['age'].describe().round(2))

# ===============================================================
# Ejercicio final: análisis completo del Titanic
# ===============================================================

variables = {'age': 'Edad', 'fare': 'Tarifa', 'sibsp': 'Hermanos/cónyuges'}


def medidas(serie):
    """Medidas de tendencia central y dispersión de una variable."""
    return {'Media': serie.mean(), 'Mediana': serie.median(), 'Moda': serie.mode()[0],
            'Varianza': serie.var(), 'Desv. est.': serie.std(), 'n': serie.count()}


print("\n=== Ejercicio final: medidas de todos los pasajeros ===")
print(pd.DataFrame({nombre: medidas(titanic[col]) for col, nombre in variables.items()})
      .round(2))

print("\n=== Ejercicio final: medidas por supervivencia ===")
for col, nombre in variables.items():
    tabla = pd.DataFrame({etiqueta: medidas(grupo[col]) for etiqueta, grupo in
                          titanic.groupby(titanic['survived'].map({0: 'No sobrevivió',
                                                                   1: 'Sobrevivió'}))})
    print(f"\n{nombre}:")
    print(tabla.round(2))

# Tasa de supervivencia por sexo y por clase
print("\n=== Ejercicio final: tasa de supervivencia por grupo ===")
print((titanic.groupby('sex')['survived'].mean() * 100).round(1).to_string())
print((titanic.groupby('pclass')['survived'].mean() * 100).round(1).to_string())

# Prueba de Mann-Whitney: ¿las distribuciones difieren entre sobrevivientes y no?
print("\n=== Ejercicio final: prueba de Mann-Whitney (sobrevivió vs. no) ===")
for col, nombre in variables.items():
    sobrevivio = titanic.loc[titanic['survived'] == 1, col].dropna()
    no_sobrevivio = titanic.loc[titanic['survived'] == 0, col].dropna()
    resultado = stats.mannwhitneyu(sobrevivio, no_sobrevivio)
    print(f"{nombre:<18} U = {resultado.statistic:>9.1f}   p = {resultado.pvalue:.2g}")

colores = {0: 'salmon', 1: 'mediumseagreen'}
etiquetas = {0: 'No sobrevivió', 1: 'Sobrevivió'}

# Histogramas de las tres variables por supervivencia
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
rangos = {'age': np.arange(0, 85, 5), 'fare': np.arange(0, 525, 25), 'sibsp': np.arange(-0.5, 9)}
for ax, (col, nombre) in zip(axes, variables.items()):
    for s in (0, 1):
        ax.hist(titanic.loc[titanic['survived'] == s, col].dropna(), bins=rangos[col],
                alpha=0.6, color=colores[s], edgecolor='black', label=etiquetas[s])
    ax.set_title(nombre)
    ax.set_ylabel('Número de pasajeros')
axes[0].legend()
guardar('final_histogramas.png')

# Diagramas de caja de las tres variables por supervivencia
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (col, nombre) in zip(axes, variables.items()):
    sns.boxplot(data=titanic, x='survived', y=col, hue='survived', ax=ax,
                palette=list(colores.values()), legend=False)
    ax.set_xticks([0, 1], list(etiquetas.values()))
    ax.set_xlabel('')
    ax.set_ylabel(nombre)
guardar('final_boxplots.png')

# Diagrama de dispersión de edad vs. tarifa por supervivencia
plt.figure(figsize=(8, 5))
for s in (0, 1):
    grupo = titanic[titanic['survived'] == s]
    plt.scatter(grupo['age'], grupo['fare'], color=colores[s], alpha=0.6,
                edgecolor='gray', linewidth=0.3, label=etiquetas[s])
plt.yscale('symlog', linthresh=10)  # La tarifa es muy asimétrica
plt.title('Edad vs. tarifa según supervivencia')
plt.xlabel('Edad (años)')
plt.ylabel('Tarifa (escala logarítmica)')
plt.legend()
guardar('final_dispersion.png')

# Matriz de correlación de las variables numéricas
numericas = titanic[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']]
correlacion = numericas.corr()
plt.figure(figsize=(7, 5.5))
sns.heatmap(correlacion, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            square=True)
plt.title('Matriz de correlación')
guardar('final_correlacion.png')

# Tasa de supervivencia por sexo y clase
plt.figure(figsize=(7, 4.5))
sns.barplot(data=titanic, x='pclass', y='survived', hue='sex', errorbar=None,
            palette=['steelblue', 'orchid'])
plt.title('Tasa de supervivencia por clase y sexo')
plt.xlabel('Clase')
plt.ylabel('Proporción que sobrevivió')
guardar('final_supervivencia.png')

print("\n=== Ejercicio final: correlación con survived ===")
print(correlacion['survived'].drop('survived').round(3).to_string())

print(f"\nFiguras guardadas en {os.path.basename(CARPETA_IMG)}/")
