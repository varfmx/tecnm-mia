import os

import matplotlib

# En Colab o Jupyter no existe __file__; ahí las figuras también se muestran en pantalla
EN_NOTEBOOK = '__file__' not in globals()
if not EN_NOTEBOOK:
    matplotlib.use('Agg')  # Como script, genera las figuras como archivos PNG sin abrir ventanas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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
# Código base: ejemplo de Naive Bayes del documento fuente
# ===============================================================

# Generar un conjunto de datos sintético para clasificación (2 clases, 2 características)
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# Definir el modelo Naive Bayes Gaussiano
modelo = GaussianNB()

# Ajustar (entrenar) el modelo con los datos
modelo.fit(X, y)

# Seleccionar una muestra para predecir
Xmuestra, ymuestra = [X[0]], y[0]

# Realizar una predicción probabilística
probabilidades_predichas = modelo.predict_proba(Xmuestra)
print('Probabilidades predichas:', probabilidades_predichas)

# Realizar una predicción de clase
clase_predicha = modelo.predict(Xmuestra)
print('Clase predicha:', clase_predicha)
print('Clase real:', ymuestra)

# ===============================================================
# a) Preparación de datos: Iris dividido en entrenamiento (70%) y prueba (30%)
# ===============================================================

iris = load_iris()
CLASES = iris.target_names
X, y = iris.data, iris.target

# stratify conserva la misma proporción de especies en ambos conjuntos
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.3, random_state=42,
                                              stratify=y)

print("\n=== a) Preparación de datos ===")
print(f"Total: {len(X)} flores, {X.shape[1]} características")
print(f"Entrenamiento: {len(X_ent)} flores   Prueba: {len(X_pru)} flores")
print("Flores por especie en prueba:",
      {str(c): int(n) for c, n in zip(CLASES, np.bincount(y_pru))})

# ===============================================================
# b) Modelado con Naive Bayes Gaussiano
# ===============================================================

modelo = GaussianNB()
modelo.fit(X_ent, y_ent)

# Predicción probabilística: una probabilidad por especie para cada flor de prueba
probabilidades = modelo.predict_proba(X_pru)
predicciones = modelo.predict(X_pru)

resultados = pd.DataFrame(probabilidades, columns=[f'P({c})' for c in CLASES])
resultados['Real'] = CLASES[y_pru]
resultados['Predicha'] = CLASES[predicciones]
resultados['Correcta'] = predicciones == y_pru

print("\n=== b) Modelado con GaussianNB ===")
print("Media de cada característica por especie (parámetro mu de cada normal):")
print(pd.DataFrame(modelo.theta_, index=CLASES,
                   columns=['sep_len', 'sep_wid', 'pet_len', 'pet_wid']).round(2))
print("\nPrimeras 8 predicciones del conjunto de prueba:")
print(resultados.head(8).round(4).to_string())
print(f"\nExactitud en prueba: {accuracy_score(y_pru, predicciones):.4f} "
      f"({(predicciones == y_pru).sum()} de {len(y_pru)})")
print("Matriz de confusión (filas = real, columnas = predicha):")
print(pd.DataFrame(confusion_matrix(y_pru, predicciones), index=CLASES, columns=CLASES))

# La exactitud depende de qué flores caen en la prueba: partición del documento fuente
# (random_state=42, sin estratificar) y variación en 200 particiones aleatorias 70/30
X_ent_doc, X_pru_doc, y_ent_doc, y_pru_doc = train_test_split(X, y, test_size=0.3,
                                                              random_state=42)
exactitud_doc = accuracy_score(y_pru_doc, GaussianNB().fit(X_ent_doc, y_ent_doc)
                               .predict(X_pru_doc))
exactitudes_particiones = []
for semilla in range(200):
    a, b, c, d = train_test_split(X, y, test_size=0.3, random_state=semilla, stratify=y)
    exactitudes_particiones.append(accuracy_score(d, GaussianNB().fit(a, c).predict(b)))
exactitudes_particiones = np.array(exactitudes_particiones)

print(f"\nExactitud con la partición del documento fuente (sin estratificar): "
      f"{exactitud_doc:.4f}")
print(f"Exactitud en 200 particiones estratificadas: media {exactitudes_particiones.mean():.4f}, "
      f"desv. estándar {exactitudes_particiones.std(ddof=1):.4f}")
print(f"  mín. {exactitudes_particiones.min():.4f}, máx. {exactitudes_particiones.max():.4f}")

# ===============================================================
# c) Evaluación de la incertidumbre
# ===============================================================


def medidas_incertidumbre(probabilidades):
    """Confianza, margen entre las dos clases más probables y entropía de cada predicción."""
    ordenadas = np.sort(probabilidades, axis=1)
    confianza = ordenadas[:, -1]
    margen = ordenadas[:, -1] - ordenadas[:, -2]
    p = np.clip(probabilidades, 1e-12, 1)
    entropia = -(p * np.log2(p)).sum(axis=1)
    return confianza, margen, entropia


confianza, margen, entropia = medidas_incertidumbre(probabilidades)
resultados['Confianza'] = confianza
resultados['Margen'] = margen
resultados['Entropía'] = entropia

# Casos ambiguos: las dos especies más probables están cerca de 0.5 cada una
UMBRAL_MARGEN = 0.5
ambiguos = resultados[resultados['Margen'] < UMBRAL_MARGEN]

print("\n=== c) Evaluación de la incertidumbre ===")
print(f"Confianza media: {confianza.mean():.4f}   mínima: {confianza.min():.4f}")
print(f"Predicciones con confianza > 0.99: {(confianza > 0.99).sum()} de {len(confianza)}")
print(f"Casos ambiguos (margen < {UMBRAL_MARGEN}): {len(ambiguos)}")
print(ambiguos[['P(setosa)', 'P(versicolor)', 'P(virginica)', 'Real', 'Predicha', 'Margen']]
      .round(4).to_string())

# Exactitud según el nivel de confianza
grupos = pd.cut(resultados['Confianza'], bins=[0, 0.9, 0.99, 1.0],
                labels=['< 0.90', '0.90-0.99', '> 0.99'])
por_confianza = resultados.groupby(grupos, observed=False)['Correcta'].agg(['count', 'mean'])
por_confianza.columns = ['Flores', 'Exactitud']
print("\nExactitud por nivel de confianza:")
print(por_confianza.round(3))

errores = resultados[~resultados['Correcta']]
print(f"\nPredicciones incorrectas: {len(errores)}")
print(errores[['Real', 'Predicha', 'Confianza', 'Margen']].round(4).to_string())

# Histogramas de las probabilidades predichas
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, (i, clase) in zip(axes, enumerate(CLASES)):
    ax.hist(probabilidades[:, i], bins=np.linspace(0, 1, 21), color=f'C{i}', edgecolor='black')
    ax.set_title(f'P({clase})')
    ax.set_xlabel('Probabilidad predicha')
    ax.set_ylabel('Flores')
axes[3].hist(confianza, bins=np.linspace(0, 1, 21), color='gray', edgecolor='black')
axes[3].axvline(0.5, color='red', linestyle='--', label='0.5')
axes[3].set_title('Confianza (probabilidad máxima)')
axes[3].set_xlabel('Probabilidad máxima')
axes[3].legend()
guardar('c_histogramas_probabilidades.png')

# Mapa de las flores de prueba según la confianza del modelo
plt.figure(figsize=(7.5, 5))
puntos = plt.scatter(X_pru[:, 2], X_pru[:, 3], c=confianza, cmap='viridis', vmin=0.5, vmax=1,
                     s=60, edgecolor='black')
plt.scatter(X_pru[margen < UMBRAL_MARGEN, 2], X_pru[margen < UMBRAL_MARGEN, 3], s=220,
            facecolors='none', edgecolors='red', linewidth=2, label='Caso ambiguo')
plt.scatter(X_pru[predicciones != y_pru, 2], X_pru[predicciones != y_pru, 3], marker='x',
            color='red', s=80, label='Predicción incorrecta')
plt.colorbar(puntos, label='Confianza')
plt.title('Flores de prueba coloreadas por la confianza del modelo')
plt.xlabel('Longitud del pétalo (cm)')
plt.ylabel('Ancho del pétalo (cm)')
plt.legend(loc='upper left')
guardar('c_confianza_petalos.png')

# ===============================================================
# d) Limitaciones: efecto del ruido en los datos
# ===============================================================

# Se agrega ruido gaussiano a las mediciones de prueba, como errores de medición
rng = np.random.default_rng(8)
NIVELES_RUIDO = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
REPETICIONES = 50

filas = []
for sd in NIVELES_RUIDO:
    exactitudes, confianzas, confianzas_error, ambiguos_pct = [], [], [], []
    for _ in range(REPETICIONES):
        X_ruido = X_pru + rng.normal(0, sd, X_pru.shape)
        prob = modelo.predict_proba(X_ruido)
        conf, marg, _ = medidas_incertidumbre(prob)
        exactitudes.append(accuracy_score(y_pru, prob.argmax(axis=1)))
        confianzas.append(conf.mean())
        incorrectas = prob.argmax(axis=1) != y_pru
        if incorrectas.any():
            confianzas_error.append(conf[incorrectas].mean())
        ambiguos_pct.append((marg < UMBRAL_MARGEN).mean() * 100)
    filas.append({'Ruido (cm)': sd, 'Exactitud': np.mean(exactitudes),
                  'Confianza media': np.mean(confianzas),
                  'Confianza en errores': np.mean(confianzas_error),
                  'Ambiguos (%)': np.mean(ambiguos_pct)})
ruido = pd.DataFrame(filas)

print("\n=== d) Efecto del ruido en las mediciones (promedio de 50 repeticiones) ===")
print(ruido.round(3).to_string(index=False))

# El supuesto "naive": las características son independientes dentro de cada especie
correlaciones = {clase: np.corrcoef(X_ent[y_ent == i][:, 2], X_ent[y_ent == i][:, 3])[0, 1]
                 for i, clase in enumerate(CLASES)}
print("\nCorrelación entre longitud y ancho del pétalo dentro de cada especie:")
print({str(c): round(float(r), 3) for c, r in correlaciones.items()})

plt.figure(figsize=(7.5, 4.5))
plt.plot(ruido['Ruido (cm)'], ruido['Exactitud'], 'o-', label='Exactitud')
plt.plot(ruido['Ruido (cm)'], ruido['Confianza media'], 's--', label='Confianza media')
plt.plot(ruido['Ruido (cm)'], ruido['Confianza en errores'], '^:',
         label='Confianza media en las predicciones incorrectas')
plt.xlabel('Desviación estándar del ruido agregado (cm)')
plt.ylabel('Proporción')
plt.ylim(0, 1.05)
plt.legend(loc='lower left')
plt.grid(linestyle='--', alpha=0.5)
plt.title('Efecto del ruido en las predicciones de GaussianNB')
guardar('d_ruido.png')

print(f"\nFiguras guardadas en {os.path.basename(CARPETA_IMG)}/")
