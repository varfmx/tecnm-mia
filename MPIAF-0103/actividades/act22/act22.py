# =============================================================================
# Actividad 22: Modelos de predicción
# Tema 3.3: regresión, métricas de bondad y ajuste (MAE, MSE, RMSE, R2) y métodos de partición
# Ejecutar: python act22.py   (figuras en img/; tablas y reporte automático en resultados/)
# Los datos se descargan del repositorio UCI, así que se necesita conexión a internet.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, LeaveOneOut, PredefinedSplit, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option('display.width', 120)
SEMILLA = 42

# Carpetas donde se guardan las figuras del reporte y las tablas del reporte automático
BASE = os.getcwd() if EN_NOTEBOOK else os.path.dirname(os.path.abspath(__file__))
CARPETA_IMG = os.path.join(BASE, 'img')
CARPETA_RES = os.path.join(BASE, 'resultados')
os.makedirs(CARPETA_IMG, exist_ok=True)
os.makedirs(CARPETA_RES, exist_ok=True)


def guardar(nombre):
    """Guarda la figura actual en la carpeta img/ y la muestra si se ejecuta en un notebook."""
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_IMG, nombre), dpi=150)
    if EN_NOTEBOOK:
        plt.show()
    plt.close()


def metricas(y_real, y_pred):
    """Métricas de bondad y ajuste del tema 3.3 para un regresor."""
    mse = mean_squared_error(y_real, y_pred)
    return {'MAE': mean_absolute_error(y_real, y_pred), 'MSE': mse,
            'RMSE': np.sqrt(mse), 'R2': r2_score(y_real, y_pred)}


def a_markdown(tabla, indice=True):
    """Convierte un DataFrame en una tabla Markdown (sin depender del paquete tabulate)."""
    tabla = tabla.reset_index() if indice else tabla
    lineas = ['| ' + ' | '.join(str(c) for c in tabla.columns) + ' |',
              '|' + '---|' * len(tabla.columns)]
    for _, fila in tabla.iterrows():
        lineas.append('| ' + ' | '.join(str(v) for v in fila) + ' |')
    return '\n'.join(lineas)


def nuevos_modelos():
    """Dos regresores: uno lineal (referencia) y uno no lineal (bosque aleatorio)."""
    return {'Regresión lineal': make_pipeline(StandardScaler(), LinearRegression()),
            'Bosque aleatorio': RandomForestRegressor(n_estimators=100, random_state=SEMILLA, n_jobs=-1)}


# ---- Datos: selección del conjunto de datos ----------------------------------
# Airfoil Self-Noise (NASA), del repositorio UCI: ruido aerodinámico de perfiles alares
# medido en túnel de viento. Objetivo: presión sonora (dB) a partir de 5 variables.
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
columnas = ['frecuencia', 'angulo', 'cuerda', 'velocidad', 'espesor', 'presion_sonora']
datos = pd.read_csv(URL, sep='\t', header=None, names=columnas)
X = datos.drop(columns='presion_sonora')
y = datos['presion_sonora']

print("=== Datos: Airfoil Self-Noise (UCI) ===")
print(f"Instancias: {len(X)}; variables predictoras: {X.shape[1]}; valores faltantes: {int(datos.isna().sum().sum())}")
print(datos.describe().loc[['mean', 'std', 'min', 'max']].round(3).T.to_string())

# ---- Tarea 1a: técnica de partición más adecuada -----------------------------
# Primero se revisa la estructura de los datos: cada fila es un ensayo en el túnel de viento y
# el mismo montaje (ángulo, cuerda, velocidad, espesor) se mide a varias frecuencias.
print("\n=== Tarea 1a: estructura de los datos y técnica de partición ===")
montaje = datos.groupby(['angulo', 'cuerda', 'velocidad', 'espesor']).ngroup()
filas_por_montaje = montaje.value_counts()
print(f"Montajes distintos: {montaje.nunique()}; filas por montaje: mín = {filas_por_montaje.min()}, "
      f"máx = {filas_por_montaje.max()}, media = {filas_por_montaje.mean():.1f}")

# Se repiten hold-out (80/20) y validación cruzada de 10 particiones con semillas distintas para
# medir cuánto cambia la estimación según la partición elegida.
rng = np.random.default_rng(SEMILLA)
N_REP = 30
filas = []
for nombre, modelo in nuevos_modelos().items():
    est_hold, est_cv = [], []
    for _ in range(N_REP):
        semilla = int(rng.integers(1_000_000))
        a, b, c, d = train_test_split(X, y, test_size=0.20, random_state=semilla)
        est_hold.append(metricas(d, modelo.fit(a, c).predict(b)))
        cv = KFold(n_splits=10, shuffle=True, random_state=semilla)
        est_cv.append(metricas(y, cross_val_predict(modelo, X, y, cv=cv)))
    for metodo, est in (('hold-out 80/20', est_hold), ('10-fold', est_cv)):
        est = pd.DataFrame(est)
        filas.append((nombre, metodo, est['RMSE'].mean(), est['RMSE'].std(), est['R2'].mean(), est['R2'].std()))
variabilidad = pd.DataFrame(filas, columns=['modelo', 'método', 'RMSE media', 'RMSE desv.', 'R2 media', 'R2 desv.'])
print(f"Variabilidad en {N_REP} repeticiones con particiones distintas:")
print(variabilidad.round(4).to_string(index=False))

# ---- Tarea 1b: construcción y entrenamiento de los modelos -------------------
# Se evalúa cada modelo con cuatro particiones:
#  - hold-out 80/20 aleatorio, - 10-fold aleatorio, - leave-one-out (solo regresión lineal,
#  porque con el bosque aleatorio exigiría entrenar 1503 bosques) y - 10-fold por montajes.
print("\n=== Tarea 1b: entrenamiento y evaluación ===")
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.20, random_state=SEMILLA)
print(f"Hold-out: entrenamiento = {len(X_ent)}, prueba = {len(X_pru)}")
# 10-fold por montajes: cada montaje se asigna completo a un fold (todas sus filas juntas). La
# asignación usa una semilla propia para que no dependa de la versión de scikit-learn.
fold_de_montaje = np.random.default_rng(SEMILLA).permutation(montaje.nunique()) % 10
cv_montajes = PredefinedSplit(test_fold=fold_de_montaje[montaje.to_numpy()])
predicciones = {}   # (modelo, partición) -> predicciones sobre los datos evaluados
reales = {}         # (modelo, partición) -> valores reales correspondientes
resultados = {}
for nombre, modelo in nuevos_modelos().items():
    modelo.fit(X_ent, y_ent)
    predicciones[(nombre, 'hold-out 80/20')] = modelo.predict(X_pru)
    reales[(nombre, 'hold-out 80/20')] = y_pru
    predicciones[(nombre, '10-fold')] = cross_val_predict(
        modelo, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=SEMILLA))
    reales[(nombre, '10-fold')] = y
    predicciones[(nombre, '10-fold por montajes')] = cross_val_predict(modelo, X, y, cv=cv_montajes)
    reales[(nombre, '10-fold por montajes')] = y
predicciones[('Regresión lineal', 'leave-one-out')] = cross_val_predict(
    nuevos_modelos()['Regresión lineal'], X, y, cv=LeaveOneOut())
reales[('Regresión lineal', 'leave-one-out')] = y
for clave, pred in predicciones.items():
    resultados[clave] = metricas(reales[clave], pred)
print(f"Evaluaciones completadas: {len(resultados)}")

# ---- Tarea 2: comparación de las métricas de bondad y ajuste -----------------
print("\n=== Tarea 2: comparación de métricas ===")
tabla = pd.DataFrame(resultados).T
tabla.index.names = ['modelo', 'partición']
tabla['RMSE / desv. est. de y'] = tabla['RMSE'] / y.std()
print(tabla.round(4).to_string())

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
particiones = ['hold-out 80/20', '10-fold', 'leave-one-out', '10-fold por montajes']
for eje, metrica, titulo in zip(ejes, ['RMSE', 'R2'], ['RMSE (dB, menor es mejor)', 'R2 (mayor es mejor)']):
    ancho = 0.38
    for k, nombre in enumerate(['Regresión lineal', 'Bosque aleatorio']):
        valores = [tabla[metrica].get((nombre, p), np.nan) for p in particiones]
        eje.bar(np.arange(4) + (k - 0.5) * ancho, valores, ancho, label=nombre)
    eje.set_xticks(np.arange(4), [p.replace(' ', '\n', 1) for p in particiones], fontsize=8)
    eje.set_title(titulo)
eje.set_ylim(0, 1.2)  # espacio arriba para que la leyenda no tape las barras
eje.legend(loc='upper left', ncol=2)
guardar('comparacion_metricas.png')

# ---- Tarea 3: análisis de resultados y validez --------------------------------
# El bosque aleatorio es el mejor modelo con la partición aleatoria; se revisa si esa estimación
# se sostiene cuando las pruebas se hacen con montajes que el modelo nunca vio.
print("\n=== Tarea 3: análisis de resultados ===")
nombre = 'Bosque aleatorio'
for particion in ('10-fold', '10-fold por montajes'):
    r = resultados[(nombre, particion)]
    error = reales[(nombre, particion)] - predicciones[(nombre, particion)]
    print(f"{particion}: R2 = {r['R2']:.4f}, RMSE = {r['RMSE']:.3f} dB, "
          f"sesgo medio = {error.mean():.3f} dB, error absoluto p95 = {np.percentile(np.abs(error), 95):.2f} dB")
caida_rmse = resultados[(nombre, '10-fold por montajes')]['RMSE'] / resultados[(nombre, '10-fold')]['RMSE'] - 1
print(f"El RMSE sube {caida_rmse:.1%} al evaluar con montajes no vistos")

fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for eje, particion in zip(ejes, ('10-fold', '10-fold por montajes')):
    eje.scatter(y, predicciones[(nombre, particion)], s=6, alpha=0.5)
    eje.plot([y.min(), y.max()], [y.min(), y.max()], color='tab:red', linestyle='--')
    eje.set_title(f"{nombre}: {particion} (R2 = {resultados[(nombre, particion)]['R2']:.3f})")
    eje.set_xlabel('Presión sonora real (dB)')
ejes[0].set_ylabel('Presión sonora predicha (dB)')
guardar('real_vs_predicho.png')

# Reporte automático: tablas en CSV y resumen en Markdown dentro de resultados/
tabla.round(4).to_csv(os.path.join(CARPETA_RES, 'metricas_por_modelo_y_particion.csv'))
variabilidad.round(4).to_csv(os.path.join(CARPETA_RES, 'variabilidad_particiones.csv'), index=False)
with open(os.path.join(CARPETA_RES, 'reporte_automatico.md'), 'w', encoding='utf-8') as f:
    f.write("# Reporte automático: regresión sobre Airfoil Self-Noise (UCI)\n\n")
    f.write(f"- Instancias: {len(X)}; variables predictoras: {X.shape[1]}; objetivo: presión sonora (dB).\n")
    f.write(f"- Montajes experimentales distintos: {montaje.nunique()} (filas no independientes entre sí).\n")
    f.write("- Modelos: regresión lineal y bosque aleatorio.\n\n")
    f.write("## Métricas por modelo y método de partición\n\n")
    f.write(a_markdown(tabla.round(4)) + "\n\n")
    f.write("## Variabilidad de la estimación según la partición (repeticiones)\n\n")
    f.write(a_markdown(variabilidad.round(4), indice=False) + "\n")
print("\nReporte automático guardado en la carpeta resultados/")
print("Archivos:", sorted(os.listdir(CARPETA_RES)))
