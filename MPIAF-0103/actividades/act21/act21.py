# =============================================================================
# Actividad 21: Modelos de clasificación
# Tema 3.3: métodos de partición (hold-out, 10-fold, leave-one-out) y métricas de evaluación
# Ejecutar: python act21.py   (figuras en img/; tablas y reporte automático en resultados/)
# En Colab no hace falta instalar nada: scikit-learn, pandas y matplotlib ya vienen incluidos.
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
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import (LeaveOneOut, StratifiedKFold, cross_val_predict,
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


def metricas(y_real, y_pred, y_prob):
    """Métricas del tema 3.3 a partir de la matriz de confusión (la clase positiva es 1)."""
    vn, fp, fn, vp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    precision = vp / (vp + fp)
    recall = vp / (vp + fn)
    especificidad = vn / (vn + fp)
    return {'VP': vp, 'FN': fn, 'FP': fp, 'VN': vn,
            'exactitud': (vp + vn) / (vp + vn + fp + fn),
            'precisión': precision, 'sensibilidad': recall, 'especificidad': especificidad,
            'F1': 2 * precision * recall / (precision + recall),
            'exactitud balanceada': (recall + especificidad) / 2,
            'AUC-ROC': roc_auc_score(y_real, y_prob)}


def a_markdown(tabla, indice=True):
    """Convierte un DataFrame en una tabla Markdown (sin depender del paquete tabulate)."""
    tabla = tabla.reset_index() if indice else tabla
    lineas = ['| ' + ' | '.join(str(c) for c in tabla.columns) + ' |',
              '|' + '---|' * len(tabla.columns)]
    for _, fila in tabla.iterrows():
        lineas.append('| ' + ' | '.join(str(v) for v in fila) + ' |')
    return '\n'.join(lineas)


def nuevo_modelo():
    """Clasificador: regresión logística con variables estandarizadas."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


# ---- Datos: selección del conjunto de datos ----------------------------------
# Breast Cancer Wisconsin (Diagnostic), del repositorio UCI (copia incluida en scikit-learn).
# 569 pacientes, 30 variables numéricas medidas en imágenes de células. Clase positiva = maligno.
cancer = load_breast_cancer(as_frame=True)
X = cancer.data
y = (cancer.target == 0).astype(int)  # en scikit-learn 0 = maligno; aquí 1 = maligno (clase de interés)

print("=== Datos: Breast Cancer Wisconsin (Diagnostic) ===")
print(f"Instancias: {len(X)}; variables: {X.shape[1]}; valores faltantes: {int(X.isna().sum().sum())}")
conteo = y.value_counts().rename({1: 'maligno (positivo)', 0: 'benigno (negativo)'})
print(pd.DataFrame({'pacientes': conteo, 'proporción': (conteo / len(y)).round(3)}).to_string())

# ---- Tarea 1a: método hold-out -----------------------------------------------
# Partición aleatoria estratificada: 70 % entrenamiento, 30 % prueba (Train y Test disjuntos).
print("\n=== Tarea 1a: hold-out (70 % / 30 %) ===")
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEMILLA)
modelo = nuevo_modelo().fit(X_ent, y_ent)
res_holdout = metricas(y_pru, modelo.predict(X_pru), modelo.predict_proba(X_pru)[:, 1])
print(f"Entrenamiento: {len(X_ent)} instancias; prueba: {len(X_pru)} instancias")
print(pd.Series(res_holdout).round(4).to_string())

# ---- Tarea 1b: validación cruzada de 10 particiones --------------------------
# Cada instancia se usa una vez como prueba; las predicciones de los 10 folds se juntan.
print("\n=== Tarea 1b: validación cruzada de 10 particiones (estratificada) ===")
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMILLA)
prob_cv = cross_val_predict(nuevo_modelo(), X, y, cv=cv10, method='predict_proba')[:, 1]
res_cv10 = metricas(y, (prob_cv >= 0.5).astype(int), prob_cv)
print(pd.Series(res_cv10).round(4).to_string())
# Variación de la exactitud entre los 10 folds
exact_folds = [np.mean((prob_cv[idx] >= 0.5) == y.to_numpy()[idx]) for _, idx in cv10.split(X, y)]
print(f"Exactitud por fold: media = {np.mean(exact_folds):.4f}, desv. est. = {np.std(exact_folds, ddof=1):.4f}, "
      f"mín = {np.min(exact_folds):.4f}, máx = {np.max(exact_folds):.4f}")

# ---- Tarea 1c: leave-one-out -------------------------------------------------
# Un fold por instancia: se entrena con n - 1 y se prueba con la instancia restante.
print("\n=== Tarea 1c: leave-one-out ===")
prob_loo = cross_val_predict(nuevo_modelo(), X, y, cv=LeaveOneOut(), method='predict_proba')[:, 1]
res_loo = metricas(y, (prob_loo >= 0.5).astype(int), prob_loo)
print(f"Modelos entrenados: {len(X)}")
print(pd.Series(res_loo).round(4).to_string())

# ---- Tarea 1d: ¿qué método conviene según la cantidad de datos? --------------
# Se repite cada método con muchas particiones aleatorias distintas y se mide cuánto cambia
# la exactitud estimada. Una estimación estable no depende de la partición elegida.
print("\n=== Tarea 1d: variabilidad de la estimación según el tamaño del conjunto ===")
rng = np.random.default_rng(SEMILLA)
N_REP = 100
filas = []
for n in (569, 60):
    if n < len(X):  # submuestra estratificada pequeña (simula un problema con pocos datos)
        idx_peq, _ = train_test_split(np.arange(len(X)), train_size=n, stratify=y, random_state=SEMILLA)
        Xn, yn = X.iloc[idx_peq], y.iloc[idx_peq]
    else:
        Xn, yn = X, y
    est_holdout, est_cv10 = [], []
    for _ in range(N_REP):
        semilla = int(rng.integers(1_000_000))
        a, b, c, d = train_test_split(Xn, yn, test_size=0.30, stratify=yn, random_state=semilla)
        est_holdout.append(nuevo_modelo().fit(a, c).score(b, d))
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=semilla)
        pred = cross_val_predict(nuevo_modelo(), Xn, yn, cv=cv)
        est_cv10.append(np.mean(pred == yn.to_numpy()))
    pred_loo = cross_val_predict(nuevo_modelo(), Xn, yn, cv=LeaveOneOut())
    exact_loo = np.mean(pred_loo == yn.to_numpy())
    for nombre, est in (('hold-out 70/30', est_holdout), ('10-fold', est_cv10)):
        filas.append((n, nombre, np.mean(est), np.std(est, ddof=1), np.min(est), np.max(est)))
    filas.append((n, 'leave-one-out', exact_loo, 0.0, exact_loo, exact_loo))
variabilidad = pd.DataFrame(filas, columns=['n', 'método', 'exactitud media', 'desv. est.', 'mínima', 'máxima'])
print(variabilidad.round(4).to_string(index=False))

fig, ejes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for eje, n in zip(ejes, (569, 60)):
    sub = variabilidad[variabilidad['n'] == n]
    eje.errorbar(sub['método'], sub['exactitud media'], yerr=sub['desv. est.'], fmt='o', capsize=6)
    eje.set_title(f'n = {n} (media y desv. est. de {N_REP} repeticiones)')
    eje.set_ylabel('Exactitud estimada')
guardar('variabilidad_metodos.png')

# ---- Tarea 2: métricas, selección de la más adecuada y reporte automático ----
print("\n=== Tarea 2: resultados de efectividad ===")
comparacion = pd.DataFrame({'hold-out 70/30': res_holdout, '10-fold': res_cv10,
                            'leave-one-out': res_loo}).T
comparacion.index.name = 'método'
comparacion[['VP', 'FN', 'FP', 'VN']] = comparacion[['VP', 'FN', 'FP', 'VN']].astype(int)
columnas = ['exactitud', 'precisión', 'sensibilidad', 'especificidad', 'F1',
            'exactitud balanceada', 'AUC-ROC']
print(comparacion[columnas].T.round(4).to_string())
print(f"\nCosto de los errores en leave-one-out: {res_loo['FN']} falsos negativos (malignos no detectados) y "
      f"{res_loo['FP']} falsos positivos (benignos señalados como malignos)")

# Matrices de confusión de los tres métodos
fig, ejes = plt.subplots(1, 3, figsize=(12, 3.8))
for eje, (nombre, r) in zip(ejes, comparacion.iterrows()):
    matriz = np.array([[r['VP'], r['FN']], [r['FP'], r['VN']]])
    eje.imshow(matriz, cmap='Blues')
    for i in range(2):
        for j in range(2):
            eje.text(j, i, int(matriz[i, j]), ha='center', va='center', fontsize=14,
                     color='white' if matriz[i, j] > matriz.max() / 2 else 'black')
    eje.set_xticks([0, 1], ['maligno', 'benigno'])
    eje.set_yticks([0, 1], ['maligno', 'benigno'])
    eje.set_xlabel('Predicción')
    eje.set_ylabel('Clase real')
    eje.set_title(nombre)
guardar('matrices_confusion.png')

# Reporte automático: tablas en CSV y resumen en Markdown dentro de resultados/
comparacion[['VP', 'FN', 'FP', 'VN'] + columnas].round(4).to_csv(os.path.join(CARPETA_RES, 'metricas_por_metodo.csv'))
variabilidad.round(4).to_csv(os.path.join(CARPETA_RES, 'variabilidad_metodos.csv'), index=False)
metrica_principal = 'sensibilidad'
with open(os.path.join(CARPETA_RES, 'reporte_automatico.md'), 'w', encoding='utf-8') as f:
    f.write("# Reporte automático: clasificación de Breast Cancer Wisconsin (Diagnostic)\n\n")
    f.write(f"- Instancias: {len(X)}; variables: {X.shape[1]}; clase positiva: maligno "
            f"({int(y.sum())} casos, {y.mean():.1%}).\n")
    f.write("- Clasificador: regresión logística con variables estandarizadas.\n")
    f.write(f"- Métrica principal: {metrica_principal} (un falso negativo es un cáncer no detectado).\n\n")
    f.write("## Métricas por método de partición\n\n")
    f.write(a_markdown(comparacion[['VP', 'FN', 'FP', 'VN'] + columnas].round(4)) + "\n\n")
    f.write("## Variabilidad de la exactitud según el método y el tamaño del conjunto\n\n")
    f.write(a_markdown(variabilidad.round(4), indice=False) + "\n")
print("\nReporte automático guardado en la carpeta resultados/")
print("Archivos:", sorted(os.listdir(CARPETA_RES)))
