# =============================================================================
# Actividad 20: "Diagnóstico Médico Asistido"
# Tema 4.3. Aprendizaje automático
#
# Dependencias (en Colab ya vienen instaladas):  pip install scikit-learn pandas matplotlib seaborn
# No descarga datos: usa el esqueleto del tema, que simula 1000 pacientes con make_classification.
# =============================================================================
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

SEMILLA = 42
AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
IMG = os.path.join(AQUI, "img")
os.makedirs(IMG, exist_ok=True)
sns.set_theme(style="whitegrid")
np.random.seed(SEMILLA)


# =============================================================================
# TAREA 1: EXPLORACIÓN DE LOS DATOS
# =============================================================================
def explorar_datos_salud():
    """Comienza tu análisis aquí"""
    # Cargar datos de salud (simulados para este ejercicio)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=8,
                               n_informative=5, n_classes=2,
                               random_state=42)

    feature_names = ['Edad', 'Presión Arterial', 'Colesterol', 'Glucosa',
                     'IMC', 'Fumador', 'Actividad Física', 'Historial Familiar']

    print("Análisis de Datos de Salud Cardíaca")
    print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
    print(f"Pacientes sanos: {sum(y==0)}, Pacientes con riesgo: {sum(y==1)}")

    return X, y, feature_names


print("=" * 70 + "\nTAREA 1: EXPLORACIÓN DE LOS DATOS\n" + "=" * 70)
X, y, nombres = explorar_datos_salud()
df = pd.DataFrame(X, columns=nombres)
df["Riesgo"] = y

print("\nValores faltantes por columna:", int(df.isna().sum().sum()))
print("\nEstadísticos descriptivos:")
print(df[nombres].describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_string())

print("\nMedia de cada característica por grupo:")
medias = df.groupby("Riesgo")[nombres].mean().T
medias.columns = ["Sano (0)", "Riesgo (1)"]
medias["diferencia"] = medias["Riesgo (1)"] - medias["Sano (0)"]
print(medias.round(2).to_string())

corr = df.corr()
corr_y = corr["Riesgo"].drop("Riesgo")
print("\nCorrelación de cada característica con el riesgo (ordenada por |r|):")
print(corr_y.reindex(corr_y.abs().sort_values(ascending=False).index).round(3).to_string())
pares = [(a, b, corr.loc[a, b]) for i, a in enumerate(nombres) for b in nombres[i + 1:] if abs(corr.loc[a, b]) >= 0.3]
print("\nPares de características con |r| >= 0.3:", [(a, b, round(float(r), 2)) for a, b, r in pares] or "ninguno")

# Figura: distribución por grupo y mapa de correlaciones
fig, ejes = plt.subplots(2, 4, figsize=(14, 6))
for ax, nombre in zip(ejes.ravel(), nombres):
    for clase, color, etiqueta in [(0, "#2a9d8f", "Sano"), (1, "#e76f51", "Riesgo")]:
        sns.kdeplot(df.loc[df.Riesgo == clase, nombre], ax=ax, color=color, fill=True, alpha=0.35, label=etiqueta)
    ax.set_title(f"{nombre} (r = {corr_y[nombre]:+.2f})", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
ejes[0, 0].legend()
plt.suptitle("Distribución de cada característica según el grupo", y=1.0)
plt.tight_layout()
plt.savefig(f"{IMG}/distribuciones.png", dpi=200)
plt.close()

plt.figure(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8})
plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig(f"{IMG}/correlaciones.png", dpi=200)
plt.close()


# =============================================================================
# PREPROCESAMIENTO: partición y escalado
# =============================================================================
print("\n" + "=" * 70 + "\nPREPROCESAMIENTO\n" + "=" * 70)
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.3, random_state=SEMILLA, stratify=y)
print(f"Entrenamiento: {len(y_ent)} pacientes (riesgo: {y_ent.mean():.1%})")
print(f"Prueba:        {len(y_prueba)} pacientes (riesgo: {y_prueba.mean():.1%})")
print("El escalado (StandardScaler) va dentro de un Pipeline: se ajusta sólo con los datos de entrenamiento de cada pliegue.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)


# =============================================================================
# TAREA 2: CONSTRUCCIÓN Y COMPARACIÓN DE MODELOS
# =============================================================================
print("\n" + "=" * 70 + "\nTAREA 2: MODELOS DE CLASIFICACIÓN\n" + "=" * 70)
modelos = {
    "Regresión Logística": make_pipeline(StandardScaler(), LogisticRegression(random_state=SEMILLA)),
    "Árbol de decisión": DecisionTreeClassifier(random_state=SEMILLA),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEMILLA),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEMILLA)),
    "K-Vecinos": make_pipeline(StandardScaler(), KNeighborsClassifier()),
}
metricas = {"exactitud": "accuracy", "precisión": "precision", "recall": "recall", "F1": "f1", "AUC": "roc_auc"}

filas_cv = []
for nombre, modelo in modelos.items():
    r = cross_validate(modelo, X_ent, y_ent, cv=cv, scoring=list(metricas.values()), return_train_score=True)
    fila = {"modelo": nombre}
    for corto, largo in metricas.items():
        fila[f"{corto} CV"] = r[f"test_{largo}"].mean()
    fila["F1 CV desv."] = r["test_f1"].std()
    fila["exactitud entren."] = r["train_accuracy"].mean()
    filas_cv.append(fila)
tabla_cv = pd.DataFrame(filas_cv).set_index("modelo")
print("\nValidación cruzada estratificada de 5 pliegues (sobre los 700 pacientes de entrenamiento):")
print(tabla_cv.round(3).to_string())

# Se ajusta cada modelo con todo el entrenamiento y se evalúa UNA vez en la prueba
filas_te, predicciones, probas = [], {}, {}
for nombre, modelo in modelos.items():
    modelo.fit(X_ent, y_ent)
    pred = modelo.predict(X_prueba)
    proba = modelo.predict_proba(X_prueba)[:, 1]
    predicciones[nombre], probas[nombre] = pred, proba
    filas_te.append({"modelo": nombre,
                     "exactitud": accuracy_score(y_prueba, pred),
                     "precisión": precision_score(y_prueba, pred),
                     "recall": recall_score(y_prueba, pred),
                     "F1": f1_score(y_prueba, pred),
                     "AUC": roc_auc_score(y_prueba, proba)})
tabla_te = pd.DataFrame(filas_te).set_index("modelo")
print("\nDesempeño en el conjunto de prueba (300 pacientes):")
print(tabla_te.round(3).to_string())

mejor = tabla_cv["F1 CV"].idxmax()
print(f"\nMejor modelo según el F1 de validación cruzada (no de prueba): {mejor}")

# Figura: matrices de confusión de los cinco modelos
fig, ejes = plt.subplots(2, 3, figsize=(12, 7.5))
for ax, (nombre, pred) in zip(ejes.ravel(), predicciones.items()):
    sns.heatmap(confusion_matrix(y_prueba, pred), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Sano", "Riesgo"], yticklabels=["Sano", "Riesgo"])
    ax.set_title(f"{nombre}\nexactitud {accuracy_score(y_prueba, pred):.1%}, recall {recall_score(y_prueba, pred):.1%}", fontsize=10)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
ejes[1, 2].axis("off")
plt.tight_layout()
plt.savefig(f"{IMG}/matrices_confusion.png", dpi=200)
plt.close()

# Figura: curvas ROC
plt.figure(figsize=(6.5, 5.5))
for nombre, proba in probas.items():
    fpr, tpr, _ = roc_curve(y_prueba, proba)
    plt.plot(fpr, tpr, linewidth=2, label=f"{nombre} (AUC {roc_auc_score(y_prueba, proba):.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos (recall)")
plt.title("Curvas ROC en el conjunto de prueba")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{IMG}/roc.png", dpi=200)
plt.close()

# Figura: comparación de métricas
ax = tabla_te[["exactitud", "precisión", "recall", "F1", "AUC"]].plot.bar(figsize=(10, 4.5), width=0.8, rot=0)
ax.set_ylim(0.6, 1.0)
ax.set_ylabel("Valor en el conjunto de prueba")
ax.set_xlabel("")
ax.set_title("Comparación de los cinco modelos")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
plt.tight_layout()
plt.savefig(f"{IMG}/comparacion.png", dpi=200)
plt.close()

# ---- Optimización de hiperparámetros del bosque y de la SVM -----------------
print("\n" + "-" * 70 + "\nOptimización de hiperparámetros (búsqueda en rejilla, F1 con validación cruzada)\n" + "-" * 70)
rejillas = {
    "Random Forest": (RandomForestClassifier(random_state=SEMILLA),
                      {"n_estimators": [100, 300], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 3, 5]}),
    "SVM": (make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEMILLA)),
            {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]}),
}
optimos, filas_opt = {}, []
for nombre, (base, rejilla) in rejillas.items():
    gs = GridSearchCV(base, rejilla, scoring="f1", cv=cv, n_jobs=-1).fit(X_ent, y_ent)
    optimos[nombre] = gs.best_estimator_
    pred = gs.best_estimator_.predict(X_prueba)
    proba = gs.best_estimator_.predict_proba(X_prueba)[:, 1]
    print(f"{nombre}: mejores hiperparámetros {gs.best_params_}")
    print(f"   F1 CV: {gs.best_score_:.3f} (por defecto: {tabla_cv.loc[nombre, 'F1 CV']:.3f})")
    filas_opt.append({"modelo": f"{nombre} optimizado", "exactitud": accuracy_score(y_prueba, pred),
                      "precisión": precision_score(y_prueba, pred), "recall": recall_score(y_prueba, pred),
                      "F1": f1_score(y_prueba, pred), "AUC": roc_auc_score(y_prueba, proba)})
print("\nModelos optimizados en el conjunto de prueba:")
print(pd.DataFrame(filas_opt).set_index("modelo").round(3).to_string())

# =============================================================================
# EVALUACIÓN DETALLADA DEL MEJOR MODELO
# =============================================================================
print("\n" + "=" * 70 + f"\nEVALUACIÓN DETALLADA: {mejor}\n" + "=" * 70)
modelo_final = modelos[mejor]
print(classification_report(y_prueba, predicciones[mejor], target_names=["Sano", "Riesgo"], digits=3))
tn, fp, fn, tp = confusion_matrix(y_prueba, predicciones[mejor]).ravel()
print(f"Verdaderos negativos: {tn}, falsos positivos: {fp}, falsos negativos: {fn}, verdaderos positivos: {tp}")
print(f"Pacientes con riesgo que el modelo NO detectó (falsos negativos): {fn} de {fn + tp}")

# Umbral de decisión: en medicina un falso negativo cuesta más que un falso positivo.
# El umbral se elige SOLO con el entrenamiento (predicciones de validación cruzada) y luego se prueba.
proba_cv = cross_val_predict(modelo_final, X_ent, y_ent, cv=cv, method="predict_proba")[:, 1]
umbrales = np.linspace(0.05, 0.95, 91)
rec_cv = np.array([recall_score(y_ent, proba_cv >= u) for u in umbrales])
pre_cv = np.array([precision_score(y_ent, proba_cv >= u, zero_division=0) for u in umbrales])
umbral_97 = umbrales[rec_cv >= 0.97].max()
print(f"\nMayor umbral con recall >= 0.97 en validación cruzada: {umbral_97:.2f}")
filas_u = []
for etiqueta, u in [("0.50 (por defecto)", 0.5), (f"{umbral_97:.2f} (recall >= 0.97 en CV)", umbral_97)]:
    pred_u = (probas[mejor] >= u).astype(int)
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_prueba, pred_u).ravel()
    filas_u.append({"umbral": etiqueta, "precisión": precision_score(y_prueba, pred_u),
                    "recall": recall_score(y_prueba, pred_u), "F1": f1_score(y_prueba, pred_u),
                    "falsos neg.": fn_u, "falsos pos.": fp_u})
print("Efecto del umbral en el conjunto de prueba:")
print(pd.DataFrame(filas_u).set_index("umbral").round(3).to_string())

plt.figure(figsize=(7, 4.5))
plt.plot(umbrales, rec_cv, label="Recall", linewidth=2, color="#e76f51")
plt.plot(umbrales, pre_cv, label="Precisión", linewidth=2, color="#264653")
plt.axvline(0.5, color="gray", linestyle=":", label="Umbral por defecto")
plt.axvline(umbral_97, color="#e76f51", linestyle="--", label=f"Umbral {umbral_97:.2f} (recall ≥ 0.97)")
plt.xlabel("Umbral de decisión sobre la probabilidad de riesgo")
plt.ylabel("Valor (validación cruzada, entrenamiento)")
plt.title(f"Compromiso entre recall y precisión: {mejor}")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{IMG}/umbral.png", dpi=200)
plt.close()


# =============================================================================
# TAREA 3: INTERPRETACIÓN DE RESULTADOS
# =============================================================================
print("\n" + "=" * 70 + "\nTAREA 3: ¿QUÉ FACTORES SON MÁS IMPORTANTES?\n" + "=" * 70)

# (a) Regresión logística: coeficientes sobre variables estandarizadas
rl = modelos["Regresión Logística"]
coef = pd.Series(rl[-1].coef_[0], index=nombres)
oddsratio = np.exp(coef)

# (b) Random Forest: importancia por reducción de impureza
rf = modelos["Random Forest"]
imp_rf = pd.Series(rf.feature_importances_, index=nombres)

# (c) Importancia por permutación (en prueba, 30 repeticiones): cuánto cae el AUC al desordenar una característica
perm = permutation_importance(rf, X_prueba, y_prueba, scoring="roc_auc", n_repeats=30, random_state=SEMILLA)
imp_perm = pd.Series(perm.importances_mean, index=nombres)
imp_perm_sd = pd.Series(perm.importances_std, index=nombres)

# (d) Árbol de decisión poco profundo, legible
arbol_p = DecisionTreeClassifier(max_depth=3, random_state=SEMILLA).fit(X_ent, y_ent)
print(f"\nÁrbol de profundidad 3: exactitud en prueba {arbol_p.score(X_prueba, y_prueba):.3f} "
      f"(árbol sin límite: {tabla_te.loc['Árbol de decisión', 'exactitud']:.3f})")
print(export_text(arbol_p, feature_names=nombres))

resumen = pd.DataFrame({"coef. logística (z)": coef, "odds ratio": oddsratio,
                        "importancia RF": imp_rf, "permutación (caída AUC)": imp_perm,
                        "desv. perm.": imp_perm_sd})
resumen = resumen.reindex(imp_perm.sort_values(ascending=False).index)
print("Importancia de las características (ordenadas por permutación):")
print(resumen.round(3).to_string())

rangos = pd.DataFrame({"logística": coef.abs().rank(ascending=False), "RF": imp_rf.rank(ascending=False),
                       "permutación": imp_perm.rank(ascending=False)})
print("\nPosición de cada característica según cada método (1 = más importante):")
print(rangos.reindex(resumen.index).astype(int).to_string())
print("\nCorrelación de Spearman entre los rangos:")
print(rangos.corr(method="spearman").round(2).to_string())

fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
orden = coef.abs().sort_values().index
colores = ["#e76f51" if coef[n] > 0 else "#2a9d8f" for n in orden]
ejes[0].barh(orden, coef[orden], color=colores)
ejes[0].axvline(0, color="k", linewidth=0.8)
ejes[0].set_title("Regresión logística\n(coeficiente, variables estandarizadas)", fontsize=10)
orden = imp_rf.sort_values().index
ejes[1].barh(orden, imp_rf[orden], color="#264653")
ejes[1].set_title("Random Forest\n(importancia por impureza)", fontsize=10)
orden = imp_perm.sort_values().index
ejes[2].barh(orden, imp_perm[orden], xerr=imp_perm_sd[orden], color="#e9c46a")
ejes[2].set_title("Importancia por permutación\n(caída del AUC en prueba)", fontsize=10)
plt.tight_layout()
plt.savefig(f"{IMG}/importancias.png", dpi=200)
plt.close()

plt.figure(figsize=(14, 6))
plot_tree(arbol_p, feature_names=nombres, class_names=["Sano", "Riesgo"], filled=True, rounded=True, fontsize=8)
plt.title("Árbol de decisión de profundidad 3")
plt.tight_layout()
plt.savefig(f"{IMG}/arbol.png", dpi=200)
plt.close()

# ---- Verificación contra la verdad conocida (sólo posible porque los datos son simulados) ----
# Con shuffle=False las columnas informativas, redundantes y de ruido quedan en orden fijo; las
# columnas de X (con shuffle=True) son una permutación de esas, así que se identifican por sus valores.
X_ord, _ = make_classification(n_samples=1000, n_features=8, n_informative=5, n_classes=2, random_state=42, shuffle=False)
roles_orden = ["informativa"] * 5 + ["redundante"] * 2 + ["ruido"] * 1       # n_redundant=2 por defecto
roles = {}
for j, n in enumerate(nombres):
    k = next(k for k in range(8) if np.allclose(np.sort(X[:, j]), np.sort(X_ord[:, k])))
    roles[n] = roles_orden[k]
verdad = pd.DataFrame({"rol real": pd.Series(roles), "|coef.| logística": coef.abs(),
                       "importancia RF": imp_rf, "permutación": imp_perm})
verdad = verdad.reindex(imp_perm.sort_values(ascending=False).index)
print("\nVerificación con la estructura real de los datos simulados:")
print(verdad.round(3).to_string())
print("\nLos nombres médicos (Edad, Colesterol...) son sólo etiquetas: make_classification asigna el rol a columnas al azar.")
print("Por eso la importancia aquí describe los datos simulados, no la fisiología de las enfermedades cardíacas.")

# ---- Ejemplo: un paciente concreto, explicado con la regresión logística ----
i = int(np.argmax(probas["Regresión Logística"]))
j = int(np.argmin(np.abs(probas["Regresión Logística"] - 0.5)))
for etiqueta, idx in [("paciente de mayor riesgo en prueba", i), ("paciente más dudoso en prueba", j)]:
    z = rl[0].transform(X_prueba[idx:idx + 1])[0]
    aporte = pd.Series(z * coef.values, index=nombres)
    print(f"\n{etiqueta}: probabilidad de riesgo {probas['Regresión Logística'][idx]:.1%} "
          f"(real: {'riesgo' if y_prueba[idx] else 'sano'})")
    print("   aportes al logit (coef. x valor estandarizado), de mayor a menor |aporte|:")
    print(aporte.reindex(aporte.abs().sort_values(ascending=False).index).round(2).head(4).to_string())

# ---- Falsos negativos del mejor modelo ----
errores = np.where((predicciones[mejor] == 0) & (y_prueba == 1))[0]
print(f"\nFalsos negativos de {mejor}: {len(errores)}; probabilidad de riesgo asignada:",
      np.round(probas[mejor][errores], 2).tolist())
print("\nListo. Figuras guardadas en", IMG)
