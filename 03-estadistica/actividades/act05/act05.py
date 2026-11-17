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


def resumen_boxplot(grupos):
    """Calcula lo que dibuja cada caja: cuartiles, bigotes y valores atípicos."""
    filas = {}
    for nombre, datos in grupos.items():
        s = boxplot_stats(datos)[0]
        filas[nombre] = {
            'Q1': s['q1'], 'Mediana': s['med'], 'Q3': s['q3'], 'RIC': s['iqr'],
            'Bigote inf.': s['whislo'], 'Bigote sup.': s['whishi'],
            'Media': s['mean'], 'Desv. est.': np.std(datos, ddof=1),
            'Atípicos': len(s['fliers']),
        }
    # Una columna por grupo para que la tabla quepa en la consola
    return pd.DataFrame(filas).round(2)


def puntos_dispersos(ax, grupos, semilla=0):
    """Dibuja cada observación sobre su caja con un pequeño desplazamiento horizontal."""
    rng = np.random.default_rng(semilla)
    for i, datos in enumerate(grupos, start=1):
        x = i + rng.uniform(-0.12, 0.12, len(datos))
        ax.scatter(x, datos, s=12, color='black', alpha=0.4, zorder=3)


def correlacion(x, y):
    return np.corrcoef(x, y)[0, 1]


# ===============================================================
# Boxplots - Ejercicio 1: calificaciones de dos grupos
# ===============================================================

# Código base del documento fuente
np.random.seed(10)
calificaciones_grupo_A = np.random.randint(60, 100, 30)
calificaciones_grupo_B = np.random.randint(50, 95, 30)

# Extensión: medias, observaciones individuales y resumen numérico
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot([calificaciones_grupo_A, calificaciones_grupo_B],
           tick_labels=['Grupo A', 'Grupo B'],
           patch_artist=True, showmeans=True,
           boxprops=dict(facecolor='lightblue'),
           meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red'))
puntos_dispersos(ax, [calificaciones_grupo_A, calificaciones_grupo_B])
ax.set_title('Comparación de calificaciones finales entre dos grupos')
ax.set_ylabel('Calificación')
ax.grid(axis='y', linestyle='--', alpha=0.7)
guardar('boxplot_ej1.png')

print("=== Boxplots - Ejercicio 1: calificaciones por grupo ===")
print(resumen_boxplot({'Grupo A': calificaciones_grupo_A,
                       'Grupo B': calificaciones_grupo_B}))

# ===============================================================
# Boxplots - Ejercicio 2: tiempos de espera en tres sucursales
# ===============================================================

# Código base del documento fuente
np.random.seed(20)
tiempos_sucursal_1 = np.random.normal(15, 3, 40)
tiempos_sucursal_2 = np.random.normal(20, 5, 40)
tiempos_sucursal_3 = np.random.normal(10, 2, 40)
tiempos = [tiempos_sucursal_1, tiempos_sucursal_2, tiempos_sucursal_3]
sucursales = ['Sucursal 1', 'Sucursal 2', 'Sucursal 3']

# Extensión: un color por sucursal, medias y observaciones individuales
fig, ax = plt.subplots(figsize=(7, 5))
cajas = ax.boxplot(tiempos, tick_labels=sucursales, patch_artist=True, showmeans=True,
                   meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red'))
for caja, color in zip(cajas['boxes'], ['skyblue', 'lightgreen', 'lightcoral']):
    caja.set_facecolor(color)
puntos_dispersos(ax, tiempos)
ax.set_title('Comparación de tiempos de espera en tres sucursales')
ax.set_ylabel('Tiempo de espera (minutos)')
ax.grid(axis='y', linestyle='--', alpha=0.7)
guardar('boxplot_ej2.png')

print("\n=== Boxplots - Ejercicio 2: tiempos de espera por sucursal ===")
resumen = resumen_boxplot(dict(zip(sucursales, tiempos)))
print(resumen)
for nombre, datos in zip(sucursales, tiempos):
    atipicos = boxplot_stats(datos)[0]['fliers']
    if len(atipicos):
        print(f"Valores atípicos en {nombre}: {np.round(atipicos, 2).tolist()}")

# ===============================================================
# Histogramas - Ejercicio 1: tiempos de entrega de 200 pedidos
# ===============================================================

# Código base del documento fuente
np.random.seed(100)
tiempos_entrega = np.concatenate([
    np.random.normal(4, 1, 180),   # mayoría de los pedidos
    np.random.uniform(8, 15, 20)   # algunos retrasados
])
tiempos_entrega = np.clip(tiempos_entrega, 1, 15)  # limitar entre 1 y 15 días

# Extensión: media, mediana y zona de retraso (más de 7 días)
media, mediana = tiempos_entrega.mean(), np.median(tiempos_entrega)
plt.figure(figsize=(8, 5))
plt.hist(tiempos_entrega, bins=14, color='skyblue', edgecolor='black')
plt.axvspan(7, 15, color='red', alpha=0.08, label='Retraso (> 7 días)')
plt.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
plt.axvline(mediana, color='green', linestyle='-', linewidth=2, label=f'Mediana: {mediana:.2f}')
plt.title('Distribución de tiempos de entrega de pedidos')
plt.xlabel('Tiempo de entrega (días)')
plt.ylabel('Número de pedidos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
guardar('histograma_ej1.png')

retrasados = tiempos_entrega > 7
print("\n=== Histogramas - Ejercicio 1: tiempos de entrega ===")
print(pd.Series(tiempos_entrega).describe().round(2).to_string())
print(f"Mediana: {mediana:.2f} días")
print(f"Pedidos entregados entre 2 y 7 días: "
      f"{((tiempos_entrega >= 2) & (tiempos_entrega <= 7)).mean():.1%}")
print(f"Pedidos con más de 7 días: {retrasados.sum()} ({retrasados.mean():.1%})")

# ===============================================================
# Histogramas - Ejercicio 2: calificaciones de 120 estudiantes
# ===============================================================

# Código base del documento fuente
np.random.seed(200)
calificaciones = np.concatenate([
    np.random.normal(75, 8, 110),   # mayoría de los estudiantes
    np.random.uniform(30, 59, 5),   # calificaciones bajas
    np.random.uniform(91, 100, 5)   # calificaciones altas
])
calificaciones = np.clip(calificaciones, 0, 100)

# Extensión: media, mediana y tabla de frecuencias por rango de calificación
media, mediana = calificaciones.mean(), np.median(calificaciones)
plt.figure(figsize=(8, 5))
plt.hist(calificaciones, bins=10, color='orange', edgecolor='black')
plt.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
plt.axvline(mediana, color='green', linestyle='-', linewidth=2, label=f'Mediana: {mediana:.2f}')
plt.title('Distribución de calificaciones en el examen final')
plt.xlabel('Calificación')
plt.ylabel('Número de estudiantes')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
guardar('histograma_ej2.png')

rangos = pd.cut(calificaciones, bins=[0, 60, 70, 80, 90, 100],
                labels=['< 60', '60-70', '70-80', '80-90', '> 90'], include_lowest=True)
tabla = pd.Series(rangos).value_counts(sort=False).to_frame('Estudiantes')
tabla['Porcentaje'] = (tabla['Estudiantes'] / len(calificaciones) * 100).round(1)
tabla.index.name = 'Rango'

print("\n=== Histogramas - Ejercicio 2: calificaciones ===")
print(pd.Series(calificaciones).describe().round(2).to_string())
print(f"Mediana: {mediana:.2f}")
print(tabla)

# ===============================================================
# Dispersión - Ejercicio 1: horas de estudio vs. calificación por turno
# ===============================================================

# Código base del documento fuente
np.random.seed(123)
horas_matutino = np.random.normal(8, 2, 25)
calif_matutino = horas_matutino * 6 + np.random.normal(40, 5, 25)
horas_vespertino = np.random.normal(6, 2, 25)
calif_vespertino = horas_vespertino * 7 + np.random.normal(38, 7, 25)

turnos = {
    'Matutino': (horas_matutino, calif_matutino, 'blue'),
    'Vespertino': (horas_vespertino, calif_vespertino, 'green'),
}

# Extensión: recta de regresión y coeficiente de correlación por turno
plt.figure(figsize=(8, 5))
filas = {}
for turno, (x, y, color) in turnos.items():
    pendiente, ordenada = np.polyfit(x, y, 1)
    r = correlacion(x, y)
    plt.scatter(x, y, color=color, label=f'{turno} (r = {r:.2f})', alpha=0.7)
    xs = np.linspace(x.min(), x.max(), 50)
    plt.plot(xs, pendiente * xs + ordenada, color=color, linewidth=2)
    filas[turno] = {'Horas (media)': x.mean(), 'Calificación (media)': y.mean(),
                    'Pendiente': pendiente, 'Ordenada': ordenada, 'r de Pearson': r}
plt.title('Horas de estudio vs. calificación por turno escolar')
plt.xlabel('Horas de estudio semanales')
plt.ylabel('Calificación final')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
guardar('dispersion_ej1.png')

print("\n=== Dispersión - Ejercicio 1: horas de estudio vs. calificación ===")
print(pd.DataFrame(filas).round(2))
# La simulación del código base no limita la calificación a 100
for turno, (x, y, color) in turnos.items():
    print(f"Calificaciones mayores a 100 ({turno}): {(y > 100).sum()}")

# ===============================================================
# Dispersión - Ejercicio 2: edad vs. colesterol por actividad física
# ===============================================================

# Código base del documento fuente
np.random.seed(321)
edad_bajo = np.random.randint(30, 60, 20)
colesterol_bajo = np.random.normal(220, 20, 20)
edad_alto = np.random.randint(30, 60, 20)
colesterol_alto = np.random.normal(190, 15, 20)

grupos = {
    'Actividad física baja': (edad_bajo, colesterol_bajo, 'red'),
    'Actividad física alta': (edad_alto, colesterol_alto, 'purple'),
}

# Extensión: colesterol medio de cada grupo y correlación edad-colesterol
plt.figure(figsize=(8, 5))
filas = {}
for grupo, (x, y, color) in grupos.items():
    r = correlacion(x, y)
    plt.scatter(x, y, color=color, label=f'{grupo} (r = {r:.2f})', alpha=0.7)
    plt.axhline(y.mean(), color=color, linestyle='--', linewidth=1.5)
    filas[grupo] = {'Edad (media)': x.mean(), 'Colesterol (media)': y.mean(),
                    'Colesterol (desv. est.)': np.std(y, ddof=1), 'r de Pearson': r}
plt.title('Edad vs. nivel de colesterol por grupo de actividad física')
plt.xlabel('Edad (años)')
plt.ylabel('Colesterol (mg/dL)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
guardar('dispersion_ej2.png')

print("\n=== Dispersión - Ejercicio 2: edad vs. colesterol ===")
print(pd.DataFrame(filas).round(2))
diferencia = colesterol_bajo.mean() - colesterol_alto.mean()
print(f"Diferencia de colesterol medio (baja - alta): {diferencia:.2f} mg/dL")

print(f"\nFiguras guardadas en {os.path.basename(CARPETA_IMG)}/")
