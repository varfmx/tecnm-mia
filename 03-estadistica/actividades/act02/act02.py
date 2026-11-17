import itertools
import numpy as np
import pandas as pd

# Dominio de cada atributo del Weather Dataset
dominios = {
    'outlook': ['sunny', 'overcast', 'rainy'],
    'temperature': ['hot', 'mild', 'cool'],
    'humidity': ['high', 'normal'],
    'windy': [False, True],
}

# ---------------------------------------------------------------
# Tarea 1: completar el dataset
# ---------------------------------------------------------------

# Las 14 instancias de la Tabla 1.1 (las 5 primeras son las del código base)
weather_data = {
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
}
df_weather = pd.DataFrame(weather_data)

# Tipos categóricos con su dominio completo
for col in ['outlook', 'temperature', 'humidity']:
    df_weather[col] = pd.Categorical(df_weather[col], categories=dominios[col])
df_weather['play'] = pd.Categorical(df_weather['play'], categories=['yes', 'no'])

# Tabla 1.1 del documento, transcrita fila por fila como referencia
TABLA_1_1 = [
    ('sunny', 'hot', 'high', False, 'no'),
    ('sunny', 'hot', 'high', True, 'no'),
    ('overcast', 'hot', 'high', False, 'yes'),
    ('rainy', 'mild', 'high', False, 'yes'),
    ('rainy', 'cool', 'normal', False, 'yes'),
    ('rainy', 'cool', 'normal', True, 'no'),
    ('overcast', 'cool', 'normal', True, 'yes'),
    ('sunny', 'mild', 'high', False, 'no'),
    ('sunny', 'cool', 'normal', False, 'yes'),
    ('rainy', 'mild', 'normal', False, 'yes'),
    ('sunny', 'mild', 'normal', True, 'yes'),
    ('overcast', 'mild', 'high', True, 'yes'),
    ('overcast', 'hot', 'normal', False, 'yes'),
    ('rainy', 'mild', 'high', True, 'no'),
]


def comparar_con_tabla(df, referencia):
    """Compara el DataFrame contra la Tabla 1.1, instancia por instancia."""
    filas = [tuple(fila) for fila in df.astype(object).itertuples(index=False)]
    coincidencias = [fila == ref for fila, ref in zip(filas, referencia)]
    faltantes = [ref for ref in referencia if ref not in filas]
    sobrantes = [fila for fila in filas if fila not in referencia]

    print(f"Instancias en la Tabla 1.1: {len(referencia)}")
    print(f"Instancias en el DataFrame: {len(filas)}")
    print(f"Instancias que coinciden en la misma posición: "
          f"{sum(coincidencias)}/{len(referencia)}")
    print(f"Instancias de la tabla que faltan en el DataFrame: {len(faltantes)}")
    print(f"Instancias del DataFrame que no están en la tabla: {len(sobrantes)}")
    for i, ok in enumerate(coincidencias, start=1):
        if not ok:
            print(f"  Diferencia en la instancia {i}: {filas[i - 1]} vs {referencia[i - 1]}")
    assert len(filas) == len(referencia) and all(coincidencias), \
        "El DataFrame no coincide con la Tabla 1.1"
    print("El DataFrame coincide exactamente con la Tabla 1.1")


def verificar_dataset(df):
    """Comprueba que el dataset tenga las 14 instancias del documento."""
    assert len(df) == 14, "Deben existir 14 instancias"
    # Un valor fuera del dominio se convierte en NaN al pasar a category
    assert not df.isna().any().any(), "Hay valores nulos o fuera del dominio"
    assert df['windy'].isin(dominios['windy']).all(), "windy debe ser booleano"
    assert not df.duplicated().any(), "Hay instancias repetidas"
    assert df['play'].value_counts().to_dict() == {'yes': 9, 'no': 5}, \
        "La clase debe tener 9 yes y 5 no"
    print("Verificación correcta: 14 instancias distintas, sin nulos, "
          "valores dentro del dominio y clase 9 yes / 5 no")


print("Dataset construido:")
print(df_weather)

print("\nEstructura del DataFrame:")
df_weather.info()

print("\nComparación con la Tabla 1.1:")
comparar_con_tabla(df_weather, TABLA_1_1)

print()
verificar_dataset(df_weather)

print("\nDistribución de la clase 'play':")
print(df_weather['play'].value_counts())

# Espacio de combinaciones posibles de los atributos
combinaciones = list(itertools.product(*dominios.values()))
presentes = df_weather[list(dominios)].astype(object).drop_duplicates()
print(f"\nCombinaciones posibles: {len(combinaciones)}")
print(f"Combinaciones presentes en el dataset: {len(presentes)}")

# ---------------------------------------------------------------
# Tarea 2: funciones de análisis
# ---------------------------------------------------------------


def contar_por_condicion(df, atributo):
    """Cuenta las instancias de cada clase para cada valor del atributo."""
    return pd.crosstab(df[atributo], df['play'], margins=True, margins_name='total')


def probabilidad_juego(df, atributo):
    """Calcula P(play = yes) y P(play = no) para cada valor del atributo."""
    tabla = pd.crosstab(df[atributo], df['play'], normalize='index')
    tabla.columns = ['P(yes)', 'P(no)']
    tabla.insert(0, 'n', df[atributo].value_counts(sort=False))
    return tabla.round(3)


def filtrar(df, **condiciones):
    """Devuelve las instancias que cumplen todas las condiciones dadas."""
    mascara = pd.Series(True, index=df.index)
    for atributo, valor in condiciones.items():
        mascara &= df[atributo] == valor
    return df[mascara]


print("\n=== Tarea 2a: conteo por condición climática ===")
print(contar_por_condicion(df_weather, 'outlook'))

print("\n=== Tarea 2b: probabilidad de jugar por categoría ===")
for col in dominios:
    print(f"\n{probabilidad_juego(df_weather, col)}")

print("\n=== Tarea 2c: filtrado por condiciones ===")
print("\noutlook = sunny, humidity = high:")
print(filtrar(df_weather, outlook='sunny', humidity='high'))
print("\noutlook = rainy, windy = True:")
print(filtrar(df_weather, outlook='rainy', windy=True))

# ---------------------------------------------------------------
# Tarea 3: generador de reglas (árbol de decisión ID3)
# ---------------------------------------------------------------


def entropia(clases):
    """Entropía de Shannon (en bits) de una serie de clases."""
    p = clases.value_counts(normalize=True)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ganancia(df, atributo):
    """Ganancia de información de dividir df según el atributo."""
    g = entropia(df['play'])
    for _, grupo in df.groupby(atributo, observed=True):
        g -= len(grupo) / len(df) * entropia(grupo['play'])
    return g


def clase_mayoritaria(df):
    return df['play'].value_counts().idxmax()


def construir_arbol(df, atributos):
    """Construye el árbol con ID3: divide por el atributo de mayor ganancia."""
    if df['play'].nunique() == 1 or not atributos:
        return clase_mayoritaria(df)
    mejor = max(atributos, key=lambda a: ganancia(df, a))
    resto = [a for a in atributos if a != mejor]
    ramas = {}
    for valor in dominios[mejor]:
        subconjunto = df[df[mejor] == valor]
        # Un valor sin ejemplos recibe la clase mayoritaria del nodo padre
        if len(subconjunto) == 0:
            ramas[valor] = clase_mayoritaria(df)
        else:
            ramas[valor] = construir_arbol(subconjunto, resto)
    return {'atributo': mejor, 'ramas': ramas}


def imprimir_arbol(arbol, nivel=0):
    for valor, sub in arbol['ramas'].items():
        prefijo = '|   ' * nivel + f"{arbol['atributo']} = {valor}"
        if isinstance(sub, dict):
            print(prefijo)
            imprimir_arbol(sub, nivel + 1)
        else:
            print(f"{prefijo}: {sub}")


def extraer_reglas(arbol, condiciones=()):
    """Convierte cada camino de la raíz a una hoja en una regla SI-ENTONCES."""
    if not isinstance(arbol, dict):
        return [(condiciones, arbol)]
    reglas = []
    for valor, sub in arbol['ramas'].items():
        reglas += extraer_reglas(sub, condiciones + ((arbol['atributo'], valor),))
    return reglas


def predecir(arbol, instancia):
    """Recorre el árbol según los valores de la instancia hasta llegar a una hoja."""
    while isinstance(arbol, dict):
        arbol = arbol['ramas'][instancia[arbol['atributo']]]
    return arbol


atributos = list(dominios)

print("\n=== Tarea 3: generador de reglas ===")
print(f"\nEntropía de la clase: {entropia(df_weather['play']):.3f} bits")
print("Ganancia de información en la raíz:")
for a in sorted(atributos, key=lambda a: -ganancia(df_weather, a)):
    print(f"  {a:<12} {ganancia(df_weather, a):.3f}")

arbol = construir_arbol(df_weather, atributos)
print("\nÁrbol de decisión:")
imprimir_arbol(arbol)

print("\nReglas extraídas:")
for i, (condiciones, clase) in enumerate(extraer_reglas(arbol), start=1):
    antecedente = ' Y '.join(f"{a} = {v}" for a, v in condiciones)
    print(f"R{i}: SI {antecedente} ENTONCES play = {clase}")

instancias = df_weather.to_dict('records')
aciertos = sum(predecir(arbol, x) == x['play'] for x in instancias)
print(f"\nExactitud sobre las 14 instancias: {aciertos}/{len(instancias)}")

nuevas = pd.DataFrame([
    {'outlook': 'sunny', 'temperature': 'cool', 'humidity': 'high', 'windy': True},
    {'outlook': 'overcast', 'temperature': 'mild', 'humidity': 'normal', 'windy': False},
    {'outlook': 'rainy', 'temperature': 'hot', 'humidity': 'high', 'windy': False},
    {'outlook': 'rainy', 'temperature': 'cool', 'humidity': 'high', 'windy': True},
])
nuevas['play'] = [predecir(arbol, x) for x in nuevas.to_dict('records')]
print("\nPredicción para condiciones nuevas:")
print(nuevas)

df_weather.to_csv('weather.csv', index=False)
print("\nDataset exportado a weather.csv")
