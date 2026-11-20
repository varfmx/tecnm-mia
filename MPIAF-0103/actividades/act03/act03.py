import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Conjunto de datos del Ejercicio 1.1
df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Color de Ojos': ['Azul', 'Verde', 'Marrón', 'Azul', 'Marrón'],
    'Satisfacción': ['Alta', 'Baja', 'Media', 'Alta', 'Baja'],
    'Edad': [25, 34, 42, 28, 37],
    'Número de Hijos': [2, 1, 3, 0, 4],
    'Altura (cm)': [170, 165, 180, 160, 175],
}).set_index('ID')

# ---------------------------------------------------------------
# Ejercicio 1.1: identificación de tipos de variables
# ---------------------------------------------------------------

clasificacion = pd.DataFrame({
    'Tipo': ['Cualitativa', 'Cualitativa', 'Cuantitativa', 'Cuantitativa', 'Cuantitativa'],
    'Subtipo': ['Nominal', 'Ordinal', 'Discreta', 'Discreta', 'Continua'],
}, index=df.columns)

# Representación computacional de cada tipo en pandas
df['Color de Ojos'] = df['Color de Ojos'].astype('category')
df['Satisfacción'] = pd.Categorical(df['Satisfacción'],
                                    categories=['Baja', 'Media', 'Alta'], ordered=True)
df['Altura (cm)'] = df['Altura (cm)'].astype(float)
clasificacion['dtype en pandas'] = df.dtypes.astype(str)

print("=== Ejercicio 1.1: tipos de variables ===")
print(df)
print()
print(clasificacion)

# Una variable ordinal admite comparaciones de orden, una nominal no
print("\nSatisfacción >= 'Media':")
print(df['Satisfacción'] >= 'Media')

# ---------------------------------------------------------------
# Ejercicio 1.2: codificación One-Hot
# ---------------------------------------------------------------

data = {
    'Color de Ojos': ['Azul', 'Verde', 'Marrón', 'Azul', 'Marrón']
}

# Crear un dataframe
df_ojos = pd.DataFrame(data)

# Realizar la codificación One-Hot
df_one_hot = pd.get_dummies(df_ojos, columns=['Color de Ojos'])

print("\n=== Ejercicio 1.2: codificación One-Hot ===")
print(df_one_hot)

# Misma codificación con 0 y 1 en lugar de False y True
df_one_hot_01 = pd.get_dummies(df_ojos, columns=['Color de Ojos'], dtype=int)
print("\nCon dtype=int:")
print(df_one_hot_01)

# Cada fila tiene exactamente un 1: la categoría a la que pertenece
print("\nSuma por fila:", df_one_hot_01.sum(axis=1).tolist())

# ---------------------------------------------------------------
# Ejercicio 1.3: normalización Min-Max
# ---------------------------------------------------------------

altura = [170, 165, 180, 160, 175]

# Crear el objeto scaler
scaler = MinMaxScaler()

# Normalizar los datos (MinMaxScaler espera una matriz de n filas x 1 columna)
df_altura = pd.DataFrame({'Altura (cm)': altura})
df_altura['Normalizada'] = scaler.fit_transform(df_altura[['Altura (cm)']]).ravel()

# Verificación con la fórmula (1.8): (X - min) / (max - min)
minimo, maximo = min(altura), max(altura)
df_altura['Fórmula (1.8)'] = [(x - minimo) / (maximo - minimo) for x in altura]

print("\n=== Ejercicio 1.3: normalización de Altura ===")
print(f"Mínimo = {scaler.data_min_[0]:.0f}, Máximo = {scaler.data_max_[0]:.0f}")
print(df_altura)
print("¿Coinciden MinMaxScaler y la fórmula?",
      (df_altura['Normalizada'] == df_altura['Fórmula (1.8)']).all())

# ---------------------------------------------------------------
# Ejercicio 1.4: análisis de frecuencia
# ---------------------------------------------------------------

satisfaccion = ['Alta', 'Baja', 'Media', 'Alta', 'Baja']

# Crear un dataframe
df_satisfaccion = pd.DataFrame({'Satisfacción': satisfaccion})

# Calcular la frecuencia de cada categoría
frecuencias = df_satisfaccion['Satisfacción'].value_counts()

print("\n=== Ejercicio 1.4: frecuencia de Satisfacción ===")
print(frecuencias)

# Tabla de frecuencias en el orden natural de la variable ordinal
orden = pd.Categorical(satisfaccion, categories=['Baja', 'Media', 'Alta'], ordered=True)
tabla = pd.Series(orden).value_counts(sort=False).to_frame('Frecuencia')
tabla['Relativa'] = tabla['Frecuencia'] / tabla['Frecuencia'].sum()
tabla['Acumulada'] = tabla['Frecuencia'].cumsum()
tabla['Relativa acumulada'] = tabla['Relativa'].cumsum()
tabla.index.name = 'Satisfacción'

print("\nTabla de frecuencias ordenada:")
print(tabla)
