import itertools
import pandas as pd

# Dominio de cada atributo del Weather Dataset
dominios = {
    'outlook': ['sunny', 'overcast', 'rainy'],
    'temperature': ['hot', 'mild', 'cool'],
    'humidity': ['high', 'normal'],
    'windy': [False, True],
}

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

print("Dataset construido:")
print(df_weather)

print("\nEstructura del DataFrame:")
df_weather.info()

print("\nDistribución de la clase 'play':")
print(df_weather['play'].value_counts())

print("\nFrecuencia de cada atributo:")
for col in dominios:
    print(f"{col}: {df_weather[col].value_counts(sort=False).to_dict()}")

print("\nTabla de contingencia outlook vs play:")
print(pd.crosstab(df_weather['outlook'], df_weather['play']))

# Espacio de combinaciones posibles de los atributos
combinaciones = list(itertools.product(*dominios.values()))
presentes = df_weather[list(dominios)].astype(object).drop_duplicates()
print(f"\nCombinaciones posibles: {len(combinaciones)}")
print(f"Combinaciones presentes en el dataset: {len(presentes)}")

df_weather.to_csv('weather.csv', index=False)
print("\nDataset exportado a weather.csv")
