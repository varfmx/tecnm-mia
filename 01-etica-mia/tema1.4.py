# Ejemplo de agente reflexivo simple
import random

# Definición del entorno
entorno = {
    "A": random.choice(["limpio", "sucio"]),
    "B": random.choice(["limpio", "sucio"])
}

# Posición inicial del agente
posicion = random.choice(["A", "B"])

def agente_reflexivo(posicion, entorno):
    print(f"\nInicio en habitación {posicion}")
    pasos = 0

    while entorno["A"] == "sucio" or entorno["B"] == "sucio":
        print(f"\nPaso {pasos + 1}")
        print(f"Posición actual: {posicion}")
        print(f"Estado del entorno: {entorno}")

        if entorno[posicion] == "sucio":
            print("Acción: Limpiar")
            entorno[posicion] = "limpio"
        else:
            # Moverse a la otra habitación
            posicion = "B" if posicion == "A" else "A"
            print(f"Acción: Moverse a la habitación {posicion}")

        pasos += 1

    print(f"\nEntorno final limpio: {entorno}")
    print(f"Total de pasos: {pasos}")

# Ejecutar el agente
agente_reflexivo(posicion, entorno)