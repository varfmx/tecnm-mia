#Lógica Proposicional:

# Base de hechos

hechos = {

    "llueve": True,

    "paraguas": False

}

# Reglas tipo (premisas → conclusión)

reglas = [

    (["llueve", "not paraguas"], "mojarse"),

    (["llueve"], "suelo_mojado")

]

def evaluar_literal(literal, hechos):

    if literal.startswith("not "):

        return not hechos.get(literal[4:], False)

    return hechos.get(literal, False)

def motor_inferencia(hechos, reglas):

    nuevos = True

    while nuevos:

        nuevos = False

        for premisas, conclusion in reglas:

            if conclusion not in hechos:

                if all(evaluar_literal(p, hechos) for p in premisas):

                    hechos[conclusion] = True

                    nuevos = True

    return hechos



resultado = motor_inferencia(hechos, reglas)

print("Hechos inferidos:", resultado)