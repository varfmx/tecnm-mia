#Lógica de Primer Orden

from kanren import Relation, facts, var, run



padre = Relation()

facts(padre,

      ("Juan", "Pedro"),

      ("Juan", "Ana"),

      ("Pedro", "Luis"))



def obtener_hijos(persona):

    x = var()

    return run(10, x, padre(persona, x))



def obtener_abuelo(nieto):

    x, y = var(), var()

    return run(10, x, padre(x, y), padre(y, nieto))



print("Hijos de Juan:", obtener_hijos("Juan"))

print("Abuelos de Luis:", obtener_abuelo("Luis"))