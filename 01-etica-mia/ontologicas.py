
#Ontologías OWL

from owlready2 import *



onto = get_ontology("http://ejemplo.org/mascotas.owl")

with onto:

    class Animal(Thing): pass

    class Perro(Animal): pass

    class Persona(Thing): pass

    class tieneMascota(Persona >> Animal): pass

    class ladra(Animal >> bool, FunctionalProperty): pass

# Crear instancias

juan = Persona("Juan")

firulais = Perro("Firulais")



juan.tieneMascota.append(firulais)

firulais.ladra = True



# Razonador

sync_reasoner()



# Consulta

for mascota in juan.tieneMascota:

    print("Juan tiene mascota:", mascota.name)



print("¿Firulais ladra?:", firulais.ladra)