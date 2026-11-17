mport networkx as nx

G = nx.DiGraph()

# Nodos (conceptos)

conceptos = ["Animal", "Mamifero", "Ave", "Perro", "Gato", "Canario"]

G.add_nodes_from(conceptos)



# Relaciones jerárquicas

G.add_edge("Mamifero", "Animal", relation="es_un")

G.add_edge("Ave", "Animal", relation="es_un")

G.add_edge("Perro", "Mamifero", relation="es_un")

G.add_edge("Gato", "Mamifero", relation="es_un")

G.add_edge("Canario", "Ave", relation="es_un")



# Relaciones adicionales

G.add_edge("Perro", "Gato", relation="enemigo_de")

G.add_edge("Canario", "Gato", relation="presa_de")



# Consultas

def obtener_superclases(concepto):

    return [v for _, v, d in G.out_edges(concepto, data=True) if d["relation"] == "es_un"]



def obtener_relaciones(concepto):

    return [(v, d["relation"]) for _, v, d in G.out_edges(concepto, data=True)]



print("Superclases de Perro:", obtener_superclases("Perro"))

print("Relaciones de Gato:", obtener_relaciones("Gato"))