import networkx as nx

G = nx.DiGraph()

# CAMBIO 1: Se agrega "Tigre" a la lista de conceptos de la red
conceptos = ["Animal", "Mamifero", "Ave", "Perro", "Gato", "Canario", "Tigre"]
G.add_nodes_from(conceptos)

# Relaciones jerárquicas originales
G.add_edge("Mamifero", "Animal", relation="es_un")
G.add_edge("Ave", "Animal", relation="es_un")
G.add_edge("Perro", "Mamifero", relation="es_un")
G.add_edge("Gato", "Mamifero", relation="es_un")
G.add_edge("Canario", "Ave", relation="es_un")

# CAMBIO 2: Se añaden las relaciones semánticas del nuevo nodo
G.add_edge("Tigre", "Mamifero", relation="es_un")
G.add_edge("Tigre", "Perro", relation="enemigo_de")

# Relaciones adicionales originales
G.add_edge("Perro", "Gato", relation="enemigo_de")
G.add_edge("Canario", "Gato", relation="presa_de")

# Funciones de consulta e inferencia en el grafo
def obtener_superclases(concepto):
    return [v for _, v, d in G.out_edges(concepto, data=True) if d["relation"] == "es_un"]

def obtener_relaciones(concepto):
    return [(v, d["relation"]) for _, v, d in G.out_edges(concepto, data=True)]

# CAMBIO 3: Ejecución de consultas para verificar el nuevo conocimiento
print("Superclases de Tigre:", obtener_superclases("Tigre"))
print("Relaciones de Tigre:", obtener_relaciones("Tigre"))