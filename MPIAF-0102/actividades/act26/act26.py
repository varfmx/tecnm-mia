# =============================================================================
# Actividad 26: "Diseña tu Primer Robot Virtual"
# Tema 4.6. Robótica
#
# Dependencias (en Colab ya están):  pip install numpy pandas matplotlib
# No requiere descargas: las habitaciones se generan al azar (con semilla fija).
# Tarda unos 2 minutos.
# =============================================================================
import heapq
import math
import os
import time
import warnings
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

SEMILLA = 42
AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
IMG = os.path.join(AQUI, "img")
os.makedirs(IMG, exist_ok=True)

BATERIA_MAX = 100       # valores de la plantilla del tema
CAPACIDAD_MAX = 100
COSTO_PASO = 1          # una unidad de batería por celda recorrida
COSTO_LIMPIEZA = 1      # una unidad por celda limpiada (como limpiar_area de la plantilla)
UMBRAL_TEMA = 20        # la plantilla regresa a la base cuando la batería es <= 20
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))   # este, norte, oeste, sur


# =============================================================================
# ROBOT DEL TEMA (clases RobotMovil y RobotAutonomo, sin pausas ni mensajes)
# =============================================================================
class RobotMovil:
    """Un robot móvil con ruedas que puede navegar en un espacio 2D (tal como en el tema)."""

    def __init__(self, nombre, verbose=False):
        self.nombre = nombre
        self.verbose = verbose
        self.posicion = [0.0, 0.0]
        self.orientacion = 0
        self.velocidad = 0
        self.trayectoria = []

    def mover_adelante(self, distancia, velocidad=1.0):
        angulo_rad = math.radians(self.orientacion)
        self.posicion[0] += distancia * math.cos(angulo_rad)
        self.posicion[1] += distancia * math.sin(angulo_rad)
        self.trayectoria.append((self.posicion[0], self.posicion[1]))

    def rotar(self, angulo):
        self.orientacion = (self.orientacion + angulo) % 360


class RobotAutonomo(RobotMovil):
    """Navegación reactiva del tema: si hay un obstáculo cerca, gira 45 grados y avanza una unidad."""

    def __init__(self, nombre, verbose=False):
        super().__init__(nombre, verbose)
        self.obstaculos = []
        self.distancia_segura = 2.0

    def agregar_obstaculo(self, x, y, radio=1.0):
        self.obstaculos.append((x, y, radio))

    def detectar_obstaculos_cercanos(self):
        cercanos = []
        for obstaculo in self.obstaculos:
            x_obs, y_obs, radio_obs = obstaculo
            distancia = math.sqrt((self.posicion[0] - x_obs) ** 2 + (self.posicion[1] - y_obs) ** 2)
            if distancia < (radio_obs + self.distancia_segura):
                cercanos.append((obstaculo, distancia))
        return cercanos

    def navegar_a_destino(self, destino_x, destino_y):
        max_pasos = 50
        tolerancia = 0.5
        for paso in range(max_pasos):
            dx = destino_x - self.posicion[0]
            dy = destino_y - self.posicion[1]
            distancia_al_destino = math.sqrt(dx ** 2 + dy ** 2)
            if distancia_al_destino < tolerancia:
                return True
            angulo_destino = math.degrees(math.atan2(dy, dx))
            if self.detectar_obstaculos_cercanos():
                self.rotar(45)
                self.mover_adelante(1.0)
            else:
                self.rotar(angulo_destino - self.orientacion)
                self.mover_adelante(min(2.0, distancia_al_destino))
        return False


class RobotLimpiadorTema(RobotAutonomo):
    """La plantilla del tema tal cual (sin la parte '¡Tu código aquí!'), para medir el punto de partida."""

    def __init__(self, nombre):
        super().__init__(nombre)
        self.areas_limpias = set()
        self.bateria = 100
        self.capacidad_basura = 100

    def limpiar_area(self, x, y):
        if (x, y) in self.areas_limpias:
            return False
        if self.bateria <= 0:
            return False
        self.bateria -= 1
        self.capacidad_basura -= 1
        self.areas_limpias.add((x, y))
        return True

    def planificar_ruta_limpieza(self, ancho, alto):
        areas_limpiadas = 0
        for y in range(alto):
            for x in range(ancho):
                if self.navegar_a_destino(x, y):
                    if self.limpiar_area(x, y):
                        areas_limpiadas += 1
                    if self.bateria <= 20:
                        return areas_limpiadas
        return areas_limpiadas


# =============================================================================
# EL ENTORNO: HABITACIÓN EN CUADRÍCULA Y NAVEGACIÓN (A*)
# =============================================================================
class Habitacion:
    """Cuadrícula ancho x alto con muebles (celdas ocupadas) y una base de carga en (0, 0)."""

    def __init__(self, ancho, alto, obstaculos=(), base=(0, 0)):
        self.ancho, self.alto, self.base = ancho, alto, base
        self.obstaculos = frozenset(obstaculos)
        self.dist_base = self.bfs(base)
        self.limpiables = sorted((c for c in self.celdas() if self.dist_base[c[1], c[0]] >= 0), key=lambda c: (c[1], c[0]))

    def celdas(self):
        return [(x, y) for y in range(self.alto) for x in range(self.ancho) if (x, y) not in self.obstaculos]

    def libre(self, c):
        return 0 <= c[0] < self.ancho and 0 <= c[1] < self.alto and c not in self.obstaculos

    def vecinos(self, c):
        return [(c[0] + dx, c[1] + dy) for dx, dy in DIRS if self.libre((c[0] + dx, c[1] + dy))]

    def bfs(self, origen):
        """Distancia (en pasos) desde el origen a cada celda; -1 si es inalcanzable."""
        d = np.full((self.alto, self.ancho), -1, dtype=int)
        d[origen[1], origen[0]] = 0
        cola = deque([origen])
        while cola:
            c = cola.popleft()
            for v in self.vecinos(c):
                if d[v[1], v[0]] < 0:
                    d[v[1], v[0]] = d[c[1], c[0]] + 1
                    cola.append(v)
        return d

    @classmethod
    def aleatoria(cls, ancho, alto, densidad, rng):
        """Coloca muebles rectangulares de 1x1 a 3x3 celdas hasta alcanzar (aprox.) la densidad pedida."""
        obst, objetivo, intentos = set(), round(densidad * ancho * alto), 0
        while len(obst) < objetivo and intentos < 2000:
            intentos += 1
            w, h = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            x, y = int(rng.integers(0, ancho - w + 1)), int(rng.integers(0, alto - h + 1))
            celdas = {(x + i, y + j) for i in range(w) for j in range(h)}
            if celdas & {(0, 0), (1, 0), (0, 1)} or len(obst | celdas) > objetivo + 2:
                continue
            obst |= celdas
        return cls(ancho, alto, obst)


def astar(hab, a, b):
    """Camino más corto de a a b (lista de celdas) y número de celdas expandidas; (None, n) si no hay camino."""
    if a == b:
        return [a], 1
    h = lambda c: abs(c[0] - b[0]) + abs(c[1] - b[1])
    abierto, g, padre, cerrado, expandidas = [(h(a), h(a), a)], {a: 0}, {}, set(), 0
    while abierto:
        _, _, c = heapq.heappop(abierto)
        if c in cerrado:
            continue
        cerrado.add(c)
        expandidas += 1
        if c == b:
            camino = [c]
            while camino[-1] != a:
                camino.append(padre[camino[-1]])
            return camino[::-1], expandidas
        for v in hab.vecinos(c):
            if g[c] + 1 < g.get(v, 1e9):
                g[v], padre[v] = g[c] + 1, c
                heapq.heappush(abierto, (g[v] + h(v), h(v), v))
    return None, expandidas


def bfs_camino(hab, a, b):
    """Búsqueda en anchura (sin heurística), para comparar cuántas celdas expande."""
    padre, cola, expandidas = {a: None}, deque([a]), 0
    while cola:
        c = cola.popleft()
        expandidas += 1
        if c == b:
            camino = [c]
            while padre[camino[-1]] is not None:
                camino.append(padre[camino[-1]])
            return camino[::-1], expandidas
        for v in hab.vecinos(c):
            if v not in padre:
                padre[v] = c
                cola.append(v)
    return None, expandidas


# =============================================================================
# TAREA 1: DISEÑAR EL COMPORTAMIENTO (RobotLimpiador con batería, basura y recarga)
# =============================================================================
class Varado(Exception):
    """El robot se quedó sin batería lejos de la base."""


class Inalcanzable(Exception):
    """Una celda no se puede limpiar ni siquiera saliendo de la base con la batería llena."""


def proyecto_robot_limpiador():
    """Completa la plantilla del tema: navegación con A*, batería segura, recarga y rutas de cobertura."""

    class RobotLimpiador(RobotAutonomo):
        def __init__(self, nombre, habitacion, politica="segura", verbose=False):
            super().__init__(nombre, verbose)
            self.hab = habitacion
            self.politica = politica            # "segura": regresa a tiempo; "ingenua": regresa con batería <= 20 como la plantilla
            self.areas_limpias = set()
            self.bateria = BATERIA_MAX
            self.capacidad_basura = CAPACIDAD_MAX
            self.posicion = list(habitacion.base)
            self.cepillo = True
            self.pasos = self.giros = self.recargas = 0
            self.varado = False
            self.ruta = [habitacion.base]
            self.nivel_bateria = [BATERIA_MAX]
            self.limpiar_area(*habitacion.base)

        # --- limpieza -------------------------------------------------------
        def limpiar_area(self, x, y):
            """Limpia una celda (si está sucia y hay batería y espacio en el depósito)."""
            if (x, y) in self.areas_limpias or self.bateria < COSTO_LIMPIEZA or self.capacidad_basura <= 0:
                return False
            self.bateria -= COSTO_LIMPIEZA
            self.capacidad_basura -= 1
            self.areas_limpias.add((x, y))
            return True

        # --- movimiento -----------------------------------------------------
        def _paso(self, c):
            dx, dy = c[0] - self.posicion[0], c[1] - self.posicion[1]
            angulo = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}[(dx, dy)]
            giro = (angulo - self.orientacion) % 360
            self.giros += (360 - giro if giro > 180 else giro) // 90
            self.orientacion = angulo
            self.bateria -= COSTO_PASO
            self.posicion = [c[0], c[1]]
            self.pasos += 1
            self.ruta.append(c)
            if self.cepillo:
                self.limpiar_area(*c)
            self.nivel_bateria.append(self.bateria)
            if self.bateria <= 0 and c != self.hab.base:
                self.varado = True
                raise Varado()

        def _puede_avanzar(self, c):
            """Política segura: avanzar sólo si después de entrar a c alcanza la batería para volver a la base."""
            if self.politica == "ingenua":
                return self.bateria > COSTO_PASO + COSTO_LIMPIEZA
            sucia = c not in self.areas_limpias
            if sucia and self.capacidad_basura <= 0:
                return False
            costo = COSTO_PASO + (COSTO_LIMPIEZA if sucia else 0)
            return self.bateria - costo >= self.hab.dist_base[c[1], c[0]] * COSTO_PASO

        def recargar(self):
            self.bateria, self.capacidad_basura = BATERIA_MAX, CAPACIDAD_MAX
            self.recargas += 1
            self.nivel_bateria.append(self.bateria)

        def volver_a_base(self):
            """Regresa por el camino más corto con el cepillo apagado y recarga (y vacía el depósito)."""
            self.cepillo = False
            try:
                camino, _ = astar(self.hab, tuple(self.posicion), self.hab.base)
                for c in camino[1:]:
                    self._paso(c)
            finally:
                self.cepillo = True
            self.recargar()

        def ir_a(self, destino):
            """Navega con A* hasta una celda; si la batería o el depósito no alcanzan, pasa por la base primero."""
            while tuple(self.posicion) != destino:
                camino, _ = astar(self.hab, tuple(self.posicion), destino)
                for c in camino[1:]:
                    if not self._puede_avanzar(c):
                        if tuple(self.posicion) == self.hab.base and self.bateria == BATERIA_MAX and self.capacidad_basura == CAPACIDAD_MAX:
                            raise Inalcanzable()
                        self.volver_a_base()
                        break
                    self._paso(c)
                    if self.politica == "ingenua" and self.bateria <= UMBRAL_TEMA and tuple(self.posicion) != self.hab.base:
                        self.volver_a_base()
                        break

        def navegar_a_destino(self, destino_x, destino_y):
            """Misma interfaz que RobotAutonomo, pero con A* sobre el mapa: llega siempre que exista camino."""
            if not self.hab.libre((destino_x, destino_y)) or self.hab.dist_base[destino_y, destino_x] < 0:
                return False
            self.ir_a((destino_x, destino_y))
            return True

        # --- cobertura ------------------------------------------------------
        def planificar_ruta_limpieza(self, ancho=None, alto=None, metodo="serpentina"):
            """Planifica una ruta de cobertura de toda la habitación, la ejecuta y vuelve a la base."""
            ruta = PLANIFICADORES[metodo](self.hab)
            try:
                for c in ruta[1:]:
                    self.ir_a(c)
                self.volver_a_base()
            except Varado:
                pass
            except Inalcanzable:
                pass
            return len(self.areas_limpias)

        def resumen(self):
            n = len(self.hab.limpiables)
            return {"cobertura": len(self.areas_limpias & set(self.hab.limpiables)) / n, "pasos": self.pasos,
                    "giros": self.giros, "recargas": self.recargas, "varado": self.varado}

    return RobotLimpiador


RobotLimpiador = proyecto_robot_limpiador()


# =============================================================================
# TAREA 3: OPTIMIZAR RUTAS (planificadores de cobertura)
# =============================================================================
def unir(hab, objetivos):
    """Ruta celda por celda que visita los objetivos en orden, usando A* entre cada par."""
    ruta = [objetivos[0]]
    for t in objetivos[1:]:
        camino, _ = astar(hab, ruta[-1], t)
        ruta += camino[1:]
    return ruta


def ruta_raster(hab):
    """Lo que hace la plantilla del tema: fila por fila, siempre de izquierda a derecha."""
    return unir(hab, [hab.base] + [c for c in hab.limpiables if c != hab.base])


def ruta_serpentina(hab):
    """Boustrophedon: las filas se recorren alternando el sentido, así no se regresa al inicio de cada fila."""
    objetivos = [hab.base]
    for y in range(hab.alto):
        fila = [c for c in hab.limpiables if c[1] == y and c != hab.base]
        objetivos += fila if y % 2 == 0 else fila[::-1]
    return unir(hab, objetivos)


def ruta_vecino(hab):
    """Voraz: siempre ir a la celda sucia más cercana (por pasos), y contar como limpias las que se cruzan."""
    actual, vistas, ruta = hab.base, {hab.base}, [hab.base]
    faltan = set(hab.limpiables) - vistas
    while faltan:
        d = hab.bfs(actual)
        meta = min(faltan, key=lambda c: (d[c[1], c[0]], c[1], c[0]))
        camino, _ = astar(hab, actual, meta)
        for c in camino[1:]:
            vistas.add(c)
            faltan.discard(c)
        ruta += camino[1:]
        actual = meta
    return ruta


def matriz_distancias(hab, celdas):
    idx = {c: i for i, c in enumerate(celdas)}
    D = np.zeros((len(celdas), len(celdas)), dtype=int)
    for i, c in enumerate(celdas):
        d = hab.bfs(c)
        D[i] = [d[v[1], v[0]] for v in celdas]
    return D


def mejorar_2opt(orden, D, max_pasadas=100):
    """2-opt sobre un camino abierto con el inicio fijo: invierte tramos mientras acorten el recorrido."""
    o = np.array(orden)
    n = len(o)
    for _ in range(max_pasadas):
        mejoro = False
        for i in range(1, n - 1):
            a, b = o[i - 1], o[i]
            j = np.arange(i + 1, n)
            sig = np.where(j < n - 1, o[np.minimum(j + 1, n - 1)], -1)
            quitado = D[a, b] + np.where(sig >= 0, D[o[j], np.maximum(sig, 0)], 0)
            puesto = D[a, o[j]] + np.where(sig >= 0, D[b, np.maximum(sig, 0)], 0)
            delta = puesto - quitado
            k = int(np.argmin(delta))
            if delta[k] < 0:
                o[i:j[k] + 1] = o[i:j[k] + 1][::-1].copy()
                mejoro = True
        if not mejoro:
            break
    return list(o)


def ruta_2opt(hab):
    """Parte de la ruta voraz y la mejora con 2-opt sobre las distancias reales (en pasos) entre celdas."""
    inicial = list(dict.fromkeys(ruta_vecino(hab)))          # orden de primera visita de cada celda
    D = matriz_distancias(hab, inicial)
    orden = mejorar_2opt(list(range(len(inicial))), D)
    return unir(hab, [inicial[i] for i in orden])


PLANIFICADORES = {"raster (plantilla)": ruta_raster, "serpentina": ruta_serpentina,
                  "vecino más cercano": ruta_vecino, "vecino + 2-opt": ruta_2opt}


def validar_ruta(hab, ruta):
    assert ruta[0] == hab.base
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and hab.libre(b) for a, b in zip(ruta, ruta[1:])), "pasos no válidos"
    assert set(ruta) >= set(hab.limpiables), "la ruta no cubre toda la habitación"


def limpieza_sin_mapa(hab, mapa_conocido=False):
    """Cobertura voraz con sensores: el robot sólo sabe que una celda vecina está ocupada cuando está a su lado.
    Con mapa_conocido=True hace lo mismo pero conociendo todos los muebles desde el inicio."""
    conocidos = set(hab.obstaculos) if mapa_conocido else set()
    pos, limpias, pasos, replanes = hab.base, {hab.base}, 0, 0

    def sentir(p):
        for dx, dy in DIRS:
            v = (p[0] + dx, p[1] + dy)
            if v in hab.obstaculos:
                conocidos.add(v)

    sentir(pos)
    while True:
        padre, cola, meta = {pos: None}, deque([pos]), None
        while cola:
            c = cola.popleft()
            if c not in limpias:
                meta = c
                break
            for dx, dy in DIRS:
                v = (c[0] + dx, c[1] + dy)
                if 0 <= v[0] < hab.ancho and 0 <= v[1] < hab.alto and v not in padre and v not in conocidos:
                    padre[v] = c
                    cola.append(v)
        if meta is None:
            break
        camino = [meta]
        while padre[camino[-1]] is not None:
            camino.append(padre[camino[-1]])
        for c in camino[::-1][1:]:
            if c in conocidos:
                replanes += 1
                break
            pos = c
            pasos += 1
            limpias.add(c)
            sentir(pos)
    return {"pasos": pasos, "limpias": len(limpias), "replanes": replanes}



# =============================================================================
# EXPERIMENTOS
# =============================================================================
def choques(hab, puntos):
    """Movimientos de una trayectoria reactiva que cruzan un mueble (círculo de radio 0.5) o salen de la habitación."""
    n_choques = fuera = 0
    for (px, py), (qx, qy) in zip(puntos, puntos[1:]):
        if not (-0.5 <= qx <= hab.ancho - 0.5 and -0.5 <= qy <= hab.alto - 0.5):
            fuera += 1
        L2 = (qx - px) ** 2 + (qy - py) ** 2
        for (cx, cy) in hab.obstaculos:
            t = 0 if L2 == 0 else max(0, min(1, ((cx - px) * (qx - px) + (cy - py) * (qy - py)) / L2))
            if math.hypot(px + t * (qx - px) - cx, py + t * (qy - py) - cy) < 0.5:
                n_choques += 1
                break
    return n_choques, fuera


def robot_reactivo(hab, clase=RobotAutonomo, inicio=(0, 0), *args):
    r = clase("tema", *args)
    r.posicion = [float(inicio[0]), float(inicio[1])]
    for (x, y) in hab.obstaculos:
        r.agregar_obstaculo(x, y, 0.5)
    return r


rng = np.random.default_rng(SEMILLA)
N_HAB = 30
TAMANOS = {"12x10": (12, 10), "20x15": (20, 15)}
DENSIDADES = (0.0, 0.10, 0.20)
habitaciones = {(t, d): [Habitacion.aleatoria(*TAMANOS[t], d, rng) for _ in range(N_HAB)] for t in TAMANOS for d in DENSIDADES}
print("Habitaciones generadas (30 por condición):")
filas = []
for (t, d), hs in habitaciones.items():
    filas.append({"tamaño": t, "densidad pedida": d, "densidad real": np.mean([len(h.obstaculos) / (h.ancho * h.alto) for h in hs]),
                  "celdas limpiables": np.mean([len(h.limpiables) for h in hs]),
                  "con celdas aisladas": np.mean([len(h.limpiables) < h.ancho * h.alto - len(h.obstaculos) for h in hs])})
print(pd.DataFrame(filas).round(3).to_string(index=False))

# =============================================================================
# PUNTO DE PARTIDA: LA PLANTILLA DEL TEMA
# =============================================================================
print("\n--- Reproducción del ejemplo de navegación del tema (obstáculos (3, 2) r = 1.5 y (1, 4) r = 1.0; destino (8, 6)) ---")
ej = RobotAutonomo("Navegador-3000")
ej.agregar_obstaculo(3, 2, 1.5)
ej.agregar_obstaculo(1, 4, 1.0)
llego = ej.navegar_a_destino(8, 6)
ciclo = [tuple(round(v, 2) for v in p) for p in ej.trayectoria]
print(f"¿Llegó? {llego}. Pasos dados: {len(ej.trayectoria)}. Posición final: ({ej.posicion[0]:.2f}, {ej.posicion[1]:.2f}). "
      f"Posiciones distintas visitadas: {len(set(ciclo))} de {len(ciclo)}")

print("\n--- Plantilla del tema sin modificar (RobotLimpiadorTema) en habitaciones de 12x10 ---")
filas = []
for d in DENSIDADES:
    res = []
    for hab in habitaciones[("12x10", d)][:20]:
        r = robot_reactivo(hab, RobotLimpiadorTema)
        limpiadas = r.planificar_ruta_limpieza(hab.ancho, hab.alto)
        c, f = choques(hab, [(0.0, 0.0)] + r.trayectoria)
        reales = len(r.areas_limpias & set(hab.limpiables))
        res.append({"cobertura real": reales / len(hab.limpiables), "marcadas en muebles": len(r.areas_limpias & set(hab.obstaculos)),
                    "choques": c, "salidas del cuarto": f, "batería final": r.bateria})
    filas.append({"densidad": d, **pd.DataFrame(res).mean().to_dict()})
print(pd.DataFrame(filas).round(3).to_string(index=False))

# =============================================================================
# TAREA 2: IMPLEMENTAR NAVEGACIÓN (evitar obstáculos)
# =============================================================================
print("\n--- Navegación reactiva del tema contra A* (200 pares inicio-destino por densidad, cuadrícula de 12x10) ---")
filas, DENS_NAV = [], (0.0, 0.10, 0.20, 0.30)
rng_nav = np.random.default_rng(SEMILLA + 1)
for d in DENS_NAV:
    llega = limpio = n = 0
    exp_a, exp_b = [], []
    for _ in range(25):
        hab = Habitacion.aleatoria(12, 10, d, rng_nav)
        for _ in range(8):
            a, b = [hab.limpiables[i] for i in rng_nav.choice(len(hab.limpiables), 2, replace=False)]
            n += 1
            r = robot_reactivo(hab, inicio=a)
            llego = r.navegar_a_destino(*b)
            c, f = choques(hab, [(float(a[0]), float(a[1]))] + r.trayectoria)
            llega += llego
            if llego and c == 0 and f == 0:
                limpio += 1
            ca, ea = astar(hab, a, b)
            cb, eb = bfs_camino(hab, a, b)
            assert len(ca) == len(cb)
            exp_a.append(ea)
            exp_b.append(eb)
    filas.append({"densidad": d, "reactivo llega": llega / n, "llega sin chocar": limpio / n,
                  "A* llega": 1.0, "celdas expandidas A*": np.mean(exp_a), "celdas expandidas BFS": np.mean(exp_b)})
nav = pd.DataFrame(filas)
print(nav.round(3).to_string(index=False))

# =============================================================================
# TAREA 3: OPTIMIZAR RUTAS
# =============================================================================
print("\n--- Rutas de cobertura (30 habitaciones por condición; política segura de batería) ---")
registros = []
for (t, d), hs in habitaciones.items():
    for k, hab in enumerate(hs):
        n = len(hab.limpiables)
        for metodo, planificar in PLANIFICADORES.items():
            t0 = time.time()
            ruta = planificar(hab)
            seg = time.time() - t0
            validar_ruta(hab, ruta)
            robot = RobotLimpiador("r", hab, "segura")
            robot.planificar_ruta_limpieza(metodo=metodo)
            registros.append({"tamaño": t, "densidad": d, "habitación": k, "método": metodo, "celdas": n, "ruta": len(ruta) - 1,
                              "exceso": (len(ruta) - 1) / (n - 1) - 1, "pasos totales": robot.pasos, "giros": robot.giros,
                              "recargas": robot.recargas, "cobertura": robot.resumen()["cobertura"], "segundos": seg})
res = pd.DataFrame(registros)
orden_metodos = list(PLANIFICADORES)
tab = res.groupby(["tamaño", "densidad", "método"]).agg(celdas=("celdas", "mean"), ruta=("ruta", "mean"), exceso=("exceso", "mean"),
                                                         pasos_totales=("pasos totales", "mean"), giros=("giros", "mean"),
                                                         recargas=("recargas", "mean"), cobertura=("cobertura", "mean"),
                                                         seg=("segundos", "mean")).reset_index()
ALIAS = {"raster (plantilla)": "raster", "serpentina": "serpentina", "vecino más cercano": "vecino", "vecino + 2-opt": "2-opt"}
tab["método"] = pd.Categorical(tab["método"], orden_metodos)
tab = tab.sort_values(["tamaño", "densidad", "método"])
tab["método"] = tab["método"].astype(str).map(ALIAS)
tab = tab.drop(columns="giros").rename(columns={"densidad": "dens."})
tab["exceso"] = (100 * tab["exceso"]).round(1)
tab = tab.rename(columns={"exceso": "exceso_%", "pasos_totales": "pasos_tot", "cobertura": "cobert."})
print(tab.round({"celdas": 1, "ruta": 1, "pasos_tot": 1, "recargas": 1, "cobert.": 3, "seg": 3}).to_string(index=False))

print("\n--- Diferencias pareadas por habitación (IC 95 % por bootstrap) ---")
rng_b = np.random.default_rng(SEMILLA)
filas = []
for medida, col in (("pasos de la ruta", "ruta"), ("pasos totales (con recargas)", "pasos totales")):
    piv = res.pivot_table(index=["tamaño", "densidad", "habitación"], columns="método", values=col)
    for t in TAMANOS:
        for a, b in (("raster (plantilla)", "serpentina"), ("serpentina", "vecino más cercano"), ("vecino más cercano", "vecino + 2-opt"), ("serpentina", "vecino + 2-opt")):
            dif = (piv.loc[t][a] - piv.loc[t][b]).values
            medias = [rng_b.choice(dif, len(dif)).mean() for _ in range(2000)]
            filas.append({"medida": medida, "tamaño": t, "comparación": f"{ALIAS[a]} - {ALIAS[b]}", "dif_media": dif.mean(),
                          "IC_inf": np.percentile(medias, 2.5), "IC_sup": np.percentile(medias, 97.5),
                          "2o_mejor": f"{(dif > 0).sum()}/{len(dif)}"})
print(pd.DataFrame(filas).round(2).to_string(index=False))

print("\n--- Batería: política de la plantilla (regresar con <= 20) contra política segura (serpentina) ---")
filas = []
for (t, d), hs in habitaciones.items():
    for pol in ("ingenua", "segura"):
        rs = []
        for hab in hs:
            robot = RobotLimpiador("r", hab, pol)
            robot.planificar_ruta_limpieza(metodo="serpentina")
            rs.append({**robot.resumen(), "máx. distancia a la base": hab.dist_base.max()})
        df = pd.DataFrame(rs)
        filas.append({"tamaño": t, "densidad": d, "política": pol, "varado (%)": 100 * df.varado.mean(),
                      "cobertura media": df.cobertura.mean(), "máx. distancia a la base": df["máx. distancia a la base"].mean()})
bat = pd.DataFrame(filas)
print(bat.round(2).to_string(index=False))

print("\n--- Sin mapa previo (el robot sólo siente las celdas contiguas) contra con mapa conocido, búsqueda voraz ---")
filas = []
for (t, d), hs in habitaciones.items():
    a = [limpieza_sin_mapa(h) for h in hs]
    b = [limpieza_sin_mapa(h, True) for h in hs]
    filas.append({"tamaño": t, "densidad": d, "pasos sin mapa": np.mean([x["pasos"] for x in a]),
                  "pasos con mapa": np.mean([x["pasos"] for x in b]), "replanificaciones": np.mean([x["replanes"] for x in a]),
                  "cobertura sin mapa": np.mean([x["limpias"] / len(h.limpiables) for x, h in zip(a, hs)])})
print(pd.DataFrame(filas).round(2).to_string(index=False))

# =============================================================================
# FIGURAS
# =============================================================================
COL = {"raster (plantilla)": "#E45756", "serpentina": "#4C78A8", "vecino más cercano": "#72B7B2", "vecino + 2-opt": "#54A24B"}
hab_ej = Habitacion.aleatoria(12, 10, 0.15, np.random.default_rng(3))
fig, ejes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, (metodo, planificar) in zip(ejes, PLANIFICADORES.items()):
    ruta = planificar(hab_ej)
    for (x, y) in hab_ej.obstaculos:
        ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color="#BAB0AC"))
    xs, ys = zip(*ruta)
    ax.plot(xs, ys, color=COL[metodo], lw=1.2, alpha=0.9)
    ax.plot(*hab_ej.base, marker="*", color="black", ms=13, zorder=5)
    ax.set_xlim(-.5, hab_ej.ancho - .5)
    ax.set_ylim(-.5, hab_ej.alto - .5)
    ax.set_aspect("equal")
    ax.set_xticks(range(hab_ej.ancho))
    ax.set_yticks(range(hab_ej.alto))
    ax.tick_params(labelsize=6)
    ax.set_title(f"{metodo}: {len(ruta) - 1} pasos", fontsize=10)
plt.tight_layout()
plt.savefig(f"{IMG}/rutas.png", dpi=200)
plt.close()

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
w = 0.38
x = np.arange(len(DENS_NAV))
ejes[0].bar(x - w / 2, nav["llega sin chocar"], w, label="reactivo del tema", color="#E45756")
ejes[0].bar(x + w / 2, nav["A* llega"], w, label="A*", color="#4C78A8")
ejes[0].set_xticks(x, [f"{int(100 * d)} %" for d in DENS_NAV])
ejes[0].set_xlabel("Muebles (% de la habitación)")
ejes[0].set_ylabel("Llega sin chocar")
ejes[0].set_ylim(0, 1.05)
ejes[0].legend()
ejes[1].bar(x - w / 2, nav["celdas expandidas BFS"], w, label="búsqueda en anchura", color="#BAB0AC")
ejes[1].bar(x + w / 2, nav["celdas expandidas A*"], w, label="A*", color="#4C78A8")
ejes[1].set_xticks(x, [f"{int(100 * d)} %" for d in DENS_NAV])
ejes[1].set_xlabel("Muebles (% de la habitación)")
ejes[1].set_ylabel("Celdas expandidas por camino")
ejes[1].legend()
plt.tight_layout()
plt.savefig(f"{IMG}/navegacion.png", dpi=200)
plt.close()

fig, ejes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, t in zip(ejes, TAMANOS):
    sub = res[res["tamaño"] == t]
    for i, m in enumerate(orden_metodos):
        g = sub[sub["método"] == m].groupby("densidad")["exceso"]
        ax.bar(np.arange(len(DENSIDADES)) + (i - 1.5) * 0.2, 100 * g.mean().values, 0.2, yerr=100 * g.std().values, capsize=2,
               label=m, color=COL[m])
    ax.set_xticks(range(len(DENSIDADES)), [f"{int(100 * d)} %" for d in DENSIDADES])
    ax.set_xlabel("Muebles (% de la habitación)")
    ax.set_title(f"Habitaciones de {t}")
ejes[0].set_ylabel("Pasos de más sobre el mínimo (%)")
ejes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG}/cobertura.png", dpi=200)
plt.close()

hab_b = habitaciones[("20x15", 0.10)][0]
robot_b = RobotLimpiador("r", hab_b, "segura")
robot_b.planificar_ruta_limpieza(metodo="serpentina")
robot_n = RobotLimpiador("r", hab_b, "ingenua")
robot_n.planificar_ruta_limpieza(metodo="serpentina")
fig, ejes = plt.subplots(1, 2, figsize=(12, 3.8))
ejes[0].plot(robot_b.nivel_bateria, color="#4C78A8", lw=1, label=f"segura ({robot_b.recargas} recargas)")
ejes[0].plot(robot_n.nivel_bateria, color="#E45756", lw=1, label=f"plantilla (varado: {robot_n.varado})")
ejes[0].axhline(UMBRAL_TEMA, color="gray", ls="--", lw=0.8)
ejes[0].set_xlabel("Paso")
ejes[0].set_ylabel("Batería")
ejes[0].legend(fontsize=8)
for i, pol in enumerate(("ingenua", "segura")):
    sub = bat[bat["política"] == pol]
    ejes[1].bar(np.arange(len(sub)) + (i - 0.5) * 0.4, sub["varado (%)"], 0.4, label="plantilla (<= 20)" if pol == "ingenua" else "segura",
                color="#E45756" if pol == "ingenua" else "#4C78A8")
ejes[1].set_xticks(range(len(sub)), [f"{t}\n{int(100 * d)} %" for t, d in zip(sub["tamaño"], sub["densidad"])], fontsize=8)
ejes[1].set_ylabel("Habitaciones con robot varado (%)")
ejes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG}/bateria.png", dpi=200)
plt.close()
print("\nListo. Figuras guardadas en", IMG)
