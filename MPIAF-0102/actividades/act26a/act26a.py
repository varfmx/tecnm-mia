# =============================================================================
# Actividad 26a: "Simulador de Enjambre de Robots"
# Tema 4.6. Robótica
#
# Dependencias (en Colab ya están):  pip install numpy pandas matplotlib
# No requiere descargas: las zonas de desastre se generan al azar (con semilla fija).
# Tarda unos minutos (ver el tiempo total al final de la ejecución).
# =============================================================================
import math
import os
import time
import warnings
from collections import Counter, deque
from dataclasses import dataclass

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
T_INICIO = time.time()

# --- Parámetros del escenario (fijados antes de evaluar) ---------------------
ANCHO = ALTO = 30        # zona de desastre de 30 x 30 celdas
DENSIDAD = 0.15          # fracción de celdas con escombros
N_VICTIMAS = 8
RADIO_SENSOR = 2         # cada robot observa el cuadrado de 5 x 5 celdas centrado en él
MAX_PASOS = 800          # tope de pasos por misión
N_MUNDOS = 30            # zonas distintas por condición (semillas SEMILLA, SEMILLA+1, ...)
SEMILLA_AJUSTE = 1000    # semillas distintas para elegir los parámetros del enjambre
PERIODO_BALIZA = 3       # cada cuántos pasos un robot anuncia su posición y objetivo
REINTENTO = 6            # cada cuántos pasos un robot sin fronteras vuelve a mirar su mapa
VIGENCIA = 15            # pasos durante los que se confía en lo que dijo otro robot
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


# =============================================================================
# ROBOT DEL TEMA (clase RobotMovil, sin pausas ni mensajes)
# =============================================================================
class RobotMovil:
    """Un robot móvil con ruedas que puede navegar en un espacio 2D (tal como en el tema)."""

    def __init__(self, nombre):
        self.nombre = nombre
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


# =============================================================================
# EL ENTORNO: ZONA DE DESASTRE CON ESCOMBROS Y VÍCTIMAS
# =============================================================================
class Mundo:
    """Cuadrícula con escombros (celdas intransitables), una base en (0, 0) y víctimas escondidas."""

    def __init__(self, ancho, alto, escombros, victimas):
        self.ancho, self.alto, self.base = ancho, alto, (0, 0)
        self.real = np.zeros((alto, ancho), dtype=int)       # 0 libre, 1 escombro
        for x, y in escombros:
            self.real[y, x] = 1
        self.victimas = list(victimas)
        self.alcanzable = self._alcanzables()
        self.n_alcanzables = int(self.alcanzable.sum())

    def _alcanzables(self):
        vis = np.zeros((self.alto, self.ancho), dtype=bool)
        vis[0, 0] = True
        cola = deque([(0, 0)])
        while cola:
            x, y = cola.popleft()
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.ancho and 0 <= ny < self.alto and not vis[ny, nx] and self.real[ny, nx] == 0:
                    vis[ny, nx] = True
                    cola.append((nx, ny))
        return vis

    @classmethod
    def aleatorio(cls, ancho, alto, densidad, n_victimas, rng):
        """Escombros rectangulares de 1x1 a 3x3 celdas hasta la densidad pedida; víctimas en celdas alcanzables."""
        obst, objetivo, intentos = set(), round(densidad * ancho * alto), 0
        while len(obst) < objetivo and intentos < 5000:
            intentos += 1
            w, h = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            x, y = int(rng.integers(0, ancho - w + 1)), int(rng.integers(0, alto - h + 1))
            celdas = {(x + i, y + j) for i in range(w) for j in range(h)}
            if celdas & {(0, 0), (1, 0), (0, 1)} or len(obst | celdas) > objetivo + 2:
                continue
            obst |= celdas
        m = cls(ancho, alto, obst, [])
        libres = [(x, y) for y in range(alto) for x in range(ancho) if m.alcanzable[y, x] and (x, y) != (0, 0)]
        idx = rng.choice(len(libres), n_victimas, replace=False)
        m.victimas = [libres[i] for i in idx]
        return m


class Registro:
    """Contabilidad global (la ve el simulador, no los robots): qué se ha observado y cuándo se halló a cada víctima."""

    def __init__(self, mundo):
        self.mundo = mundo
        self.obs = np.zeros((mundo.alto, mundo.ancho), dtype=bool)
        self.detectadas = {}

    def observar(self, x0, x1, y0, y1, t):
        sub = self.obs[y0:y1, x0:x1]
        nuevas = int((~sub).sum())
        sub[:] = True
        for v in self.mundo.victimas:
            if x0 <= v[0] < x1 and y0 <= v[1] < y1 and v not in self.detectadas:
                self.detectadas[v] = t
        return nuevas


# =============================================================================
# TAREA 2: COMUNICACIÓN. CANAL DE MENSAJES ENTRE ROBOTS
# =============================================================================
@dataclass
class Mensaje:
    tipo: str        # POSICION (baliza), MAPA (celdas nuevas) o VICTIMA (hallazgo urgente)
    de: int
    a: object        # id del destinatario o "*" (difusión a todo el que esté en alcance)
    carga: object
    t: int


class Canal:
    """Radio de alcance limitado: un mensaje llega a quien esté a distancia <= rc cuando se envía, con un paso de retraso."""

    def __init__(self, rc, traza=0):
        self.rc = rc                      # 0 = sin radio; 999 = alcance práctico infinito
        self.activo = rc != 0
        self.cuenta = Counter()
        self.celdas = 0
        self.en_vuelo = []
        self.traza, self.max_traza = [], traza

    def enviar(self, msg, emisor, robots):
        if not self.activo:
            return
        dest = [r for r in robots if r.vivo and r.id != emisor.id and (msg.a == "*" or msg.a == r.id)
                and math.dist(r.celda, emisor.celda) <= self.rc]
        self.cuenta[msg.tipo] += 1
        if msg.tipo in ("MAPA", "VICTIMA"):
            self.celdas += len(msg.carga)
        if len(self.traza) < self.max_traza:
            self.traza.append((msg, emisor.celda, [r.id for r in dest]))
        self.en_vuelo += [(r, msg) for r in dest]

    def entregar(self):
        pendientes, self.en_vuelo = self.en_vuelo, []
        for r, msg in pendientes:
            if r.vivo:
                r.recibir(msg)

    @property
    def total(self):
        return sum(self.cuenta.values())


# =============================================================================
# EL ROBOT DEL ENJAMBRE (extiende RobotMovil del tema)
# =============================================================================
class RobotEnjambre(RobotMovil):
    """Percibe 5x5 celdas, recuerda lo que vio (su propio mapa) y se mueve hacia la frontera que elija su política.
    Las subclases solo cambian cómo puntúan las fronteras (y si usan la radio)."""
    usa_radio = False

    def __init__(self, id, n_total, mundo, canal, registro, rng, robots):
        super().__init__(f"R{id}")
        self.id, self.n_total, self.mundo, self.canal, self.registro, self.rng = id, n_total, mundo, canal, registro, rng
        self.robots = robots
        self.celda = (0, 0)
        self.conocido = np.full((mundo.alto, mundo.ancho), -1, dtype=int)   # -1 desconocido, 0 libre, 1 escombro
        self.victimas_conocidas = set()
        self.log = []               # hechos aprendidos: (x, y, valor, de quién), valor 2 = hay una víctima
        self.enviado = {}           # hasta qué punto del log le mandé información a cada compañero
        self.otros = {}             # lo último que supe de cada compañero: (celda, objetivo, paso)
        self.escuchados = set()
        self.nuevas_victimas = []
        self.objetivo, self.ruta, self.ultimo_plan = None, [], -99
        self.vivo, self.ocioso = True, False
        self.pasos_utiles = self.movimientos = 0

    # ---- percepción -----------------------------------------------------------
    def aprender(self, x, y, v, de):
        if self.conocido[y, x] != -1:
            return
        self.conocido[y, x] = min(v, 1)
        if v == 2:
            self.victimas_conocidas.add((x, y))
        self.log.append((x, y, v, de))

    def percibir(self, t):
        x, y = self.celda
        x0, x1 = max(0, x - RADIO_SENSOR), min(self.mundo.ancho, x + RADIO_SENSOR + 1)
        y0, y1 = max(0, y - RADIO_SENSOR), min(self.mundo.alto, y + RADIO_SENSOR + 1)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if self.conocido[yy, xx] == -1:
                    v = 2 if (xx, yy) in self.mundo.victimas else int(self.mundo.real[yy, xx])
                    self.aprender(xx, yy, v, self.id)
                    if v == 2:
                        self.nuevas_victimas.append((xx, yy))
        if self.registro.observar(x0, x1, y0, y1, t) > 0:
            self.pasos_utiles += 1

    # ---- comunicación ---------------------------------------------------------
    def comunicar(self, t):
        if not (self.usa_radio and self.canal.activo):
            return
        if t % PERIODO_BALIZA == self.id % PERIODO_BALIZA:
            self.canal.enviar(Mensaje("POSICION", self.id, "*", (self.celda, self.objetivo), t), self, self.robots)
        if self.nuevas_victimas:
            carga = [(x, y, 2, self.id) for x, y in self.nuevas_victimas]
            self.canal.enviar(Mensaje("VICTIMA", self.id, "*", carga, t), self, self.robots)
        self.nuevas_victimas = []
        for j in sorted(self.escuchados):
            delta = [f for f in self.log[self.enviado.get(j, 0):] if f[3] != j]
            self.enviado[j] = len(self.log)
            if delta:
                self.canal.enviar(Mensaje("MAPA", self.id, j, delta, t), self, self.robots)
        self.escuchados = set()

    def recibir(self, msg):
        if msg.tipo == "POSICION":
            self.otros[msg.de] = (*msg.carga, msg.t)
            self.escuchados.add(msg.de)
        else:                                   # MAPA o VICTIMA: celdas nuevas para mí
            for x, y, v, _ in msg.carga:
                self.aprender(x, y, v, msg.de)

    # ---- planificación --------------------------------------------------------
    def bfs(self):
        libre = (self.conocido == 0).tolist()
        dist = [[-1] * self.mundo.ancho for _ in range(self.mundo.alto)]
        padre = {}
        x0, y0 = self.celda
        dist[y0][x0] = 0
        cola = deque([(x0, y0)])
        while cola:
            x, y = cola.popleft()
            d = dist[y][x] + 1
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.mundo.ancho and 0 <= ny < self.mundo.alto and libre[ny][nx] and dist[ny][nx] < 0:
                    dist[ny][nx] = d
                    padre[(nx, ny)] = (x, y)
                    cola.append((nx, ny))
        return np.array(dist), padre

    def es_frontera(self, c):
        x, y = c
        if self.conocido[y, x] != 0:
            return False
        return any(0 <= x + dx < self.mundo.ancho and 0 <= y + dy < self.mundo.alto and self.conocido[y + dy, x + dx] == -1
                   for dx, dy in DIRS)

    def fronteras(self):
        k = self.conocido
        desc = k == -1
        vec = np.zeros_like(desc)
        vec[1:, :] |= desc[:-1, :]
        vec[:-1, :] |= desc[1:, :]
        vec[:, 1:] |= desc[:, :-1]
        vec[:, :-1] |= desc[:, 1:]
        return (k == 0) & vec

    def pierde_conflicto(self, t):
        return False

    def puntaje(self, xs, ys, d, t):
        """Costo de cada frontera (menor = mejor). La política base: la más cercana."""
        return d.astype(float)

    def planificar(self, t):
        dist, padre = self.bfs()
        ys, xs = np.nonzero(self.fronteras() & (dist >= 0))
        self.ultimo_plan = t
        if len(xs) == 0:
            self.objetivo, self.ruta, self.ocioso = None, [], True
            return
        self.ocioso = False
        s = self.puntaje(xs, ys, dist[ys, xs], t) + self.rng.random(len(xs)) * 1e-3     # desempate al azar
        i = int(np.argmin(s))
        obj = (int(xs[i]), int(ys[i]))
        ruta = [obj]
        while ruta[-1] != self.celda:
            ruta.append(padre[ruta[-1]])
        self.objetivo, self.ruta = obj, ruta[::-1][1:]

    # ---- acción ---------------------------------------------------------------
    def mover_a(self, c):
        dx, dy = c[0] - self.celda[0], c[1] - self.celda[1]
        self.rotar(math.degrees(math.atan2(dy, dx)) - self.orientacion)
        self.mover_adelante(1.0)
        self.celda = c
        self.posicion = [float(c[0]), float(c[1])]
        self.movimientos += 1

    def decidir_y_mover(self, t):
        if self.ocioso and t - self.ultimo_plan < REINTENTO:
            return
        # compromiso: se conserva el objetivo hasta alcanzarlo (o hasta que deje de ser frontera o pierda un conflicto)
        if self.objetivo is None or not self.es_frontera(self.objetivo) or self.pierde_conflicto(t):
            self.planificar(t)
        if self.ruta:
            self.mover_a(self.ruta.pop(0))


class RobotAleatorio(RobotEnjambre):
    """Camina al azar entre las celdas libres vecinas. No planifica ni habla."""

    def decidir_y_mover(self, t):
        x, y = self.celda
        libres = [(x + dx, y + dy) for dx, dy in DIRS
                  if 0 <= x + dx < self.mundo.ancho and 0 <= y + dy < self.mundo.alto and self.conocido[y + dy, x + dx] == 0]
        if libres:
            self.mover_a(libres[int(self.rng.integers(len(libres)))])


class RobotIndependiente(RobotEnjambre):
    """Va a la frontera más cercana de SU mapa. Hay varios robots pero no se coordinan: es el control sin enjambre."""


class RobotBandas(RobotEnjambre):
    """Coordinación sin radio: cada robot tiene una franja horizontal fija (según su número) y prefiere fronteras en ella."""

    def puntaje(self, xs, ys, d, t):
        ylo = round(self.id * self.mundo.alto / self.n_total)
        yhi = round((self.id + 1) * self.mundo.alto / self.n_total) - 1
        fuera = np.maximum(0, np.maximum(ylo - ys, ys - yhi))
        return fuera * 1000.0 + d


class RobotMapaCompartido(RobotEnjambre):
    """Comunicación sin coordinación de objetivos: comparte lo que ve, pero cada robot sigue yendo a la frontera más cercana."""
    usa_radio = True


class RobotEnjambreCompleto(RobotMapaCompartido):
    """Comparte el mapa y además penaliza las fronteras cercanas a donde está o a donde va otro compañero."""
    SEP, PEN_OBJ, PEN_POS = 6, 1.5, 0.0     # elegidos con las semillas de ajuste

    def pierde_conflicto(self, t):
        """Subasta simple: si un compañero reclama (casi) mi mismo objetivo y está más cerca, lo cedo (desempata el menor id)."""
        for j, (celda, obj, tt) in self.otros.items():
            if obj is None or t - tt > VIGENCIA:
                continue
            if abs(obj[0] - self.objetivo[0]) + abs(obj[1] - self.objetivo[1]) <= 2:
                mia = abs(self.celda[0] - self.objetivo[0]) + abs(self.celda[1] - self.objetivo[1])
                suya = abs(celda[0] - obj[0]) + abs(celda[1] - obj[1])
                if (suya, j) < (mia, self.id):
                    return True
        return False

    def puntaje(self, xs, ys, d, t):
        s = d.astype(float)
        for celda, obj, tt in self.otros.values():
            if t - tt > VIGENCIA:
                continue
            if obj is not None:
                s += self.PEN_OBJ * np.maximum(0, self.SEP - (abs(xs - obj[0]) + abs(ys - obj[1])))
            s += self.PEN_POS * np.maximum(0, self.SEP - (abs(xs - celda[0]) + abs(ys - celda[1])))
        return s


ESTRATEGIAS = {
    "aleatorio": RobotAleatorio,
    "independiente": RobotIndependiente,
    "bandas fijas": RobotBandas,
    "mapa compartido": RobotMapaCompartido,
    "enjambre": RobotEnjambreCompleto,
}


# =============================================================================
# TAREA 3: ESTRATEGIA COLECTIVA. SIMULADOR DE LA MISIÓN
# =============================================================================
def simular(mundo, estrategia, n, rc=10, semilla=0, fallas=0, t_falla=15, max_pasos=MAX_PASOS, traza=0, clase=None):
    """Una misión: n robots salen de la base; termina cuando todas las víctimas fueron detectadas (por alguien)."""
    rng = np.random.default_rng(semilla)
    canal, reg, robots = Canal(rc, traza), Registro(mundo), []
    clase = clase or ESTRATEGIAS[estrategia]
    for i in range(n):
        robots.append(clase(i, n, mundo, canal, reg, np.random.default_rng(int(rng.integers(2 ** 31))), robots))
    caidos = set(int(i) for i in rng.choice(n, fallas, replace=False)) if fallas else set()
    t_todas = None
    for t in range(max_pasos):
        if t == t_falla:
            for i in caidos:
                robots[i].vivo = False
        vivos = [r for r in robots if r.vivo]
        if not vivos:
            break
        canal.entregar()
        for r in vivos:
            r.percibir(t)
        if len(reg.detectadas) == len(mundo.victimas):
            t_todas = t
            break
        for r in vivos:
            r.comunicar(t)
        for r in vivos:
            r.decidir_y_mover(t)
        if all(r.ocioso for r in vivos) and not canal.en_vuelo:
            break
    tiempos = sorted(reg.detectadas.values())
    pasos_robot = sum(r.movimientos for r in robots)
    nv = len(mundo.victimas)
    return {
        "completa": t_todas is not None,
        "t_todas": t_todas if t_todas is not None else max_pasos,
        "t_mitad": tiempos[nv // 2 - 1] if len(tiempos) >= nv // 2 else max_pasos,
        "halladas": len(reg.detectadas) / nv,
        "cobertura": float((reg.obs & mundo.alcanzable).sum()) / mundo.n_alcanzables,
        "redundancia": 1 - sum(r.pasos_utiles for r in robots) / max(1, pasos_robot),
        "mensajes": canal.total,
        "celdas_tx": canal.celdas,
        "pasos_robot": pasos_robot,
        "_robots": robots, "_canal": canal, "_registro": reg,
    }


def mundo_k(k, base=SEMILLA):
    return Mundo.aleatorio(ANCHO, ALTO, DENSIDAD, N_VICTIMAS, np.random.default_rng(base + k))


def evaluar(estrategia, n, rc, mundos, fallas=0, clase=None, base=SEMILLA):
    filas = []
    for k, m in enumerate(mundos):
        r = simular(m, estrategia, n, rc, semilla=base * 7 + k, fallas=fallas, clase=clase)
        filas.append({"mundo": k, **{a: b for a, b in r.items() if not a.startswith("_")}})
    return pd.DataFrame(filas)


def resumen(df):
    return {"compl. %": 100 * df.completa.mean(), "t_todas": df.t_todas.mean(), "desv": df.t_todas.std(),
            "mediana": df.t_todas.median(), "t_mitad": df.t_mitad.mean(),
            "redund. %": 100 * df.redundancia.mean(), "mensajes": df.mensajes.mean(), "celdas": df.celdas_tx.mean()}


# =============================================================================
# AJUSTE DE PARÁMETROS (con semillas distintas a las de la evaluación)
# =============================================================================
print("=" * 78)
print("AJUSTE de la penalización del enjambre (10 zonas con semillas distintas de las de evaluación)")
print("=" * 78)
mundos_aj = [mundo_k(k, SEMILLA_AJUSTE) for k in range(10)]
filas = []
for sep, po, pp in ((10, 0.0, 0.0), (10, 1.5, 0.0), (10, 1.5, 0.7), (10, 3.0, 0.0), (6, 1.5, 0.0), (6, 1.5, 0.7), (14, 1.5, 0.0), (14, 3.0, 0.7)):
    cls = type("Aj", (RobotEnjambreCompleto,), {"SEP": sep, "PEN_OBJ": po, "PEN_POS": pp})
    d4 = evaluar("enjambre", 4, 10, mundos_aj, clase=cls, base=SEMILLA_AJUSTE)
    d8 = evaluar("enjambre", 8, 10, mundos_aj, clase=cls, base=SEMILLA_AJUSTE)
    filas.append({"SEP": sep, "PEN_OBJ": po, "PEN_POS": pp, "t_todas (4 robots)": d4.t_todas.mean(),
                  "t_todas (8 robots)": d8.t_todas.mean(), "completa % (8)": 100 * d8.completa.mean()})
print(pd.DataFrame(filas).round(1).to_string(index=False))

# =============================================================================
# EXPERIMENTO 1: ESTRATEGIAS (4 robots, radio de 10 celdas)
# =============================================================================
print("\n" + "=" * 78)
print(f"EXPERIMENTO 1: cinco estrategias, 4 robots, radio 10, {N_MUNDOS} zonas de {ANCHO}x{ALTO} con {N_VICTIMAS} víctimas")
print("=" * 78)
mundos = [mundo_k(k) for k in range(N_MUNDOS)]
print(f"Celdas alcanzables por zona: media {np.mean([m.n_alcanzables for m in mundos]):.0f} de {ANCHO * ALTO}")
res1 = {e: evaluar(e, 4, 10, mundos) for e in ESTRATEGIAS}
tab1 = pd.DataFrame({e: resumen(df) for e, df in res1.items()}).T
print(tab1.round(1).to_string())

print("\nDiferencias pareadas por zona en t_todas (IC 95 % por bootstrap, 2000 remuestreos)")
rng_b = np.random.default_rng(SEMILLA)
filas = []
for a, b in (("independiente", "enjambre"), ("bandas fijas", "enjambre"), ("mapa compartido", "enjambre"),
             ("independiente", "mapa compartido"), ("independiente", "bandas fijas")):
    dif = (res1[a].t_todas - res1[b].t_todas).values
    medias = [rng_b.choice(dif, len(dif)).mean() for _ in range(2000)]
    filas.append({"comparación": f"{a} - {b}", "dif_media": dif.mean(), "IC_inf": np.percentile(medias, 2.5),
                  "IC_sup": np.percentile(medias, 97.5), "2o_mejor": f"{(dif > 0).sum()}/{len(dif)}"})
print(pd.DataFrame(filas).round(1).to_string(index=False))

# =============================================================================
# EXPERIMENTO 2: ESCALABILIDAD (cuántos robots)
# =============================================================================
print("\n" + "=" * 78)
print("EXPERIMENTO 2: escalabilidad con el número de robots (radio 10)")
print("=" * 78)
TAMS = (1, 2, 4, 8, 16)
filas = []
res2 = {}
for e in ("independiente", "bandas fijas", "enjambre"):
    for n in TAMS:
        df = res2[(e, n)] = evaluar(e, n, 10, mundos)
        filas.append({"estrategia": e, "robots": n, "t_todas": df.t_todas.mean(), "completa %": 100 * df.completa.mean(),
                      "redund. %": 100 * df.redundancia.mean(), "mensajes": df.mensajes.mean()})
esc = pd.DataFrame(filas)
t1 = {e: esc[(esc.estrategia == e) & (esc.robots == 1)].t_todas.iloc[0] for e in esc.estrategia.unique()}
esc["aceleración"] = [t1[e] / t for e, t in zip(esc.estrategia, esc.t_todas)]
esc["eficiencia"] = esc["aceleración"] / esc.robots
print(esc.round(2).to_string(index=False))

# =============================================================================
# EXPERIMENTO 3: ALCANCE DE LA RADIO (8 robots)
# =============================================================================
print("\n" + "=" * 78)
print("EXPERIMENTO 3: alcance de la radio del enjambre (8 robots)")
print("=" * 78)
RCS = (0, 3, 6, 10, 20, 999)
filas = []
res3 = {}
for rc in RCS:
    df = res3[rc] = evaluar("enjambre", 8, rc, mundos)
    filas.append({"alcance": rc, **resumen(df), })
com = pd.DataFrame(filas)
print(com.round(1).to_string(index=False))

# =============================================================================
# EXPERIMENTO 4: ROBUSTEZ (8 robots; fallan f robots en el paso 15)
# =============================================================================
print("\n" + "=" * 78)
print("EXPERIMENTO 4: robustez ante la pérdida de robots (8 robots; los caídos dejan de moverse y de hablar en el paso 15)")
print("=" * 78)
filas = []
res4 = {}
for e in ("independiente", "bandas fijas", "enjambre"):
    for f in (0, 2, 4, 6):
        df = res4[(e, f)] = evaluar(e, 8, 10, mundos, fallas=f)
        filas.append({"estrategia": e, "robots caídos": f, "completa %": 100 * df.completa.mean(), "halladas %": 100 * df.halladas.mean(),
                      "t_todas": df.t_todas.mean()})
rob = pd.DataFrame(filas)
print(rob.round(1).to_string(index=False))

# =============================================================================
# TRAZA DE MENSAJES DE UNA MISIÓN PEQUEÑA
# =============================================================================
print("\n" + "=" * 78)
print("TRAZA: primeros mensajes de una misión de 3 robots (zona 0, radio 10)")
print("=" * 78)
r = simular(mundos[0], "enjambre", 3, 10, semilla=SEMILLA * 7, traza=400)
canal = r["_canal"]
print(f"Misión completa en {r['t_todas']} pasos; mensajes por tipo: {dict(canal.cuenta)}; celdas transmitidas: {canal.celdas}")
vistos, mostrados = set(), []
for msg, desde, dest in canal.traza:
    if msg.tipo == "MAPA" and "MAPA" in vistos and len(mostrados) > 5:
        continue
    mostrados.append((msg, desde, dest))
    vistos.add(msg.tipo)
    if len(mostrados) >= 14:
        break
for msg, desde, dest in mostrados:
    if msg.tipo == "POSICION":
        c = f"posición {msg.carga[0]}, objetivo {msg.carga[1]}"
    else:
        c = f"{len(msg.carga)} celdas" + (f" (víctima en {[(x, y) for x, y, v, _ in msg.carga if v == 2]})" if msg.tipo == "VICTIMA" else "")
    print(f"t={msg.t:3d}  R{msg.de} -> {msg.a!s:>2}  {msg.tipo:<8}  {c}   [llega a {dest}]")
primera_victima = [(m, d, ds) for m, d, ds in canal.traza if m.tipo == "VICTIMA"]
if primera_victima:
    m, d, ds = primera_victima[0]
    print(f"Primer aviso de víctima: t={m.t}, R{m.de}, llega a {ds}")

# =============================================================================
# FIGURAS
# =============================================================================
COL = {"aleatorio": "#BAB0AC", "independiente": "#E45756", "bandas fijas": "#F58518", "mapa compartido": "#72B7B2", "enjambre": "#4C78A8"}
COLR = ["#4C78A8", "#E45756", "#54A24B", "#F58518"]


def dibujar_mision(ax, m, r, titulo):
    cmap = np.where(m.real == 1, 0.75, 1.0)
    ax.imshow(cmap, origin="lower", cmap="gray", vmin=0, vmax=1, extent=(-.5, m.ancho - .5, -.5, m.alto - .5))
    obs = r["_registro"].obs
    ax.imshow(np.where(obs & (m.real == 0), 0.0, np.nan), origin="lower", cmap="Blues", vmin=0, vmax=2.5, alpha=0.35,
              extent=(-.5, m.ancho - .5, -.5, m.alto - .5))
    for i, rob in enumerate(r["_robots"]):
        xs, ys = [0] + [p[0] for p in rob.trayectoria], [0] + [p[1] for p in rob.trayectoria]
        ax.plot(xs, ys, lw=1.1, color=COLR[i % 4], alpha=0.9)
        ax.plot(xs[-1], ys[-1], "o", ms=4, color=COLR[i % 4])
    for v in m.victimas:
        ax.plot(*v, marker="*", ms=11, color="#D62728" if v in r["_registro"].detectadas else "black", mec="black", mew=0.5, zorder=5)
    ax.plot(0, 0, marker="s", ms=8, color="black")
    ax.set_xlim(-.5, m.ancho - .5)
    ax.set_ylim(-.5, m.alto - .5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titulo, fontsize=9)


m_ej = mundos[1]
fig, ejes = plt.subplots(1, 3, figsize=(13, 4.6))
for ax, e in zip(ejes, ("independiente", "bandas fijas", "enjambre")):
    r = simular(m_ej, e, 4, 10, semilla=SEMILLA * 7 + 1)
    dibujar_mision(ax, m_ej, r, f"{e}: {r['t_todas']} pasos, {r['redundancia'] * 100:.0f} % de pasos redundantes")
plt.tight_layout()
plt.savefig(f"{IMG}/misiones.png", dpi=200)
plt.close()

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
orden = list(ESTRATEGIAS)
datos = [res1[e].t_todas.values for e in orden]
bp = ejes[0].boxplot(datos, patch_artist=True, medianprops=dict(color="black"))
ejes[0].set_xticklabels([e.replace(" ", "\n") for e in orden])
for p, e in zip(bp["boxes"], orden):
    p.set_facecolor(COL[e])
ejes[0].set_ylabel("Pasos hasta hallar a las 8 víctimas")
ejes[0].axhline(MAX_PASOS, color="gray", ls="--", lw=0.8)
w = 0.38
x = np.arange(len(orden))
ejes[1].bar(x - w / 2, [100 * res1[e].redundancia.mean() for e in orden], w, color=[COL[e] for e in orden])
ejes[1].set_xticks(x, [e.replace(" ", "\n") for e in orden])
ejes[1].set_ylabel("Pasos de robot sin descubrir nada nuevo (%)")
plt.tight_layout()
plt.savefig(f"{IMG}/estrategias.png", dpi=200)
plt.close()

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
for e in ("independiente", "bandas fijas", "enjambre"):
    sub = esc[esc.estrategia == e]
    ejes[0].plot(sub.robots, sub.t_todas, marker="o", label=e, color=COL[e])
    ejes[1].plot(sub.robots, sub["aceleración"], marker="o", label=e, color=COL[e])
ejes[1].plot(TAMS, TAMS, ls="--", color="gray", lw=0.8, label="ideal")
for ax in ejes:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(TAMS, [str(t) for t in TAMS])
    ax.set_xlabel("Número de robots")
    ax.legend(fontsize=8)
ejes[0].set_ylabel("Pasos hasta hallar a las 8 víctimas")
ejes[1].set_ylabel("Aceleración respecto a un solo robot")
plt.tight_layout()
plt.savefig(f"{IMG}/escalabilidad.png", dpi=200)
plt.close()

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
etq = [("∞" if rc == 999 else str(rc)) for rc in RCS]
ejes[0].plot(range(len(RCS)), com.t_todas, marker="o", color=COL["enjambre"], label="pasos hasta hallar a todas")
ejes[0].set_xticks(range(len(RCS)), etq)
ejes[0].set_xlabel("Alcance de la radio (celdas)")
ejes[0].set_ylabel("Pasos hasta hallar a las 8 víctimas")
ax2 = ejes[0].twinx()
ax2.plot(range(len(RCS)), com.mensajes, marker="s", ls="--", color="#F58518", label="mensajes")
ax2.set_ylabel("Mensajes por misión")
for e in ("independiente", "bandas fijas", "enjambre"):
    sub = rob[rob.estrategia == e]
    ejes[1].plot(sub["robots caídos"], sub.t_todas, marker="o", label=e, color=COL[e])
ejes[1].set_xticks((0, 2, 4, 6))
ejes[1].set_xlabel("Robots caídos de 8 (en el paso 15)")
ejes[1].set_ylabel("Pasos hasta hallar a las 8 víctimas")
ejes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG}/radio_robustez.png", dpi=200)
plt.close()
print(f"\nListo en {time.time() - T_INICIO:.0f} s. Figuras guardadas en {IMG}")
