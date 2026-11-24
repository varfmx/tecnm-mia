# =============================================================================
# Actividad 17: Proyecto "Chatbot de Atención al Cliente Básico"
# Tema 4.1. Procesamiento del lenguaje natural
#
# Este archivo contiene el motor del chatbot (intenciones, comprensión de
# lenguaje, contexto y fallback) y su evaluación. La interfaz web está en app.py:
#   streamlit run app.py
#
# Dependencias (en Colab anteponer "!"):  pip install -U nltk scikit-learn pandas matplotlib seaborn streamlit
# =============================================================================
import os
import re
import unicodedata
from dataclasses import dataclass, field

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

try:
    stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords", quiet=True)

AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
IMG = os.path.join(AQUI, "img")


# =============================================================================
# TAREA 1: DISEÑO DE INTENCIONES
# Cada intención tiene: patrones (expresiones regulares), ejemplos de enunciados
# (para la similitud TF-IDF) y respuestas. Empresa ficticia: TecnoMIA (tienda
# en línea de electrónica).
# =============================================================================
INTENCIONES = {
    "saludo": {
        "patrones": [r"^(hola+|ola|buenas?|buenos dias|buen dia|buenas tardes|buenas noches|que tal|hey|saludos)\b"],
        "ejemplos": ["hola", "buenos días", "buenas tardes", "buenas noches", "qué tal", "hey",
                     "saludos", "hola necesito ayuda", "hola, ¿me pueden ayudar?"],
        "respuestas": ["¡Hola! Soy el asistente de TecnoMIA. ¿En qué puedo ayudarte hoy?",
                       "¡Hola de nuevo! ¿Con qué te ayudo?"],
    },
    "despedida": {
        "patrones": [r"\b(adios|hasta (luego|pronto|manana)|nos vemos|chao|bye|me despido|eso es todo|es todo por ahora)\b"],
        "ejemplos": ["adiós", "hasta luego", "hasta pronto", "nos vemos", "chao", "bye",
                     "me despido", "eso es todo, gracias", "ya no necesito nada más"],
        "respuestas": ["¡Hasta pronto! Si necesitas algo más, aquí estaré.",
                       "¡Que tengas un excelente día!"],
    },
    "agradecimiento": {
        "patrones": [r"\b(gracias|te agradezco|se agradece|muy amable)\b"],
        "ejemplos": ["gracias", "muchas gracias", "te lo agradezco", "muy amable",
                     "gracias por tu ayuda", "excelente servicio, gracias", "perfecto, gracias"],
        "respuestas": ["¡Con gusto! ¿Hay algo más en lo que pueda ayudarte?",
                       "Para eso estoy. ¿Necesitas algo más?"],
    },
    "horario": {
        "patrones": [r"\bhorarios?\b", r"\bhoras? de (atencion|servicio)\b",
                     r"\ba que hora (abren|cierran|atienden)\b", r"\b(abren|cierran|abierto|abiertos)\b"],
        "ejemplos": ["¿cuál es su horario?", "¿a qué hora abren?", "¿a qué hora cierran?",
                     "horario de atención", "¿están abiertos los sábados?",
                     "¿atienden en domingo?", "¿qué días trabajan?", "¿hasta qué hora atienden?"],
        "respuestas": ["Atendemos de lunes a viernes de 9:00 a 19:00 h y los sábados de 10:00 a 14:00 h. "
                       "El chat en línea funciona las 24 horas."],
    },
    "envio_tiempo": {
        "patrones": [r"\b(envio|envios|entrega|entregan|paquete|compra|pedido)\b"
                     r".*\b(tarda|tardan|demora|demoran|dias|tiempo|llega|llegar)\b",
                     r"\b(tarda|tardan|demora|demoran)\b.*\b(envio|entrega|paquete)\b"],
        "ejemplos": ["¿cuánto tarda el envío?", "¿cuántos días tarda en llegar mi pedido?",
                     "¿en cuánto tiempo llega mi compra?", "tiempo de entrega",
                     "¿cuándo me llega el paquete?", "¿cuánto demora la entrega?",
                     "¿qué tan rápido es el envío?", "¿hacen envíos express?"],
        "respuestas": ["El envío estándar tarda de 3 a 5 días hábiles y el envío express de 24 a 48 horas "
                       "(zonas metropolitanas)."],
    },
    "envio_costo": {
        "patrones": [r"\b(envio|envios|entrega|enviar|domicilio)\b"
                     r".*\b(cuesta|costo|precio|gratis|cobran|cobro|tarifa)\b",
                     r"\b(cuesta|costo|cobran|gratis)\b.*\b(envio|enviar|entrega)\b"],
        "ejemplos": ["¿cuánto cuesta el envío?", "¿el envío es gratis?", "costo de envío",
                     "¿cobran por la entrega?", "¿tiene costo el envío a domicilio?",
                     "¿cuánto cobran por enviar?", "tarifa de envío", "envío gratis"],
        "respuestas": ["El envío estándar cuesta $99 MXN y es gratis en compras mayores a $999 MXN. "
                       "El envío express cuesta $199 MXN."],
    },
    "rastreo": {
        "patrones": [r"\b(rastrear|rastreo|seguimiento|guia)\b", r"\bdonde (esta|viene|va) (mi|el) (pedido|paquete|compra)\b",
                     r"\bestado de (mi )?(pedido|compra|paquete)\b"],
        "ejemplos": ["¿dónde está mi pedido?", "quiero rastrear mi paquete", "estado de mi pedido",
                     "número de guía", "seguimiento de mi compra", "¿ya salió mi pedido?",
                     "quiero saber cómo va mi compra", "no me ha llegado mi paquete"],
        "respuestas": ["Con gusto reviso tu pedido."],   # se completa con el número de pedido
    },
    "devolucion": {
        "patrones": [r"\b(devolver|devolucion|devoluciones|reembolso|"
                     r"regresar|devuelvo|devuelven)\b",
                     r"\bcambiar(lo)?\b.*\b(producto|articulo)\b"],
        "ejemplos": ["quiero devolver un producto", "¿cómo hago una devolución?", "necesito un reembolso",
                     "¿puedo regresar mi compra?", "política de devoluciones", "quiero cambiar mi producto",
                     "no me gustó lo que compré, ¿qué hago?", "¿me devuelven mi dinero?"],
        "respuestas": ["Puedes devolver cualquier producto sin uso dentro de los 30 días posteriores a la entrega. "
                       "Solicítalo en Mis pedidos > Devolver y el reembolso se aplica en 5 a 7 días hábiles."],
    },
    "pago": {
        "patrones": [r"\b(pago|pagos|pagar|tarjeta|paypal|transferencia|"
                     r"oxxo|efectivo|meses sin intereses|msi)\b"],
        "ejemplos": ["¿qué formas de pago aceptan?", "¿puedo pagar con tarjeta?", "¿aceptan paypal?",
                     "¿tienen meses sin intereses?", "¿se puede pagar en oxxo?", "métodos de pago",
                     "¿puedo pagar con transferencia?", "¿cómo pago mi pedido?"],
        "respuestas": ["Aceptamos tarjetas de crédito y débito, PayPal, transferencia y pago en OXXO. "
                       "Con tarjetas participantes ofrecemos 3, 6 y 12 meses sin intereses."],
    },
    "garantia": {
        "patrones": [r"\bgarantia\b", r"\b(defectuoso|descompuso|falla|fallando|no funciona|roto)\b"],
        "ejemplos": ["¿tienen garantía?", "mi producto llegó defectuoso", "el celular no funciona",
                     "se descompuso mi licuadora", "¿cuánto dura la garantía?", "quiero hacer válida la garantía",
                     "el producto falla", "llegó roto"],
        "respuestas": ["Todos los productos tienen 12 meses de garantía por defectos de fábrica. "
                       "Para hacerla válida conserva tu comprobante y escríbenos a garantias@tecnomia.example."],
    },
    "cancelar": {
        "patrones": [r"\b(cancelar|cancelo|cancelacion|anular)\b"],
        "ejemplos": ["quiero cancelar mi pedido", "¿cómo cancelo mi compra?", "necesito anular mi pedido",
                     "cancelar pedido", "me arrepentí de mi compra", "ya no quiero mi pedido",
                     "¿puedo cancelar una compra?"],
        "respuestas": ["Puedes cancelar tu pedido mientras no haya sido enviado, desde Mis pedidos > Cancelar. "
                       "Si ya salió, solicita una devolución al recibirlo."],
    },
    "cuenta": {
        "patrones": [r"\b(contrasena|password|iniciar sesion|registrarme|crear (una )?cuenta|mi cuenta)\b"],
        "ejemplos": ["olvidé mi contraseña", "no puedo iniciar sesión", "¿cómo creo una cuenta?",
                     "quiero registrarme", "problemas con mi cuenta", "cambiar mi contraseña",
                     "no puedo entrar a mi cuenta", "¿cómo me registro?"],
        "respuestas": ["Para recuperar tu contraseña usa «¿Olvidaste tu contraseña?» en la pantalla de inicio de sesión; "
                       "te llegará un enlace al correo registrado. Para crear una cuenta, pulsa «Registrarme»."],
    },
    "promociones": {
        "patrones": [r"\b(promocion|promociones|promo|descuento|descuentos|"
                     r"oferta|ofertas|cupon|rebaja)\b"],
        "ejemplos": ["¿tienen promociones?", "¿hay descuentos?", "¿qué ofertas tienen?", "cupón de descuento",
                     "¿hay algún cupón?", "¿están en oferta?", "promociones de la semana", "quiero un descuento"],
        "respuestas": ["Esta semana hay 15 % de descuento en audífonos y bocinas. En tu primera compra "
                       "usa el cupón BIENVENIDA10 para obtener 10 % adicional."],
    },
    "humano": {
        "patrones": [r"\b(agente|asesor|humano|persona real|operador|hablar con (alguien|una persona))\b"],
        "ejemplos": ["quiero hablar con un agente", "comunícame con un asesor", "necesito hablar con una persona",
                     "atención humana", "quiero hablar con alguien", "pásame con un operador",
                     "¿puedo hablar con una persona real?"],
        "respuestas": ["Te transfiero con un agente. Su tiempo de espera es de unos 5 minutos "
                       "(lunes a viernes, 9:00 a 19:00 h). También puedes escribir a ayuda@tecnomia.example."],
    },
    "costo_generico": {   # ambigua: se resuelve con el contexto de la conversación
        "patrones": [r"\b(cuanto (cuesta|cuestan|vale|valen|cobran)|precio|costo)\b"],
        "ejemplos": ["¿cuánto cuesta?", "¿cuál es el precio?", "¿y el costo?", "¿cuánto vale?"],
        "respuestas": ["¿El costo de qué? Puedo informarte sobre el costo de envío, o decirte qué "
                       "promociones hay vigentes."],
    },
}

# Palabra ancla de cada intención: sirve para dar contexto a preguntas de seguimiento
TEMAS = {"envio_tiempo": "envío", "envio_costo": "envío", "rastreo": "pedido",
         "devolucion": "devolución", "pago": "pago", "garantia": "garantía",
         "cancelar": "pedido", "cuenta": "cuenta", "promociones": "promoción",
         "horario": "horario"}

# Base de pedidos simulada para la intención de rastreo
PEDIDOS = {"TM-1001": "En camino: salió del almacén y llega mañana antes de las 18:00 h.",
           "TM-1002": "Entregado el pasado lunes.",
           "TM-1003": "En preparación: se enviará en las próximas 24 horas.",
           "TM-1004": "Con retraso por clima adverso; la nueva fecha estimada es en 2 días."}


# =============================================================================
# TAREA 2: PROCESAMIENTO DE CONSULTAS
# =============================================================================
STEMMER = SnowballStemmer("spanish")


def quitar_acentos(t):
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


# Las palabras interrogativas y los negadores sí aportan información en un chatbot
# (se descartan las genéricas «qué», «cómo», «cuál», «por»..., que producen similitudes falsas)
_CONSERVAR = {"no", "nada", "sin", "cuando", "donde", "cuanto", "cuanta", "cuantos"}
STOP = {quitar_acentos(w) for w in stopwords.words("spanish")} - _CONSERVAR


def normalizar(texto):
    """Minúsculas, sin acentos ni puntuación (se usa para las expresiones regulares)."""
    t = quitar_acentos(texto.lower())
    t = re.sub(r"[^\w\s-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokenizar(texto):
    """Normaliza, quita stopwords y aplica stemming (se usa para TF-IDF)."""
    toks = [w for w in normalizar(texto).split() if w not in STOP]
    toks = toks or normalizar(texto).split()
    return " ".join(STEMMER.stem(w) for w in toks)


AFIRMACION = re.compile(r"^(si|claro|ok|okay|vale|por favor|dale|sip|afirmativo|esta bien)\b")
NEGACION = re.compile(r"^(no|nop|nel|negativo|no gracias|ahorita no)\b")
SEGUIMIENTO = re.compile(r"^(y|e|entonces|ademas|tambien|y si|pero)\b")
RE_PEDIDO =re.compile(r"\b(?:tm-?)?(\d{4})\b", re.I)
RE_NOMBRE = re.compile(r"\b(?:me llamo|mi nombre es|soy)\s+([a-záéíóúñ]+)", re.I)


class ModeloIntenciones:
    """Clasificador de intenciones: (a) patrones regex, (b) similitud coseno con TF-IDF."""

    def __init__(self, intenciones=INTENCIONES):
        self.intenciones = intenciones
        self.regex = {i: [re.compile(p) for p in d["patrones"]] for i, d in intenciones.items()}
        self.ejemplos, self.etiquetas = [], []
        for i, d in intenciones.items():
            for e in d["ejemplos"]:
                self.ejemplos.append(tokenizar(e))
                self.etiquetas.append(i)
        self.etiquetas = np.array(self.etiquetas)
        self.vectorizador = TfidfVectorizer(ngram_range=(1, 2))
        self.matriz = self.vectorizador.fit_transform(self.ejemplos)

    def por_patrones(self, texto):
        """Devuelve (intención, puntaje). Un patrón compuesto (con '.*') pesa 2."""
        t = normalizar(texto)
        puntajes = {}
        for i, pats in self.regex.items():
            p = sum(2 if ".*" in r.pattern else 1 for r in pats if r.search(t))
            if p:
                puntajes[i] = p
        if not puntajes:
            return None, 0
        mejor = max(puntajes, key=puntajes.get)
        return mejor, puntajes[mejor]

    def por_similitud(self, texto):
        """Devuelve (intención, similitud máxima) y el ranking de intenciones."""
        consulta = self.vectorizador.transform([tokenizar(texto)])
        sims = cosine_similarity(consulta, self.matriz)[0]
        ranking = {}
        for e, s in zip(self.etiquetas, sims):
            ranking[e] = max(ranking.get(e, 0.0), s)
        ranking = dict(sorted(ranking.items(), key=lambda kv: -kv[1]))
        mejor = next(iter(ranking))
        return mejor, ranking[mejor], ranking

    def clasificar(self, texto, umbral=0.45, metodo="hibrido"):
        """metodo: 'patrones', 'tfidf' o 'hibrido'. Devuelve (intención|None, confianza, método usado)."""
        if metodo in ("patrones", "hibrido"):
            i, p = self.por_patrones(texto)
            if i:
                return i, min(0.6 + 0.2 * p, 1.0), "patrones"
            if metodo == "patrones":
                return None, 0.0, "patrones"
        i, s, _ = self.por_similitud(texto)
        return (i, s, "tfidf") if s >= umbral else (None, s, "tfidf")


@dataclass
class Turno:
    usuario: str
    respuesta: str
    intencion: str
    confianza: float
    metodo: str


@dataclass
class Chatbot:
    """Chatbot con memoria de la conversación (contexto) y política de fallback."""
    modelo: ModeloIntenciones = field(default_factory=ModeloIntenciones)
    umbral: float = 0.45
    nombre: str = ""
    ultimo_tema: str = ""            # tema de la última intención (para preguntas de seguimiento)
    esperando: str = ""              # 'pedido' si el bot pidió un número de pedido
    pendiente: str = ""              # intención propuesta que espera un sí/no
    fallos_seguidos: int = 0
    historial: list = field(default_factory=list)
    _contador: dict = field(default_factory=dict)

    # -- utilidades ------------------------------------------------------------
    def _respuesta(self, intencion):
        r = INTENCIONES[intencion]["respuestas"]
        n = self._contador.get(intencion, 0)
        self._contador[intencion] = n + 1
        return r[n % len(r)]

    def _menu(self):
        return ("Puedo ayudarte con: envíos, rastreo de pedidos, devoluciones, formas de pago, "
                "garantía, cancelaciones, tu cuenta, promociones y horarios.")

    def _consultar_pedido(self, numero):
        clave = f"TM-{numero}"
        estado = PEDIDOS.get(clave)
        if estado:
            return f"Pedido {clave}. {estado}"
        return f"No encuentro el pedido {clave}. Verifica el número (formato TM-1234) o escribe «agente»."

    def _registrar(self, texto, resp, intencion, conf, metodo):
        self.historial.append(Turno(texto, resp, intencion, round(float(conf), 3), metodo))
        return resp

    # -- comprensión -----------------------------------------------------------
    def entender(self, texto):
        """Clasificación con contexto. Devuelve (intención|None, confianza, método)."""
        intencion, conf, metodo = self.modelo.clasificar(texto, self.umbral)
        # Contexto: una pregunta ambigua («¿cuánto cuesta?») o de seguimiento («¿y el pago?»)
        # se interpreta con el último tema de la conversación
        seguimiento = bool(SEGUIMIENTO.match(normalizar(texto))) or intencion == "costo_generico"
        if self.ultimo_tema and seguimiento and intencion in (None, "costo_generico"):
            i2, c2, _ = self.modelo.clasificar(f"{texto} {self.ultimo_tema}", self.umbral)
            if i2 and i2 != "costo_generico":
                return i2, c2, "contexto"
        return intencion, conf, metodo

    # -- diálogo ----------------------------------------------------------------
    def responder(self, texto):
        norm = normalizar(texto)
        m = RE_NOMBRE.search(texto)
        if m:
            self.nombre = m.group(1).capitalize()
        pedido = RE_PEDIDO.search(texto)

        # 1) El bot esperaba un número de pedido
        if self.esperando == "pedido":
            if pedido:
                self.esperando = ""
                self.fallos_seguidos = 0
                return self._registrar(texto, self._consultar_pedido(pedido.group(1)), "rastreo", 1.0, "slot")
            self.esperando = ""        # el usuario cambió de tema: se procesa normalmente

        # 2) El bot esperaba un sí/no
        if self.pendiente:
            propuesta, self.pendiente = self.pendiente, ""
            if AFIRMACION.match(norm):
                self.fallos_seguidos = 0
                return self._registrar(texto, self._respuesta(propuesta), propuesta, 1.0, "confirmacion")
            if NEGACION.match(norm):
                return self._registrar(texto, "De acuerdo. " + self._menu(), "negacion", 1.0, "confirmacion")

        # 3) Clasificación de la intención
        intencion, conf, metodo = self.entender(texto)

        if intencion is None:
            return self._fallback(texto, conf)

        self.fallos_seguidos = 0
        if intencion in TEMAS:
            self.ultimo_tema = TEMAS[intencion]

        if intencion == "rastreo":
            if pedido:
                resp = self._consultar_pedido(pedido.group(1))
            else:
                self.esperando = "pedido"
                resp = "Con gusto. ¿Cuál es tu número de pedido? (formato TM-1234)"
        elif intencion == "saludo" and self.nombre:
            resp = f"¡Hola, {self.nombre}! ¿En qué puedo ayudarte?"
        else:
            resp = self._respuesta(intencion)
        return self._registrar(texto, resp, intencion, conf, metodo)

    def _fallback(self, texto, conf):
        """Fallback escalonado: reformular -> sugerir opciones -> ofrecer un agente humano."""
        self.fallos_seguidos += 1
        if self.fallos_seguidos == 1:
            resp = "Lo siento, no estoy seguro de haber entendido. ¿Podrías decirlo con otras palabras?"
        elif self.fallos_seguidos == 2:
            resp = "Sigo sin entender tu consulta. " + self._menu()
        else:
            self.pendiente = "humano"
            resp = "Parece que no logro ayudarte con esto. ¿Quieres que te transfiera con un agente humano?"
        return self._registrar(texto, resp, "fallback", conf, "fallback")

    def reiniciar(self):
        self.nombre = self.ultimo_tema = self.esperando = self.pendiente = ""
        self.fallos_seguidos = 0
        self.historial.clear()
        self._contador.clear()


# =============================================================================
# EVALUACIÓN (conjunto de prueba escrito aparte de los ejemplos de entrenamiento)
# =============================================================================
PRUEBA = {
    "saludo": ["Hola, buenas tardes", "Qué onda, ¿hay alguien ahí?", "Buen día", "Holaaa"],
    "despedida": ["Bueno, me retiro", "Chao", "Nos vemos luego", "Eso sería todo por hoy"],
    "agradecimiento": ["Mil gracias por tu ayuda", "Te lo agradezco mucho", "Qué amable, gracias", "Muy útil, gracias"],
    "horario": ["¿Hasta qué hora están abiertos?", "¿Atienden los domingos?", "¿Cuál es su horario de atención?", "¿A qué hora cierran hoy?"],
    "envio_tiempo": ["¿Cuántos días tarda en llegar mi paquete?", "¿En cuánto tiempo me entregan la compra?",
                     "¿Cuánto debo esperar para recibir mi pedido?", "¿Qué tan rápido hacen los envíos?"],
    "envio_costo": ["¿Cobran por el envío?", "¿El envío es gratis?", "¿Cuánto cuesta enviar a Monterrey?", "¿Tiene costo la entrega a domicilio?"],
    "rastreo": ["¿Dónde está mi pedido?", "Quiero saber el estado de mi compra", "Necesito rastrear mi paquete", "¿Me pueden dar la guía de seguimiento?"],
    "devolucion": ["Quiero devolver un producto", "¿Cómo puedo regresar lo que compré?", "Necesito un reembolso", "No me gustó, ¿puedo cambiarlo?"],
    "pago": ["¿Qué formas de pago aceptan?", "¿Puedo pagar con tarjeta de crédito?", "¿Tienen meses sin intereses?", "¿Aceptan pago en efectivo?"],
    "garantia": ["Mi celular dejó de funcionar, ¿tiene garantía?", "El producto llegó defectuoso", "Se descompuso a los dos meses", "¿Cuánto dura la garantía?"],
    "cancelar": ["Quiero cancelar mi compra", "Necesito anular mi pedido", "Me arrepentí, ¿puedo cancelar?", "¿Cómo cancelo el pedido que hice ayer?"],
    "cuenta": ["Olvidé mi contraseña", "No puedo iniciar sesión", "¿Cómo creo una cuenta?", "Quiero registrarme en la tienda"],
    "promociones": ["¿Tienen descuentos esta semana?", "¿Hay algún cupón disponible?", "¿Qué ofertas manejan?", "¿Están de rebaja los audífonos?"],
    "humano": ["Quiero hablar con un asesor", "Pásame con una persona real", "Necesito atención humana", "¿Me comunican con un agente?"],
    "fallback": ["¿Cuál es la capital de Francia?", "Cuéntame un chiste", "¿Quién ganó el partido ayer?",
                 "Recomiéndame una película", "¿Cómo se hace una paella?", "¿Qué clima hará mañana?",
                 "Explícame la teoría de la relatividad", "asdfgh jkl", "¿Cuánto es 25 por 4?",
                 "Háblame de política", "Me duele la cabeza", "¿Vendes boletos de avión?"],
}


def evaluar_metodo(modelo, metodo, umbral=0.45):
    filas = []
    for real, textos in PRUEBA.items():
        for t in textos:
            pred, conf, _ = modelo.clasificar(t, umbral, metodo)
            filas.append({"texto": t, "real": real, "pred": pred or "fallback"})
    return pd.DataFrame(filas)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    os.makedirs(IMG, exist_ok=True)
    pd.set_option("display.width", 150, "display.max_colwidth", 70)

    modelo = ModeloIntenciones()

    print("=" * 70)
    print("TAREA 1: DISEÑO DE INTENCIONES")
    print("=" * 70)
    tabla = pd.DataFrame([{"intención": i, "patrones": len(d["patrones"]), "ejemplos": len(d["ejemplos"]),
                           "respuestas": len(d["respuestas"])} for i, d in INTENCIONES.items()])
    print(tabla.to_string(index=False))
    print(f"\nTotal: {len(INTENCIONES)} intenciones, {tabla['ejemplos'].sum()} ejemplos, "
          f"{tabla['patrones'].sum()} patrones")

    print("\nNormalización (para patrones) y tokenización con stemming (para TF-IDF):")
    for t in ["¿Cuánto TARDA el envío a Monterrey?", "Quiero DEVOLVER mi pedido!!!", "No me llegó mi paquete"]:
        print(f"  {t!r}\n    -> {normalizar(t)!r}\n    -> {tokenizar(t)!r}")

    print("\nCoincidencia de patrones y similitud sobre consultas de ejemplo:")
    for t in ["¿Cuántos días tarda en llegar mi paquete?", "¿Cuánto debo esperar para recibir mi pedido?"]:
        ip, p = modelo.por_patrones(t)
        it, s, rank = modelo.por_similitud(t)
        print(f"  {t}\n    patrones: {ip} (puntaje {p}) | tf-idf: {it} (similitud {s:.2f}); "
              f"siguiente: {list(rank)[1]} ({list(rank.values())[1]:.2f})")

    print("\n" + "=" * 70)
    print("TAREA 2: PROCESAMIENTO DE CONSULTAS")
    print("=" * 70)
    n_total = sum(len(v) for v in PRUEBA.values())
    n_ood = len(PRUEBA["fallback"])
    print(f"Conjunto de prueba: {n_total} consultas ({n_total - n_ood} dentro del dominio, {n_ood} fuera de dominio)")

    resultados = {}
    for etiqueta, metodo in [("Solo patrones", "patrones"), ("Solo TF-IDF", "tfidf"), ("Híbrido", "hibrido")]:
        df = evaluar_metodo(modelo, metodo)
        dentro = df[df["real"] != "fallback"]
        fuera = df[df["real"] == "fallback"]
        resultados[etiqueta] = {
            "exactitud": (df["real"] == df["pred"]).mean(),
            "dentro del dominio": (dentro["real"] == dentro["pred"]).mean(),
            "rechazo fuera de dominio": (fuera["pred"] == "fallback").mean(),
        }
        resultados[etiqueta]["_df"] = df
    res = pd.DataFrame({k: {m: v for m, v in d.items() if m != "_df"} for k, d in resultados.items()}).T
    print("\nComparación de estrategias de comprensión (umbral TF-IDF = 0.45):")
    print(res.round(3).to_string())

    df_h = resultados["Híbrido"]["_df"]
    print("\nReporte por intención (método híbrido):")
    print(classification_report(df_h["real"], df_h["pred"], zero_division=0, digits=2))
    errores = df_h[df_h["real"] != df_h["pred"]]
    print(f"Consultas mal clasificadas por el método híbrido: {len(errores)} de {len(df_h)}")
    print(errores.to_string(index=False))

    # Barras de comparación
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    res.plot.bar(ax=ax, color=["#1971c2", "#2f9e44", "#e8590c"], rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("proporción")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{IMG}/estrategias.png", dpi=200)
    plt.close()

    # Barrido del umbral de fallback (TF-IDF puro)
    filas = []
    for u in np.arange(0.0, 0.81, 0.05):
        df = evaluar_metodo(modelo, "tfidf", u)
        dentro, fuera = df[df["real"] != "fallback"], df[df["real"] == "fallback"]
        filas.append({"umbral": round(u, 2), "acierto dentro del dominio": (dentro["real"] == dentro["pred"]).mean(),
                      "rechazo fuera de dominio": (fuera["pred"] == "fallback").mean()})
    barrido = pd.DataFrame(filas)
    print("\nBarrido del umbral de confianza (solo TF-IDF):")
    print(barrido.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(barrido["umbral"], barrido["acierto dentro del dominio"], marker="o", color="#2f9e44", label="acierto en dominio")
    ax.plot(barrido["umbral"], barrido["rechazo fuera de dominio"], marker="s", color="#e8590c", label="rechazo fuera de dominio")
    ax.axvline(0.45, ls="--", color="gray")
    ax.set_xlabel("umbral de similitud")
    ax.set_ylabel("proporción")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{IMG}/umbral.png", dpi=200)
    plt.close()

    # Conversaciones de demostración: contexto, slots y fallback
    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN DE CONTEXTO Y FALLBACK")
    print("=" * 70)
    guiones = {
        "A) Contexto: preguntas de seguimiento": [
            "Hola, me llamo Ana", "¿Cuánto tarda el envío?", "¿y cuánto cuesta?", "¿y qué formas de pago hay?"],
        "B) Slot de pedido": ["Quiero rastrear mi paquete", "Es el TM-1001", "¿y el pedido 1004?"],
        "C) Fallback escalonado": [
            "¿Cuál es la capital de Francia?", "Cuéntame un chiste", "¿Cómo se hace una paella?", "sí"],
    }
    for titulo, mensajes in guiones.items():
        bot = Chatbot(modelo)
        print(f"\n{titulo}")
        for m in mensajes:
            r = bot.responder(m)
            t = bot.historial[-1]
            print(f"  Usuario: {m}\n  Bot    : {r}\n           [intención={t.intencion}, método={t.metodo}, confianza={t.confianza}]")


if __name__ == "__main__":
    main()
