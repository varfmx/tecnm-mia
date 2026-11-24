# =============================================================================
# Actividad 17 - Tarea 3: Interfaz de usuario (Streamlit)
# Ejecutar desde esta carpeta:  streamlit run app.py
# =============================================================================
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from act17 import Chatbot, ModeloIntenciones

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
ARCHIVO_CSAT = os.path.join(LOGS, "satisfaccion.csv")
SUGERENCIAS = ["¿Cuánto tarda el envío?", "¿Qué formas de pago aceptan?", "Quiero devolver un producto",
               "Quiero rastrear mi paquete", "¿Tienen promociones?"]

st.set_page_config(page_title="Asistente TecnoMIA", page_icon="💬")


@st.cache_resource
def cargar_modelo():
    """El modelo (TF-IDF + patrones) se entrena una sola vez y se comparte entre sesiones."""
    return ModeloIntenciones()


def nueva_conversacion():
    st.session_state.bot = Chatbot(cargar_modelo())
    st.session_state.mensajes = [{"rol": "assistant", "texto": "¡Hola! Soy el asistente de TecnoMIA. "
                                  "Puedo ayudarte con envíos, pagos, devoluciones y más. ¿Qué necesitas?"}]
    st.session_state.valoraciones = {}      # índice de mensaje -> 0 (pulgar abajo) / 1 (pulgar arriba)
    st.session_state.enviada = False
    st.session_state.version = st.session_state.get("version", 0) + 1   # reinicia los widgets de valoración


if "bot" not in st.session_state:
    nueva_conversacion()
bot = st.session_state.bot


def procesar(texto):
    st.session_state.mensajes.append({"rol": "user", "texto": texto})
    respuesta = bot.responder(texto)
    t = bot.historial[-1]
    st.session_state.mensajes.append({"rol": "assistant", "texto": respuesta,
                                      "intencion": t.intencion, "confianza": t.confianza, "metodo": t.metodo})


# ----------------------------- Barra lateral ----------------------------------
with st.sidebar:
    st.header("Opciones")
    st.caption("Preguntas frecuentes")
    for s in SUGERENCIAS:
        if st.button(s, use_container_width=True):
            procesar(s)
    if st.button("Nueva conversación", type="primary", use_container_width=True):
        nueva_conversacion()
        st.rerun()

    st.divider()
    st.caption("Estado del contexto (depuración)")
    st.json({"nombre": bot.nombre or None, "último tema": bot.ultimo_tema or None,
             "esperando": bot.esperando or None, "fallos seguidos": bot.fallos_seguidos})

    n = len(bot.historial)
    if n:
        df = pd.DataFrame([vars(t) for t in bot.historial])
        entendidas = (df["intencion"] != "fallback").mean()
        st.metric("Consultas entendidas", f"{entendidas:.0%}", help=f"{n} turnos en esta conversación")
        st.download_button("Descargar historial (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="historial.csv", mime="text/csv", use_container_width=True)

# ----------------------------- Conversación ------------------------------------
st.title("Asistente virtual TecnoMIA")
st.caption("Chatbot de atención al cliente: pregunta por envíos, pagos, devoluciones, garantía y más.")

for i, m in enumerate(st.session_state.mensajes):
    with st.chat_message(m["rol"]):
        st.write(m["texto"])
        if m["rol"] == "assistant" and i > 0:
            st.caption(f"intención: {m['intencion']} · método: {m['metodo']} · confianza: {m['confianza']:.2f}")
            v = st.feedback("thumbs", key=f"fb_{st.session_state.version}_{i}")
            if v is not None:
                st.session_state.valoraciones[i] = v

consulta = st.chat_input("Escribe tu pregunta...")
if consulta:
    procesar(consulta)
    st.rerun()

# ----------------------------- Satisfacción --------------------------------------
st.divider()
with st.expander("Evaluar la conversación", expanded=len(bot.historial) >= 3 and not st.session_state.enviada):
    if st.session_state.enviada:
        st.success("¡Gracias por tu valoración!")
    else:
        estrellas = st.feedback("stars", key=f"estrellas_{st.session_state.version}")
        comentario = st.text_area("Comentario (opcional)", key=f"coment_{st.session_state.version}")
        if st.button("Enviar valoración", disabled=estrellas is None or not bot.historial):
            os.makedirs(LOGS, exist_ok=True)
            fila = pd.DataFrame([{
                "fecha": datetime.now().isoformat(timespec="seconds"),
                "estrellas": estrellas + 1,
                "turnos": len(bot.historial),
                "entendidas": sum(t.intencion != "fallback" for t in bot.historial),
                "pulgares_arriba": sum(v == 1 for v in st.session_state.valoraciones.values()),
                "pulgares_abajo": sum(v == 0 for v in st.session_state.valoraciones.values()),
                "comentario": comentario}])
            fila.to_csv(ARCHIVO_CSAT, mode="a", header=not os.path.exists(ARCHIVO_CSAT), index=False)
            st.session_state.enviada = True
            st.rerun()

if os.path.exists(ARCHIVO_CSAT):
    csat = pd.read_csv(ARCHIVO_CSAT)
    with st.sidebar:
        st.divider()
        st.metric("Satisfacción promedio", f"{csat['estrellas'].mean():.1f} / 5",
                  help=f"{len(csat)} conversaciones evaluadas")
