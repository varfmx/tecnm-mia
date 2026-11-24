# =============================================================================
# Actividad 19 - Tarea 2: Interfaz de usuario (Streamlit)
# Ejecutar desde esta carpeta:  streamlit run app.py
# =============================================================================
import os
import time

import cv2
import pandas as pd
import streamlit as st

from act19 import DATOS, RESULTADOS, Config, flujo_video, preparar_modelos

VIDEO_PRUEBA = os.path.join(DATOS, "video_prueba.mp4")

st.set_page_config(page_title="Detección de objetos en tiempo real", page_icon="🎥", layout="wide")
st.title("Detección de objetos en tiempo real")
st.caption("YOLOv3 con OpenCV: cámara web o archivo de video, métricas en vivo y grabación de resultados.")


@st.cache_resource
def modelos_listos():
    preparar_modelos()
    return True


modelos_listos()

# ----------------------------- Barra lateral ----------------------------------
with st.sidebar:
    st.header("Configuración")
    origen = st.radio("Fuente de video", ["Cámara web", "Video de prueba", "Subir un video"])
    archivo = st.file_uploader("Video (mp4, avi, mov)", type=["mp4", "avi", "mov"]) if origen == "Subir un video" else None
    modelo = st.selectbox("Modelo", ["yolov3-tiny", "yolov3"], help="yolov3-tiny es mucho más rápido; yolov3 es más preciso.")
    tam = st.select_slider("Tamaño de entrada", [320, 416, 608], value=416, help="Menor tamaño = más rápido y menos preciso.")
    conf = st.slider("Umbral de confianza", 0.1, 0.9, 0.3, 0.05)
    saltar = st.slider("Inferencia cada N fotogramas", 1, 5, 1, help="Los fotogramas intermedios reutilizan las últimas cajas.")
    grabar = st.checkbox("Grabar resultados (video + CSV)")
    st.divider()
    activo = st.toggle("Iniciar detección", value=False)

# ----------------------------- Paneles ----------------------------------------
col_video, col_datos = st.columns([3, 2])
marco = col_video.empty()
with col_datos:
    c1, c2 = st.columns(2)
    m_fps, m_ms = c1.empty(), c2.empty()
    c3, c4 = st.columns(2)
    m_obj, m_total = c3.empty(), c4.empty()
    tabla = st.empty()
    estado = st.empty()


def fuente_de_video():
    if origen == "Cámara web":
        return 0
    if origen == "Video de prueba":
        if not os.path.exists(VIDEO_PRUEBA):
            st.warning("El video de prueba se crea al ejecutar «python act19.py».")
            return None
        return VIDEO_PRUEBA
    if archivo is None:
        st.info("Sube un archivo de video para continuar.")
        return None
    ruta = os.path.join(DATOS, "subido_" + archivo.name)
    with open(ruta, "wb") as f:
        f.write(archivo.getbuffer())
    return ruta


if not activo:
    estado.info("Activa «Iniciar detección» en la barra lateral.")
else:
    fuente = fuente_de_video()
    if fuente is not None:
        base = os.path.join(RESULTADOS, time.strftime("grabacion_%Y%m%d_%H%M%S")) if grabar else None
        cfg = Config(modelo=modelo, tam=tam, conf=conf, saltar=saltar)
        historial = []
        try:
            for frame, m in flujo_video(fuente, cfg, grabar=base):
                marco.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                historial.append(m["fps"])
                m_fps.metric("FPS", f"{m['fps']:.1f}")
                m_ms.metric("Latencia de inferencia", f"{m['ms']:.0f} ms")
                m_obj.metric("Objetos en el fotograma", m["objetos"])
                m_total.metric("Fotogramas", m["fotograma"] + 1)
                top = pd.Series(m["acumulado"], dtype=int).sort_values(ascending=False).head(8)
                tabla.dataframe(top.rename("detecciones acumuladas").to_frame(), use_container_width=True)
            estado.success("El video terminó." + (f" Grabación guardada en {os.path.basename(base)}.mp4 y .csv" if base else ""))
        except RuntimeError as error:
            estado.error(f"{error}. Si usas la cámara web, concede el permiso de cámara a la terminal o al navegador.")
