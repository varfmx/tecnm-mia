# =============================================================================
# Actividad 23 - Tarea 3: Interfaz de usuario (Streamlit)
# Ejecutar desde esta carpeta:  streamlit run app.py
# Requiere haber ejecutado antes act23.py (guarda el modelo en modelo/).
# =============================================================================
import io
import os

import altair as alt
import pandas as pd
import streamlit as st

from act23 import EMOCIONES, MODELO, ClasificadorEmociones

ICONOS = {"tristeza": "😢", "alegría": "😄", "amor": "❤️", "enojo": "😠", "miedo": "😨", "sorpresa": "😲"}
UMBRAL = 0.60      # por debajo de esta confianza se sugiere revisión humana
EJEMPLOS = ["i feel so thrilled and grateful for this team", "i am terrified of what the doctor will say",
            "this delay makes me furious", "i was surprised by how fast the package arrived",
            "i feel lonely and ignored by customer support", "i really love how easy the app is to use"]

st.set_page_config(page_title="Emociones en texto", page_icon="💬")


@st.cache_resource
def cargar():
    return ClasificadorEmociones()


st.title("Clasificador de emociones en texto")
st.caption("Red recurrente (GRU bidireccional) entrenada con el conjunto dair-ai/emotion: reconoce tristeza, alegría, "
           "amor, enojo, miedo y sorpresa en textos en inglés.")

if not os.path.exists(os.path.join(MODELO, "emociones.keras")):
    st.error("No se encontró el modelo. Ejecuta primero `python act23.py`.")
    st.stop()
clasificador = cargar()

with st.sidebar:
    st.header("Sobre el modelo")
    st.write("Escribe un comentario de un cliente, o carga varios a la vez en la pestaña de lote. "
             f"Cuando la confianza es menor a {UMBRAL:.0%} el sistema sugiere revisión humana.")
    st.write("Idioma: **inglés**. Las palabras que el modelo no conoce se ignoran (se indican abajo).")

tab_uno, tab_lote = st.tabs(["Un comentario", "Lote de comentarios"])

with tab_uno:
    if "texto" not in st.session_state:
        st.session_state.texto = ""
    st.write("Ejemplos:")
    columnas = st.columns(3)
    for i, ej in enumerate(EJEMPLOS):
        if columnas[i % 3].button(ej, key=f"ej{i}", width="stretch"):
            st.session_state.texto = ej
    texto = st.text_area("Comentario del cliente", key="texto", height=100, placeholder="i feel ...")
    if st.button("Clasificar", type="primary") or texto.strip():
        if not texto.strip():
            st.warning("Escribe un texto.")
        else:
            emocion, confianza, probs, desconocidas = clasificador.clasificar(texto)
            st.subheader(f"{ICONOS[emocion]} {emocion.capitalize()}  ({confianza:.0%})")
            if confianza < UMBRAL:
                st.warning("Confianza baja: se recomienda revisión humana.")
            if desconocidas:
                st.info("Palabras desconocidas para el modelo: " + ", ".join(desconocidas))
            datos = pd.DataFrame({"emoción": list(probs), "probabilidad": list(probs.values())})
            st.altair_chart(alt.Chart(datos).mark_bar().encode(
                x=alt.X("probabilidad:Q", scale=alt.Scale(domain=[0, 1]), title="Probabilidad"),
                y=alt.Y("emoción:N", sort=EMOCIONES, title=None), tooltip=["emoción", "probabilidad"]).properties(height=220),
                width="stretch")

with tab_lote:
    st.write("Pega un comentario por línea, o carga un archivo CSV con una columna llamada `texto`.")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    pegado = st.text_area("o pega los comentarios (uno por línea)", height=140, key="lote")
    textos = []
    if archivo is not None:
        df_in = pd.read_csv(archivo)
        if "texto" not in df_in.columns:
            st.error("El CSV debe tener una columna llamada `texto`.")
        else:
            textos = df_in["texto"].astype(str).tolist()
    elif pegado.strip():
        textos = [t for t in pegado.splitlines() if t.strip()]
    if textos:
        p = clasificador.probabilidades(textos)
        df = pd.DataFrame({"texto": textos, "emoción": [EMOCIONES[i] for i in p.argmax(1)], "confianza": p.max(1).round(3)})
        df["revisar"] = df["confianza"] < UMBRAL
        st.write(f"{len(df)} comentarios; {int(df['revisar'].sum())} con confianza baja.")
        conteo = df["emoción"].value_counts().reindex(EMOCIONES, fill_value=0).rename_axis("emoción").reset_index(name="comentarios")
        st.altair_chart(alt.Chart(conteo).mark_bar().encode(x=alt.X("emoción:N", sort=EMOCIONES, title=None, axis=alt.Axis(labelAngle=0)),
                                                            y=alt.Y("comentarios:Q", title="Comentarios")).properties(height=220), width="stretch")
        st.dataframe(df, width="stretch")
        salida = io.StringIO()
        df.to_csv(salida, index=False)
        st.download_button("Descargar resultados (CSV)", salida.getvalue(), "emociones.csv", "text/csv")
