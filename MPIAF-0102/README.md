# MPIAF-0102: Inteligencia artificial y su ética

## Instalación

Requiere Python 3.12 (spaCy y gensim aún no publican paquetes para 3.14).

```bash
cd MPIAF-0102
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm   # modelo en español (act16)
```

## Uso

Ejecutar el script de una actividad (las figuras se guardan en `actividades/actNN/img/`):

```bash
cd actividades/act16
python act16.py
```

Chatbot de la actividad 17 (el motor está en `act17.py`; la interfaz web, en `app.py`):

```bash
cd actividades/act17
python act17.py          # evaluación del motor
streamlit run app.py     # interfaz web en http://localhost:8501
```

Compilar el reporte PDF (requiere `latexmk` y `biber`):

```bash
cd actividades
./compilar.sh act16
```
