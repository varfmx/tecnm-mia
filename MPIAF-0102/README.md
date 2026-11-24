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

Actividad 18 (clasificador de animales con TensorFlow/Keras): la primera vez descarga unas 600
imágenes (~80 MB) a `actividades/act18/datos/` (ignorada por git) y tarda ~35 min en CPU.

```bash
cd actividades/act18
python act18.py
```

Actividad 19 (detección de objetos con YOLO y OpenCV): la primera vez descarga los modelos (~280 MB)
y 200 imágenes de validación de COCO a `actividades/act19/datos/` (ignorada por git). Requiere
OpenCV 4.x (la versión 5 ya no lee modelos Darknet). Lee `actividades/act19/LEEME.txt`.

```bash
cd actividades/act19
python act19.py                  # evaluación (~5 min en CPU)
python act19.py --camara         # cámara web (q para salir)
streamlit run app.py             # interfaz web
```

Actividad 20 (diagnóstico médico asistido con scikit-learn; datos simulados, sin descargas, ~1 min):

```bash
cd actividades/act20
python act20.py
```

Actividad 21 (recomendación de películas con K-Means sobre MovieLens): descarga ~1 MB a `actividades/act21/datos/`
(ignorada por git); tarda ~2.5 min.

```bash
cd actividades/act21
python act21.py
```

Compilar el reporte PDF (requiere `latexmk` y `biber`):

```bash
cd actividades
./compilar.sh act16
```
