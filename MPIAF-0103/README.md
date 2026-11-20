# MPIAF-0103: Modelos estadísticos para el aprendizaje automático y ciencia de datos

## Instalación

Requiere Python 3.13.

```bash
cd MPIAF-0103
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

Ejecutar el script de una actividad (las figuras se guardan en `actividades/actNN/img/`):

```bash
cd actividades/act12
python act12.py
```

Compilar el reporte PDF (requiere `latexmk` y `biber`):

```bash
cd actividades
./compilar.sh act12
```
