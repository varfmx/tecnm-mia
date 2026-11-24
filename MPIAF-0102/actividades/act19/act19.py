# =============================================================================
# Actividad 19: Proyecto "Sistema de Detección de Objetos en Tiempo Real"
# Tema 4.2. Visión artificial
#
# Uso:
#   python act19.py                         Evaluación completa (Tareas 1 y 3) y prueba del flujo de video
#   python act19.py --camara                Detección en tiempo real con la cámara web (ventana de OpenCV)
#   python act19.py --video ruta.mp4        Igual, pero con un archivo de video
#   (opciones: --modelo yolov3-tiny|yolov3  --grabar  --saltar N ; en la ventana: q = salir, g = grabar)
#   streamlit run app.py                    Interfaz web (Tarea 2)
# Dependencias (en Colab anteponer "!"):  pip install "opencv-python<4.13" numpy pandas matplotlib
# Los modelos YOLO (~280 MB) y 200 imágenes de validación de COCO se descargan solos la primera vez.
# =============================================================================
import argparse
import io
import json
import os
import random
import time
import urllib.request
import warnings
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEMILLA = 42
UMBRALES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_IMAGENES = 200
AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()   # Colab no define __file__
DATOS = os.path.join(AQUI, "datos")
MODELOS = os.path.join(DATOS, "modelos")
VALIDACION = os.path.join(DATOS, "validacion")
IMG = os.path.join(AQUI, "img")
RESULTADOS = os.path.join(AQUI, "resultados")
for carpeta in (MODELOS, VALIDACION, IMG, RESULTADOS):
    os.makedirs(carpeta, exist_ok=True)

DARKNET = "https://raw.githubusercontent.com/pjreddie/darknet/master/"
ARCHIVOS_MODELO = {
    "coco.names": DARKNET + "data/coco.names",
    "yolov3.cfg": DARKNET + "cfg/yolov3.cfg",
    "yolov3-tiny.cfg": DARKNET + "cfg/yolov3-tiny.cfg",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3-tiny.weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
}
URL_ANOTACIONES = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
URL_IMAGENES = "http://images.cocodataset.org/val2017/"


# =============================================================================
# DATOS: modelos YOLO y conjunto de validación con etiquetas reales (COCO val2017)
# =============================================================================
def _descargar(url, destino):
    if not os.path.exists(destino):
        urllib.request.urlretrieve(url, destino)


def preparar_modelos():
    """Descarga los archivos de YOLOv3 y YOLOv3-tiny (formato Darknet) si no existen."""
    for nombre, url in ARCHIVOS_MODELO.items():
        _descargar(url, os.path.join(MODELOS, nombre))


class _ArchivoRemoto(io.RawIOBase):
    """Archivo de solo lectura sobre HTTP (peticiones Range): permite abrir un .zip remoto
    y extraer un solo archivo sin descargar los 241 MB completos."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        self.n = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, desplazamiento, desde=0):
        self.pos = desplazamiento if desde == 0 else self.pos + desplazamiento if desde == 1 else self.n + desplazamiento
        return self.pos

    def readinto(self, b):
        fin = min(self.pos + len(b), self.n) - 1
        if self.pos > fin:
            return 0
        peticion = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{fin}"})
        datos = urllib.request.urlopen(peticion, timeout=120).read()
        b[:len(datos)] = datos
        self.pos += len(datos)
        return len(datos)


def preparar_validacion():
    """Elige N_IMAGENES imágenes al azar de COCO val2017 (no usadas para entrenar YOLOv3), las descarga y
    guarda sus etiquetas reales en validacion/etiquetas.json: [{archivo, ancho, alto, cajas:[[x,y,w,h,clase,multitud]]}]."""
    ruta = os.path.join(VALIDACION, "etiquetas.json")
    if os.path.exists(ruta):
        return json.load(open(ruta))
    z = zipfile.ZipFile(io.BufferedReader(_ArchivoRemoto(URL_ANOTACIONES), buffer_size=1 << 20))
    coco = json.loads(z.read("annotations/instances_val2017.json"))
    nombres = [l.strip() for l in open(os.path.join(MODELOS, "coco.names"))]
    categorias = sorted(c["id"] for c in coco["categories"])                 # id de COCO (1..90) -> índice 0..79
    indice = {cid: i for i, cid in enumerate(categorias)}
    por_id = {c["id"]: c["name"] for c in coco["categories"]}
    equivalentes = {"motorcycle": "motorbike", "airplane": "aeroplane", "couch": "sofa",        # nombres distintos en Darknet
                    "potted plant": "pottedplant", "dining table": "diningtable", "tv": "tvmonitor"}
    assert all(equivalentes.get(por_id[cid], por_id[cid]) == nombres[i] for cid, i in indice.items()), \
        "las clases de COCO no coinciden con coco.names"
    cajas = {}
    for a in coco["annotations"]:
        cajas.setdefault(a["image_id"], []).append(a["bbox"] + [indice[a["category_id"]], a["iscrowd"]])
    candidatas = sorted((im for im in coco["images"] if any(not c[5] for c in cajas.get(im["id"], []))), key=lambda im: im["id"])
    elegidas = random.Random(SEMILLA).sample(candidatas, N_IMAGENES)
    etiquetas = [{"archivo": im["file_name"], "ancho": im["width"], "alto": im["height"], "cajas": cajas[im["id"]]} for im in elegidas]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda e: _descargar(URL_IMAGENES + e["archivo"], os.path.join(VALIDACION, e["archivo"])), etiquetas))
    json.dump(etiquetas, open(ruta, "w"))
    return etiquetas


# =============================================================================
# TAREA 1: IMPLEMENTACIÓN DE YOLO (código base del tema, con las correcciones necesarias)
# =============================================================================
def configurar_yolo(modelo="yolov3"):
    """Configura YOLO para detección de objetos (código base). Cambios respecto al base: se elige
    entre yolov3 y yolov3-tiny, y las capas de salida se obtienen con .flatten() porque en las
    versiones actuales de OpenCV getUnconnectedOutLayers() devuelve un arreglo plano (i[0] falla)."""
    with open(os.path.join(MODELOS, "coco.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet(os.path.join(MODELOS, f"{modelo}.weights"), os.path.join(MODELOS, f"{modelo}.cfg"))
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers, classes


def detectar(imagen, net, output_layers, conf_min=0.5, nms_umbral=0.4, tam=416, por_clase=False):
    """Detección con YOLO (misma lógica del código base) que devuelve una lista de
    (caja [x, y, w, h], confianza, índice de clase). Con por_clase=True la supresión de no
    máximos se aplica dentro de cada clase, no entre todas."""
    height, width = imagen.shape[:2]
    blob = cv2.dnn.blobFromImage(imagen, 0.00392, (tam, tam), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_min:
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(detection[0] * width - w / 2), int(detection[1] * height - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)
    if not boxes:
        return []
    if por_clase:
        indexes = cv2.dnn.NMSBoxesBatched(boxes, confidences, class_ids, conf_min, nms_umbral)
    else:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_min, nms_umbral)
    return [(boxes[i], confidences[i], class_ids[i]) for i in np.array(indexes).flatten()]


def colores_clases(n=80):
    return np.random.RandomState(SEMILLA).uniform(60, 255, size=(n, 3))


def dibujar(imagen, detecciones, classes, colors=None, grosor=2):
    """Dibuja las cajas y etiquetas (como en el código base). Modifica y devuelve la imagen."""
    colors = colores_clases(len(classes)) if colors is None else colors
    escala = max(0.4, min(imagen.shape[:2]) / 800)
    for (x, y, w, h), confianza, cid in detecciones:
        color = tuple(int(c) for c in colors[cid])
        cv2.rectangle(imagen, (x, y), (x + w, y + h), color, grosor)
        texto = f"{classes[cid]} {confianza:.2f}"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
        cv2.rectangle(imagen, (x, max(y - th - 6, 0)), (x + tw + 4, max(y, th + 6)), color, -1)
        cv2.putText(imagen, texto, (x + 2, max(y - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 1, cv2.LINE_AA)
    return imagen


def detectar_objetos_yolo(imagen, net, output_layers, classes):
    """Equivalente de la función del código base: detecta y devuelve la imagen con las cajas dibujadas."""
    return dibujar(imagen, detectar(imagen, net, output_layers), classes)


# ---- Flujo de video en tiempo real (cámara web o archivo) ----------------------------------------
@dataclass
class Config:
    modelo: str = "yolov3-tiny"     # yolov3 (preciso, lento) o yolov3-tiny (rápido)
    tam: int = 416                  # lado de la entrada de la red: 320, 416 o 608
    conf: float = 0.3               # umbral de confianza (0.3 dio el mejor F1 de YOLOv3 en la evaluación; el código base usa 0.5)
    nms: float = 0.4                # umbral de IoU de la supresión de no máximos
    por_clase: bool = True          # NMS por clase (permite objetos de distinta clase superpuestos)
    saltar: int = 1                 # inferencia cada N fotogramas (los demás reutilizan las últimas cajas)
    ancho_max: int = 640            # los fotogramas más anchos se reducen antes de procesarse


class Grabador:
    """Guarda el video con las detecciones dibujadas (.mp4) y un CSV con las métricas de cada fotograma."""
    def __init__(self, base, fps, tamano):
        self.video = cv2.VideoWriter(base + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, tamano)
        self.csv = open(base + ".csv", "w")
        self.csv.write("fotograma,segundos,fps,ms_inferencia,objetos,clases\n")
        self.t0 = time.perf_counter()

    def escribir(self, frame, m, nombres):
        self.video.write(frame)
        clases = "|".join(sorted(set(nombres)))
        self.csv.write(f"{m['fotograma']},{time.perf_counter() - self.t0:.3f},{m['fps']:.1f},{m['ms']:.1f},{m['objetos']},{clases}\n")

    def cerrar(self):
        self.video.release()
        self.csv.close()


def flujo_video(fuente, cfg, grabar=None, parar=lambda: False):
    """Generador: lee fotogramas de la fuente (0 = cámara web, o ruta de un video), detecta objetos y
    produce (fotograma dibujado, métricas). Métricas: fps del ciclo completo, ms de inferencia, objetos
    en el fotograma y detecciones acumuladas por clase. Si grabar es una ruta base, guarda .mp4 y .csv."""
    net, salidas, nombres = configurar_yolo(cfg.modelo)
    colores = colores_clases(len(nombres))
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")
    grabador, detecciones, ms, fps, n = None, [], 0.0, 0.0, 0
    acumulado = Counter()
    t_prev = time.perf_counter()
    try:
        while not parar():
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] > cfg.ancho_max:
                frame = cv2.resize(frame, (cfg.ancho_max, int(frame.shape[0] * cfg.ancho_max / frame.shape[1])))
            if n % cfg.saltar == 0:
                t0 = time.perf_counter()
                detecciones = detectar(frame, net, salidas, cfg.conf, cfg.nms, cfg.tam, cfg.por_clase)
                ms = (time.perf_counter() - t0) * 1000
                acumulado.update(nombres[d[2]] for d in detecciones)
            dibujar(frame, detecciones, nombres, colores)
            ahora = time.perf_counter()
            fps = 1 / (ahora - t_prev) if fps == 0 else 0.9 * fps + 0.1 / (ahora - t_prev)
            t_prev = ahora
            metricas = {"fotograma": n, "fps": fps, "ms": ms, "objetos": len(detecciones), "acumulado": dict(acumulado)}
            cv2.putText(frame, f"{fps:.1f} FPS | {ms:.0f} ms | {len(detecciones)} objetos", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            if grabar:
                if grabador is None:
                    origen = cap.get(cv2.CAP_PROP_FPS)
                    grabador = Grabador(grabar, origen if 1 < origen < 120 else 20, (frame.shape[1], frame.shape[0]))
                grabador.escribir(frame, metricas, [nombres[d[2]] for d in detecciones])
            n += 1
            yield frame, metricas
    finally:
        cap.release()
        if grabador:
            grabador.cerrar()


def modo_ventana(fuente, cfg, grabar):
    """Detección en tiempo real en una ventana de OpenCV. q = salir, g = iniciar/detener la grabación."""
    base = os.path.join(RESULTADOS, time.strftime("grabacion_%Y%m%d_%H%M%S")) if grabar else None
    for frame, m in flujo_video(fuente, cfg, grabar=base):
        cv2.imshow("Actividad 19 - Deteccion de objetos (q para salir)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    if base:
        print("Grabación guardada en", base + ".mp4 / .csv")


def crear_video_prueba(etiquetas, ruta, segundos_por_imagen=2.0, fps=15, n_imagenes=6):
    """Genera un video de prueba reproducible: acercamiento lento sobre las imágenes de validación con más
    objetos. No sustituye a la cámara (las imágenes no tienen movimiento real), pero permite probar el flujo."""
    elegidas = sorted(etiquetas, key=lambda e: -len(e["cajas"]))[:n_imagenes]
    escritor = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for e in elegidas:
        img = cv2.imread(os.path.join(VALIDACION, e["archivo"]))
        n = int(segundos_por_imagen * fps)
        for k in range(n):
            z = 1 + 0.15 * k / n
            h, w = img.shape[:2]
            ch, cw = int(h / z), int(w / z)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            escritor.write(cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (640, 480)))
    escritor.release()


# =============================================================================
# TAREA 3: ANÁLISIS DE RESULTADOS (precisión, recall, falsos positivos y negativos)
# =============================================================================
def iou(a, b):
    """IoU de dos cajas [x, y, w, h]."""
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def emparejar(dets, gts, iou_min=0.5):
    """Empareja las detecciones de UNA imagen con las cajas reales. Se recorren de mayor a menor confianza y
    cada una se asigna a la caja real libre de su clase con mayor IoU (>= iou_min): verdadero positivo (TP).
    Si no hay pareja es un falso positivo (FP), que se clasifica como: duplicada, clase equivocada,
    mala localización (IoU 0.1-0.5 con un objeto de su clase) o fondo (no hay objeto). Las detecciones sobre
    regiones 'multitud' de COCO se ignoran. Devuelve (filas de detecciones, confianza con que se halló cada objeto)."""
    reales = [g for g in gts if not g[5]]
    multitud = [g for g in gts if g[5]]
    hallado = [None] * len(reales)
    filas = []
    for caja, conf, c in sorted(dets, key=lambda d: -d[1]):
        mejor, j_mejor = 0.0, None
        for j, g in enumerate(reales):
            if g[4] == c and hallado[j] is None:
                v = iou(caja, g[:4])
                if v > mejor:
                    mejor, j_mejor = v, j
        if j_mejor is not None and mejor >= iou_min:
            hallado[j_mejor] = conf
            filas.append((c, conf, "TP", ""))
            continue
        if any(g[4] == c and iou(caja, g[:4]) >= 0.3 for g in multitud):
            filas.append((c, conf, "ignorada", ""))
            continue
        misma = max([iou(caja, g[:4]) for g in reales if g[4] == c], default=0.0)
        otra = max([iou(caja, g[:4]) for g in reales if g[4] != c], default=0.0)
        tipo = "duplicada" if misma >= iou_min else "clase equivocada" if otra >= iou_min else "mala localización" if misma >= 0.1 else "fondo"
        filas.append((c, conf, "FP", tipo))
    return filas, hallado, reales


def evaluar_configuracion(detecciones, etiquetas):
    """Empareja todas las imágenes. Devuelve dos tablas: una fila por detección y una por objeto real."""
    d_filas, g_filas = [], []
    for i, (dets, e) in enumerate(zip(detecciones, etiquetas)):
        filas, hallado, reales = emparejar(dets, e["cajas"])
        d_filas += [(i,) + f for f in filas]
        g_filas += [(i, g[4], g[2] * g[3], h) for g, h in zip(reales, hallado)]
    return (pd.DataFrame(d_filas, columns=["imagen", "clase", "conf", "estado", "tipo"]),
            pd.DataFrame(g_filas, columns=["imagen", "clase", "area", "conf_hallado"]))


def metricas(d, g, umbral):
    """Precisión, recall y F1 globales con el umbral de confianza dado."""
    sel = d[(d["conf"] >= umbral) & (d["estado"] != "ignorada")]
    tp, fp = int((sel["estado"] == "TP").sum()), int((sel["estado"] == "FP").sum())
    fn = len(g) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / len(g) if len(g) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precisión": p, "recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0}


def ap50(d, g):
    """AP@0.5: área bajo la curva precisión-recall (interpolación de todos los puntos), promediada sobre las
    clases que aparecen en las etiquetas (una clase sin detecciones aporta 0)."""
    aps = []
    for c, n_real in g.groupby("clase").size().items():
        s = d[(d["clase"] == c) & (d["estado"] != "ignorada")].sort_values("conf", ascending=False)
        tp = np.cumsum((s["estado"] == "TP").to_numpy())
        fp = np.cumsum((s["estado"] == "FP").to_numpy())
        rec, pre = np.concatenate([[0], tp / n_real, [1]]), np.concatenate([[0], tp / np.maximum(tp + fp, 1), [0]])
        for k in range(len(pre) - 2, -1, -1):
            pre[k] = max(pre[k], pre[k + 1])
        idx = np.where(rec[1:] != rec[:-1])[0]
        aps.append(float(np.sum((rec[idx + 1] - rec[idx]) * pre[idx + 1])))
    return float(np.mean(aps))


def correr_modelo(nombre, tam, por_clase, etiquetas, conf_min=0.1):
    """Detecta sobre todas las imágenes (con umbral bajo, para poder variar el umbral después) y mide la latencia."""
    net, salidas, _ = configurar_yolo(nombre)
    imagenes = [cv2.imread(os.path.join(VALIDACION, e["archivo"])) for e in etiquetas]
    detectar(imagenes[0], net, salidas, conf_min, 0.4, tam, por_clase)          # calentamiento
    dets, tiempos = [], []
    for im in imagenes:
        t0 = time.perf_counter()
        dets.append(detectar(im, net, salidas, conf_min, 0.4, tam, por_clase))
        tiempos.append((time.perf_counter() - t0) * 1000)
    return dets, float(np.median(tiempos))


def main():
    preparar_modelos()
    etiquetas = preparar_validacion()
    nombres = [l.strip() for l in open(os.path.join(MODELOS, "coco.names"))]
    n_objetos = sum(1 for e in etiquetas for c in e["cajas"] if not c[5])
    print("=" * 70)
    print("DATOS")
    print("=" * 70)
    print(f"Imágenes de validación (COCO val2017, semilla {SEMILLA}): {len(etiquetas)}; objetos etiquetados: {n_objetos}; "
          f"clases presentes: {len({c[4] for e in etiquetas for c in e['cajas'] if not c[5]})} de 80")
    print(f"OpenCV {cv2.__version__}")

    # ---- Tarea 1: modelos, tamaños de entrada y NMS ------------------------------------------------
    print("\n" + "=" * 70)
    print("TAREA 1: IMPLEMENTACIÓN DE YOLO Y RENDIMIENTO")
    print("=" * 70)
    configuraciones = [("yolov3-tiny", 320, False), ("yolov3-tiny", 416, False), ("yolov3-tiny", 608, False),
                       ("yolov3", 320, False), ("yolov3", 416, False), ("yolov3", 608, False),
                       ("yolov3", 416, True)]
    corridas, filas = {}, []
    for nombre, tam, por_clase in configuraciones:
        dets, ms = correr_modelo(nombre, tam, por_clase, etiquetas)
        d, g = evaluar_configuracion(dets, etiquetas)
        m = metricas(d, g, 0.5)
        mejor_u = max(UMBRALES, key=lambda u: metricas(d, g, u)["F1"])         # umbral con el mejor F1 de esta configuración
        clave = f"{nombre}-{tam}" + ("-nms_clase" if por_clase else "")
        corridas[clave] = (dets, d, g, ms)
        filas.append({"configuración": clave, "ms/imagen": round(ms), "FPS": round(1000 / ms, 1), "precisión": round(m["precisión"], 3),
                      "recall": round(m["recall"], 3), "F1": round(m["F1"], 3), "AP50": round(ap50(d, g), 3),
                      "mejor umbral": mejor_u, "F1 máx": round(metricas(d, g, mejor_u)["F1"], 3)})
    tabla = pd.DataFrame(filas).set_index("configuración")
    print("Velocidad y calidad por configuración (IoU 0.5; prec., recall y F1 con umbral 0.5;")
    print("ms = latencia mediana en CPU; umbral y F1 máx = el de mejor F1 de cada una):")
    print(tabla.rename(columns={"ms/imagen": "ms", "precisión": "prec.", "mejor umbral": "umbral"}).to_string())

    print("\nNMS entre todas las clases (código base) frente a NMS por clase, YOLOv3-416:")
    for clave in ("yolov3-416", "yolov3-416-nms_clase"):
        _, d, g, _ = corridas[clave]
        m = metricas(d, g, 0.5)
        print(f"  {clave:22s} TP={m['TP']:4d}  FP={m['FP']:4d}  FN={m['FN']:4d}  precisión={m['precisión']:.3f}  recall={m['recall']:.3f}")

    # Elección para tiempo real: mejor F1 entre las configuraciones que alcanzan al menos 15 FPS
    candidatas = tabla[tabla["FPS"] >= 15]
    elegida = candidatas["F1 máx"].idxmax()
    print(f"\nMejor F1 con al menos 15 FPS: {elegida} (umbral {tabla.loc[elegida, 'mejor umbral']})")
    print(f"Mejor F1 sin restricción de FPS: {tabla['F1 máx'].idxmax()} (umbral {tabla.loc[tabla['F1 máx'].idxmax(), 'mejor umbral']})")

    # Ciclo de video completo: efecto de saltar fotogramas
    print("\nCiclo de video (leer + detectar + dibujar), video de prueba de 640x480:")
    video = os.path.join(DATOS, "video_prueba.mp4")
    crear_video_prueba(etiquetas, video)
    filas_v = []
    for modelo in ("yolov3-tiny", "yolov3"):
        for saltar in (1, 2, 3):
            t0, n = time.perf_counter(), 0
            for _ in flujo_video(video, Config(modelo=modelo, saltar=saltar)):
                n += 1
            filas_v.append({"modelo": modelo, "inferencia cada": f"{saltar} fotograma(s)", "fotogramas": n,
                            "FPS del ciclo": round(n / (time.perf_counter() - t0), 1)})
    print(pd.DataFrame(filas_v).to_string(index=False))

    # ---- Tarea 2 (motor): grabación de resultados ---------------------------------------------------
    print("\n" + "=" * 70)
    print("TAREA 2: FLUJO DE VIDEO Y GRABACIÓN DE RESULTADOS")
    print("=" * 70)
    base = os.path.join(RESULTADOS, "demo_grabacion")
    cfg = Config(modelo="yolov3-tiny")
    ultima = None
    fps_med, objs = [], []
    for frame, m in flujo_video(video, cfg, grabar=base):
        fps_med.append(m["fps"])
        objs.append(m["objetos"])
        ultima = m
    reg = pd.read_csv(base + ".csv")
    print(f"Fotogramas procesados y grabados: {len(reg)}; FPS promedio: {np.mean(fps_med):.1f}; objetos por fotograma: {np.mean(objs):.1f}")
    print(f"Archivos: demo_grabacion.mp4 ({os.path.getsize(base + '.mp4') / 1e6:.1f} MB) y demo_grabacion.csv")
    print("Más frecuentes:", dict(Counter(ultima['acumulado']).most_common(3)))

    # ---- Tarea 3: análisis de la configuración de máxima calidad y de la elegida ---------------------
    print("\n" + "=" * 70)
    print("TAREA 3: ANÁLISIS DE RESULTADOS")
    print("=" * 70)
    ref = tabla["F1 máx"].idxmax()
    u = float(tabla.loc[ref, "mejor umbral"])
    dets, d, g, _ = corridas[ref]
    print(f"Configuración analizada: {ref}, con su mejor umbral de confianza ({u})")
    barrido = pd.DataFrame([{"umbral": u, **metricas(d, g, u)} for u in np.round(np.arange(0.1, 0.95, 0.1), 1)]).set_index("umbral")
    print("\nPrecisión y recall según el umbral de confianza:")
    print(barrido[["TP", "FP", "FN", "precisión", "recall", "F1"]].round(3).to_string())
    m = metricas(d, g, u)
    print(f"\nCon umbral {u}: TP={m['TP']}, FP={m['FP']}, FN={m['FN']}; precisión={m['precisión']:.3f}, recall={m['recall']:.3f}, F1={m['F1']:.3f}; AP50={ap50(d, g):.3f}")

    sel = d[(d["conf"] >= u) & (d["estado"] == "FP")]
    tipos = sel["tipo"].value_counts()
    print(f"\nFalsos positivos por tipo (umbral {u}):")
    print(tipos.to_string())
    print("\nClases con más falsos positivos:")
    print(sel["clase"].map(lambda c: nombres[c]).value_counts().head(6).to_string())

    g = g.assign(detectado=g["conf_hallado"].fillna(0) >= u)
    g["tamaño"] = pd.cut(g["area"], [0, 32 ** 2, 96 ** 2, np.inf], labels=["pequeño (<32²)", "mediano", "grande (>96²)"])
    por_tam = g.groupby("tamaño", observed=True)["detectado"].agg(["size", "sum"])
    por_tam["recall"] = por_tam["sum"] / por_tam["size"]
    print(f"\nRecall según el tamaño del objeto en píxeles (umbral {u}):")
    print(por_tam.rename(columns={"size": "objetos", "sum": "detectados"}).round(3).to_string())
    fn = g[~g["detectado"]]
    print("\nClases con más falsos negativos:")
    print(fn["clase"].map(lambda c: nombres[c]).value_counts().head(6).to_string())

    # Reporte de desempeño por clase (CSV)
    sel_tp = d[(d["conf"] >= u) & (d["estado"] == "TP")].groupby("clase").size()
    sel_fp = d[(d["conf"] >= u) & (d["estado"] == "FP")].groupby("clase").size()
    n_real = g.groupby("clase").size()
    rep = pd.DataFrame({"objetos": n_real, "TP": sel_tp, "FP": sel_fp}).fillna(0).astype(int)
    rep["FN"] = rep["objetos"] - rep["TP"]
    rep["precisión"] = (rep["TP"] / (rep["TP"] + rep["FP"]).replace(0, np.nan)).round(3)
    rep["recall"] = (rep["TP"] / rep["objetos"]).round(3)
    rep.index = [nombres[c] for c in rep.index]
    rep.index.name = "clase"
    rep.sort_values("objetos", ascending=False).to_csv(os.path.join(RESULTADOS, "reporte_desempeno.csv"))
    json.dump({"configuración": ref, "umbral": u, "imágenes": len(etiquetas), "objetos": int(len(g)), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
               "AP50": round(ap50(d, g), 4)}, open(os.path.join(RESULTADOS, "resumen.json"), "w"), ensure_ascii=False, indent=1)
    print("\nReporte por clase (10 clases con más objetos):")
    print(rep.sort_values("objetos", ascending=False).head(10).to_string())
    print("\nArchivos: resultados/reporte_desempeno.csv, resultados/resumen.json")

    # ---- Figuras ---------------------------------------------------------------------------------------
    # 1. Latencia frente a calidad
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for clave, fila in tabla.iterrows():
        ax.scatter(fila["ms/imagen"], fila["F1 máx"], s=50, color="#1f4e79" if "yolov3-" in clave and "tiny" not in clave else "#c55a11")
        ax.annotate(clave.replace("yolov3-", "v3-").replace("-nms_clase", " (nms clase)"), (fila["ms/imagen"], fila["F1 máx"]), fontsize=7,
                    xytext=(4, -10) if "nms_clase" in clave else (4, 3), textcoords="offset points")
    ax.axvline(1000 / 15, color="gray", ls="--", lw=1)
    ax.text(1000 / 15 + 3, ax.get_ylim()[0] + 0.01, "15 FPS", fontsize=8, color="gray")
    ax.set_xlabel("latencia por imagen (ms, CPU)")
    ax.set_ylabel("F1 máximo (mejor umbral de cada configuración)")
    ax.set_title("Compromiso entre velocidad y calidad")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, "rendimiento.png"), dpi=200)
    plt.close()

    # 2. Precisión y recall según el umbral + curva PR
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(barrido.index, barrido["precisión"], "o-", label="precisión")
    ax[0].plot(barrido.index, barrido["recall"], "s-", label="recall")
    ax[0].plot(barrido.index, barrido["F1"], "^--", label="F1")
    ax[0].set_xlabel("umbral de confianza")
    ax[0].set_title(f"Efecto del umbral ({ref})")
    ax[0].legend()
    for clave in ("yolov3-tiny-416", "yolov3-416", "yolov3-608"):
        _, dd, gg, _ = corridas[clave]
        puntos = [metricas(dd, gg, u) for u in np.arange(0.1, 0.96, 0.05)]
        ax[1].plot([p["recall"] for p in puntos], [p["precisión"] for p in puntos], label=clave)
    ax[1].set_xlabel("recall")
    ax[1].set_ylabel("precisión")
    ax[1].set_title("Curva precisión-recall")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, "umbral.png"), dpi=200)
    plt.close()

    # 3. Errores: tipos de FP y recall por tamaño
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].barh(tipos.index[::-1], tipos.values[::-1], color="#c55a11")
    ax[0].set_title("Falsos positivos por tipo")
    ax[1].bar(por_tam.index.astype(str), por_tam["recall"], color="#1f4e79")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Recall según el tamaño del objeto")
    for k, v in enumerate(por_tam["recall"]):
        ax[1].text(k, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, "errores.png"), dpi=200)
    plt.close()

    # 4. Ejemplos: más objetos detectados, más FP y más FN (cajas reales en verde fino, detecciones de color)
    por_img = d[(d["conf"] >= u) & (d["estado"] != "ignorada")]
    n_tp = por_img[por_img["estado"] == "TP"].groupby("imagen").size()
    n_fp = por_img[por_img["estado"] == "FP"].groupby("imagen").size()
    n_fn = g[~g["detectado"]].groupby("imagen").size()
    elegidas, usadas = [], set()
    for titulo, serie in (("Más falsos positivos", n_fp), ("Más falsos negativos", n_fn), ("Más objetos detectados", n_tp)):
        i = next(int(k) for k in serie.sort_values(ascending=False).index if int(k) not in usadas)    # sin repetir imágenes
        usadas.add(i)
        elegidas.append((titulo, i))
    resto = [i for i in range(len(etiquetas)) if i not in usadas]
    elegidas += [("Ejemplo al azar", k) for k in random.Random(SEMILLA).sample(resto, 3)]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.6))
    colores = colores_clases(80)
    for ax, (titulo, i) in zip(axs.flat, elegidas):
        im = cv2.imread(os.path.join(VALIDACION, etiquetas[i]["archivo"]))
        for x, y, w, h, c, mult in etiquetas[i]["cajas"]:
            if not mult:
                cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
        dibujar(im, [dd for dd in dets[i] if dd[1] >= u], nombres, colores)
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{titulo} (#{i}): TP {int(n_tp.get(i, 0))}, FP {int(n_fp.get(i, 0))}, FN {int(n_fn.get(i, 0))}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, "ejemplos.png"), dpi=200)
    plt.close()
    print("Figuras guardadas en img/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Actividad 19: detección de objetos en tiempo real con YOLO")
    ap.add_argument("--camara", action="store_true", help="usar la cámara web")
    ap.add_argument("--video", help="usar un archivo de video")
    ap.add_argument("--modelo", default="yolov3-tiny", choices=["yolov3-tiny", "yolov3"])
    ap.add_argument("--saltar", type=int, default=1, help="inferencia cada N fotogramas")
    ap.add_argument("--grabar", action="store_true", help="guardar el video con detecciones y un CSV de métricas")
    args = ap.parse_args()
    if args.camara or args.video:
        preparar_modelos()
        modo_ventana(args.video if args.video else 0, Config(modelo=args.modelo, saltar=args.saltar), args.grabar)
    else:
        main()
