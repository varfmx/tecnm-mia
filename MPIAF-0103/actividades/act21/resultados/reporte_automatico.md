# Reporte automático: clasificación de Breast Cancer Wisconsin (Diagnostic)

- Instancias: 569; variables: 30; clase positiva: maligno (212 casos, 37.3%).
- Clasificador: regresión logística con variables estandarizadas.
- Métrica principal: sensibilidad (un falso negativo es un cáncer no detectado).

## Métricas por método de partición

| método | VP | FN | FP | VN | exactitud | precisión | sensibilidad | especificidad | F1 | exactitud balanceada | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hold-out 70/30 | 60 | 4 | 1 | 106 | 0.9708 | 0.9836 | 0.9375 | 0.9907 | 0.96 | 0.9641 | 0.9975 |
| 10-fold | 202 | 10 | 4 | 353 | 0.9754 | 0.9806 | 0.9528 | 0.9888 | 0.9665 | 0.9708 | 0.9947 |
| leave-one-out | 205 | 7 | 5 | 352 | 0.9789 | 0.9762 | 0.967 | 0.986 | 0.9716 | 0.9765 | 0.9947 |

## Variabilidad de la exactitud según el método y el tamaño del conjunto

| n | método | exactitud media | desv. est. | mínima | máxima |
|---|---|---|---|---|---|
| 569 | hold-out 70/30 | 0.9791 | 0.011 | 0.9415 | 1.0 |
| 569 | 10-fold | 0.9784 | 0.0026 | 0.9701 | 0.9842 |
| 569 | leave-one-out | 0.9789 | 0.0 | 0.9789 | 0.9789 |
| 60 | hold-out 70/30 | 0.9372 | 0.0615 | 0.7778 | 1.0 |
| 60 | 10-fold | 0.9537 | 0.0129 | 0.9333 | 0.9833 |
| 60 | leave-one-out | 0.95 | 0.0 | 0.95 | 0.95 |
