# Reporte automático: regresión sobre Airfoil Self-Noise (UCI)

- Instancias: 1503; variables predictoras: 5; objetivo: presión sonora (dB).
- Montajes experimentales distintos: 106 (filas no independientes entre sí).
- Modelos: regresión lineal y bosque aleatorio.

## Métricas por modelo y método de partición

| modelo | partición | MAE | MSE | RMSE | R2 | RMSE / desv. est. de y |
|---|---|---|---|---|---|---|
| Regresión lineal | hold-out 80/20 | 3.6724 | 22.1286 | 4.7041 | 0.5583 | 0.6819 |
| Regresión lineal | 10-fold | 3.7524 | 23.3628 | 4.8335 | 0.5088 | 0.7006 |
| Regresión lineal | 10-fold por montajes | 3.785 | 23.7832 | 4.8768 | 0.4999 | 0.7069 |
| Bosque aleatorio | hold-out 80/20 | 1.3148 | 3.2952 | 1.8153 | 0.9342 | 0.2631 |
| Bosque aleatorio | 10-fold | 1.1898 | 2.8222 | 1.6799 | 0.9407 | 0.2435 |
| Bosque aleatorio | 10-fold por montajes | 1.89 | 6.8925 | 2.6254 | 0.8551 | 0.3806 |
| Regresión lineal | leave-one-out | 3.7447 | 23.2498 | 4.8218 | 0.5111 | 0.6989 |

## Variabilidad de la estimación según la partición (repeticiones)

| modelo | método | RMSE media | RMSE desv. | R2 media | R2 desv. |
|---|---|---|---|---|---|
| Regresión lineal | hold-out 80/20 | 4.8562 | 0.1949 | 0.5085 | 0.0395 |
| Regresión lineal | 10-fold | 4.823 | 0.0041 | 0.5109 | 0.0008 |
| Bosque aleatorio | hold-out 80/20 | 1.7579 | 0.1134 | 0.9352 | 0.0089 |
| Bosque aleatorio | 10-fold | 1.6671 | 0.0209 | 0.9416 | 0.0015 |
