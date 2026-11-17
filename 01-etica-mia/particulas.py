import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. REPRESENTACIÓN DEL PROBLEMA
# ==========================================

def rastrigin(x):
    """
    Función objetivo de Rastrigin en 2D.
    Mínimo global en (0, 0) con valor f(0, 0) = 0.
    """
    A = 10
    return A * 2 + (x[0]**2 - A * np.cos(2 * np.pi * x[0])) + (x[1]**2 - A * np.cos(2 * np.pi * x[1]))

# Restricciones del espacio de búsqueda
BOUNDS = np.array([-5.12, 5.12])

# ==========================================
# 2. HIPERPARÁMETROS Y CONFIGURACIÓN
# ==========================================

NUM_PARTICLES = 30
DIMENSIONS = 2
MAX_ITERATIONS = 100

W = 0.7    # Inercia
C1 = 1.5   # Componente cognitivo
C2 = 1.5   # Componente social

# ==========================================
# 3. ALGORITMO PSO
# ==========================================

def run_pso():
    # Inicialización de posiciones y velocidades
    np.random.seed(42)  # Semilla para reproducibilidad
    positions = np.random.uniform(BOUNDS[0], BOUNDS[1], (NUM_PARTICLES, DIMENSIONS))
    velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, DIMENSIONS))

    # Inicialización de pbest (mejores individuales)
    pbest_positions = np.copy(positions)
    pbest_scores = np.array([rastrigin(p) for p in positions])

    # Inicialización de gbest (mejor global)
    gbest_index = np.argmin(pbest_scores)
    gbest_position = np.copy(pbest_positions[gbest_index])
    gbest_score = pbest_scores[gbest_index]

    # Historial para visualización de convergencia
    history_gbest = [gbest_score]

    # Criterio de parada: número máximo de iteraciones
    for iteration in range(MAX_ITERATIONS):
        for i in range(NUM_PARTICLES):
            # Términos estocásticos r1 y r2 ~ U(0, 1)
            r1 = np.random.rand(DIMENSIONS)
            r2 = np.random.rand(DIMENSIONS)

            # Actualización de velocidad
            cognitive = C1 * r1 * (pbest_positions[i] - positions[i])
            social = C2 * r2 * (gbest_position - positions[i])
            velocities[i] = W * velocities[i] + cognitive + social

            # Actualización de posición
            positions[i] = positions[i] + velocities[i]

            # Aplicar restricciones de frontera (clipping)
            positions[i] = np.clip(positions[i], BOUNDS[0], BOUNDS[1])

            # Evaluación de la nueva posición
            current_score = rastrigin(positions[i])

            # Actualizar pbest si la nueva posición es mejor
            if current_score < pbest_scores[i]:
                pbest_scores[i] = current_score
                pbest_positions[i] = np.copy(positions[i])

                # Actualizar gbest si supera al récord global
                if current_score < gbest_score:
                    gbest_score = current_score
                    gbest_position = np.copy(positions[i])

        history_gbest.append(gbest_score)

    return gbest_position, gbest_score, history_gbest

# ==========================================
# 4. EJECUCIÓN Y VISUALIZACIÓN
# ==========================================

best_solution, best_value, history = run_pso()

print("=" * 45)
print(" RESULTADOS DE LA OPTIMIZACIÓN PSO")
print("=" * 45)
print(f"Mejor solución encontrada (x, y) : {best_solution}")
print(f"Valor óptimo alcanzado f(x, y)   : {best_value:.6f}")
print("=" * 45)

# Gráfica de convergencia
plt.figure(figsize=(8, 4))
plt.plot(history, color='navy', linewidth=2, label='Mejor valor global ($g_{best}$)')
plt.title('Curva de Convergencia - PSO en Función de Rastrigin')
plt.xlabel('Iteración')
plt.ylabel('Valor de la Función Objetivo')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()