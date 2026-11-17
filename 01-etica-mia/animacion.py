import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. FUNCIÓN OBJETIVO Y CONFIGURACIÓN
# ==========================================

def rastrigin(x, y):
    """Función de Rastrigin en 2D con mínimo global en (0, 0)."""
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

# Límites del espacio de búsqueda
BOUNDS = np.array([-5.12, 5.12])

# Hiperparámetros de PSO
NUM_PARTICLES = 40
DIMENSIONS = 2
MAX_ITERATIONS = 100

W = 0.7    # Inercia
C1 = 1.5   # Coeficiente cognitivo
C2 = 1.5   # Coeficiente social

# ==========================================
# 2. INICIALIZACIÓN DEL ENJAMBRE
# ==========================================

np.random.seed(42)
positions = np.random.uniform(BOUNDS[0], BOUNDS[1], (NUM_PARTICLES, DIMENSIONS))
velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, DIMENSIONS))

pbest_positions = np.copy(positions)
pbest_scores = np.array([rastrigin(p[0], p[1]) for p in positions])

gbest_index = np.argmin(pbest_scores)
gbest_position = np.copy(pbest_positions[gbest_index])
gbest_score = pbest_scores[gbest_index]

# ==========================================
# 3. PREPARACIÓN DE LA FIGURA 2D
# ==========================================

fig, ax = plt.subplots(figsize=(8, 6))

# Crear el mapa de contorno de la Función de Rastrigin
X = np.linspace(BOUNDS[0], BOUNDS[1], 200)
Y = np.linspace(BOUNDS[0], BOUNDS[1], 200)
X_mesh, Y_mesh = np.meshgrid(X, Y)
Z_mesh = rastrigin(X_mesh, Y_mesh)

contour = ax.contourf(X_mesh, Y_mesh, Z_mesh, levels=30, cmap='viridis')
fig.colorbar(contour, label='Valor de f(x, y)')

# Elementos gráficos a actualizar
particles_scatter = ax.scatter(positions[:, 0], positions[:, 1], c='cyan', edgecolors='black', s=50, label='Partículas')
gbest_scatter = ax.scatter(gbest_position[0], gbest_position[1], c='red', marker='*', s=200, label='Mejor Global ($g_{best}$)')

ax.set_xlim(BOUNDS)
ax.set_ylim(BOUNDS)
ax.set_xlabel('Variable X')
ax.set_ylabel('Variable Y')
ax.set_title('Optimización por Enjambre de Partículas (PSO) - Iteración 0')
ax.legend(loc='upper right')

# ==========================================
# 4. FUNCIÓN DE ACTUALIZACIÓN DE ANIMACIÓN
# ==========================================

def update(frame):
    global positions, velocities, pbest_positions, pbest_scores, gbest_position, gbest_score

    # Actualizar partículas
    for i in range(NUM_PARTICLES):
        r1 = np.random.rand(DIMENSIONS)
        r2 = np.random.rand(DIMENSIONS)

        cognitive = C1 * r1 * (pbest_positions[i] - positions[i])
        social = C2 * r2 * (gbest_position - positions[i])
        velocities[i] = W * velocities[i] + cognitive + social

        positions[i] = positions[i] + velocities[i]
        positions[i] = np.clip(positions[i], BOUNDS[0], BOUNDS[1])

        current_score = rastrigin(positions[i][0], positions[i][1])

        if current_score < pbest_scores[i]:
            pbest_scores[i] = current_score
            pbest_positions[i] = np.copy(positions[i])

            if current_score < gbest_score:
                gbest_score = current_score
                gbest_position = np.copy(positions[i])

    # Actualizar la representación gráfica
    particles_scatter.set_offsets(positions)
    gbest_scatter.set_offsets(gbest_position)
    ax.set_title(f'Optimización por Enjambre de Partículas (PSO) - Iteración {frame + 1}\n'
                 f'Mejor Valor Global: {gbest_score:.4f}')

    return particles_scatter, gbest_scatter

# Crear la animación
ani = animation.FuncAnimation(fig, update, frames=MAX_ITERATIONS, interval=150, blit=False, repeat=False)

plt.tight_layout()
plt.show()