import numpy as np
import time
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ─── Simulationsparameter ───────────────────────────────────────────────────

N       = 100
t_end   = 60.0
steps   = 800
G       = 1.0
eps     = 0.5          # Softening-Parameter
EPS2    = eps * eps    # vorab berechnet

rng = np.random.default_rng(42)

# Initialzustände als flache NumPy-Arrays
pos0 = rng.uniform(-100, 100, (N, 2))
vel0 = rng.uniform(-1,   1,   (N, 2))
masses = rng.uniform(0.01, 5,  N)

# Zustandsvektor für odeint: [x0,y0, x1,y1, ..., vx0,vy0, vx1,vy1, ...]
var0 = np.concatenate([pos0.ravel(), vel0.ravel()])


# ─── Numba-Kernfunktionen ────────────────────────────────────────────────────

def vectorfield_nb(var, t, masses, G, eps2):
    """Rechte Seite der Bewegungsgleichungen – kompiliert zu Maschinencode."""
    n = len(masses)
    f = np.empty(4 * n)

    # Geschwindigkeiten direkt übernehmen (d/dt pos = vel)
    for i in range(n):
        f[2*i]     = var[2*n + 2*i]      # vx_i
        f[2*i + 1] = var[2*n + 2*i + 1]  # vy_i

    # Beschleunigungen aus Gravitationswechselwirkungen
    for i in range(n):
        ax, ay = 0.0, 0.0
        xi, yi = var[2*i], var[2*i + 1]
        for j in range(n):
            if i == j:
                continue
            dx = var[2*j]     - xi
            dy = var[2*j + 1] - yi
            dist32 = (dx*dx + dy*dy + eps2) ** 1.5
            fac = G * masses[j] / dist32
            ax += fac * dx
            ay += fac * dy
        f[2*n + 2*i]     = ax
        f[2*n + 2*i + 1] = ay

    return f


# ─── Wrapper für odeint (braucht Signatur f(y, t)) ──────────────────────────

def vectorfield(var, t):
    return vectorfield_nb(var, t, masses, G, EPS2)


# ─── Simulation ─────────────────────────────────────────────────────────────

# Einmaliger JIT-Warmup (kompiliert beim ersten Aufruf)
start = time.time()

_ = vectorfield_nb(var0, 0.0, masses, G, EPS2)

t_span = np.linspace(0, t_end, steps + 1)
sol = odeint(vectorfield, var0, t_span)          # sol: (steps+1, 4*N)

# ─── Ergebnisse aufbereiten ─────────────────────────────────────────────────

sol_pos = sol[:, :2*N].reshape(steps + 1, N, 2)   # (T, N, 2)
sol_vel = sol[:, 2*N:].reshape(steps + 1, N, 2)

duration = time.time() - start
print(f"Simulation abgeschlossen in {duration:.2f} s")


# ─── Visualisierung ─────────────────────────────────────────────────────────

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(19.2, 10.8))

manager = plt.get_current_fig_manager()
manager.window.state('zoomed')


# ax.set_aspect('equal')
# # ax.set_xlim(-100, 100)
# # ax.set_ylim(-100, 100)
ax.axis('off')

circles = []
lines   = []
for i in range(N):
    c = plt.Circle(sol_pos[0, i], 0.08, ec='w', lw=2.5, zorder=20)
    ax.add_patch(c)
    circles.append(c)
    (ln,) = ax.plot([], [])
    lines.append(ln)

def update(time_val):
    idx = int(np.rint(time_val * steps / t_end))
    for j in range(N):
        circles[j].center = sol_pos[idx, j]
        lines[j].set_data(sol_pos[:idx+1, j, 0], sol_pos[:idx+1, j, 1])
    fig.canvas.draw_idle()

for k in range(60):        
    update(k)
    plt.pause(0.09)
plt.show()