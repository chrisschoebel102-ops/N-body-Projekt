import numpy as np
import time

from matplotlib.pyplot import pause
from numba import cuda, float32
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# ─── Parameter ───────────────────────────────────────────────────────────────
N      = 10000
t_end  = np.float32(60.0)
steps  = 800
G      = np.float32(1.0)
EPS2   = np.float32(0.25)   # 0.5² vorab

rng    = np.random.default_rng(42)
pos0   = rng.uniform(-100, 100, (N, 2)).astype(np.float32)
vel0   = rng.uniform(-1,   1,   (N, 2)).astype(np.float32)
masses = rng.uniform(0.01,  5,   N    ).astype(np.float32)

THREADS = 128
BLOCKS  = math.ceil(N / THREADS)
TILE    = THREADS   # Tile-Größe = Thread-Anzahl pro Block

# ─── Kernel 1: Beschleunigungen (mit Shared Memory) ──────────────────────────
@cuda.jit(fastmath=True)
def gravity_kernel(pos, masses, acc, G, eps2):
    i  = cuda.grid(1)
    tx = cuda.threadIdx.x
    n  = pos.shape[0]

    s_pos  = cuda.shared.array((128, 2), dtype=float32)
    s_mass = cuda.shared.array((128,),   dtype=float32)

    zero = float32(0.0)
    one = float32(1.0)

    xi = zero
    yi = zero
    if i < n:
        xi = pos[i, 0]
        yi = pos[i, 1]

    ax = zero
    ay = zero

    for tile_start in range(0, n, TILE):
        j_load = tile_start + tx
        if j_load < n:
            s_pos[tx, 0] = pos[j_load, 0]
            s_pos[tx, 1] = pos[j_load, 1]
            s_mass[tx]   = masses[j_load]
        else:
            s_pos[tx, 0] = zero
            s_pos[tx, 1] = zero
            s_mass[tx]   = zero

        cuda.syncthreads()

        if i < n:
            for k in range(TILE):
                dx = s_pos[k, 0] - xi
                dy = s_pos[k, 1] - yi

                r2 = dx * dx + dy * dy + eps2

                inv_r = one / math.sqrt(r2)
                inv_r3 = inv_r * inv_r * inv_r

                fac = G * s_mass[k] * inv_r3

                ax += fac * dx
                ay += fac * dy

        cuda.syncthreads()

    if i < n:
        acc[i, 0] = ax
        acc[i, 1] = ay


# ─── Kernel 2: Leapfrog-Update (komplett auf GPU) ────────────────────────────
@cuda.jit(fastmath=True)
def drift_kernel(pos, vel, acc, dt):
    i = cuda.grid(1)
    if i >= pos.shape[0]:
        return

    half = float32(0.5)

    vel[i, 0] += half * dt * acc[i, 0]
    vel[i, 1] += half * dt * acc[i, 1]

    pos[i, 0] += dt * vel[i, 0]
    pos[i, 1] += dt * vel[i, 1]


@cuda.jit(fastmath=True)
def kick_kernel(vel, acc, dt):
    i = cuda.grid(1)
    if i >= vel.shape[0]:
        return

    half = float32(0.5)

    vel[i, 0] += half * dt * acc[i, 0]
    vel[i, 1] += half * dt * acc[i, 1]

# ─── Kernel 3: Trajektorie speichern ─────────────────────────────────────────
@cuda.jit(fastmath=True)
def save_traj_kernel(pos, traj, step):
    i = cuda.grid(1)
    if i < pos.shape[0]:
        traj[step, i, 0] = pos[i, 0]
        traj[step, i, 1] = pos[i, 1]


# ─── Hauptschleife: alles auf GPU ────────────────────────────────────────────
def simulate_cuda(pos0, vel0, masses):
    dt = np.float32(t_end / steps)

    pos      = cuda.to_device(pos0.copy())
    vel      = cuda.to_device(vel0.copy())
    m        = cuda.to_device(masses)
    acc      = cuda.device_array((N, 2), dtype=np.float32)
    traj_gpu = cuda.device_array((steps + 1, N, 2), dtype=np.float32)

    # Schritt 0 speichern + Initialbeschleunigung
    save_traj_kernel[BLOCKS, THREADS](pos, traj_gpu, 0)
    gravity_kernel[BLOCKS, THREADS](pos, m, acc, G, EPS2)

    for step in range(steps):
        # 1. Geschwindigkeit halb aktualisieren und Position bewegen
        drift_kernel[BLOCKS, THREADS](pos, vel, acc, dt)

        # 2. Beschleunigung an der neuen Position berechnen
        gravity_kernel[BLOCKS, THREADS](pos, m, acc, G, EPS2)

        # 3. Geschwindigkeit mit neuer Beschleunigung fertig aktualisieren
        kick_kernel[BLOCKS, THREADS](vel, acc, dt)

        # 4. Position speichern
        save_traj_kernel[BLOCKS, THREADS](pos, traj_gpu, step + 1)

    cuda.synchronize()
    return traj_gpu.copy_to_host()   # Nur 1 Transfer am Ende!


# ─── Starten ─────────────────────────────────────────────────────────────────
start = time.time()
traj  = simulate_cuda(pos0, vel0, masses)
print(f"Simulation: {time.time() - start:.2f} s")

# ─── Visualisierung ───────────────────────────────────────────────────────────
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(19.2, 10.8))

try:
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
except Exception:
    pass

# Ganze Fensterfläche für den Plot benutzen
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_position([0, 0, 1, 1])
ax.axis('off')

# Wie viele Frames animiert werden sollen
MAX_FRAMES = 60
FRAME_SKIP = 1

# ─── Bildausschnitt so setzen, dass die Daten das Fenster füllen ─────────────
view_traj = traj[:MAX_FRAMES]

# x_min, x_max = np.percentile(view_traj[:, :, 0], [0.5, 99.5])
# y_min, y_max = np.percentile(view_traj[:, :, 1], [0.5, 99.5])

# Für 100.000 Teilchen
x_min, x_max = np.percentile(view_traj[:, :, 0], [5.0, 95.0])
y_min, y_max = np.percentile(view_traj[:, :, 1], [5.0, 95.0])

margin = 0.03

x_range = x_max - x_min
y_range = y_max - y_min

x_min -= x_range * margin
x_max += x_range * margin
y_min -= y_range * margin
y_max += y_range * margin

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_aspect('auto')


# ─── Teilchen als ein einziges Scatter-Objekt ─────────────────────────────────
particles = ax.scatter(
    traj[0, :, 0],
    traj[0, :, 1],
    s=0.4,
    c='white',
    linewidths=0,
    zorder=20
)


# ─── Trails als eine einzige LineCollection ──────────────────────────────────
colors = plt.cm.hsv(np.linspace(0, 1, N))

trails = LineCollection(
    [],
    colors=colors,
    linewidths=0.35,
    alpha=0.9,
    zorder=10
)

ax.add_collection(trails)


def update(frame):
    particles.set_offsets(traj[frame])

    # Form: (N, frame+1, 2)
    segments = np.transpose(traj[:frame + 1], (1, 0, 2))
    trails.set_segments(segments)

    return particles, trails


ani = animation.FuncAnimation(
    fig,
    update,
    frames=range(0, MAX_FRAMES, FRAME_SKIP),
    interval=1,
    blit=False,
    cache_frame_data=False,
    repeat=False
)

plt.show()