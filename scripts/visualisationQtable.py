import numpy as np
import matplotlib            # importer matplotlib AVANT pyplot
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from maze_layout import GRID_WIDTH, GRID_HEIGHT, build_maze


# -------------------
# Constantes
# -------------------
ACTIONS = ['up', 'down', 'left', 'right']
BASE_POSITION = (1, 1)

# -------------------
# Définition des murs (labyrinthe complet)
# -------------------
WALLS = build_maze()

# -------------------
# Chargement de la Q-table
# -------------------
Q = np.load("q_table.npy")

# -------------------
# États valides & mapping
# -------------------
valid_states = [
    (x, y)
    for x in range(GRID_WIDTH)
    for y in range(GRID_HEIGHT)
    if (x, y) not in WALLS
]
state_to_idx = {s: i for i, s in enumerate(valid_states)}
idx_to_state = {i: s for s, i in state_to_idx.items()}

# -------------------
# Map directionnelle
# -------------------
direction_map = {
    0: (0, 1),   # up
    1: (0, -1),  # down
    2: (-1, 0),  # left
    3: (1, 0),   # right
}

# -------------------
# Préparation du champ
# -------------------
X, Y, U, V, C, LW = [], [], [], [], [], []
max_q = np.max(Q)
max_dist = np.hypot(GRID_WIDTH, GRID_HEIGHT)

for idx, q_vals in enumerate(Q):
    x, y = idx_to_state[idx]
    best_a = int(np.argmax(q_vals))
    dx, dy = direction_map[best_a]
    # couleur normalisée (0..1)
    c = q_vals[best_a] / max_q
    # épaisseur selon proximité base
    dist = np.hypot(x - BASE_POSITION[0], y - BASE_POSITION[1])
    lw = 0.5 + (1 - dist / max_dist) * 2.5  # [0.5,3.0]

    X.append(x); Y.append(y)
    U.append(dx); V.append(dy)
    C.append(c); LW.append(lw)

# -------------------
# Tracé
# -------------------
plt.figure(figsize=(10, 10))
quiv = plt.quiver(
    X, Y, U, V, C,
    angles='xy', scale_units='xy', scale=1,
    cmap='viridis', linewidth=LW
)
plt.colorbar(quiv, label='Q-value normalisée')

# Base et murs
plt.plot(*BASE_POSITION, 'ro', markersize=10, label='Base')
for w in WALLS:
    plt.plot(*w, 'ks', markersize=6)

plt.grid(True)
plt.xlim(-1, GRID_WIDTH)
plt.ylim(-1, GRID_HEIGHT)
plt.gca().set_aspect('equal')
plt.title("Politique (couleur=Q, épaisseur∝proximité base)")
plt.legend()
plt.savefig("outputs_images/policy_vector_field_enhanced.png")
plt.close()
