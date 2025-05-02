import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from maze_layout import GRID_WIDTH, GRID_HEIGHT, build_maze
# Paramètres de grille
ACTIONS = ['up', 'down', 'left', 'right']
# Définition des murs
WALLS = build_maze()

#Q-table

Q = np.load("q_table.npy")  # shape (n_states, 4)

valid_states = [
    (x, y)
    for x in range(GRID_WIDTH)
    for y in range(GRID_HEIGHT)
    if (x, y) not in WALLS
]
state_to_idx = {s: i for i, s in enumerate(valid_states)}

#1)Affichage des 4 heatmaps par action

vmin = np.nanmin(Q)
vmax = np.nanmax(Q)
for action_idx, action in enumerate(ACTIONS):
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), np.nan)
    for (x, y) in valid_states:
        idx = state_to_idx[(x, y)]
        grid[y, x] = Q[idx, action_idx]
    plt.figure(figsize=(6, 5))
    im = plt.imshow(grid, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Q-value')
    plt.title(f"Heatmap Q pour l'action '{action}'")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(False)
    plt.savefig(f"outputs_images/heatmap_Q_{action}.png")
    plt.close()


# 2)Affichage de la heatmap de V(s) = max_a Q(s,a)

V_grid = np.full((GRID_HEIGHT, GRID_WIDTH), np.nan)
for (x, y) in valid_states:
    idx = state_to_idx[(x, y)]
    V_grid[y, x] = np.max(Q[idx])

plt.figure(figsize=(7, 6))
im = plt.imshow(V_grid, origin='lower', cmap='plasma',
                vmin=np.nanmin(V_grid), vmax=np.nanmax(V_grid))
plt.colorbar(im, label='V(s) = maxₐ Q(s,a)')
plt.title("Heatmap des valeurs d'état V(s)")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(False)
plt.savefig("outputs_images/value_map_Vs.png")
plt.close()
