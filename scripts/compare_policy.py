import numpy as np
import random
import logging
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from maze_layout import GRID_WIDTH, GRID_HEIGHT, build_maze

#PARAMÈTRES GLOBAUX

ACTIONS = ['up', 'down', 'left', 'right']
ACTION_DELTAS = {'up': (0,1), 'down': (0,-1), 'left': (-1,0), 'right': (1,0)}
ALPHA = 0.2; gamma = 0.95
EPSILON_START = 0.3; EPSILON_MIN = 0.01; decay_rate = 1000
MAX_STEPS = 100; N_EPISODES = 2000
BASE_POSITION = (1,1)
SNAPSHOTS = [1, 100, 500, 1000, 2000]


#DESSIN DU LABYRINTHE

WALLS = build_maze()


#ÉTATS & Q-TABLE

valid_states = [(x,y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT) if (x,y) not in WALLS]
state_to_idx = {s:i for i,s in enumerate(valid_states)}
idx_to_state = {i:s for s,i in state_to_idx.items()}
Q = np.random.uniform(-1,1,(len(valid_states), len(ACTIONS)))

#ENVIRONNEMENT & ACTION

def is_valid(pos):
    x,y = pos
    return 0<=x<GRID_WIDTH and 0<=y<GRID_HEIGHT and pos not in WALLS

def take_action(pos, action):
    dx,dy = ACTION_DELTAS[action]
    np_ = (pos[0]+dx, pos[1]+dy)
    if not is_valid(np_): return pos, -5
    if np_==BASE_POSITION: return np_, +100
    return np_, -1

def choose_action(idx, eps):
    if random.random()<eps: return random.randrange(len(ACTIONS))
    return int(np.argmax(Q[idx]))

# TRAINING + SNAPSHOTS

logging.basicConfig(level=logging.INFO, format='%(message)s')
snap_q = {}

for ep in range(1, N_EPISODES+1):
    eps = max(EPSILON_MIN, EPSILON_START * np.exp(-ep/decay_rate))
    start = random.choice(valid_states)
    while start==BASE_POSITION: start = random.choice(valid_states)
    state, sidx = start, state_to_idx[start]
    for step in range(MAX_STEPS):
        aidx = choose_action(sidx, eps)
        action = ACTIONS[aidx]
        nxt, r = take_action(state, action)
        ni = state_to_idx.get(nxt, None)
        fut = np.max(Q[ni]) if ni is not None else 0
        Q[sidx,aidx] = (1-ALPHA)*Q[sidx,aidx] + ALPHA*(r + gamma*fut)
        state, sidx = (nxt, ni) if ni is not None else (state, sidx)
        if nxt==BASE_POSITION: break
    if ep in SNAPSHOTS:
        snap_q[ep] = Q.copy()
    if ep%50==0 or ep==1:
        logging.info(f"Épisode {ep}/{N_EPISODES}")

#FONCTION DE TRAÇAGE (CHAMP DE VECTEURS)

def plot_policy(ax, Qmat, title):
    X,Y,U,V = [],[],[],[]
    for idx, row in enumerate(Qmat):
        state = idx_to_state[idx]
        best = int(np.argmax(row))
        dx,dy = ACTION_DELTAS[ACTIONS[best]]
        X.append(state[0]); Y.append(state[1])
        U.append(dx); V.append(dy)
    ax.quiver(X,Y,U,V, angles='xy', scale_units='xy', scale=1)
    # murs & base
    for w in WALLS: ax.plot(*w,'ks',markersize=3)
    ax.plot(*BASE_POSITION,'ro',markersize=5)
    ax.set_xlim(-1,GRID_WIDTH); ax.set_ylim(-1,GRID_HEIGHT)
    ax.set_aspect('equal'); ax.set_title(title); ax.axis('off')

#AFFICHAGE COMPARATIF

fig, axes = plt.subplots(2,3, figsize=(12,8))
axes = axes.flatten()

for i, ep in enumerate(SNAPSHOTS):
    plot_policy(axes[i], snap_q[ep], f"Ép. {ep}")
axes[-1].axis('off')  # case vide
plt.tight_layout()
plt.savefig("outputs_images/compare_policy.png")
