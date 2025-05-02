import numpy as np
import random
import logging
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from maze_layout import GRID_WIDTH, GRID_HEIGHT, BASE_POS, build_maze


#CONFIG GLOBALE
logging.basicConfig(level=logging.INFO, format='%(message)s')
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_DELTAS = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
ALPHA = 0.2
GAMMA = 0.95
EPS_START = 0.3
EPS_MIN= 0.01
EPS_DECAY = 1000
MAX_STEPS= 100
N_EPISODES= 2000


WALLS = build_maze()

#États valides et mappages
valid_states = [(x,y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT) if (x,y) not in WALLS]
state_to_idx = {s:i for i,s in enumerate(valid_states)}

# AeFFICHAGE DU LABYRINTHE

def show_maze():
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-1, GRID_WIDTH); ax.set_ylim(-1, GRID_HEIGHT)
    ax.plot(*BASE_POS, 'ro', label='Base')
    for w in WALLS:
        ax.plot(*w, 'ks', markersize=6)
    ax.legend(); ax.grid(True); ax.set_title("Labyrinthe")
    plt.draw(); plt.pause(2); plt.close(fig)

#ENVIRONNEMENT
def is_valid(pos):
    x,y = pos
    return 0<=x<GRID_WIDTH and 0<=y<GRID_HEIGHT and pos not in WALLS

def step_env(pos, action):
    dx,dy = ACTION_DELTAS[action]
    nxt   = (pos[0]+dx, pos[1]+dy)
    if not is_valid(nxt):
        return pos, -5
    if nxt == BASE_POS:
        return nxt, +100
    return nxt, -1


#q-learning
def choose_action(q_table, idx, eps):
    if random.random() < eps:
        return random.randrange(len(ACTIONS))
    return int(np.argmax(q_table[idx]))

def train_q_learning():
    n_states = len(valid_states)
    q_table  = np.random.uniform(-1,1,(n_states,len(ACTIONS)))

    stats = {
        'rewards':   np.zeros(N_EPISODES),
        'steps':     np.zeros(N_EPISODES),
        'collisions':np.zeros(N_EPISODES),
        'goals':     np.zeros(N_EPISODES)
    }

    for ep in range(1, N_EPISODES+1):
        eps = max(EPS_MIN, EPS_START * np.exp(-ep/EPS_DECAY))
        #choisir un départ aléatoire différent de la base
        start = random.choice(valid_states)
        while start == BASE_POS:
            start = random.choice(valid_states)
        state_idx = state_to_idx[start]

        total_reward = collisions = 0

        for step in range(1, MAX_STEPS+1):
            aidx  = choose_action(q_table, state_idx, eps)
            action= ACTIONS[aidx]
            nxt,r = step_env(valid_states[state_idx], action)

            total_reward += r
            if nxt == valid_states[state_idx] and r<0:
                collisions += 1

            nxt_idx = state_to_idx.get(nxt, None)
            future_v= np.max(q_table[nxt_idx]) if nxt_idx is not None else 0

            # Q update
            q_table[state_idx,aidx] = (
                (1-ALPHA)*q_table[state_idx,aidx]
                + ALPHA*(r + GAMMA*future_v)
            )

            # transition
            if nxt_idx is None:
                #un mur invalide improbable
                nxt_idx = state_idx
            state_idx = nxt_idx

            if nxt == BASE_POS:
                stats['goals'][ep-1] = 1
                break

        stats['rewards'][ep-1]    = total_reward
        stats['steps'][ep-1]       = step
        stats['collisions'][ep-1]  = collisions

        if ep==1 or ep%50==0:
            logging.info(f"Ép {ep}/{N_EPISODES} | R={total_reward:.1f} | steps={step} | colls={collisions}")

    return q_table, stats


#SAUVEGARDE & CHARGEMENT

def save_q(q_table, fname="q_table.npy"):
    np.save(fname, q_table)

def load_q(fname="q_table.npy"):
    return np.load(fname)

#EXTRACTION DE POLITIQUE

def extract_policy(q_table):
    return {
        state: ACTIONS[int(np.argmax(q_table[state_to_idx[state]]))]
        for state in valid_states
    }

#SIMULATION & VISUALISATION

def simulate(policy, start):
    path = [start]; pos=start
    for _ in range(MAX_STEPS):
        if pos==BASE_POS: break
        act = policy.get(pos)
        if act is None: break
        pos,_ = step_env(pos, act)
        path.append(pos)
    return path

def plot_path(path):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-1, GRID_WIDTH)
    ax.set_ylim(-1, GRID_HEIGHT)
    ax.grid(True)
    #dessine murs & base
    for w in WALLS:
        ax.plot(*w, 'ks', markersize=6)
    ax.plot(*BASE_POS, 'ro', markersize=10)
    #préparer l'objet robot et la ligne de tracé
    robot, = ax.plot([], [], 'bo', markersize=8)
    line,  = ax.plot([], [], 'b-', linewidth=2)
    #animation pas à pas
    xs, ys = [], []
    for (x, y) in path:
        xs.append(x)
        ys.append(y)
        # met à jour la ligne (trace) et la position du robot
        line.set_data(xs, ys)
        robot.set_data([x], [y])
        plt.draw()
        plt.pause(0.4)
    #pause finale pour bien voir le chemin complet
    plt.pause(3)
    plt.close(fig)

def plot_stats(stats):
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(stats['rewards'])
    ax.set(title="Récompenses par épisode", xlabel="Épisode", ylabel="Récompense")
    plt.draw(); plt.pause(3); plt.close(fig)
#MAIN
if __name__=="__main__":
    # 1)Affiche le labyrinthe
    show_maze()
    # 2)Entraîne et récupère Q_final
    Q_final, stats = train_q_learning()
    # 3)Sauvegarde la Q-table finale
    save_q(Q_final, "q_table.npy")
    # 4)Charge-la (pour démonstration)
    Q = load_q("q_table.npy")
    # 5)Extrait la politique gloutonne
    policy = extract_policy(Q)
    # 6)Simule depuis un départ aléatoire
    start = random.choice([s for s in valid_states if s!=BASE_POS])
    print("Départ :", start)
    path = simulate(policy, start)
    # 7)Affiche la trajectoire et les stats
    plot_path(path)
    plot_stats(stats)
