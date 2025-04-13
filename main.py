import numpy as np
import random
import pickle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------------------
# PARAMÈTRES GLOBAUX
# ---------------------
GRID_WIDTH = 20
GRID_HEIGHT = 22
ACTIONS = ['up', 'down', 'left', 'right']
ALPHA = 0.2
GAMMA = 0.95
EPSILON = 0.3
MAX_STEPS = 100
N_EPISODES = 2000
BASE_POSITION = (1, 1)

# ---------------------
# DÉFINITION DES MURS
# ---------------------
WALLS = {
    (10,y) for y in range(GRID_HEIGHT+1) if y not in [5,6,7,8]  # Passage plus large
}

# ---------------------
# INITIALISATION Q-TABLE
# ---------------------
def init_q_table():
    return {(x, y): {a: np.random.uniform(-1, 1) for a in ACTIONS}  # Valeurs plus petites
            for x in range(GRID_WIDTH)
            for y in range(GRID_HEIGHT)
            if (x, y) not in WALLS}

# ---------------------
# ENVIRONNEMENT
# ---------------------
def is_valid_position(pos):
    x, y = pos
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT and pos not in WALLS

def take_action(pos, action):
    x, y = pos
    if action == 'up': y += 1
    elif action == 'down': y -= 1
    elif action == 'left': x -= 1
    elif action == 'right': x += 1
    new_pos = (x, y)
    if not is_valid_position(new_pos):
        return pos, -5
    elif new_pos == BASE_POSITION:
        return new_pos, 100
    else:
        return new_pos, -0.3

def choose_action(state, Q):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    return max(Q[state], key=Q[state].get)

# ---------------------
# STATISTIQUES
# ---------------------
episode_stats = {
    'episode': [],
    'total_reward': [],
    'steps': [],
    'outcome': [],  # 'goal', 'wall', 'timeout'
    'random_actions': [],
    'wall_hits': []
}

# ---------------------
# APPRENTISSAGE Q-LEARNING
# ---------------------
def train_q_learning(Q):
    visited_states = set()
    for episode in range(N_EPISODES):
        visited_states = set()
        epsilon = max(0.01, EPSILON * (1 - episode / N_EPISODES))
        state = random.choice([s for s in Q if s != BASE_POSITION])
        total_reward = 0
        nb_random = 0
        wall_hits = 0
        outcome = 'timeout'
        visited_states.add(state)
        for step in range(MAX_STEPS):
            # Stratégie epsilon-greedy
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
                nb_random += 1
            else:
                action = choose_action(state, Q)

            next_state, reward = take_action(state, action)

            # Mise à jour Q-table
            old_q = Q[state][action]
            max_next_q = max(Q[next_state].values()) if next_state in Q else 0
            Q[state][action] = (1 - ALPHA) * old_q + ALPHA * (reward + GAMMA * max_next_q)

            total_reward += reward
            if reward == -1:
                wall_hits += 1

            if next_state == BASE_POSITION:
                outcome = 'goal'
                break
            elif next_state == state and reward == -1:
                outcome = 'wall'
                break
            elif next_state in visited_states:
                outcome = 'loop'
                break
            state = next_state

        # Statistiques de l'épisode
        episode_stats['episode'].append(episode + 1)
        episode_stats['total_reward'].append(total_reward)
        episode_stats['steps'].append(step + 1)
        episode_stats['outcome'].append(outcome)
        episode_stats['random_actions'].append(nb_random)
        episode_stats['wall_hits'].append(wall_hits)

        if (episode + 1) % 50 == 0 or episode == 0:
            print(f"Épisode {episode+1}/{N_EPISODES} | Récompense: {total_reward:.1f} | Fin: {outcome} | Steps: {step+1}")
    return Q

# ---------------------
# SAUVEGARDE ET CHARGEMENT
# ---------------------
def save_q_table(Q, filename="q_table.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(Q, f)

def load_q_table(filename="q_table.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# ---------------------
# EXTRAIRE POLITIQUE
# ---------------------
def get_policy(Q):
    return {s: max(Q[s], key=Q[s].get) for s in Q}

# ---------------------
# SIMULATION D'UN TRAJET
# ---------------------
def simulate(policy, start):
    path = [start]
    pos = start
    for _ in range(MAX_STEPS):
        if pos not in policy:
            break
        action = policy[pos]
        pos, _ = take_action(pos, action)
        path.append(pos)
        if pos == BASE_POSITION:
            break
    return path

# ---------------------
# VISUALISATION DU TRAJET
# ---------------------
def plot_path(path, walls=WALLS):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, GRID_WIDTH)
    ax.set_ylim(-1, GRID_HEIGHT)
    ax.set_aspect('equal')
    plt.grid(True)

    for wall in walls:
        ax.plot(*wall, 'ks', markersize=6)

    ax.plot(*BASE_POSITION, 'ro', markersize=12, label="Base")
    robot, = ax.plot([], [], 'bo', markersize=8)

    def init():
        robot.set_data([], [])
        return robot,

    def animate(i):
        x, y = path[i]
        robot.set_data([x], [y])
        return robot,

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(path), interval=400, blit=True, repeat=False)

    plt.legend()
    plt.show()

# ---------------------
# STATISTIQUES EN GRAPHIQUES
# ---------------------
def plot_stats(stats):
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.flatten()

    axs[0].plot(stats['episode'], stats['total_reward'])
    axs[0].set_title("Récompense totale par épisode")

    axs[1].plot(stats['episode'], stats['steps'])
    axs[1].set_title("Nombre de pas par épisode")

    axs[2].plot(stats['episode'], stats['random_actions'])
    axs[2].set_title("Actions aléatoires par épisode")

    axs[3].plot(stats['episode'], stats['wall_hits'])
    axs[3].set_title("Collisions contre les murs")

    outcomes = {'goal': [], 'wall': [], 'timeout': []}
    for i, o in enumerate(stats['outcome']):
        for k in outcomes:
            outcomes[k].append(1 if o == k else 0)
    for i, (k, v) in enumerate(outcomes.items(), 4):
        axs[i].plot(stats['episode'], v, label=k)
        axs[i].set_title(f"Épisodes terminés par {k}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()

# ---------------------
# PIPELINE COMPLÈTE
# ---------------------
if __name__ == "__main__":
    Q = init_q_table()
    Q = train_q_learning(Q)
    save_q_table(Q)

    policy = get_policy(Q)
    start_position = (15, 15)
    path = simulate(policy, start_position)

    plot_path(path)
    plot_stats(episode_stats)
