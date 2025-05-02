# Q-Learning pour navigation d'un robot dans un labyrinthe

Ce projet implémente un agent intelligent capable d'apprendre à naviguer dans un environnement discret en utilisant l'algorithme de **Q-learning**. L'agent apprend à atteindre une base en évitant les murs d'un labyrinthe codé manuellement.


## Prérequis

- Python 3.7+
- NumPy
- Matplotlib (backend TkAgg conseillé)

Installation (si besoin) :
```bash
pip install numpy matplotlib
```
# Exécution
Lancer le script principal :

```bash
python main.py
```
Des visualisations seront générées automatiquement dans outputs_images/ en éxécutant les autres fichiers, lire le rapport pour comprendre le rôle de chaque.

# Objectif
Apprendre une politique optimale pour atteindre la base (1,1) dans un environnement à récompenses fixes. Le robot explore l’environnement à l’aide d’une stratégie ε-greedy, puis apprend par renforcement les meilleures actions à adopter selon sa position.
