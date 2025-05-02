# maze_layout.py

GRID_WIDTH  = 20
GRID_HEIGHT = 22
BASE_POS    = (1, 1)

def build_maze():
    walls = set()

    for x in range(GRID_WIDTH):
        walls.add((x, 0))
        walls.add((x, GRID_HEIGHT-1))
    for y in range(GRID_HEIGHT):
        walls.add((0, y))
        walls.add((GRID_WIDTH-1, y))

    # 2) Premier mur horizontal sous la bordure
    #    (de x=4 à x=15, à y=18)
    for x in range(4, 16):
        walls.add((x, 18))

    # 3) Deux montants verticaux sous ce mur
    #
    for y in range(12, 15):
        walls.add((4, y))
        walls.add((12, y))

    # 4) Montant vertical central prolongeant ce mur
    #    (à x=9, pour y=14..19)
    for y in range(14, 20):
        walls.add((9, y))

    # 5) Grand mur horizontal médian
    #    (de x=2 à x=17, à y=11)
    for x in range(2, 18):
        walls.add((x, 11))

    # 6) Barres horizontales basses
    #    - gauche  : x=2..7 à y=5
    #    - droite  : x=12..17 à y=5
    for x in range(2, 8):
        walls.add((x, 5))
    for x in range(12, 18):
        walls.add((x, 5))

    # 7) Montant vertical bas-centre
    #    (à x=9, pour y=1..5)
    for y in range(1, 6):
        walls.add((9, y))

    return walls
