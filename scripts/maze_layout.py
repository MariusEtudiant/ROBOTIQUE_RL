# maze_layout.py
#peu comporter toutes les config possibles, à modifier comme on le souhaite

GRID_WIDTH = 20
GRID_HEIGHT= 22
BASE_POS = (1, 1)

def build_maze():
    walls = set()
    for x in range(GRID_WIDTH):
        walls.add((x, 0))
        walls.add((x, GRID_HEIGHT-1))
    for y in range(GRID_HEIGHT):
        walls.add((0, y))
        walls.add((GRID_WIDTH-1, y))
    # 2) Premier mur horizontal sous la bordure
    for x in range(4, 16):
        walls.add((x, 18))
    # 3) Deux montants verticaux sous ce mur
    for y in range(12, 15):
        walls.add((4, y))
        walls.add((12, y))
    # 4) Montant vertical central prolongeant ce mur
    for y in range(14, 20):
        walls.add((9, y))
    # 5) Grand mur horizontal médian
    for x in range(2, 18):
        walls.add((x, 11))
    # 6) Barres horizontales basses
    for x in range(2, 8):
        walls.add((x, 5))
    for x in range(12, 18):
        walls.add((x, 5))
    # 7) Montant vertical bas-centre
    for y in range(1, 6):
        walls.add((9, y))

    return walls
