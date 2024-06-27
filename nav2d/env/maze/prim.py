import numpy as np
import random

def generate_random_maze(width, height):
    maze = np.ones((height, width), dtype=np.int8)
    start_row = random.randrange(1, height, 2)
    start_col = random.randrange(1, width, 2)
    maze[start_row][start_col] = 0
    frontiers = [(start_row, start_col)]
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    while frontiers:
        current = random.choice(frontiers)
        neighbors = []
        for direction in directions:
            nr, nc = current[0] + direction[0], current[1] + direction[1]
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and maze[nr][nc] == 1:
                if sum(maze[nr + dr][nc + dc] for dr, dc in directions if 0 <= nr + dr < height and 0 <= nc + dc < width) >= 2 * 1:
                    neighbors.append((nr, nc))
        if neighbors:
            chosen = random.choice(neighbors)
            maze[chosen[0]][chosen[1]] = 0
            maze[current[0] + (chosen[0] - current[0]) // 2][current[1] + (chosen[1] - current[1]) // 2] = 0
            frontiers.append(chosen)
        else:
            frontiers.remove(current)

    # TODO: sometimes only one empty cell
    return maze, (start_row, start_col)



if __name__ == "__main__":
    # Example usage:
    maze = generate_random_maze(21, 21)
    print(maze)
