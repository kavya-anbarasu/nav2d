"""
https://chat.openai.com/share/8d683d43-8040-434f-8e5e-34f939687eb8
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


class Grid:
    def __init__(self, width, height, grid=None):
        self.width = width
        self.height = height
        if grid is None:
            self.grid = np.zeros((height, width), dtype=int)
        else:
            self.grid = grid

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return self.grid[id[1]][id[0]] == 0

    def neighbors(self, id):
        (x, y) = id
        results = [
            (x+1, y), (x, y-1), (x-1, y), (x, y+1),  # Orthogonal neighbors
            # (x+1, y+1), (x-1, y-1), (x-1, y+1), (x+1, y-1)  # Diagonal neighbors
        ]
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def a_star_search(grid, start, goal):
    if not isinstance(grid, Grid):
        grid = Grid(*grid.shape, grid=grid)

    start = tuple(start)
    goal = tuple(goal)

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while frontier:
        current = heapq.heappop(frontier)[1]

        if current == goal:
            break

        for next in grid.neighbors(current):
            new_cost = cost_so_far[current] + (heuristic(current, next))
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current
    
    # Reconstruct path
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path



def plot_discrete_path(path, grid):
    # Plotting
    fig, ax = plt.subplots()
    ax.imshow(grid.grid, cmap='Greys', origin='lower')

    # Plot the path
    for (x, y) in path:
        ax.plot(x, y, marker='.', color='blue')

    # Highlight the start and goal
    ax.plot(*start, marker='o', color='green', markersize=3)  # start
    ax.plot(*goal, marker='o', color='red', markersize=3)    # goal


def plot_continuous_path(path, grid):
    # Extract x and y coordinates from the path
    x = [p[0] for p in path]
    y = [p[1] for p in path]

    # Create a parameter t which goes from 0 to 1 along the points
    t = np.linspace(0, 1, num=len(path))

    # Fit cubic spline to the path points
    cs_x = CubicSpline(t, x, bc_type='natural')
    cs_y = CubicSpline(t, y, bc_type='natural')

    # Evaluate spline over a finer resolution to make the path continuous
    fine_t = np.linspace(0, 1, num=300)
    fine_x = cs_x(fine_t)
    fine_y = cs_y(fine_t)

    # Plotting
    fig, ax = plt.subplots()
    ax.imshow(grid.grid, cmap='Greys', origin='lower')

    # Plot the interpolated path
    ax.plot(fine_x, fine_y, 'b-', label='Interpolated Path')

    # Highlight the original discrete path points
    # ax.plot(x, y, 'ro', label='Original Path')

    # Highlight the start and goal
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start')  # start
    ax.plot(x[-1], y[-1], 'mo', markersize=10, label='Goal')  # goal

    ax.legend()


def generate_data(n_samples, grid_size, max_path_length=100, start=None, goal=None):
    grid = Grid(grid_size, grid_size)
    X = []
    Y = []
    for _ in range(n_samples):
        if start is None:
            _start = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        else:
            _start = start

        if goal is None:
            _goal = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        else:
            _goal = goal

        path = a_star_search(grid, _start, _goal)
        # Limit path length
        path = path[:min(len(path), max_path_length)]
        path += [path[-1]]*(max_path_length - len(path))

        grid_img = np.ones((grid_size, grid_size, 3))*255
        grid_img[_start] = (0, 255, 0)
        grid_img[_goal] = (255, 0, 0)
        X.append(grid_img)
        Y.append(np.array(path))

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    # print(y.shape)
    return X, Y


def plot_data(X, y):
    print("X.shape:", X.shape)
    print(y[0])
    print(y)
    # print("y.shape:", y.shape)
    plt.figure()
    plt.imshow(X[0], origin="lower")
    plt.plot(y[0][:, 1], y[0][:, 0], c="black", marker=".")


if __name__ == "__main__":
    # grid = Grid(100, 100)
    # start = (1, 1)
    # goal = (10, 90)
    # path = a_star_search(grid, start, goal)

    # plot_discrete_path(path, grid)
    # plot_continuous_path(path, grid)

    X, y = generate_data(1, grid_size=20, max_path_length=30)

    plot_data(X, y)
    plt.show()
