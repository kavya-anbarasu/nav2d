from matplotlib import pyplot as plt
from nav2d.env.maze.kruskal import KruskalMaze
from nav2d.env.maze.maze import generate_maze
from nav2d.env.maze.maze_visualizer import MazePlot


def test_generate_maze():
    maze = generate_maze()


def test_kruskal():
    width = 5
    height = 5
    maze = KruskalMaze(width, height)
    print(maze.maze)
    print(maze.legal_edges)

    maze_plot = MazePlot(width, height, maze)
    maze_plot.create_plot()

    # self = maze
    # grid = [[1 for _ in range(self.height*2 + 1)] for _ in range(self.width*2 + 1)]
    # for x, y in self.nodes:
    #     grid[x*2+1][y*2+1] = 0  # Paths are open where nodes are
    # for x, y in self.maze:
    #     grid[(x[0] + y[0])//2 + 1][(x[1] + y[1])//2 + 1] = 0  # Open paths for edges in the solution

    # print(grid)
    # for row in grid:
    #     print(' '.join(str(cell) for cell in row))
    # plt.figure()
    # plt.imshow(grid, cmap='gray')
    # plt.show()

    grid = maze.grid
    # transform grid to image (0, 0, 0) for black, (255, 255, 255) for white
    grid = [[(0, 0, 0) if cell == 1 else (255, 255, 255) for cell in row] for row in grid]
    print(grid)

    plt.figure()
    plt.imshow(grid, cmap='gray', origin="lower", aspect="equal")
    plt.show()


if __name__ == "__main__":
    test_kruskal()
