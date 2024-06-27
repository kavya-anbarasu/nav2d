# import gymnasium as gym
from dijkstra import WeightedSearch
from maze_visualizer import MazePlot
from kruskal import KruskalMaze


def generate_maze():
    # Use Kruskal's algo to create MST to generate mazes

    # Convert to Point Map env
    env = gym.make('PointMaze_UMaze-v3', maze_map=generated_maze)


def sample_goal():
    pass


def generate_path():
    pass


class MazeRunner:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = KruskalMaze(self.width, self.height)
        self.graph = [[0 for x in range(self.width)] for y in range(self.height)]  # noqa: E501
        self.visualization = self.generate_visualization(name="maze")
        self.visualization.save_as_image()
        self.solve_maze_weighted()

    def generate_visualization(self, name=None):
        if name:
            visualization = MazePlot(self.width,self.height,self.maze,maze_name=name)
        else:
            visualization = MazePlot(self.width,self.height,self.maze)
        return visualization

    def solve_maze_weighted(self):
        self.visualization.clear_maze()
        self.visualization.maze_name = "weighted_search"
        entry_exit_points = self.visualization.entry_exit_points
        self.maze_solver = WeightedSearch(self.graph, entry_exit_points[0], entry_exit_points[1], self.maze.legal_edges)
        self.visualization.path = self.maze_solver.path
        self.visualization.animate()
        return


if __name__ == "__main__":
    MazeRunner(20, 20)
