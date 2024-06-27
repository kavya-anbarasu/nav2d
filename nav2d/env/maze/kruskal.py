import random

import numpy as np

'''
Generates a random maze from the given width and height using
Kruskal's algorithm for minimum spanning trees

Output:
    maze - the edges of the spanning tree
    legal_edges - a dict of nodes as keys and a list of legal edges
'''


class KruskalMaze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.nodes, self.edges = self.create_graph()
        self.maze = self.generate_maze()
        self.legal_edges = self.get_legal_traversal_edges()
        self.grid = self.generate_grid()

    def get_random_edge_weights(self):
        '''
        assigns random weights to each edge of the graph
        '''
        edge_weights = [(random.randint(1, 4), x, y) for (x, y) in self.edges]
        return edge_weights

    def get_legal_traversal_edges(self):
        '''
        gets legal edges for each node based on the spanning tree. illegal edges are walls
        '''
        legal_edges = {}
        for s in sorted(self.maze):
            if s[0] not in legal_edges:
                legal_edges[s[0]] = [s[1]]
            else:
                legal_edges[s[0]].append(s[1])
            if s[1] not in legal_edges:
                legal_edges[s[1]] = [s[0]]
            else:
                legal_edges[s[1]].append(s[0])
        return legal_edges

    def create_graph(self):
        x = self.width
        y = self.height
        nodes = set()
        edges = set()
        for i in range(x):
            for j in range(y):
                nodes.add((i, j))
                if i > 0:
                    e1 = (i - 1, j)
                    edges.add(((i, j), e1))
                if i < x - 1:
                    e2 = (i + 1, j)
                    edges.add(((i, j), e2))
                if j > 0:
                    e3 = (i, j - 1)
                    edges.add(((i, j), e3))
                if j < y - 1:
                    e4 = (i, j + 1)
                    edges.add(((i, j), e4))
        return nodes, edges

    def generate_maze(self):
        edge_weights = self.get_random_edge_weights()
        clusters = {n: n for n in self.nodes}
        ranks = {n: 0 for n in self.nodes}
        solution = set()

        def find(u):
            if clusters[u] != u:
                clusters[u] = find(clusters[u])
            return clusters[u]

        def union(x, y):
            x, y = find(x), find(y)
            if ranks[x] > ranks[y]:
                clusters[y] = x
            else:
                clusters[x] = y
            if ranks[x] == ranks[y]:
                ranks[y] += 1

        for w, x, y in sorted(edge_weights):
            if x != y:
                if find(x) != find(y):
                    #add edge to solution
                    solution.add((x, y))
                    union(x, y)
        return solution

    def generate_grid(self):
        # Create a grid with all walls including borders
        grid = [[1 for _ in range(2 * self.height + 1)] for _ in range(2 * self.width + 1)]

        # Carve out the paths for the maze
        for node in self.nodes:
            x, y = node
            grid_x, grid_y = 2 * x + 1, 2 * y + 1  # Center nodes in the grid to allow wall spaces
            grid[grid_x][grid_y] = 0  # Mark the node itself as a pathway

            # Check each direction from the node and mark as pathways if legal
            if (x, y) in self.legal_edges:
                for connected_node in self.legal_edges[(x, y)]:
                    connected_x, connected_y = connected_node
                    if connected_x == x + 1:
                        grid[grid_x + 1][grid_y] = 0  # Right
                    elif connected_x == x - 1:
                        grid[grid_x - 1][grid_y] = 0  # Left
                    if connected_y == y + 1:
                        grid[grid_x][grid_y + 1] = 0  # Down
                    elif connected_y == y - 1:
                        grid[grid_x][grid_y - 1] = 0  # Up

        # width = number of columns, height = number of rows
        grid = np.array(grid).T
        return grid
