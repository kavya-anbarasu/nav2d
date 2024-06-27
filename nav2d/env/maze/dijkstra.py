import heapq

'''
Solves a maze using Dijkstra's weighted graph search algorithm
 and returns a list of the x,y coordinates for each step in the path.

Input:
    graph - matrix (same size as graph used to create the maze)
    legal edges - edges that do not have a maze wall between the nodes

Output:
    seen- a list of (x,y) coordinates for each step in the path
    from the entry to exit
'''


class NodeWeight:
    """node class for dijkstra/A* search"""
    def __init__(self, x, y, weight=0):
        self.x = x
        self.y = y
        self.weight = weight

    def __eq__(self, other):
        return self.weight == other.weight

    def __lt__(self, other):
        return self.weight < other.weight

    def __str__(self):
        return "{}, {}".format(self.x, self.y)


class WeightedSearch():
    def __init__(self, graph, start, goal, legal_edges=None):
        self.graph = graph
        self.legal_edges = legal_edges
        self.start = start
        self.goal = goal

        self.path = self.solve_maze()

    def get_neighbors(self, node):
        neighbors = self.legal_edges[node]
        return neighbors

    def get_weight(self, node):
        x1 = node[0]
        y1 = node[1]
        weight = abs(x1 - self.goal[0]) + abs(y1 - self.goal[1])
        return weight

    def solve_maze(self):
        seen = []
        print(self.start)
        start_node = NodeWeight(self.start[0], self.start[1],
                                self.get_weight(self.start))
        queue = [start_node]
        heapq.heapify(queue)
        while queue:
            current = heapq.heappop(queue)
            seen.append((current.x, current.y))
            if (current.x, current.y) == self.goal:
                return seen
            neighbors = self.get_neighbors((current.x, current.y))
            for n in neighbors:
                if (n not in seen and
                        not any(nd.x == n[0] and nd.y == n[1]
                                for nd in queue)):
                    new_node = NodeWeight(n[0], n[1], self.get_weight(n))
                    heapq.heappush(queue, new_node)
        return seen
