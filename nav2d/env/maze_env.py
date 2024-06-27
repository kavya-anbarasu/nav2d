from functools import partial
from pathlib import Path
import gym
from gym import spaces
import numpy as np
import pygame
from PIL import Image

from nav2d.env.constants import OBJECT_TYPES
from nav2d.env.maze.kruskal import KruskalMaze
from nav2d.env.renderer import render_maze
from nav2d.env.texture import load_texture_set
from nav2d.env.env_state import EnvState, EnvParams


def step_maze(env_params, state, action):
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    next_position = [state.agent_position[0] + moves[action][0], state.agent_position[1] + moves[action][1]]

    reward = 0
    done = False

    height, width = env_params.maze_size
    if 0 <= next_position[0] <= height and 0 <= next_position[1] <= width:
        if state.grid[tuple(next_position)] != OBJECT_TYPES["wall"]:
            # move the agent object
            state.grid[tuple(state.agent_position)] = OBJECT_TYPES["empty"]
            state.agent_position = next_position
            state.grid[tuple(state.agent_position)] = OBJECT_TYPES["agent"]

            if tuple(state.agent_position) == tuple(state.goal_position):
                reward = 1
                done = True

    return state, reward, done


class GridWorldEnv(gym.Env):
    def __init__(self, renderer, reset, env_params: EnvParams, step=step_maze):
        super().__init__()
        self.renderer = renderer
        self.reset_fn = reset
        self.step_fn = step
        self.env_params = env_params

        self.width, self.height = env_params.maze_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=max(OBJECT_TYPES.values()),
            shape=(self.width, self.height),
            dtype=np.uint8
        )

    def reset(self, *args, **kwargs):
        self.state = self.reset_fn(self.env_params, *args, **kwargs)
        obs = self.get_obs()
        return obs

    def step(self, action):
        state, reward, done = self.step_fn(self.env_params, self.state, action)
        self.state = state
        obs = self.get_obs()
        return obs, reward, done, {}

    def get_obs(self):
        return self.state.grid

    def render(self, mode='human'):
        img = self.renderer(self.state)

        if mode == 'human':
            img.show()
        elif mode == 'rgb_array':
            return np.array(img.convert('RGB'))


def play(env, keys_to_action=None, fps=30, new_api=False, width=400, height=400):
    if keys_to_action is None:
        keys_to_action = {
            pygame.K_UP: 0,
            pygame.K_RIGHT: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3
        }

    running = True
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    env.reset()
    while running:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in keys_to_action:
                    action = keys_to_action[event.key]
                # if ESC is pressed, reset the environment
                elif event.key == pygame.K_ESCAPE:
                    env.reset()

        if action is not None:
            if new_api:
                _, reward, done, _, _ = env.step(action)
            else:
                _, reward, done, _ = env.step(action)
            if done:
                print("Goal reached! Resetting environment...")
                env.reset()

        if new_api:
            obs = env.render()
        else:
            obs = env.render(mode='rgb_array')
        surface = pygame.surfarray.make_surface(obs.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


keys_to_action = {
    pygame.K_UP: 0,
    pygame.K_RIGHT: 1,
    pygame.K_DOWN: 2,
    pygame.K_LEFT: 3
}


def reset_maze(params: EnvParams) -> EnvState:
    width, height = params.maze_size
    # TODO: KruskalMaze width and height doesn't take the walls into account
    maze = KruskalMaze(int((width-1)/2), int((height-1)/2))
    grid_mask = maze.grid

    # import nav2d.env.maze.prim
    # grid, agent_position = nav2d.env.maze.prim.generate_random_maze(width, height)
    # agent_position = tuple(agent_position)

    # replace 1 by wall object type
    grid = [[OBJECT_TYPES["wall"] if cell == 1 else OBJECT_TYPES["empty"] for cell in row] for row in grid_mask]
    grid = np.array(grid)

    # # sample agent and goal positions (must be empty cells and different)
    empty_cells = np.argwhere(grid == OBJECT_TYPES["empty"])
    agent_position = empty_cells[np.random.choice(len(empty_cells))]
    goal_position = agent_position
    i = 0
    while np.all(agent_position == goal_position):
        goal_position = empty_cells[np.random.choice(len(empty_cells))]
        i += 1
        if i > 1e4:
            raise ValueError("Couldn't find different agent and goal positions")

    # add goal to the grid
    grid[tuple(goal_position)] = OBJECT_TYPES["goal"]
    # add agent to the grid
    grid[tuple(agent_position)] = OBJECT_TYPES["agent"]

    # select a random texture set
    texture_set = params.texture_sets[np.random.choice(len(params.texture_sets))]

    return EnvState(
        grid=grid,
        grid_mask=grid_mask,
        agent_position=agent_position,
        goal_position=goal_position,
        texture_set=texture_set
    )


if __name__ == "__main__":
    # texture_dir = Path(__file__).parent / ".." / "output" / "texture" / "textures_pokemon_sprite2"
    texture_dir = Path(__file__).parent / ".." / "output" / "texture" / "textures_subthemes"

    texture_size = 45
    texture_sets = []
    # for p in texture_dir.glob("*/*"):
    for p in texture_dir.glob("*/*/*"):
        texture_set = load_texture_set(p, texture_size)
        texture_sets.append(texture_set)

    env_params = EnvParams(texture_sets, (13, 11))
    env = GridWorldEnv(render_maze, reset_maze, env_params)

    play(env, keys_to_action, width=env.width*texture_size, height=env.height*texture_size)
