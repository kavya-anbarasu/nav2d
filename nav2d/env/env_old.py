# from pathlib import Path
# import gym
# from gym import spaces
# import numpy as np
# from PIL import Image
# import pygame


# TEXTURE_DIR = Path(__file__).parent / "data"
# TEXTURE_DIR /= "set6"
# EXTENSION = "png"


# class GridWorld(gym.Env):
#     def __init__(self, size=5, walls=None, scale_factor=0.5, goal_position=(4, 4)):
#         super(GridWorld, self).__init__()
#         self.size = size
#         self.action_space = spaces.Discrete(4)
#         self.observation_space = spaces.Box(low=0, high=3, shape=(size, size), dtype=np.uint8)

#         self.walls = walls or [(1, 1), (1, 2), (2, 1)]
#         self.goal_position = goal_position

#         # Load and resize textures
#         self.agent_texture = Image.open(TEXTURE_DIR / f"agent.{EXTENSION}").convert('RGBA')
#         self.wall_texture = Image.open(TEXTURE_DIR / f"wall.{EXTENSION}").convert('RGBA')
#         self.empty_texture = Image.open(TEXTURE_DIR / f"empty.{EXTENSION}").convert('RGBA')
#         self.goal_texture = Image.open(TEXTURE_DIR / f"goal.{EXTENSION}").convert('RGBA')

#         self.texture_size = int(self.wall_texture.width * scale_factor)
#         self.agent_texture = self.agent_texture.resize((self.texture_size, self.texture_size), Image.Resampling.LANCZOS)
#         self.wall_texture = self.wall_texture.resize((self.texture_size, self.texture_size), Image.Resampling.LANCZOS)
#         self.empty_texture = self.empty_texture.resize((self.texture_size, self.texture_size), Image.Resampling.LANCZOS)
#         self.goal_texture = self.goal_texture.resize((self.texture_size, self.texture_size), Image.Resampling.LANCZOS)

#         self.reset()

#     def reset(self):
#         self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
#         self.agent_position = [0, 0]
#         self.grid[tuple(self.agent_position)] = 1
#         for wall in self.walls:
#             self.grid[tuple(wall)] = 2
#         self.grid[self.goal_position] = 3
#         return self.grid

#     def step(self, action):
#         moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
#         next_position = [self.agent_position[0] + moves[action][0], self.agent_position[1] + moves[action][1]]

#         reward = 0
#         done = False

#         if 0 <= next_position[0] < self.size and 0 <= next_position[1] < self.size:
#             if self.grid[tuple(next_position)] != 2:
#                 self.grid[tuple(self.agent_position)] = 0
#                 self.agent_position = next_position
#                 self.grid[tuple(self.agent_position)] = 1

#                 if tuple(self.agent_position) == self.goal_position:
#                     reward = 1
#                     done = True

#         return self.grid, reward, done, {}

#     def render(self, mode='human'):
#         game_image = Image.new('RGBA', (self.size * self.texture_size, self.size * self.texture_size))
#         for r in range(self.size):
#             for c in range(self.size):
#                 if self.grid[r, c] == 1:
#                     texture = self.agent_texture
#                 elif self.grid[r, c] == 2:
#                     texture = self.wall_texture
#                 elif self.grid[r, c] == 3:
#                     texture = self.goal_texture
#                 else:
#                     texture = self.empty_texture

#                 game_image.paste(texture, (c * self.texture_size, r * self.texture_size))

#         if mode == 'human':
#             game_image.show()
#         elif mode == 'rgb_array':
#             return np.array(game_image.convert('RGB'))


# def play(env, keys_to_action=None, fps=30, new_api=False, width=400, height=400):
#     if keys_to_action is None:
#         keys_to_action = {
#             pygame.K_UP: 0,
#             pygame.K_RIGHT: 1,
#             pygame.K_DOWN: 2,
#             pygame.K_LEFT: 3
#         }

#     running = True
#     pygame.init()
#     if hasattr(env, 'texture_size') and hasattr(env, 'agent_texture') and hasattr(env, 'size'):
#         screen = pygame.display.set_mode((env.size * env.texture_size, env.size * env.texture_size))
#     else:
#         screen = pygame.display.set_mode((width, height))
#     clock = pygame.time.Clock()

#     env.reset()
#     while running:
#         action = None

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type == pygame.KEYDOWN:
#                 if event.key in keys_to_action:
#                     action = keys_to_action[event.key]

#         if action is not None:
#             if new_api:
#                 _, reward, done, _, _ = env.step(action)
#             else:
#                 _, reward, done, _ = env.step(action)
#             if done:
#                 print("Goal reached! Resetting environment...")
#                 env.reset()

#         if new_api:
#             obs = env.render()
#         else:
#             obs = env.render(mode='rgb_array')
#         surface = pygame.surfarray.make_surface(obs.transpose((1, 0, 2)))
#         screen.blit(surface, (0, 0))

#         pygame.display.flip()
#         clock.tick(fps)

#     pygame.quit()


# keys_to_action = {
#     pygame.K_UP: 0,
#     pygame.K_RIGHT: 1,
#     pygame.K_DOWN: 2,
#     pygame.K_LEFT: 3
# }

# if __name__ == "__main__":
#     # env = GridWorld(size=5, scale_factor=0.05)
#     env = GridWorld(size=10, scale_factor=0.04)
#     play(env, keys_to_action)
