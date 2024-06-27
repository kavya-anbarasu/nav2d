from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import gymnasium as gym
from gymnasium import spaces

from nav2d.env.maze.kruskal import KruskalMaze
from nav2d import make


fig_dir = Path(__file__).parent / "figures"
fig_dir.mkdir(exist_ok=True, parents=True)


def render_mazes(env, agent_positions, goal_positions, img_width=96, img_height=96, marker_size=6):
    """
    Args:
        env: maze gym env
        agent_positions: list of (x, y) positions for the agent
        goal_positions: list of (x, y) positions for the goal
        img_width: width of the output image in pixels
        img_height: height of the output image in pixels
    Returns:
        List of RGB arrays, each being a rendered maze image of specified size
    """
    maze_map = np.array(env.unwrapped.maze.maze_map)
    maze = env.unwrapped.maze
    height = maze.map_length
    width = maze.map_width
    extent = (-width / 2, width / 2, -height / 2, height / 2)

    dpi = 100
    figsize = (img_width / dpi, img_height / dpi)

    # Create a single figure and axes to reuse for all renderings
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    canvas = FigureCanvas(fig)
    ax.imshow(maze_map * 0.5, extent=extent, origin="upper", cmap=plt.cm.binary, vmin=0, vmax=1)
    ax.axis('off')
    fig.tight_layout(pad=0)

    images = []
    scatter_artist = None  # Initialize variable to store scatter plot references

    for agent_pos, goal_pos in zip(agent_positions, goal_positions):
        if len(images) > 0 and len(images) % 500 == 0:
            print("Frame:", len(images))
        # Clear previous scatter plots by removing each artist
        if scatter_artist:
            for artist in scatter_artist:
                artist.remove()

        # Plot new positions and store the scatter plot references
        scatter_agent = ax.scatter(*agent_pos, marker="o", color="tab:green", s=marker_size / 4)
        # scatter_goal = ax.scatter(*goal_pos, marker="o", color="tab:red", s=marker_size)
        scatter_goal = ax.scatter(*goal_pos, marker="d", color="black", s=marker_size)
        scatter_artist = [scatter_agent, scatter_goal]

        # Draw the current state and grab the buffer
        canvas.draw()
        buf = canvas.buffer_rgba()
        X = np.asarray(buf)
        X = X[:, :, :3]  # Drop the alpha channel
        images.append(X.copy())

    # Close the figure to free up resources after finishing all renderings
    plt.close(fig)
    return images


class MinariMazeEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(
            self,
            env_id,
            render_size=96,
            **kwargs
        ):
        self.env = gym.make(env_id, **kwargs)
        self.render_size = render_size
        # self.marker_size = self.render_size / 5
        self.marker_size = self.render_size / 2

        # https://minari.farama.org/tutorials/dataset_creation/point_maze_dataset/
        self.k_p, self.k_v = 10, -1

        # TODO: what's the window size here?
        self.window_size = ws = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, render_size, render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        # convert to gym space
        self.action_space = spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype
        )

    def reset(self, *args, **kwargs):
        # obs: observation, achieved_goal, desired_goal
        obs, info = self.env.reset(*args, **kwargs)
        # store initial agent pos
        agent_pos = obs["observation"][:2]
        goal_pos = obs["desired_goal"]

        # img = render_maze(self.env, agent_pos, goal_pos, self.render_size, self.render_size)
        img = render_mazes(self.env, [agent_pos], [goal_pos], self.render_size, self.render_size, self.marker_size)[0]

        self.agent_pos = agent_pos
        self.agent_speed = obs["observation"][2:]
        self.goal_pos = goal_pos
        self.image = img
        return {
            "agent_pos": agent_pos,
            "image": img
        }

    def step(self, action):
        # action: desired new_agent_pos
        # https://minari.farama.org/tutorials/dataset_creation/point_maze_dataset/#integral-term-i
        acceleration = self.k_p * (action - self.agent_pos) + self.k_v * self.agent_speed
        # acceleration = np.clip(acceleration, -1, 1)
        obs, reward, terminated, truncated, info = self.env.step(acceleration)

        agent_pos = obs["observation"][:2]
        goal_pos = obs["desired_goal"]
        # img = render_maze(self.env, agent_pos, goal_pos, self.render_size, self.render_size)
        img = render_mazes(self.env, [agent_pos], [goal_pos], self.render_size, self.render_size, self.marker_size)[0]
        observation = {
            "agent_pos": agent_pos,
            "image": img
        }

        # update agent pos
        self.agent_pos = obs["observation"][:2]
        self.agent_speed = obs["observation"][2:]
        self.goal_pos = goal_pos
        self.image = img
        done = terminated or truncated
        # diffusion_policy uses gym done step api
        return observation, reward, done, info

    def render(self):
        return self.image


def plot_env():
    width, height = (7, 7)
    maze = KruskalMaze(int((width-1)/2), int((height-1)/2))
    maze_map = maze.grid
    render_size = 168

    env = MinariMazeEnv('PointMaze_UMaze-v3', maze_map=maze_map, render_size=render_size)
    obs = env.reset()
    print(obs)

    obs, rew, done, info = env.step(env.agent_pos + 1)
    print(obs)
    img = env.render()
    print(img.shape)

    plt.imshow(img)
    plt.axis('off')
    plt.savefig(fig_dir / "point_maze.png")

    imgs = []
    for _ in range(10):
        env.reset()
        img = env.render()
        imgs.append(img)

    # # load image maze.png
    # from PIL import Image
    # img = Image.open(fig_dir / "maze.png")
    # # resize to render_size
    # img = img.resize((render_size, render_size))
    # # convert to rgb array
    # img = img.convert("RGB")
    # img = np.array(img)
    # imgs = img[None]

    model = make("model.maze_a_star.ecwq2mwi")
    model(np.array(imgs), save_dir=fig_dir / "res")


if __name__ == "__main__":
    import nav2d.exp.maze_a_star_eval
    plot_env()
