from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from nav2d.env.constants import OBJECT_TYPES, OBJECT_TYPES_INV
from nav2d.env.env_state import EnvState
from nav2d.env.maze_env import GridWorldEnv, play, reset_maze, keys_to_action
from nav2d.env.texture import load_texture_set
from nav2d.exp import maze_a_star


texture_dir = Path(__file__).parent / ".." / ".." / "output" / "texture" / "textures_subthemes"
fig_dir = Path(__file__).parent / "figures" / "diffusion_to_policy"
fig_dir.mkdir(parents=True, exist_ok=True)


def step_continuous(env_params, state, action):
    # action to direction
    if isinstance(action, int):
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        action = np.array(directions[action])

    # implement continuous movement
    acc = action / env_params.mass
    new_agent_position = state.agent_position + state.agent_velocity
    new_agent_velocity = (state.agent_velocity + acc) * env_params.friction

    # check for collision
    new_agent_pos_rounded = np.round(new_agent_position).astype(int)
    if state.grid[new_agent_pos_rounded[0], new_agent_pos_rounded[1]] == OBJECT_TYPES["wall"]:
        new_agent_velocity = np.array([0, 0])
        new_agent_position = state.agent_position

    # check for out of bounds
    if new_agent_position[0] < 0 or new_agent_position[0] >= state.grid.shape[0] or new_agent_position[1] < 0 or new_agent_position[1] >= state.grid.shape[1]:
        new_agent_velocity = np.array([0, 0])
        new_agent_position = state.agent_position

    state.agent_position = new_agent_position
    state.agent_velocity = new_agent_velocity

    # check for goal
    reward = 0
    done = False
    if np.all(np.abs(state.agent_position - state.goal_position) < 0.5):
        reward = 1
        done = True

    return state, reward, done


def render_continous(state: EnvState, background_color=(225, 225, 225)) -> Image.Image:
    grid = np.array(state.grid)
    grid = grid.T

    # make a copy of the texture set
    texture_set = state.texture_set.copy()
    # replace empty background texture if background color is provided
    if background_color is not None:
        texture_set["empty"] = Image.new("RGBA", (texture_set["empty"].width, state.texture_set["empty"].height), (*background_color, 255))

    image_size = (grid.shape[0] * texture_set["empty"].width, grid.shape[1] * texture_set["empty"].height)
    img = Image.new("RGBA", image_size)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # don't render agent or goal here
            if grid[i, j] in [OBJECT_TYPES["agent"], OBJECT_TYPES["goal"]]:
                img.paste(texture_set["empty"], (i * texture_set["empty"].width, j * texture_set["empty"].height))
                continue

            texture = texture_set[OBJECT_TYPES_INV[grid[i, j]]]
            img.paste(texture, (i * texture.width, j * texture.height))

    # draw goal
    goal_texture = texture_set["goal"]
    goal_position = state.goal_position
    img.paste(goal_texture, (int(goal_position[1] * goal_texture.width), int(goal_position[0] * goal_texture.height)))

    # draw agent
    agent_texture = texture_set["agent"]
    agent_position = state.agent_position
    img.paste(agent_texture, (int(agent_position[1] * agent_texture.width), int(agent_position[0] * agent_texture.height)))


    return img


@dataclass
class EnvParams:
    texture_sets: list[dict[str, Image.Image]]
    maze_size: tuple[int, int]  # width, height
    start_position: tuple[int, int] = None
    goal_position: tuple[int, int] = None
    # physics
    friction: float = 0.9
    mass: float = 100


class PDController:
    def __init__(self, kp=1, kd=2):
        self.kp = kp
        self.kd = kd

    def __call__(self, position, velocity, target_position):
        error = target_position - position
        action = self.kp * error - self.kd * velocity
        return action


def animate_frames(frames, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=0)
    if save_path:
        anim.save(save_path, writer="imagemagick", fps=30)
    else:
        plt.show()


def visualize_gt_policy():
    texture_size = 45
    texture_sets = []
    # for p in texture_dir.glob("*/*"):
    for p in texture_dir.glob("*/*/*"):
        texture_set = load_texture_set(p, texture_size)
        texture_sets.append(texture_set)

    grid_size = (7, 7)
    image_size = (grid_size[0]*texture_size, grid_size[1]*texture_size)
    path_length = 64
    env_params = EnvParams(texture_sets, grid_size)
    env = GridWorldEnv(render_continous, reset_maze, env_params, step=step_continuous)

    env.reset()
    env_state = env.state
    path = maze_a_star.optimal_path(env_state, image_size, grid_size, path_length, path_upsample=4)

    ratio = image_size[0] // grid_size[0]
    path = (np.array(path) - ratio//2) / ratio

    controller = PDController(kp=1, kd=2)

    frames = []
    for i in range(path_length):
        target_position = path[i]
        for _ in range(10):
            action = controller(env_state.agent_position, env_state.agent_velocity, target_position)
            _, reward, done, info = env.step(action)
            env_state = env.state
            frame = env.render(mode="rgb_array")
            frames.append(frame)

            if done:
                break

        if done:
            break

    frames = frames[::5]
    animate_frames(frames, save_path=fig_dir / "gt_policy.gif")

    # play(env, keys_to_action, width=env.width*texture_size, height=env.height*texture_size)


if __name__ == "__main__":
    visualize_gt_policy()
