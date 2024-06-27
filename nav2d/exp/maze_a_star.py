from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from nav2d.env.env_state import EnvParams
from nav2d.env.maze_env import GridWorldEnv, play, keys_to_action, reset_maze
from nav2d.env.renderer import render_maze
from nav2d.env.texture import load_texture_set
from nav2d.exp.grid_a_star import a_star_search


texture_dir = Path(__file__).parent / ".." / "output" / "texture" / "textures_subthemes"


def load_texture_sets(texture_size: int, path: Path = None, val_ratio: float = 0, val_themes=None):
    if path is None:
        path = texture_dir

    # list all themes (first level in texture dir)
    themes = [p for p in path.glob("*") if p.is_dir()]

    # train test split themes
    if val_themes is not None:
        theme_names = [p.name for p in themes]
        val_indices = [theme_names.index(name) for name in val_themes]
        train_themes = [p for i, p in enumerate(themes) if i not in val_indices]
        val_themes = [p for i, p in enumerate(themes) if i in val_indices]

        assert val_ratio == len(val_themes) / len(themes), "val_ratio does not match val_themes"
    else:
        n_val = int(len(themes) * val_ratio)
        n_train = len(themes) - n_val

        train_themes = themes[:n_train]
        val_themes = themes[n_train:]

    print("Train themes:", [p.name for p in train_themes])
    print("Val themes:", [p.name for p in val_themes])

    def load_theme_textures(theme_path):
        textures = []
        texture_paths = []
        for p in theme_path.glob("*/*"):
            texture_set = load_texture_set(p, texture_size)
            textures.append(texture_set)
            texture_paths.append(p.resolve())
        return textures, texture_paths

    train_textures = []
    train_texture_paths = []
    for p in train_themes:
        tex_images, tex_paths = load_theme_textures(p)
        train_textures.extend(tex_images)
        train_texture_paths.extend(tex_paths)

    val_textures = []
    val_texture_paths = []
    for p in val_themes:
        tex_images, tex_paths = load_theme_textures(p)
        val_textures.extend(tex_images)
        val_texture_paths.extend(tex_paths)

    return train_textures, val_textures, train_texture_paths, val_texture_paths


def play_env():
    texture_size = 24
    texture_sets = load_texture_sets(texture_dir, texture_size)
    env_params = EnvParams(texture_sets, (7, 7))
    env = GridWorldEnv(render_maze, reset_maze, env_params)
    play(env, keys_to_action, width=env.width*texture_size, height=env.height*texture_size)


@dataclass
class DatasetParams:
    grid_size: tuple
    image_size: tuple
    texture_sets: list
    path_length: int
    path_upsample: int
    start_position: tuple[int, int] = None
    goal_position: tuple[int, int] = None


def optimal_path(env_state, image_size, grid_size, path_length, path_upsample):
    # target trajectory
    path = a_star_search(env_state.grid_mask.T, env_state.agent_position, env_state.goal_position)
    ratio = image_size[0] // grid_size[0]
    path = np.array(path)*ratio + ratio//2

    # upsample path with linear interpolation
    if len(path) > 1:
        x, y = path[:, 0], path[:, 1]
        t = np.linspace(0, 1, len(path))
        new_t = np.linspace(0, 1, path_upsample * len(path))  # Increase the number of points by 10 times
        interpolator_x = interp1d(t, x, kind='linear')
        interpolator_y = interp1d(t, y, kind='linear')
        upsampled_x = interpolator_x(new_t)
        upsampled_y = interpolator_y(new_t)
        path = np.vstack([upsampled_x, upsampled_y]).T

    # limit path length
    path = path[:path_length]
    # pad if necessary (repeat last point)
    if len(path) < path_length:
        path = np.pad(path, ((0, path_length - len(path)), (0, 0)), mode="edge")
    return path


def generate_sample(params: DatasetParams):
    env_params = EnvParams(
        texture_sets=params.texture_sets,
        maze_size=params.grid_size,
        start_position=params.start_position,
        goal_position=params.goal_position
    )
    env_state = reset_maze(env_params)

    path = optimal_path(env_state, params.image_size, params.grid_size, params.path_length, params.path_upsample)

    img = render_maze(env_state)
    assert img.size == params.image_size, f"Expected image size {params.image_size}, got {img.size}"
    # convert to rgb array
    img = img.convert("RGB")
    img = np.array(img) / 255
    return img, path


def test_generate_sample():
    image_size = (168, 168)
    grid_size = (7, 7)
    # compute texture size
    texture_size = image_size[0] // grid_size[0]
    texture_sets, _, _, _ = load_texture_sets(texture_size, val_themes=["Desert Oasis"], val_ratio=0.1)
    params = DatasetParams(
        grid_size=grid_size,
        image_size=image_size,
        texture_sets=texture_sets,
        path_length=64,
        path_upsample=4
    )
    for _ in range(10):
        img, path = generate_sample(params)
        plt.figure()
        plt.imshow(img)

        colors = np.linspace(0, 1, len(path))
        plt.scatter(path[:, 1], path[:, 0], c=colors, cmap="magma", marker=".")
        plt.show()


if __name__ == "__main__":
    # play_env()
    test_generate_sample()
