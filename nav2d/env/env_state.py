from dataclasses import dataclass
from PIL import Image
import numpy as np


@dataclass
class EnvState:
    grid: np.ndarray
    grid_mask: np.ndarray
    agent_position: tuple[int, int]
    goal_position: tuple[int, int]
    texture_set: dict[str, Image.Image]
    agent_velocity: tuple[int, int] = np.array([0, 0])


@dataclass
class EnvParams:
    texture_sets: list[dict[str, Image.Image]]
    maze_size: tuple[int, int]  # width, height
    start_position: tuple[int, int] = None
    goal_position: tuple[int, int] = None
