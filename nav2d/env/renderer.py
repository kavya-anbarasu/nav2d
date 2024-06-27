from pathlib import Path
from PIL import Image
import numpy as np

from nav2d.env.constants import OBJECT_TYPES_INV
from nav2d.env.texture import load_texture_set
from nav2d.env.env_state import EnvState


def render_maze(state: EnvState, background_color=(225, 225, 225)) -> Image.Image:
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
            texture = texture_set[OBJECT_TYPES_INV[grid[i, j]]]
            img.paste(texture, (i * texture.width, j * texture.height))

    return img


def test_render_maze():
    grid = [
        [1, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 2, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0]
    ]
    path = Path(__file__).parent / ".." / "output" / "texture" / "textures_pokemon_sprite2" / "medieval" / "run0"
    texture_set = load_texture_set(path, 64)
    state = EnvState(grid, (0, 0), (4, 4), texture_set)
    img = render_maze(state)
    img.show()


if __name__ == "__main__":
    test_render_maze()
