import json
from pathlib import Path
from PIL import Image
from nav2d.env.env_state import EnvState
import numpy as np

import nav2d.model
from nav2d import make


output_dir = Path(__file__).parent / ".." / "output" / "texture"
prompt_dir = Path(__file__).parent / ".." / "prompts"


def load_texture_set(path, texture_size=None, extension="png"):
    def _load_img(img_path):
        return Image.open(img_path).convert('RGBA')

    texture_set = {
        key: _load_img(path / f"{key}.{extension}")
        for key in ["agent", "wall", "goal", "empty"]
    }

    # resize if needed
    if texture_size is not None:
        for key, texture in texture_set.items():
            texture_set[key] = texture.resize((texture_size, texture_size), Image.Resampling.LANCZOS)

    return texture_set



def generate_textures(pipe, prompts, n_runs, start_run=0, save_dir=None):
    imgs = {}
    for run in range(start_run, start_run + n_runs):
        texture_set_name = f"run{run}"
        if save_dir is not None:
            _save_dir = save_dir / texture_set_name
            _save_dir.mkdir(parents=True, exist_ok=True)

        for key, prompt in prompts.items():
            image = pipe(prompt)[0]
            imgs[key] = image

            if save_dir is not None:
                image.save(_save_dir / f"{key}.png")

    return imgs


def generate_textures_single_batch(pipe, prompts, n_runs, start_run=0, save_dir=None):
    # TODO: takes a lot of memory (8Gb for 2 runs)
    # generate list of prompts with corresponding keys
    prompt_list = []
    prompt_keys = []
    runs = []
    for run in range(start_run, start_run + n_runs):
        for key, prompt in prompts.items():
            prompt_list.append(prompt)
            prompt_keys.append(key)
            runs.append(run)

    images = pipe(prompt_list)
    if save_dir is not None:
        for i, image in enumerate(images):
            _save_dir = save_dir / f"run{runs[i]}"
            _save_dir.mkdir(parents=True, exist_ok=True)
            image.save(_save_dir / f"{prompt_keys[i]}.png")
    return images


def generate_textures2(pipe, prompts, n_runs, start_run=0, save_dir=None):
    prompt_list = list(prompts.values())
    prompt_keys = list(prompts.keys())

    all_images = []
    for run in range(start_run, start_run + n_runs):
        images = pipe(prompt_list)
        if save_dir is not None:
            for i, image in enumerate(images):
                _save_dir = save_dir / f"run{run}"
                _save_dir.mkdir(parents=True, exist_ok=True)
                image.save(_save_dir / f"{prompt_keys[i]}.png")
        all_images.extend(images)
    return images


# def generate_themes():
#     # Use a pipeline as a high-level helper
#     from transformers import pipeline

#     # TODO: takes too much time to run
#     pipe = pipeline("text-generation", model="mistralai/Mixtral-8x22B-Instruct-v0.1")

#     # load prompts/generate_themes.txt
#     prompt_save_dir = Path(__file__).parent / "prompts"
#     prompt_save_dir.mkdir(parents=True, exist_ok=True)
#     with open(prompt_save_dir / "generate_themes.txt", "r") as f:
#         prompt = f.readlines()
#     print(prompt)

#     # generate themes
#     themes = pipe(prompt, max_length=100, num_return_sequences=1)
#     print(themes)
#     # save in output/themes.txt
#     with open("output/themes.txt", "w") as f:
#         f.write(themes)


if __name__ == "__main__":
    from nav2d.env.renderer import render_maze

    # generate_themes()

    grid = [
        [1, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 2, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0]
    ]

    n_runs = 1
    start_run = 4
    device = "cuda:7"


    # prompt_file_name = "textures_pokemon_sprite2"
    # https://chat.openai.com/share/b39f7ca3-beb4-43b1-8010-395c130c797f
    # prompt_path = Path(__file__).parent / "prompts/texturqes.json"
    # prompt_path = Path(__file__).parent / "prompts/textures_test_closeup.json"

    # https://chat.openai.com/share/cf2037f7-3bdb-4824-a3e1-ffc4d1b8f7b0
    prompt_file_name = "textures_subthemes"

    prompt_path = prompt_dir / f"{prompt_file_name}.json"
    prompts = json.load(open(prompt_path, "r"))

    pipeline_id = "text2img.pokemon-trainer-sprite-pixelart"
    pipe = make(f"pipeline.{pipeline_id}", device=device)

    output_dir = output_dir / prompt_file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    conf = {
        "pipeline_id": pipeline_id,
        "prompts": prompts
    }
    # save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(conf, f, indent=4)


    for theme, theme_prompts in prompts.items():
        if not isinstance(theme_prompts, list):
            theme_prompts = [theme_prompts]

        for i, subtheme_prompts in enumerate(theme_prompts):
            _save_dir = output_dir / theme / f"subtheme{i}"
            # TODO: memory error
            generate_textures(pipe, subtheme_prompts, n_runs=n_runs, start_run=start_run, save_dir=_save_dir)
            # generate_textures2(pipe, subtheme_prompts, n_runs=n_runs, start_run=start_run, save_dir=_save_dir)

            for run in range(start_run, start_run + n_runs):
                texture_set_path = _save_dir / f"run{run}"

                texture_set = load_texture_set(texture_set_path, 64)
                img = render_maze(EnvState(grid, (0, 0), (0, 0), texture_set))
                img.save(texture_set_path / "maze.png")
