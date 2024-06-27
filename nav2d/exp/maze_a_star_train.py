import os
import copy
from pathlib import Path
from time import perf_counter
from typing import Dict
import zarr
from concurrent.futures import ProcessPoolExecutor, as_completed

import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

from nav2d.exp import maze_a_star


conf_dir = Path(__file__).parents[1] / "config"
fig_dir = Path(__file__).parent / "figures" / "maze_a_star"
fig_dir.mkdir(parents=True, exist_ok=True)
data_path = Path(__file__).parent / "data" / "maze_a_star"
data_path.mkdir(parents=True, exist_ok=True)


def generate_single_sample(params):
    """Function to generate a single sample."""
    X, y = maze_a_star.generate_sample(params)
    X = X.transpose(2, 0, 1)  # CHW format
    X = X[None].repeat(params.path_length, axis=0)  # Repeat across time dimension
    return X, y


# class MazeAStarImageDataset(BaseImageDataset):
#     def __init__(
#             self,
#             n_samples=10000,
#             horizon=64,
#             grid_size=7,
#             render_size=168,
#             path_upsample=4,
#             val_ratio=0.1,
#             dataset_path=data_path / "dataset.zarr"
#         ):
#         super().__init__()
#         self.horizon = horizon
#         self.grid_size = grid_size
#         self.render_size = render_size

#         self.n_samples = n_samples
#         self.val_ratio = val_ratio
#         self.n_train_samples = int(n_samples * (1 - val_ratio))
#         self.n_val_samples = n_samples - self.n_train_samples

#         # compute texture size
#         texture_size = render_size // grid_size
#         self.train_textures, self.val_textures, self.train_texture_paths, self.val_texture_paths = maze_a_star.load_texture_sets(texture_size, val_ratio=val_ratio)
#         print("Loaded {} train textures and {} val textures".format(len(self.train_textures), len(self.val_textures)))

#         self.train_params = maze_a_star.DatasetParams(
#             grid_size=(grid_size, grid_size),
#             image_size=(render_size, render_size),
#             texture_sets=self.train_textures,
#             path_length=horizon,
#             path_upsample=path_upsample
#         )
#         self.val_params = copy.copy(self.train_params)
#         self.val_params.texture_sets = self.val_textures

#         self.dataset_path = Path(dataset_path)
#         if not self.dataset_path.exists():
#             print("No dataset found. Generate dataset...")
#             self.generate_dataset()
#         else:
#             print(f"Load dataset from {self.dataset_path}")
#         self.dataset = zarr.open(self.dataset_path, mode='r')
#         self.train_data = self.dataset['train']

#     def generate_dataset(self):
#         root = zarr.open(self.dataset_path, mode='w')
#         train_group = root.create_group("train")
#         val_group = root.create_group("val")

#         img_shape = (self.horizon, 3, self.render_size, self.render_size)
#         train_images = train_group.zeros('image', shape=(self.n_train_samples, *img_shape), dtype=np.float32, chunks=(1, *img_shape[1:]))
#         train_actions = train_group.zeros('actions', shape=(self.n_train_samples, self.horizon, 2), dtype=np.float32)
        
#         val_images = val_group.zeros('image', shape=(self.n_val_samples, *img_shape), dtype=np.float32, chunks=(1, *img_shape[1:]))
#         val_actions = val_group.zeros('actions', shape=(self.n_val_samples, self.horizon, 2), dtype=np.float32)

#         # print(f"Generate {self.n_train_samples} training samples...")
#         # tic = perf_counter()
#         # for i in range(self.n_train_samples):
#         #     X, y = maze_a_star.generate_sample(self.train_params)
#         #     X = X.transpose(2, 0, 1)
#         #     X = X[None].repeat(self.horizon, axis=0)
#         #     train_images[i] = X
#         #     train_actions[i] = y
#         #     print("Sample", i + 1, "processed.")
#         # print(f"Training samples generated in {perf_counter() - tic:.2f} seconds.")

#         # print(f"Generate {self.n_val_samples} validation samples...")
#         # for i in range(self.n_val_samples):
#         #     X, y = maze_a_star.generate_sample(self.val_params)
#         #     X = X.transpose(2, 0, 1)
#         #     X = X[None].repeat(self.horizon, axis=0)
#         #     val_images[i] = X
#         #     val_actions[i] = y


#         # Determine number of workers to use
#         num_workers = os.cpu_count()  # Use all available CPUs
#         print(f"Using {num_workers} workers for parallel data generation.")

#         def populate_group(samples, group_images, group_actions, params):
#             with ProcessPoolExecutor(max_workers=num_workers) as executor:
#                 futures = [executor.submit(generate_single_sample, params) for _ in range(samples)]
#                 for i, future in enumerate(as_completed(futures)):
#                     X, y = future.result()
#                     group_images[i] = X
#                     group_actions[i] = y
#                     print(f"Sample {i + 1}/{samples} processed.")

#         # Generate training and validation samples in parallel
#         print("Generating training samples...")
#         tic = perf_counter()
#         populate_group(self.n_train_samples, train_images, train_actions, self.train_params)
#         print(f"Training samples generated in {perf_counter() - tic:.2f} seconds.")

#         print("Generating validation samples...")
#         populate_group(self.n_val_samples, val_images, val_actions, self.val_params)


#     def get_normalizer(self, mode='limits', **kwargs):
#         # generate some data to fit the normalizer
#         # action trajectories are in the range [0, 167]
#         action = np.array([
#             np.arange(0, 168, dtype=np.float32),
#             np.arange(0, 168, dtype=np.float32)
#         ]).T
#         action = action[None].repeat(self.horizon, axis=0)
#         action = action[None]  # batch axis

#         data = {
#             "action": action,
#             "agent_pos": np.zeros((self.horizon, 2), dtype=np.float32)
#         }
#         normalizer = LinearNormalizer()
#         normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
#         normalizer['image'] = get_image_range_normalizer()
#         return normalizer
 
#     def get_validation_dataset(self):
#         val_dataset = copy.copy(self)
#         val_dataset.train_params = self.val_params
#         val_dataset.n_train_samples = self.n_val_samples
#         val_dataset.train_textures = self.val_textures
#         val_dataset.train_data = self.dataset['val']
#         return val_dataset

#     def __len__(self) -> int:
#         return self.train_data["image"].shape[0]
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         # X, y = maze_a_star.generate_sample(self.params)
#         # X = X.transpose(2, 0, 1)  # (render_size, render_size, 3) -> (3, render_size, render_size)
#         # # copy input image for horizon
#         # X = X[None].repeat(self.horizon, axis=0)  # (horizon, 3, render_size, render_size)

#         data = {
#             "obs": {
#                 "image": self.train_data["image"][idx],
#                 # agent position not used
#                 "agent_pos": np.zeros((self.horizon, 2), dtype=np.float32)
#             },
#             "action": self.train_data["actions"][idx]
#         }
#         assert data["obs"]["image"].shape == (self.horizon, 3, self.render_size, self.render_size)
#         assert data["action"].shape == (self.horizon, 2)
#         torch_data = dict_apply(data, torch.from_numpy)


#         return torch_data


class MazeAStarImageDatasetOnline(BaseImageDataset):
    def __init__(
            self,
            n_samples=10000,
            horizon=64,
            grid_size=7,
            render_size=168,
            path_upsample=4,
            val_ratio=0.1,
            val_themes=None
        ):
        super().__init__()
        self.horizon = horizon
        self.grid_size = grid_size
        self.render_size = render_size
        self.n_samples = n_samples
        self.val_ratio = val_ratio
        self.path_upsample = path_upsample

        # compute texture size
        texture_size = render_size // grid_size
        self.train_textures, self.val_textures, self.train_texture_paths, self.val_texture_paths = maze_a_star.load_texture_sets(texture_size, val_ratio=val_ratio, val_themes=val_themes)
        print("Loaded {} train textures and {} val textures".format(len(self.train_textures), len(self.val_textures)))
        print("Val textures:", self.val_texture_paths)

        self.params = maze_a_star.DatasetParams(
            grid_size=(grid_size, grid_size),
            image_size=(render_size, render_size),
            texture_sets=self.train_textures,
            path_length=horizon,
            path_upsample=path_upsample
        )

    def get_normalizer(self, mode='limits', **kwargs):
        # generate some data to fit the normalizer
        # action trajectories are in the range [0, 167]
        action = np.array([
            np.arange(0, 168, dtype=np.float32),
            np.arange(0, 168, dtype=np.float32)
        ]).T
        action = action[None].repeat(self.horizon, axis=0)
        action = action[None]  # batch axis

        data = {
            "action": action,
            "agent_pos": np.zeros((self.horizon, 2), dtype=np.float32)
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer
 
    def get_validation_dataset(self):
        val_dataset = copy.copy(self)
        val_dataset.params.texture_sets = self.val_textures
        val_dataset.n_samples = int(self.n_samples * self.val_ratio)
        return val_dataset

    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # generate a new maze for each sample
        X, y = maze_a_star.generate_sample(self.params)
        X = X.transpose(2, 0, 1)  # (render_size, render_size, 3) -> (3, render_size, render_size)
        # copy input image for horizon
        # X = X[None].repeat(self.horizon, axis=0)  # (horizon, 3, render_size, render_size)

        data = {
            "obs": {
                "image": X,
                # agent position not used
                "agent_pos": np.zeros((self.horizon, 2), dtype=np.float32)
            },
            "action": y
        }
        torch_data = dict_apply(data, torch.from_numpy)

        # expand time dim
        torch_data["obs"]["image"] = torch_data["obs"]["image"][None].expand(self.horizon, -1, -1, -1)

        assert torch_data["obs"]["image"].shape == (self.horizon, 3, self.render_size, self.render_size)
        assert torch_data["action"].shape == (self.horizon, 2)
        return torch_data


def test_dataset():
    dataset = MazeAStarImageDataset(n_samples=1000)
    val_dataset = dataset.get_validation_dataset()

    normalizer = dataset.get_normalizer()
    for k, v in normalizer.get_input_stats().items():
        print(k)
        # v: ParameterDict
        for kk, vv in v.items():
            print(kk, vv)

    def _plot_data(dataloader, save_dir, save_name):
        for i, batch in enumerate(dataloader):
            # print(batch["obs"]["image"].shape)
            # print(batch["action"].shape)
            # print("min, max:", batch["obs"]["image"].min(), batch["obs"]["image"].max())
            # print("min, max:", batch["action"].min(), batch["action"].max())

            img = batch["obs"]["image"][0, 0].permute(1, 2, 0).numpy()
            path = batch["action"][0].numpy()

            plt.figure()
            plt.imshow(img)
            plt.scatter(path[:, 1], path[:, 0], c=np.linspace(0, 1, len(path)), cmap="magma", marker=".")
            plt.axis('off')
            plt.savefig(save_dir / f"{save_name}{i}.png")

    save_dir = fig_dir / "dataset"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    _plot_data(dataloader, save_dir, "train_sample")

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    _plot_data(val_dataloader, save_dir, "val_sample")


@hydra.main(version_base=None)
def train(_):
    cfg = OmegaConf.load(conf_dir / "maze_a_star_diffusion_policy_cnn.yaml")
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    # test_dataset()
    train()
