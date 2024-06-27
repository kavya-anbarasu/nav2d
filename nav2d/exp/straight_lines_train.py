from pathlib import Path
from typing import Dict
from nav2d.env.texture import load_texture_set
import numpy as np
import copy

import torch
import hydra
from omegaconf import OmegaConf

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import diffusion_policy

from nav2d import make, register


fig_dir = Path(__file__).parent / "figures" / "straight_lines"
ckpt_dir = Path(__file__).parent / "outputs" / "straight_lines"
conf_dir = Path(__file__).parents[1] / "config"
texture_dir = Path(__file__).parent / ".." / "output" / "texture" / "textures_subthemes"


def generate_straight_lines_data(n_samples, grid_size, velocity=1, n_steps=100, start=None, goal=None, margin=None):
    """
    Randomly sample start and goal points and generate a straight line path between them
    """
    X = np.empty((n_samples, grid_size, grid_size, 3))
    Y = np.empty((n_samples, n_steps, 2))
    for i in range(n_samples):
        if margin is not None:
            min_idx = margin
            max_idx = grid_size - margin
        else:
            min_idx = 0
            max_idx = grid_size

        if start is None:
            _start = (np.random.randint(min_idx, max_idx), np.random.randint(min_idx, max_idx))
        else:
            _start = start

        if goal is None:
            _goal = (np.random.randint(min_idx, max_idx), np.random.randint(min_idx, max_idx))
            # make sure start and goal are different
            while _start == _goal:
                _goal = (np.random.randint(min_idx, max_idx), np.random.randint(min_idx, max_idx))
        else:
            _goal = goal

        x1, y1 = _start
        x2, y2 = _goal
        # create a straight line between the points
        path_len = int(np.sqrt((x1-x2)**2 + (y1-y2)**2) / velocity)
        x = np.linspace(x1, x2, num=path_len)
        y = np.linspace(y1, y2, num=path_len)
        # crop and pad path to n_steps
        if len(x) > n_steps:
            x = x[:n_steps]
            y = y[:n_steps]
        else:
            # pad with last x and y values
            x = np.pad(x, (0, n_steps-len(x)), mode='edge')
            y = np.pad(y, (0, n_steps-len(y)), mode='edge')

        grid_img = np.ones((grid_size, grid_size, 3))*255
        grid_img[_start] = (0, 255, 0)
        grid_img[_goal] = (255, 0, 0)

        X[i] = grid_img
        Y[i] = np.array([x, y]).T

    return X, Y


def generate_straight_lines_texture_data(n_samples, grid_size, texture_size=30, velocity=1, n_steps=100, start=None, goal=None):
    # remove margin to have space for texture on the border
    print("Start generating data")
    X, y = generate_straight_lines_data(n_samples, grid_size=grid_size, velocity=velocity, n_steps=n_steps, start=start, goal=goal, margin=texture_size//2)
    # X: (n_samples, grid_size, grid_size, 3)
    # y: (n_samples, n_steps, 2)
    print("done")

    texture_path = texture_dir / "Medieval Castle" / "subtheme0" / "run2"
    texture_set = load_texture_set(texture_path, texture_size)

    # convert PIL Image to rgb (remove alpha channel)
    agent_image = np.array(texture_set["agent"])[:, :, :3]
    goal_image = np.array(texture_set["goal"])[:, :, :3]

    # add texture to the grid
    print("Add texture to data")
    images = []
    for i in range(n_samples):
        grid_img = X[i]
        start_pos = y[i][0]
        goal_pos = y[i][-1]
        # convert to int
        start_pos = start_pos.astype(int)
        goal_pos = goal_pos.astype(int)
        # add agent image centered at start position
        grid_img[start_pos[0]-texture_size//2:start_pos[0]+texture_size//2, start_pos[1]-texture_size//2:start_pos[1]+texture_size//2] = agent_image
        # add goal image centered at goal position
        grid_img[goal_pos[0]-texture_size//2:goal_pos[0]+texture_size//2, goal_pos[1]-texture_size//2:goal_pos[1]+texture_size//2] = goal_image

        images.append(grid_img)
    
    images = np.stack(images).astype(np.uint8)
    print("images shape", images.shape)
    return images, y



class StraightLinesImageDataset(BaseImageDataset):
    def __init__(self,
            n_samples=1000,
            horizon=50,
            grid_size=20,
            texture=False,
            val_ratio=0.2
            ):
        super().__init__()
        self.horizon = horizon
        self.grid_size = grid_size

        print("Generate {} samples of straight lines dataset".format(n_samples))
        if not texture:
            X, y = generate_straight_lines_data(n_samples, grid_size=grid_size, n_steps=horizon)
        else:
            X, y = generate_straight_lines_texture_data(n_samples, grid_size=grid_size, n_steps=horizon)

        X = X / 255
        # repeat copies the data
        print("Start repeating data for horizon ", horizon)
        X = X[:, None].repeat(horizon, axis=1)  # (n_samples, horizon, grid_size, grid_size, 3)
        print("done")

        # TODO: broadcast doesn't copy but get pytorch warning that the array is not writeable
        # X = np.broadcast_to(X[:, None], (n_samples, horizon, grid_size, grid_size, 3))

        X = X.transpose(0, 1, 4, 2, 3)  # (n_samples, horizon, 3, grid_size, grid_size)
        assert y.shape == (n_samples, horizon, 2)
        assert X.shape == (n_samples, horizon, 3, grid_size, grid_size)
        # no agent pos input
        X_agent_pos = np.zeros((n_samples, horizon, 2), dtype=np.float32)
        
        # train test split
        n_val = int(n_samples * val_ratio)
        n_train = n_samples - n_val

        self.train_data = {
            "action": y[:n_train],
            "agent_pos": X_agent_pos[:n_train],
            "image": X[:n_train]
        }
        self.val_data = {
            "action": y[n_train:],
            "agent_pos": X_agent_pos[n_train:],
            "image": X[n_train:]
        }
        print(f"Training data shape: {self.train_data['image'].shape}")
        print(f"Validation data shape: {self.val_data['image'].shape}")

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            "action": self.train_data["action"],
            "agent_pos": self.train_data["agent_pos"]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer
 
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_data = self.val_data

        return val_set

    def __len__(self) -> int:
        return self.train_data["image"].shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        print("getitem", idx)
        data = {
            "obs": {
                "image": self.train_data["image"][idx],
                "agent_pos": self.train_data["agent_pos"][idx]
            },
            "action": self.train_data["action"][idx]
        }
        assert data["obs"]["image"].shape == (self.horizon, 3, self.grid_size, self.grid_size)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data



@register("dataset.image_straight_lines")
def dataset_straight_lines():
    cfg = make("config.image_straight_lines#diffusion_policy#cnn")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset


@register("config.image_straight_lines#diffusion_policy#cnn")
def config():
    cfg = OmegaConf.load(conf_dir / "straight_lines_diffusion_policy_cnn.yaml")
    return cfg


@register("config.image_straight_lines_texture#diffusion_policy#cnn")
def config():
    cfg = OmegaConf.load(conf_dir / "straight_lines_texture_diffusion_policy_cnn.yaml")
    return cfg


@hydra.main(version_base=None)
def train(_):
    # cfg = make("config.image_straight_lines#diffusion_policy#cnn")
    cfg = make("config.image_straight_lines_texture#diffusion_policy#cnn")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    train()

