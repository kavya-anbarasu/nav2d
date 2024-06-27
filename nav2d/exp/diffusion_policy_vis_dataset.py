import gym
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import torch
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.data import DataLoader
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from diffusion_policy_exp.point_maze.maze_image_dataset import MinariMazeImageDataset
from nav2d import make, register


fig_dir = Path(__file__).parent / "figures" / "vis_dataset"
conf_dir = Path(__file__).parents[1] / "config"


@register("analysis.plot_dataset")
def plot_dataset(dataset, dataloader_kwargs, save_dir=None):
    dataset = make("dataset." + dataset) if isinstance(dataset, str) else dataset
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    normalizer = dataset.get_normalizer()
    print("create dataloader")
    train_dataloader = DataLoader(dataset, **dataloader_kwargs)
    print("get batch")
    batch = next(iter(train_dataloader))
    print("done")

    def _log(v):
        if isinstance(v, torch.Tensor):
            return {
                "shape": v.shape,
                "max": v.max().item(),
                "min": v.min().item(),
                "dtype": str(v.dtype),
            }
        elif isinstance(v, dict):
            return {k: _log(vv) for k, vv in v.items()}
        else:
            return str(v)

    if save_dir is not None:
        json.dump(_log(batch), open(save_dir / "dataset_batch.json", "w"), indent=2, sort_keys=True)

    obs = normalizer(batch["obs"])
    action = normalizer({"action": batch["action"]})["action"]

    for i in range(3):
        if save_dir is not None:
            _save_dir = save_dir / f"sample{i}"
            _save_dir.mkdir(parents=True, exist_ok=True)

        img = batch["obs"]["image"][i, 0].permute(1, 2, 0).numpy()
        plt.figure()
        plt.imshow(img, origin="lower")
        if _save_dir is not None:
            plt.savefig(_save_dir / f"img{i}.png")

        agent_pos = obs["agent_pos"][i].detach().numpy()
        a = action[i].detach().numpy()
        plt.figure()
        for t in range(agent_pos.shape[0]):
            plt.scatter(*agent_pos[t], color="tab:green")
            plt.scatter(*a[t], color="tab:blue")
        plt.axis('equal')
        if _save_dir is not None:
            plt.savefig(_save_dir / f"action{i}.png")


        # plot agent_pos vs action, one subplot for x and another for y
        plt.subplot(2, 1, 1)
        plt.plot(agent_pos[:, 0], label="agent_pos", marker=".")
        plt.plot(a[:, 0], label="action", marker=".")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("x")

        plt.subplot(2, 1, 2)
        plt.plot(agent_pos[:, 1], label="agent_pos", marker=".")
        plt.plot(a[:, 1], label="action", marker=".")
        plt.xlabel("t")
        plt.ylabel("y")

        if _save_dir is not None:
            plt.savefig(_save_dir / f"action_vs_agent_pos{i}.png")        
        
        
        


    # pushT
    # obs.image: (64, 16, 3, 96, 96)
    # obs.agent_pos: (64, 16, 2)
    # action: (64, 16, 2) 

    # minari maze
    # obs.image: (64, 16, 3, 9, 12)
    # obs.agent_pos: (64, 16, 2)
    # action: (64, 16, 2)
    print(batch["obs"]["image"].shape, batch["obs"]["agent_pos"].shape)
    print(batch["action"].shape)


def plot_env(env, n_steps=100, save_dir=None):
    env = make("env." + env) if isinstance(env, str) else env

    if save_dir is not None:
        # save observation and action space log
        obs_space = env.observation_space
        action_space = env.action_space

        def _space_to_dict(space):
            if isinstance(space, gym.spaces.Dict):
                return {k: _space_to_dict(v) for k, v in space.items()}
            else:
                return {
                    "low": space.low.tolist(),
                    "high": space.high.tolist(),
                    "shape": space.shape,
                    "dtype": str(space.dtype),
                }

        obs_space_log = _space_to_dict(obs_space)
        action_space_log = _space_to_dict(action_space)
        json.dump(obs_space_log, open(save_dir / "obs_space.json", "w"), indent=2, sort_keys=True)
        json.dump(action_space_log, open(save_dir / "action_space.json", "w"), indent=2, sort_keys=True)

    obs = env.reset()
    results = defaultdict(list)
    for i in range(n_steps):
        a = env.action_space.sample()
        print(a)
        obs, reward, done, info = env.step(a)
        img = env.render(mode="rgb_array")
        print(img.shape)
        results["image"].append(img)
        results["obs"].append(obs)
        results["reward"].append(reward)
        results["done"].append(done)
        results["info"].append(info)
        results["action"].append(a)

    fig, ax = plt.subplots()
    ax.axis("off")
    ims = [[ax.imshow(img)] for img in results["image"]]
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    if save_dir is not None:
        ani.save(save_dir / "env.gif", writer="imagemagick")

        # save results TODO np array not serializable
        # json.dump(results, open(save_dir / "env.json", "w"), indent=2, sort_keys=True)
        
        # save actions
        actions = [a.tolist() for a in results["action"]]
        json.dump(actions, open(save_dir / "actions.json", "w"), indent=2, sort_keys=True)




@register("config.image_dummy#diffusion_policy#cnn")
def config():
    cfg = OmegaConf.load(conf_dir / "image_dummy_diffusion_policy_cnn.yaml")
    return cfg


@register("config.image_dummy-binary#diffusion_policy#cnn")
def config():
    cfg = OmegaConf.load(conf_dir / "image_dummy_diffusion_policy_cnn.yaml")
    cfg.task.dataset.dataset_id = "binary"
    return cfg


@register("config.image_dummy-a_star#diffusion_policy#cnn")
def config():
    cfg = OmegaConf.load(conf_dir / "image_dummy_diffusion_policy_cnn.yaml")
    cfg.task.dataset.dataset_id = "a_star"

    horizon = 100
    cfg.task.dataset.horizon = horizon
    cfg.policy.horizon = horizon
    cfg.horizon = horizon
    return cfg


@register("config.image_dummy-a_star#diffusion_policy#cnn_24")
def config():
    cfg = OmegaConf.load(conf_dir / "image_dummy_diffusion_policy_cnn_24.yaml")
    cfg.task.dataset.dataset_id = "a_star"

    # horizon = 30
    # cfg.task.dataset.horizon = horizon
    # cfg.policy.horizon = horizon
    # cfg.horizon = horizon
    return cfg


@register("config.image_maze#diffusion_policy#cnn")
def config():
    cfg = OmegaConf.load(conf_dir / "image_maze_diffusion_policy_cnn.yaml")
    return cfg


@register("config.image_pusht#diffusion_policy#cnn")
def config():
    cfg = OmegaConf.load(conf_dir / "image_pusht_diffusion_policy_cnn.yaml")
    return cfg


@register("dataset.image_dummy")
def dataset_image_dummy():
    cfg = make("config.image_dummy#diffusion_policy#cnn")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset


@register("dataset.image_dummy-binary")
def dataset_image_dummy():
    cfg = make("config.image_dummy#diffusion_policy#cnn")
    cfg.task.dataset.dataset_id = "binary"
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset


@register("dataset.image_dummy-a_star")
def dataset_image_dummy():
    cfg = make("config.image_dummy-a_star#diffusion_policy#cnn")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset


@register("dataset.image_maze-1episode")
def dataset_image_maze():
    dataset = MinariMazeImageDataset(
        dataset_id="pointmaze-large-v2",
        horizon=16,
        max_episodes=1,
        skip_steps=10,
        pad_after=7,
        pad_before=1,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=90
    )
    return dataset


@register("dataset.image_maze")
def dataset_image_maze():
    cfg = make("config.image_maze#diffusion_policy#cnn")
    dataset = hydra.utils.instantiate(cfg.task.dataset, max_episodes=100)
    return dataset


@register("dataset.image_pusht")
def dataset_image_pusht():
    from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
    dataset_kwargs = {
        "horizon": 16,
        "max_train_episodes": 90,
        "pad_after": 7,
        "pad_before": 1,
        "seed": 42,
        "val_ratio": 0.02,
        "zarr_path": (Path(__file__).parent.parent / "data/pusht/pusht_cchi_v7_replay.zarr").resolve(),
    }
    dataset = PushTImageDataset(**dataset_kwargs)
    return dataset


@register("env.image_dummy")
def env_image_dummy():
    from nav2d.dataset.dummy_dataset import DummyEnv
    env = DummyEnv()
    return env


@register("env.image_maze")
def env_image_maze():
    from diffusion_policy_exp.point_maze.maze_image_dataset import MinariMazeEnv
    env = MinariMazeEnv(dataset_id="pointmaze-large-v2")
    return env


@register("env.image_pusht")
def env_image_pusht():
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    env = PushTImageEnv(render_size=96)
    return env


if __name__ == '__main__':
    import nav2d.exp.straight_lines_train

    env_ids = [
        # "straight_lines",
        "maze_a_star",
        # "image_dummy",
        # "image_maze",
        # "image_pusht",
    ]
    # dataset_id = None
    # dataset_id = "image_maze-1episode"
    # dataset_id = "image_dummy-binary"
    # dataset_id = "image_dummy-a_star"
    # dataset_id = "straight_lines"
    dataset_id = "maze_a_star"

    for env_id in env_ids:
        if dataset_id is None:
            dataset_id = env_id

        save_dir = fig_dir / dataset_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # plot_env(env_id, save_dir=save_dir)

        # cfg = make(f"config.{env_id}#diffusion_policy#cnn")

        cfg = OmegaConf.load(conf_dir / "maze_a_star_diffusion_policy_cnn.yaml")
        dataloader_kwargs = cfg.dataloader
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        plot_dataset(
            # dataset=dataset_id,
            dataset=dataset,
            dataloader_kwargs=dataloader_kwargs,
            save_dir=save_dir / "dataset"
        )


