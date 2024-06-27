import dill
import hydra
from matplotlib.collections import LineCollection
from nav2d.exp.grid_a_star import generate_data
from nav2d.exp.straight_lines_train import generate_straight_lines_data
from omegaconf import OmegaConf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import torch
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from nav2d.exp.diffusion_policy_vis_diffusion import eval_policy, plot_eval_policy


fig_dir = Path(__file__).parent / "figures" / "vis_diffusion" / "grid_a_star"
ckpt_dir = Path(__file__).resolve().parents[0]
# device = 'cuda:0'
device = 'cpu'
device = torch.device(device)


def animate_diffusion(img_obs, target, diffusion_traj, save_path=None):
    # diffusion_traj: (100 x horizon x 2)
    plt.figure()
    img = img_obs[0].transpose(1, 2, 0)
    plt.plot(target[:, 1], target[:, 0], color="black")
    plt.imshow(img, origin="lower")
    # animate scatter
    # color gradient for time
    colormap = plt.cm.viridis
    colors = np.linspace(0, 1, diffusion_traj.shape[1])
    scat = plt.scatter(diffusion_traj[0, :, 1], diffusion_traj[0, :, 0], marker=".", c=colors, cmap=colormap)

    def update(frame):
        scat.set_offsets(
            np.c_[diffusion_traj[frame, :, 1], diffusion_traj[frame, :, 0]]
        )
        return scat,


    ani = animation.FuncAnimation(plt.gcf(), update, frames=len(diffusion_traj), blit=True)
    if save_path:
        ani.save(save_path, writer="imagemagick")

    return

        # fig = plt.figure()
        # img = img_obs[trial, 0].transpose(1, 2, 0)
        # tgt_path = target[trial]
        # plt.plot(tgt_path[:, 1], tgt_path[:, 0], marker=".", color="black")
        # plt.imshow(img, origin="lower")

        # ims = []
        # for t in trajectories:
        #     # im = plt.plot(t[trial, :, 1], t[trial, :, 0], marker=".", color="tab:blue", zorder=-1)
        #     # color gradient for time
        #     colormap = plt.cm.viridis
        #     # plotting the trajectory with color gradient
        #     x = t[trial, :, 1]
        #     y = t[trial, :, 0]
        #     points = np.array([x, y]).T.reshape(-1, 1, 2)
        #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #     lc = LineCollection(segments, cmap=colormap, norm=plt.Normalize(0, 1))
        #     lc.set_array(np.linspace(0, 1, len(x)))
        #     lc.set_linewidth(2)
        #     plt.gca().add_collection(lc)

        #     ims.append(im)
        # ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=5000)
        # if save_dir:
        #     ani.save(save_dir / f"diffusion{trial}.gif", writer="imagemagick")




def plot_diffusion_grid(policy_res, batch, n_samples=10, save_dir=None):
    # batch x horizon x 2
    action_pred = policy_res["action_pred"].detach().cpu().numpy()
    target = batch["action"].detach().cpu().numpy()
    # batch x horizon x 3 x 96 x 96
    img_obs = batch["obs"]["image"].detach().cpu().numpy()

    # diffusion output (100 x batch x horizon x 2)
    trajectories = policy_res["trajectories"]
    trajectories = np.array([t.detach().cpu().numpy() for t in trajectories])
    # skip beginning
    trajectories = trajectories[80:]

    for trial in range(n_samples):
        animate_diffusion(
            img_obs[trial], target[trial], trajectories[:, trial], save_path=save_dir / f"diffusion{trial}.gif"
        )


def analyze_error_pattern(policy, save_dir=None):
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 100
    # n_samples = 300
    grid_size = 96
    horizon = policy.horizon
    # fix start position
    # TODO: fix start pos in training data as well so that it's easier to analyze
    start_pos = (50, 50)
    X, y = generate_data(
        n_samples,
        grid_size=grid_size,
        max_path_length=horizon,
        start=start_pos
    )
    goal_positions = [_y[-1] for _y in y]
    print(goal_positions)
    X = X / 255
    # repeat image for horizon
    X = X[:, None].repeat(horizon, axis=1)  # (n_samples, horizon, 96, 96, 3)
    X = X.transpose(0, 1, 4, 2, 3)  # (n_samples, horizon, 3, 96, 96)
    assert y.shape == (n_samples, horizon, 2)
    assert X.shape == (n_samples, horizon, 3, grid_size, grid_size)
    X_agent_pos = np.zeros((n_samples, horizon, 2), dtype=np.float32)

    batch = {
        "obs": {
            "agent_pos": torch.from_numpy(X_agent_pos),
            "image": torch.from_numpy(X)
        },
        "action": torch.from_numpy(y)
    }
    obs_dict = {
        "agent_pos": batch["obs"]["agent_pos"],
        "image": batch["obs"]["image"],
        "action": batch["action"]
    }
   
    res = eval_policy(policy, obs_dict)
    plot_diffusion_grid(res, batch, save_dir=save_dir)

    # batch x horizon x 2
    action_pred = res["action_pred"].detach().cpu().numpy()
    target = batch["action"].detach().cpu().numpy()
    error = np.mean((action_pred - target)**2, axis=(1, 2))

    plt.figure()
    for goal_pos, err in zip(goal_positions, error):
        # scatter marker color proportional to error
        plt.scatter(goal_pos[1], goal_pos[0], c=err, cmap="viridis", vmin=np.min(error), vmax=np.max(error))
    plt.colorbar()
    plt.title("Error pattern")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "error_pattern.png")


def analyze_error_pattern_straight_lines(policy, cfg, save_dir=None):
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 50
    # n_samples = 500
    # grid_size = 24
    grid_size = cfg.task.dataset.grid_size
    horizon = policy.horizon
    # fix start position
    # TODO: fix start pos in training data as well so that it's easier to analyze
    # start_pos = (0, 0)
    start_pos = (12, 12)
    X, y = generate_straight_lines_data(
        n_samples,
        grid_size=grid_size,
        n_steps=horizon,
        start=start_pos
    )
    goal_positions = [_y[-1] for _y in y]
    print(goal_positions)

    X = X / 255
    # repeat image for horizon
    X = X[:, None].repeat(horizon, axis=1)  # (n_samples, horizon, 96, 96, 3)
    X = X.transpose(0, 1, 4, 2, 3)  # (n_samples, horizon, 3, 96, 96)
    assert y.shape == (n_samples, horizon, 2)
    assert X.shape == (n_samples, horizon, 3, grid_size, grid_size)
    X_agent_pos = np.zeros((n_samples, horizon, 2), dtype=np.float32)

    batch = {
        "obs": {
            "agent_pos": torch.from_numpy(X_agent_pos),
            "image": torch.from_numpy(X)
        },
        "action": torch.from_numpy(y)
    }
    obs_dict = {
        "agent_pos": batch["obs"]["agent_pos"],
        "image": batch["obs"]["image"],
        "action": batch["action"]
    }
   
    res = eval_policy(policy, obs_dict)
    plot_diffusion_grid(res, batch, save_dir=save_dir)

    # batch x horizon x 2
    action_pred = res["action_pred"].detach().cpu().numpy()
    target = batch["action"].detach().cpu().numpy()
    error = np.mean((action_pred - target)**2, axis=(1, 2))

    plt.figure()
    for goal_pos, err in zip(goal_positions, error):
        # scatter marker color proportional to error
        plt.scatter(goal_pos[1], goal_pos[0], c=err, cmap="viridis", vmin=np.min(error), vmax=np.max(error))
    plt.colorbar()
    plt.title("Error pattern")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "error_pattern.png")


if __name__ == '__main__':
    # env_id = "image_dummy-a_star"
    # run_name = "2024-04-28/10-09-01"
    # run_name = "2024-04-28/14-03-12"

    env_id = "straight_lines"
    # run_name = "2024-04-29/14-05-29"  # wncmp8u3
    # run_name = "2024-04-29/14-14-54"  # l3ieogxk
    # run_name = "2024-04-30/15-57-56"  # o9dwfw43
    # run_name = "2024-04-30/15-56-29"  # n1obcuze
    # run_name = "2024-05-01/12-05-00"  # s739olt1
    run_name = "2024-05-01/14-44-46"  # 1j7cp2pl

    checkpoint = ckpt_dir / f"outputs/{run_name}/checkpoints/latest.ckpt"
    save_dir = fig_dir / env_id / run_name
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=save_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.to(device)
    policy.eval()

    if "straight_lines" in env_id:
        analyze_error_pattern_straight_lines(policy, cfg, save_dir=save_dir / "error_pattern")
    else:
        analyze_error_pattern(policy, save_dir=save_dir / "error_pattern")


    if "n_samples" in cfg.task.dataset:
        cfg.task.dataset.n_samples = 12

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataloader = DataLoader(dataset, **cfg.dataloader)

    batch = next(iter(dataloader))
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    obs_dict = {
        "agent_pos": batch["obs"]["agent_pos"],
        "image": batch["obs"]["image"],
        "action": batch["action"]
    }
    res = eval_policy(policy, obs_dict)

    plot_diffusion_grid(res, batch, save_dir=save_dir)


