from collections import defaultdict
import dill
import hydra
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from nav2d.env.texture import load_texture_set
import numpy as np
import json
import pandas as pd
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from nav2d.exp.diffusion_policy_vis_diffusion import eval_policy, plot_eval_policy
from nav2d.exp import maze_a_star
from nav2d import register


fig_dir = Path(__file__).parent / "figures" / "maze_a_star" / "eval"
ckpt_dir = Path(__file__).resolve().parents[0]
# device = 'cuda:0'
device = 'cpu'
device = torch.device(device)


def rcparams_default():
    plt.rcParams.update({
        "figure.figsize": (4, 2.5),
        "figure.dpi": 150,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.autolayout": True,
    })


def animate_diffusion(img_obs, target, diffusion_traj, save_path=None):
    # diffusion_traj: (100 x horizon x 2)
    plt.figure()
    img = img_obs[0].transpose(1, 2, 0)
    # plt.plot(target[:, 1], target[:, 0], color="black")
    plt.imshow(img)
    # animate scatter
    # color gradient for time
    colormap = plt.cm.magma
    colors = np.linspace(0, 1, diffusion_traj.shape[1])
    scat = plt.scatter(diffusion_traj[0, :, 1], diffusion_traj[0, :, 0], marker=".", c=colors, cmap=colormap)
    plt.axis('off')

    def update(frame):
        scat.set_offsets(
            np.c_[diffusion_traj[frame, :, 1], diffusion_traj[frame, :, 0]]
        )
        return scat,


    ani = animation.FuncAnimation(plt.gcf(), update, frames=len(diffusion_traj), blit=True)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(save_path, writer="imagemagick")


def plot_diffusion_maze(policy_res, batch, n_samples=50, animate=True, save_dir=None):
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
        if animate:
            animate_diffusion(
                img_obs[trial], target[trial], trajectories[:, trial], save_path=save_dir / "anim" / f"diffusion{trial}.gif"
            )

        # plot end trajectory
        img = img_obs[trial][0].transpose(1, 2, 0)
        traj = trajectories[:, trial]
        tgt = target[trial]

        plt.figure()
        plt.plot(tgt[:, 1], tgt[:, 0], color="black", zorder=-1)
        plt.imshow(img)
        plt.scatter(traj[-1, :, 1], traj[-1, :, 0], marker=".", c=np.linspace(0, 1, traj.shape[1]), cmap=plt.cm.magma)
        plt.axis('off')
        if save_dir is not None:
            plt.savefig(save_dir / f"diffusion{trial}.png")


def analyze_error_by_texture(policy, cfg, n_samples=20, textures=None, texture_paths=None, mode="train", save_dir=None):
    def analyze_single_texture(texture, save_dir=None):
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        params = dataset.params
        params.texture_sets = [texture]

        X_data = []
        y_data = []
        for i in range(n_samples):
            print("Generate sample", i)
            X, y = maze_a_star.generate_sample(params)

            # save image
            if save_dir is not None:
                plt.figure()
                plt.imshow(X)
                plt.axis('off')
                (save_dir / "images").mkdir(parents=True, exist_ok=True)
                plt.savefig(save_dir / "images" / f"sample_{i}.png")

            X = X.transpose(2, 0, 1)
            X = torch.from_numpy(X)
            X = X[None].expand(horizon, -1, -1, -1)
            X_data.append(X)
            y_data.append(y)
        X_data = torch.stack(X_data, dim=0)
        y_data = torch.from_numpy(np.stack(y_data, axis=0))

        batch = {
            "obs": {
                "agent_pos": torch.from_numpy(np.zeros((n_samples, horizon, 2), dtype=np.float32)),
                "image": X_data
            },
            "action": y_data
        }
        assert batch["obs"]["image"].shape == (n_samples, dataset.horizon, 3, dataset.render_size, dataset.render_size)
        assert batch["action"].shape == (n_samples, dataset.horizon, 2)
        obs_dict = {
            "agent_pos": batch["obs"]["agent_pos"],
            "image": batch["obs"]["image"],
            "action": batch["action"]
        }

        print("Eval policy")
        res = eval_policy(policy, obs_dict)
        print("Make plots")
        plot_diffusion_maze(res, batch, n_samples=n_samples, animate=False, save_dir=save_dir)

        # batch x horizon x 2
        action_pred = res["action_pred"].detach().cpu().numpy()
        target = batch["action"].detach().cpu().numpy()
        error = np.mean((action_pred - target)**2, axis=(1, 2))
        print("Mean error", np.mean(error))
        print("Max error", np.max(error))
        print("Min error", np.min(error))

        # separate trajectory into moving and idling parts
        goal_pos = target[:, -1]
        # first time step before idling
        # epsilon = 1
        epsilon = 1e-1
        # is_idling = np.all(np.abs(goal_pos[:, None] - action_pred) < epsilon, axis=-1)
        is_idling = np.all(np.abs(goal_pos[:, None] - target) < epsilon, axis=-1)

        moving_error = []
        reversed_targets = []
        for i in range(n_samples):
            # idx = np.max(np.argwhere(is_idling[i] == 0))
            if np.all(is_idling[i] == 0):
                idx = horizon  # didn't reach goal
            else:
                idx = np.min(np.argwhere(is_idling[i] == 1))

            moving_traj = action_pred[i, :idx]
            moving_tgt = target[i, :idx]

            idling_traj = action_pred[i, idx:]
            idling_tgt = target[i, idx:]

            moving_mse = np.mean((moving_traj - moving_tgt)**2)
            idling_mse = np.mean((idling_traj - idling_tgt)**2)

            moving_error.append(moving_mse)
            # assert np.allclose(idling_traj, idling_tgt, atol=epsilon)

            # reversed_traj = moving_traj[::-1]
            reversed_tgt = moving_tgt[::-1]
            # concat start pos
            start_pos = target[i, 0]
            # repeat to match length
            start_pos = np.repeat(start_pos[None], len(idling_traj), axis=0)
            reversed_tgt = np.concatenate([reversed_tgt, start_pos], axis=0)
            reversed_targets.append(reversed_tgt)

            # debuging
            plt.figure()
            plt.plot(moving_traj[:, 0], label="pred_moving")
            plt.plot(action_pred[i, :, 0], label="pred")
            plt.plot(reversed_tgt[:, 0], label="reversed")
            plt.plot(np.abs(goal_pos[i, None, 0] - action_pred[i, :, 0]))
            plt.vlines(idx, 0, 150)
            plt.hlines(goal_pos[i, 0], 0, 64, color="black", label="goal")
            plt.legend()
            _save_dir = save_dir / "debug"
            _save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(_save_dir / f"{i}_action_pred_x.png")

            plt.figure()
            plt.plot(moving_traj[:, 1], label="pred_moving")
            plt.plot(action_pred[i, :, 1])
            plt.plot(reversed_tgt[:, 1], label="reversed")
            plt.plot(np.abs(goal_pos[i, None, 1] - action_pred[i, :, 1]))
            plt.vlines(idx, 0, 150)
            plt.hlines(goal_pos[i, 1], 0, 64, color="black", label="goal")
            plt.legend()
            plt.savefig(_save_dir / f"{i}_action_pred_y.png")


        moving_error = np.array(moving_error)
        reversed_targets = np.stack(reversed_targets, axis=0)
        error_reversed_tgt = np.mean((action_pred - reversed_targets)**2, axis=(1, 2))

        plt.figure()
        plt.plot(error)
        plt.xlabel("Sample")
        plt.ylabel("Mean squared error")
        plt.yscale("log")
        if save_dir is not None:
            plt.savefig(save_dir / "error.png")

        plt.figure()
        plt.plot(error_reversed_tgt)
        plt.xlabel("Sample")
        plt.ylabel("MSE (reversed target)")
        plt.yscale("log")
        if save_dir is not None:
            plt.savefig(save_dir / "error_reversed_tgt.png")

        plt.figure()
        plt.plot(moving_error)
        plt.xlabel("Sample")
        plt.ylabel("Mean squared error (moving)")
        plt.yscale("log")
        if save_dir is not None:
            plt.savefig(save_dir / "error_moving.png")

        # count nb of success
        threshold_error = 10
        acc = np.count_nonzero(error < threshold_error) / n_samples
        acc_moving = np.count_nonzero(moving_error < threshold_error) / n_samples
        acc_reversed_tgt = np.count_nonzero(error_reversed_tgt < threshold_error) / n_samples
        acc_res = {
            "acc": acc,
            "acc_moving": acc_moving,
            "acc_reversed_tgt": acc_reversed_tgt
        }
        # save in json
        with open(save_dir / "acc.json", "w") as f:
            json.dump(acc_res, f)

        error_mean = np.mean(error)
        error_reversed_tgt = np.mean(error_reversed_tgt)
        return {
            "error": error_mean,
            "error_std": np.std(error),
            "moving_error": np.mean(moving_error),
            "moving_erorr_str": np.std(moving_error),
            "error_reversed_tgt": error_reversed_tgt,
            "error_reversed_tgt_std": np.std(error_reversed_tgt),
            "shape_error": np.min([error_mean, error_reversed_tgt], axis=0),
            **acc_res
        }


    horizon = policy.horizon
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    if textures is None:
        # TODO: make sure matches with training
        train_textures = dataset.train_textures
        val_textures = dataset.val_textures
        textures = train_textures if mode == "train" else val_textures
        texture_paths = dataset.train_texture_paths if mode == "train" else dataset.val_texture_paths
    else:
        assert texture_paths is not None

    if (save_dir / "results.csv").exists():
        # load results
        df = pd.read_csv(save_dir / "results.csv")
    
    else:
        results = defaultdict(list)

        # for texture_idx in range(200, 250):
        # for texture_idx in range(50, 70):
        for texture_idx in range(len(textures)):
            texture_path = texture_paths[texture_idx]

            texture_cfg = {
                "path": str(texture_path),
                "run": texture_path.name,
                "subtheme": texture_path.parent.name,
                "theme": texture_path.parent.parent.name
            }
            texture_name = "{}-{}-{}".format(texture_cfg["theme"], texture_cfg["subtheme"], texture_cfg["run"])
            _save_dir = save_dir / texture_name
            _save_dir.mkdir(parents=True, exist_ok=True)

            with open(_save_dir / "texture.json", "w") as f:
                json.dump(texture_cfg, f)

            res = analyze_single_texture(textures[texture_idx], save_dir=_save_dir)
            for k, v in res.items():
                results[k].append(v)
            results["texture"].append(texture_name)
            results["theme"].append(texture_cfg["theme"])
            results["subtheme"].append(texture_cfg["subtheme"])
            results["run"].append(texture_cfg["run"])

            plt.close('all')

        df = pd.DataFrame(results)
        df.to_csv(save_dir / "results.csv")

    # sort by texture name
    df = df.sort_values("texture")

    # matplotlib bar plot
    plt.figure(figsize=(10, 8), dpi=150)
    plt.bar(df["texture"], df["error"], yerr=df["error_std"])
    plt.xticks(rotation=90)
    plt.yscale("log")
    plt.ylabel("Mean squared error")
    plt.tight_layout()
    plt.savefig(save_dir / "error_by_texture.png")

    plt.figure(figsize=(10, 8), dpi=150)
    plt.bar(df["texture"], df["error_reversed_tgt"], yerr=df["error_reversed_tgt_std"])
    plt.xticks(rotation=90)
    plt.yscale("log")
    plt.ylabel("MSE (reversed target)")
    plt.tight_layout()
    plt.savefig(save_dir / "error_by_texture_reversed.png")

    plt.figure(figsize=(10, 8), dpi=150)
    plt.bar(df["texture"], df["shape_error"])
    plt.xticks(rotation=90)
    plt.yscale("log")
    plt.ylabel("MSE (ind of direction)")
    plt.tight_layout()
    plt.savefig(save_dir / "error_by_texture_shape.png")

    # plot acc
    plt.figure(figsize=(10, 8), dpi=150)
    plt.bar(df["texture"], df["acc"])
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(save_dir / "acc_by_texture.png")

    plt.figure(figsize=(10, 8), dpi=150)
    plt.bar(df["texture"], df["acc_moving"])
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy (moving)")
    plt.tight_layout()
    plt.savefig(save_dir / "acc_by_texture_moving.png")

    plt.figure(figsize=(10, 8), dpi=150)
    plt.bar(df["texture"], df["acc_reversed_tgt"])
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy (reversed target)")
    plt.tight_layout()
    plt.savefig(save_dir / "acc_by_texture_reversed.png")


    plt.figure(figsize=(10, 8), dpi=150)
    plt.bar(df["texture"], df["acc"] + df["acc_reversed_tgt"])
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy (normal + reversed target)")
    plt.tight_layout()
    plt.savefig(save_dir / "acc_by_texture_normal_or_reversed.png")



def analyze_error_pattern(policy, cfg, n_samples=20, save_dir=None):
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    grid_size = cfg.task.dataset.grid_size
    horizon = policy.horizon
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    # TODO: make sure matches with training
    train_textures = dataset.train_textures
    val_textures = dataset.val_textures

    texture_idx = 2
    textures = [train_textures[texture_idx]]
    texture_path = dataset.train_texture_paths[texture_idx]
    texture_cfg = {
        "path": str(texture_path),
        "run": texture_path.parent.name,
        "subtheme": texture_path.parent.parent.name,
        "theme": texture_path.parent.parent.parent.name
    }
    texture_name = "{}-{}-{}".format(texture_cfg["theme"], texture_cfg["subtheme"], texture_cfg["run"])

    with open(save_dir / "textures.json", "w") as f:
        import json
        # convert path to str
        json.dump(texture_cfg, f)


    params = dataset.params
    params.texture_sets = textures

    X_data = []
    y_data = []
    for i in range(n_samples):
        print("Generate sample", i)
        X, y = maze_a_star.generate_sample(params)
        X = X.transpose(2, 0, 1)
        X = torch.from_numpy(X)
        X = X[None].expand(horizon, -1, -1, -1)
        X_data.append(X)
        y_data.append(y)
    X_data = torch.stack(X_data, dim=0)
    y_data = torch.from_numpy(np.stack(y_data, axis=0))

    batch = {
        "obs": {
            "agent_pos": torch.from_numpy(np.zeros((n_samples, horizon, 2), dtype=np.float32)),
            "image": X_data
        },
        "action": y_data
    }
    assert batch["obs"]["image"].shape == (n_samples, dataset.horizon, 3, dataset.render_size, dataset.render_size)
    assert batch["action"].shape == (n_samples, dataset.horizon, 2)
    obs_dict = {
        "agent_pos": batch["obs"]["agent_pos"],
        "image": batch["obs"]["image"],
        "action": batch["action"]
    }

    print("Eval policy")
    res = eval_policy(policy, obs_dict)
    print("Make plots")
    plot_diffusion_maze(res, batch, n_samples=n_samples, animate=False, save_dir=save_dir)

    # batch x horizon x 2
    action_pred = res["action_pred"].detach().cpu().numpy()
    target = batch["action"].detach().cpu().numpy()
    error = np.mean((action_pred - target)**2, axis=(1, 2))
    print("Mean error", np.mean(error))
    print("Max error", np.max(error))
    print("Min error", np.min(error))

    error_reversed_tgt = np.mean((action_pred - target[:, ::-1])**2, axis=(1, 2))

    plt.figure()
    plt.plot(error)
    plt.xlabel("Sample")
    plt.ylabel("Mean squared error")
    if save_dir is not None:
        plt.savefig(save_dir / "error.png")

    plt.figure()
    plt.plot(error_reversed_tgt)
    plt.xlabel("Sample")
    plt.ylabel("MSE (reversed target)")
    if save_dir is not None:
        plt.savefig(save_dir / "error_reversed_tgt.png")


@register("model.maze_a_star.ecwq2mwi")
def model():
    env_id = "maze_a_star"
    run_name = "2024-05-02/13-35-58"  # ecwq2mwi

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

    horizon = policy.horizon

    def _eval(img, save_dir=None):
        # img: (batch, width, height, 3)
        img = torch.as_tensor(img, dtype=torch.float32)
        img /= 255
        assert img.dim() == 4, "Expecting (batch, width, height, 3) image tensor"
        n_samples = img.shape[0]
        X = img.permute(0, 3, 1, 2)
        X = X[:, None].expand(-1, horizon, -1, -1, -1)

        batch = {
            "obs": {
                "agent_pos": torch.from_numpy(np.zeros((n_samples, horizon, 2), dtype=np.float32)),
                "image": X
            },
            "action": torch.zeros((n_samples, horizon, 2), dtype=torch.float32)
        }
        obs_dict = {
            "agent_pos": batch["obs"]["agent_pos"],
            "image": batch["obs"]["image"],
            "action": batch["action"]
        }

        print("Eval policy")
        res = eval_policy(policy, obs_dict)
        print("Make plots")
        if save_dir is not None:
            plot_diffusion_maze(res, batch, n_samples=n_samples, animate=True, save_dir=save_dir)
        return res

    return _eval



if __name__ == '__main__':
    # random seed
    seed = 5
    np.random.seed(seed)
    torch.manual_seed(seed)

    rcparams_default()

    env_id = "maze_a_star"
    # run_name = "2024-05-02/13-35-58"  # ecwq2mwi

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

    if "n_samples" in cfg.task.dataset:
        cfg.task.dataset.n_samples = 50

    # analyze_error_by_texture(policy, cfg, save_dir=save_dir / "error_texture")
    # analyze_error_by_texture(policy, cfg, mode="val", save_dir=save_dir / "error_texture_val")


    texture_size = cfg.task.dataset.render_size // cfg.task.dataset.grid_size 
    new_texture_paths = [
        maze_a_star.texture_dir / ".." / "test2" / "artic" / "run0",
        maze_a_star.texture_dir / ".." / "test2" / "desert" / "run0",
        maze_a_star.texture_dir / ".." / "test2" / "desert" / "run1",
        maze_a_star.texture_dir / ".." / "test" / "arcade" / "run0",
        maze_a_star.texture_dir / ".." / "test" / "winter" / "run0",
        maze_a_star.texture_dir / ".." / "test" / "desert" / "run0",
        maze_a_star.texture_dir / "../test2/set4",
    ]
    new_texture_paths = [p.resolve() for p in new_texture_paths]
    new_textures = [
        load_texture_set(p, texture_size) for p in new_texture_paths
    ]
    analyze_error_by_texture(policy, cfg, textures=new_textures, texture_paths=new_texture_paths, save_dir=save_dir / "new_textures")

    # dataset = hydra.utils.instantiate(cfg.task.dataset)
    # dataloader = DataLoader(dataset, **cfg.dataloader)

    # batch = next(iter(dataloader))
    # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    # obs_dict = {
    #     "agent_pos": batch["obs"]["agent_pos"],
    #     "image": batch["obs"]["image"],
    #     "action": batch["action"]
    # }
    # res = eval_policy(policy, obs_dict)

    # plot_diffusion_maze(res, batch, save_dir=save_dir / "samples")
