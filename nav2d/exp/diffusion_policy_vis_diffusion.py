import os
from pathlib import Path
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy_exp.point_maze.maze_image_dataset import MinariMazeEnv
import hydra
import numpy as np
import torch
import dill
import wandb
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.data import DataLoader
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from diffusion_policy_exp import make, register


fig_dir = Path(__file__).parent / "figures" / "vis_diffusion"
# root_dir = Path(__file__).resolve().parents[1]
root_dir = Path(__file__).resolve().parents[0]


def eval_policy(policy, obs_dict):
    """
    Adapted from DiffusionUnetHybridImagePolicy.predict_action
    """
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        Modified from DiffusionUnetHybridImagePolicy.conditional_sample to store intermetiate trajectories
        """
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        trajectories = []
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample

            trajectories.append(trajectory.clone())

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory, trajectories
        # import matplotlib.pyplot as plt
        # from matplotlib.animation import FuncAnimation
        # fig, ax = plt.subplots()
        # scatter = ax.scatter(res[0][0, :, 0], res[0][0, :, 1])

        # def update(frame_number):
        #     scatter.set_offsets(res[frame_number][0, :, :2])
        #     return scatter,
        # from matplotlib.animation import FuncAnimation, PillowWriter
        # ani = FuncAnimation(fig, update, frames=len(res), blit=True)
        # writer = PillowWriter(fps=15)  # Adjust fps as needed
        # ani.save("trajectory_animation.gif", writer=writer)


    # assert isinstance(policy, DiffusionUnetHybridImagePolicy)
    assert 'past_action' not in obs_dict # not implemented yet
    # normalize input
    nobs = policy.normalizer.normalize(obs_dict)
    value = next(iter(nobs.values()))
    B, To = value.shape[:2]
    T = policy.horizon
    Da = policy.action_dim
    Do = policy.obs_feature_dim
    To = policy.n_obs_steps

    # build input
    device = policy.device
    dtype = policy.dtype

    # handle different ways of passing observation
    local_cond = None
    global_cond = None
    if policy.obs_as_global_cond:
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = policy.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    else:
        # condition through impainting
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = policy.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        nobs_features = nobs_features.reshape(B, To, -1)
        cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        cond_data[:,:To,Da:] = nobs_features
        cond_mask[:,:To,Da:] = True

    # run sampling
    nsample, trajectories = conditional_sample(policy,
        cond_data, 
        cond_mask,
        local_cond=local_cond,
        global_cond=global_cond,
        **policy.kwargs)
    
    # unnormalize prediction
    naction_pred = nsample[...,:Da]
    action_pred = policy.normalizer['action'].unnormalize(naction_pred)

    # unnormalize trajectories
    trajectories = [policy.normalizer['action'].unnormalize(t) for t in trajectories]

    # get action
    start = To - 1
    end = start + policy.n_action_steps
    action = action_pred[:,start:end]
    
    result = {
        'action': action,
        'action_pred': action_pred,
        "trajectories": trajectories,
    }
    return result



def plot_batch(batch, idx, save_dir):
    # obs = normalizer(batch["obs"])  # image, agent_pos keys
    # action = normalizer({"action": batch["action"]})["action"]

    # 16 x 2
    agent_pos = batch["obs"]["agent_pos"][idx].detach().cpu().numpy()
    # 16 x 3 x 96 x 96
    images = batch["obs"]["image"][idx].detach().cpu().numpy()
    # 16 x 2
    action = batch["action"][idx].detach().cpu().numpy()

    # plot agent pos and action
    plt.figure()
    plt.plot(agent_pos[:, 0], agent_pos[:, 1], label="agent_pos", marker=".")
    plt.plot(action[:, 0], action[:, 1], label="action", marker=".")
    plt.legend()
    plt.axis('equal')
    if save_dir:
        plt.savefig(save_dir / "agent_pos_action.png")

    # plot images
    fig, ax = plt.subplots()
    ax.axis("off")
    # transpose to (16, 96, 96, 3)
    images = images.transpose(0, 2, 3, 1)
    ims = [[ax.imshow(img)] for img in images]
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    if save_dir is not None:
        ani.save(save_dir / "img_obs.gif", writer="imagemagick")


@register("workspace.image_maze-checkpoint")
def workspace_image_maze_checkpoint(save_dir=None):
    checkpoint_path = root_dir / "outputs/2024-04-13/21-41-56/checkpoints/latest.ckpt"
    # load checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=save_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    return workspace


def plot_eval_policy(normalizer, policy_res, n_samples=5, batch=None, save_dir=None, device="cpu"):
    for idx in range(n_samples):
        print(f"Sample {idx}")
        _save_dir = save_dir / f"sample{idx}" if save_dir else None
        _save_dir.mkdir(parents=True, exist_ok=True) if _save_dir else None

        if batch:
            plot_batch(batch, idx, _save_dir)

        action_pred = policy_res["action_pred"].detach().cpu().numpy()
        plt.figure()
        plt.plot(action_pred[idx, :, 0], action_pred[idx, :, 1], label="action_pred", marker=".")
        if batch:
            target = batch["action"].detach().cpu().numpy()
            plt.plot(target[idx, :, 0], target[idx, :, 1], label="target", marker=".")
        plt.legend()
        plt.axis('equal')
        if _save_dir:
            plt.savefig(_save_dir / "action_pred_target.png")

        # plot trajectories
        # 100 x 64 x 16 x 2
        trajectories = policy_res["trajectories"]
        trajectories = np.array([t.detach().cpu().numpy() for t in trajectories])
        trajectories = trajectories[:, idx]
        # skip beginning
        trajectories = trajectories[80:]

        # animate 2d plot
        fig, ax = plt.subplots()
        ims = [ax.plot(t[:, 0], t[:, 1], marker=".", color="tab:blue") for t in trajectories]
        # for t in trajectories:
        #     im = ax.plot(t[:, 0], t[:, 1], marker=".", color="tab:blue")
        #     ims.append(im)

        # target = nobs["action"][idx]
        if batch:
            target = batch["action"][idx].detach().cpu().numpy()
            ax.plot(target[:, 0], target[:, 1], marker=".", color="tab:orange")
        ax.axis('equal')
        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=5000)
        if _save_dir:
            ani.save(_save_dir / "trajectories.gif", writer="imagemagick")

        # same plot but normalized trajectories and target
        ntrajectories = normalizer["action"].normalize(torch.tensor(trajectories, device=device))
        fig, ax = plt.subplots()
        ims = [ax.plot(t[:, 0], t[:, 1], marker=".", color="tab:blue") for t in ntrajectories]
        if batch:
            ntarget = normalizer["action"].normalize(torch.tensor(target, device=device))
            ax.plot(ntarget[:, 0], ntarget[:, 1], marker=".", color="tab:orange")
        ax.axis('equal')
        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=5000)
        if _save_dir:
            ani.save(_save_dir / "normalized_trajectories.gif", writer="imagemagick")

        plt.close('all')


def main(checkpoint, device, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=save_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy: DiffusionUnetHybridImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.to(device)
    policy.eval()

    if "max_episodes" in cfg.task.dataset:
        # don't want to render all the maze images
        cfg.task.dataset.max_episodes = 3
    if "n_samples" in cfg.task.dataset:
        cfg.task.dataset.n_samples = 3

    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)
    dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer()

    batch = next(iter(dataloader))
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    obs_dict = {
        "agent_pos": batch["obs"]["agent_pos"],
        "image": batch["obs"]["image"],
        "action": batch["action"]
    }
    nobs = normalizer.normalize(obs_dict)
    res = eval_policy(policy, obs_dict)
    plot_eval_policy(normalizer, res, save_dir=save_dir, batch=batch)

    # run eval
    # env_runner = hydra.utils.instantiate(
    #     cfg.task.env_runner,
    #     output_dir=save_dir)
    # runner_log = env_runner.run(policy)

    # # dump log to json
    # json_log = dict()
    # for key, value in runner_log.items():
    #     if isinstance(value, wandb.sdk.data_types.video.Video):
    #         json_log[key] = value._path
    #     else:
    #         json_log[key] = value
    # out_path = os.path.join(save_dir, 'eval_log.json')
    # json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    # env_id = "image_dummy-binary"
    # env_id = "image_dummy-a_star"
    env_id = "straight_lines"
    # checkpoint = root_dir / 'outputs/2024-04-16/22-06-12/checkpoints/latest.ckpt'

    # env_id = "image_pusht"
    # checkpoint = root_dir / 'data/outputs/pusht_image/checkpoints/latest.ckpt'

    # env_id = "image_maze"
    # run_name = "2024-04-18/09-06-21"
    # run_name = "2024-04-27/19-57-06"
    # run_name = "2024-04-28/10-09-01"
    run_name = "2024-04-29/14-14-54"

    # checkpoint = root_dir / "outputs/2024-04-13/15-40-37/checkpoints/latest.ckpt"
    # checkpoint = root_dir / "outputs/2024-04-13/21-41-56/checkpoints/latest.ckpt"
    checkpoint = root_dir / f"outputs/{run_name}/checkpoints/latest.ckpt"

    # device = 'cuda:0'
    device = 'cpu'

    save_dir = fig_dir / env_id / run_name
    main(checkpoint, device, save_dir=save_dir)
