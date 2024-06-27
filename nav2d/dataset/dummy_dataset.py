# Adapted from diffusion_policy pusht image
from typing import Dict
import torch
import numpy as np
import copy
from nav2d.exp.grid_a_star import generate_data
import wandb
import collections
import pathlib
import tqdm
import dill
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import gym
from gym import spaces
import minari

import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseLowdimDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class DummyImageDataset(BaseImageDataset):
    def __init__(self,
            dataset_id="default",
            n_samples=1000,
            horizon=50,
            pad_before=0,
            pad_after=0,
            render_size=96
            ):
        print("DummyImageDataset init, dataset_id:", dataset_id, "n_samples:", n_samples, "horizon:", horizon)
        super().__init__()
        self.n_samples = n_samples
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        if dataset_id == "default":
            self.X_image = np.zeros((n_samples, horizon, 3, render_size, render_size), dtype=np.float32)
            self.X_agent_pos = np.zeros((n_samples, horizon, 2), dtype=np.float32)
            
            y = np.arange(horizon, dtype=np.float32)[:, None].repeat(2, axis=1)
            self.y = y[None].repeat(n_samples, axis=0)

        elif dataset_id == "binary":
            # when X_image is white, y goes one way, when X_image is black, y goes the other way
            self.X_image = np.zeros((n_samples, horizon, 3, render_size, render_size), dtype=np.float32)
            self.X_agent_pos = np.zeros((n_samples, horizon, 2), dtype=np.float32)
            self.y = np.zeros((n_samples, horizon, 2), dtype=np.float32)

            # half of the sample
            self.X_image[:n_samples//2] = 1
            self.y[:n_samples//2, :, 0] = np.arange(horizon, dtype=np.float32)
            # the other half
            self.y[n_samples//2:, :, 1] = -np.arange(horizon, dtype=np.float32)

        elif dataset_id == "a_star":
            print("Generate {} samples of A* dataset".format(n_samples))
            grid_size = render_size
            X, y = generate_data(n_samples, grid_size=grid_size, max_path_length=horizon)
            print("Done")
            X = X / 255
            X = X.transpose(0, 3, 1, 2)  # (n_samples, 3, render_size, render_size)
            # repeat image for horizon
            X = X[:, None].repeat(horizon, axis=1)  # (n_samples, horizon, render_size, render_size, 3)
            assert y.shape == (n_samples, horizon, 2), y.shape
            assert X.shape == (n_samples, horizon, 3, grid_size, grid_size), X.shape
            self.X_image = X
            self.X_agent_pos = np.zeros((n_samples, horizon, 2), dtype=np.float32)
            self.y = y
            print(self.X_image.shape, self.X_agent_pos.shape, self.y.shape)

        else:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            "action": self.y,
            "agent_pos": self.X_agent_pos
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer
 
    # def get_validation_dataset(self):
    #     pass

    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = {
            "obs": {
                "image": self.X_image[idx],
                "agent_pos": self.X_agent_pos[idx]
            },
            "action": self.y[idx]
        }

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class DummyEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            render_action=True,
            render_size=96,
        ):
        self.render_size = render_size
        self.window_size = ws = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, render_size, render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float64),
            high=np.array([ws, ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

    def reset(self):
        return {
            "image": np.zeros((3, self.render_size, self.render_size), dtype=np.float32),
            "agent_pos": np.zeros(2, dtype=np.float32)
        }

    def step(self, action):
        obs = {
            "image": np.zeros((3, self.render_size, self.render_size), dtype=np.float32),
            "agent_pos": np.zeros((2), dtype=np.float32)
        }
        reward = 0
        done = False
        info = {}
        return obs, reward, done, info

    def render(self, mode):
        assert mode == 'rgb_array'
        return np.zeros((self.render_size, self.render_size, 3), dtype=np.float32)


class DummyImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        _env = DummyEnv(
            render_size=render_size
        )

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    _env,
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseImagePolicy):
        return {
            "val_loss": 0,
            "train_action_mse_error": 0,
            "test_mean_score": 0
        }

        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval DummyImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

