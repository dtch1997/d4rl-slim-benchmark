import os
import random
from typing import Optional, Union

import d4rl_slim as d4rl
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from d4rl_slim import infos

def get_environment(dataset_name: str, version: str = 'v2', **kwargs) -> gym.Env:
    """Load a Gymnasium environment compatible with a given dataset."""
    # Modified to allow specifying version of environment

    if dataset_name not in infos.list_datasets():
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # mujoco environments are delegated to gymnasium mujoco v4 environments.
    if dataset_name.startswith(("ant-", "halfcheetah-", "hopper-", "walker2d-")):
        env_id, kwargs = {
            "halfcheetah": (f"HalfCheetah-{version}", {}),
            "hopper": (f"Hopper-{version}", {}),
            "walker2d": (f"Walker2d-{version}", {}),
            # v4 ant defaults to not using contact forces.
            "ant": (f"Ant-{version}", {"use_contact_forces": True}),
        }[dataset_name.split("-")[0]]
        env = gym.make(env_id, **kwargs)
        # D4RL uses a NormalizedBox which is only used for normalizing actions
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    else:
        # Fall back to loading from d4rl_slim namespace.
        slim_env_name = f"d4rl_slim/{dataset_name}"
        env = gym.make(slim_env_name, **kwargs)

    return env


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.reset(seed=seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def load_env(dataset_name):
    env = get_environment(dataset_name, version = 'v2')
    dataset = d4rl.get_dataset(dataset_name)
    return env, dataset


def get_normalize_score_fn(dataset_name):
    def normalize_score_fn(score):
        return d4rl.get_normalized_score(dataset_name, score)

    return normalize_score_fn


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env
