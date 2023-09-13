import os
import uuid
from pathlib import Path

import torch
import wandb
import yaml
from absl import app
from absl import flags
from ml_collections import config_flags

from d4rl_slim_benchmark.algorithms import get_algo_factory
from d4rl_slim_benchmark.buffer import ReplayBuffer
from d4rl_slim_benchmark.utils import compute_mean_std
from d4rl_slim_benchmark.utils import modify_reward
from d4rl_slim_benchmark.utils import normalize_states
from d4rl_slim_benchmark.utils import wandb_init

FLAGS = flags.FLAGS

flags.DEFINE_string("algo", None, "Algorithm to use")
config_flags.DEFINE_config_file("config", None, "File path to the configuration file.")


def train(_):
    config = FLAGS.config
    algo_factory = get_algo_factory(FLAGS.algo)

    # Manually generate the name
    config.name = f"{config.name}-{config.env}-{str(uuid.uuid4())[:8]}"
    if config.checkpoints_path is not None:
        config.checkpoints_path = os.path.join(config.checkpoints_path, config.name)

    # Configure d4rl (gym) or d4rl-slim (gymnasium)
    if config.use_d4rl_slim:
        from d4rl_slim_benchmark.d4rl_slim_utils import eval_actor
        from d4rl_slim_benchmark.d4rl_slim_utils import get_normalize_score_fn
        from d4rl_slim_benchmark.d4rl_slim_utils import load_env
        from d4rl_slim_benchmark.d4rl_slim_utils import set_seed
        from d4rl_slim_benchmark.d4rl_slim_utils import wrap_env
    else:
        from d4rl_slim_benchmark.d4rl_utils import eval_actor
        from d4rl_slim_benchmark.d4rl_utils import get_normalize_score_fn
        from d4rl_slim_benchmark.d4rl_utils import load_env
        from d4rl_slim_benchmark.d4rl_utils import set_seed
        from d4rl_slim_benchmark.d4rl_utils import wrap_env

    env, dataset = load_env(config.env)
    normalize_score_fn = get_normalize_score_fn(config.env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    trainer = algo_factory(config, env)
    actor = trainer.actor

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(config.to_dict())

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = normalize_score_fn(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score}, step=trainer.total_it
            )


if __name__ == "__main__":
    app.run(train)
