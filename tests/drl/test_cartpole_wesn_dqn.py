"""CartPole-v1 sanity check for our OWN DQN machinery (drl/dqn.py,
algorithm/wesn.py, drl/replay_buffer.py's ReplayBuffer) -- a standard, well-
understood RL benchmark, completely independent of the real tilt-control
pipeline's top_k_neighbor state/reward design.

CartPole is a single agent -- Dqn is used with num_sectors=1,
its per-sector array interface intact (Wesn is exercised exactly as it is
in production, just with num_features=4/num_actions=2 instead of 25/5).

Run: python tests/drl/test_cartpole_wesn_dqn.py [mlp|wesn]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import gymnasium as gym

from drl.factory import create_policy
from helpers.utils import LiveMetricsPlot, RepoLogging, get_logger

RepoLogging.configure("INFO")
logger = get_logger(__name__)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "wesn"
NUM_EPISODES = 1000
ALGORITHM_SEED = 1
SOLVED_REWARD = 195.0  # classic CartPole "solved" bar, mean over the last 100 episodes
MOVING_AVERAGE_WINDOW = 20

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "cartpole_dqn_sanity", MODEL)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    env = gym.make("CartPole-v1")
    num_features = env.observation_space.shape[0]  # 4: cart pos/vel, pole angle/angular vel
    num_actions = env.action_space.n  # 2: push left/right

    dqn_kwargs = dict(
        model=MODEL,
        hidden_sizes=(64,),
        sequence_length=16,
        wesn=dict(units=512, connectivity=0.5, spectral_radius=0.8, leaky=1.0, win_len=16,
                 mlp_hidden_sizes=[64], mlp_position="early"),
        learning_rate=2.5e-4,
        gamma=0.95,
        batch_size=32,
        replay_capacity=20_000,
        train_steps_per_interval=10,
        target_update_steps=200,
        epsilon_start=1.0,
        epsilon_end=0.005,
        epsilon_decay_rate=0.995,
    )
    logger.info("CartPole sanity check: model=%s num_features=%d num_actions=%d episodes=%d",
               MODEL, num_features, num_actions, NUM_EPISODES)
    policy = create_policy("dqn", 1, num_features, num_actions,
                           dqn_kwargs=dqn_kwargs, algorithm_seed=ALGORITHM_SEED, device="cpu")

    plot = LiveMetricsPlot(OUT_DIR, xlabel="episode", moving_average_window=MOVING_AVERAGE_WINDOW)

    episode_rewards = []
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset(seed=ALGORITHM_SEED + episode)
        obs = np.asarray(obs, dtype=np.float32)[None, :]
        done = False
        total_reward = 0.0
        losses_before = len(policy.step_losses)

        while not done:
            action = policy.act(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(int(action[0]))
            next_obs = np.asarray(next_obs, dtype=np.float32)[None, :]
            # terminal=terminated (not truncated): a time-limit cutoff isn't
            # a real terminal state, so bootstrapping should still happen.
            policy.observe(obs, action, np.array([reward]), next_obs, terminal=terminated)
            obs = next_obs
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        loss = float(np.mean(policy.step_losses[losses_before:])) if len(policy.step_losses) > losses_before else np.nan
        plot.update(episode, np.array([total_reward]), loss)

        if episode % 10 == 0:
            recent_mean = np.mean(episode_rewards[-20:])
            logger.info("episode %3d: reward=%.1f recent20_mean=%.1f", episode, total_reward, recent_mean)

    plot.close()
    env.close()

    final_mean = float(np.mean(episode_rewards[-100:]))
    solved = final_mean >= SOLVED_REWARD
    logger.info("FINAL: mean reward over last 100 episodes=%.1f solved=%s", final_mean, solved)
    print(f"FINAL: model={MODEL} mean_reward_last100={final_mean:.1f} solved={solved}")


if __name__ == "__main__":
    main()
