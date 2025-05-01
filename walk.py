#!/usr/bin/env python3
"""
walk.py

Train a PPO agent on MuJoCo's Humanoid-v2 using Stable-Baselines3.
"""

import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure

def make_env(env_id: str):
    """
    Utility to create a single Gym environment.
    """
    def _init():
        env = gym.make(env_id)
        return env
    return _init

def main():
    # Parameters
    ENV_ID = "Humanoid-v4"
    TOTAL_TIMESTEPS = 5_000_000
    LOG_DIR = "./logs/humanoid_ppo"
    SAVE_DIR = "./models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Create vectorized and monitored environments
    n_envs = 4
    env_fns = [make_env(ENV_ID) for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, LOG_DIR)

    # Configure SB3 logger (TensorBoard)
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])

    # Define a callback to stop training once we reach a reward threshold
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=5000,  # adjust based on desired performance
        verbose=1
    )
    eval_env = gym.make(ENV_ID)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=SAVE_DIR,
        log_path=LOG_DIR,
        eval_freq=50_000,          # evaluate every 50k steps
        n_eval_episodes=5,
        verbose=1
    )

    # Instantiate the agent
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=2048 // n_envs,    # so total batch size remains 2048
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        n_epochs=10,
        gae_lambda=0.95,
        gamma=0.99,
    )
    model.set_logger(new_logger)

    # Train the agent
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name="ppo_humanoid"
    )

    # Save final model
    model_path = os.path.join(SAVE_DIR, "humanoid_ppo_final")
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}.zip")

if __name__ == "__main__":
    main()
