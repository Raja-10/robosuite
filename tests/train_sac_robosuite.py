import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config
)
from stable_baselines3.common.vec_env import VecNormalize

# =========================
# ENV WRAPPER
# =========================

controller_config = load_composite_controller_config(
    controller="BASIC",
    robot="UR5e"
)

class RobosuiteGymWrapper(gym.Env):
    def __init__(self):
        super().__init__()

        self.env = suite.make(
            env_name="Lift",
            robots="UR5e",
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=10,
        )

        self.obs_keys = [
            "robot0_proprio-state",
            "object-state",
        ]

        obs = self.env.reset()
        flat_obs = self._flatten_obs(obs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_dim,),
            dtype=np.float32,
        )

    def _flatten_obs(self, obs_dict):
        return np.concatenate(
            [obs_dict[k] for k in self.obs_keys]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return self._flatten_obs(obs), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        terminated = done
        truncated = False

        return self._flatten_obs(obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()


# =========================
# ENV FACTORY
# =========================
def make_env(rank):
    def _init():
        return RobosuiteGymWrapper()
    return _init


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    n_envs = 16   # Start with 4–8; increase if CPU allows

    # Parallel training env
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    env = VecNormalize(
        SubprocVecEnv([make_env(i) for i in range(n_envs)]),
        norm_obs=True,
        norm_reward=False,
    )

    # Single eval env
    eval_env = DummyVecEnv([make_env(0)])

    eval_env = VecNormalize(
        DummyVecEnv([make_env(0)]),
        norm_obs=True,
        norm_reward=False,
        training=False,
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=10000,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        learning_starts=1000,
        device="cuda",
    )

    logger = configure("./logs", ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=100000,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save("sac_lift")
    env.save("vecnormalize.pkl")

    print("✅ Training finished!")

    env.close()
    eval_env.close()