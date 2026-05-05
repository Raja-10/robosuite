import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robosuite.controllers import load_controller_config


controller_config = load_controller_config(default_controller="OSC_POSE")


class RobosuiteGymWrapper(gym.Env):
    def __init__(self, render=True):
        super().__init__()

        self.env = suite.make(
            env_name="Lift",
            robots="UR5e",
            controller_configs=controller_config,
            has_renderer=render,
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
        return self._flatten_obs(obs), reward, done, False, info


if __name__ == "__main__":

    base_env = DummyVecEnv([lambda: RobosuiteGymWrapper(render=True)])

    env = VecNormalize.load("vecnormalize.pkl", base_env)
    env.training = False
    env.norm_reward = False

    model = SAC.load("sac_lift_parallel", env=env)

    print("✅ Model loaded!")

    obs = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done[0]:
            obs = env.reset()