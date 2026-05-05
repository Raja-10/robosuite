import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


# =========================
# ENV WRAPPER
# =========================
class RobosuiteGymWrapper(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        self.env = suite.make(
            env_name="Lift",
            robots="UR5e",
            has_renderer=render,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
        )

        obs = self.env.reset()
        self.obs_key = "robot0_proprio-state"

        obs_dim = obs[self.obs_key].shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.env.action_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return obs[self.obs_key], {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs[self.obs_key], reward, done, False, info


# =========================
# ENV FACTORIES
# =========================
def make_env():
    return RobosuiteGymWrapper(render=False)

def make_test_env():
    return RobosuiteGymWrapper(render=True)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # -------------------------
    # TRAINING ENV
    # -------------------------
    num_envs = 6
    train_env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # -------------------------
    # LOAD EXISTING MODEL
    # -------------------------
    model = SAC.load("sac_lift", env=train_env, device="cuda")
    print("✅ Loaded existing model")

    # -------------------------
    # OPTIONAL: LOAD REPLAY BUFFER
    # -------------------------
    try:
        model.load_replay_buffer("replay_buffer.pkl")
        print("✅ Loaded replay buffer")
    except:
        print("⚠ No replay buffer found, continuing without it")

    # -------------------------
    # CONTINUE TRAINING
    # -------------------------
    additional_timesteps = 300000

    model.learn(
        total_timesteps=additional_timesteps,
        reset_num_timesteps=False   # 🔥 important (continue count)
    )

    # -------------------------
    # SAVE AGAIN
    # -------------------------
    model.save("sac_lift_v2")
    model.save_replay_buffer("replay_buffer.pkl")
    print("✅ Model saved after continued training")

    # -------------------------
    # TEST ENV (SINGLE + RENDER)
    # -------------------------
    test_env = DummyVecEnv([make_test_env])

    obs = test_env.reset()
    print("🎮 Running trained policy...")

    success = 0
    episodes = 10

    for ep in range(episodes):
        obs = test_env.reset()

        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)

            if info[0].get("success", False):
                success += 1
                print(f"Episode {ep}: SUCCESS")
                break

            if done:
                break

    print(f"🏆 Success Rate: {success}/{episodes}")