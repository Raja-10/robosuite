import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(10000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.2
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display