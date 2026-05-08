import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Nero7",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(100000):
    action = np.random.randn(*env.action_spec[0].shape) * 0
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display


# import mujoco
# import mujoco.viewer

# model = mujoco.MjModel.from_xml_path("/home/cobot/Desktop/Raja/Projects/robosuite/robosuite/models/assets/robots/nero/robot.xml")
# data = mujoco.MjData(model)

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         mujoco.mj_step(model, data)