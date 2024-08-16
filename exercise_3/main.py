import numpy as np

from env import TwoBodyProblemDiscreteEnv

G = 6.67430e-11
M = 5.972e24
m = 1000
r = 1.9e6
dt = 100
dv = 10
low = 1.5e5
high = 5e8
target_altitude = 3.8e7
v_max = 100000
max_steps = 1000

v_orbital = np.sqrt(G * M / r)
initial_obs = np.array([r, 0, 0, 0, v_orbital, 0], dtype=np.float32)

env = TwoBodyProblemDiscreteEnv(G, M, m, initial_obs, dt, dv, low, high, target_altitude, v_max, max_steps)

