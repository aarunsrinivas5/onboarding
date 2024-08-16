import numpy as np

from env import TwoBodyEnv

G = 6.67430e-11
M = 5.972e24
m = 1000
r = 1900000
dt = 10

v_orbital = np.sqrt(G * M / r)
initial_obs = np.array([r, 0, 0, 0, v_orbital, 0])

states = [initial_obs]
env = TwoBodyEnv(G, M, m, initial_obs, dt)



for i in range(10000):
    done = False
    env.reset()
    while not done:
        state, done = env.step()
        states.append(state)
    states = np.vstack(states)

