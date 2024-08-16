import os
import numpy as np
import matplotlib.pyplot as plt

from env import TwoBodyProblemDiscreteEnv
from heuristic import HeuristicPolicy

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
max_steps = 2000

v_orbital = np.sqrt(G * M / r)
initial_obs = np.array([r, 0, 0, 0, v_orbital, 0], dtype=np.float32)

env = TwoBodyProblemDiscreteEnv(G, M, m, initial_obs, dt, dv, low, high, target_altitude, v_max, max_steps)
policy = HeuristicPolicy(env)

states = []
done = False
truncated = False

obs, info = env.reset()
state = info['pos']

while not done:
    action = policy.predict(state)
    next_obs, reward, done, truncated, info = env.step(action)
    state = info['pos']
    obs = next_obs
    states.append(state)

states = np.vstack(states)

path = './figures'
os.makedirs(path, exist_ok=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)

x = target_altitude * np.outer(np.cos(phi), np.sin(theta))
y = target_altitude * np.outer(np.sin(phi), np.sin(theta))
z = target_altitude * np.outer(np.ones(100), np.cos(theta))

ax.plot_surface(x, y, z, color='r', alpha=0.1, rstride=5, cstride=5, edgecolor='k')

ax.plot([0], [0], [0], 'ro', markersize=10, label='Fixed Body')
ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', label='Orbiting Body', markersize=10)

ax.set_xlabel('X position (m)')
ax.set_ylabel('Y position (m)')
ax.set_zlabel('Z position (m)')
ax.set_title('Two-Body Problem in 3D')

ax.legend()
ax.set_box_aspect([1,1,1])
ax.grid(True)

plt.savefig(os.path.join(path, '3d_plot.png'))
plt.show()

