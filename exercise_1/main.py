import os
import numpy as np
import matplotlib.pyplot as plt

from env import TwoBodyEnv

# generate orbit

G = 6.67430e-11
M = 5.972e24
m = 1000
r = 1900000
dt = 10

v_orbital = np.sqrt(G * M / r)
initial_obs = np.array([r, 0, 0, 0, v_orbital, 0])

states = [initial_obs]
env = TwoBodyEnv(G, M, m, initial_obs, dt)

done = False
env.reset()
while not done:
    state, done = env.step()
    states.append(state)
states = np.vstack(states)

# plot orbit

path = './figures'
os.makedirs(path, exist_ok=True)

coordinate_names = [
    'X Coordinate',
    'Y Coordinate',
    'Z Coordinate',
    'Velocity-X Coordinate',
    'Velocity-Y Coordinate',
    'Velocity-Z Coordinate'
]
time = np.arange(len(states)) * dt

for idx, coordinate_name in enumerate(coordinate_names):
    coordinate = states[:, idx]
    plt.plot(time, coordinate)

    plt.xlabel('Time (s)')
    plt.ylabel(coordinate_name)
    plt.title(f'{coordinate_name} Over Time')

    filename = coordinate_name.lower().replace('-', '_').split()[0]
    plt.savefig(os.path.join(path, f'{filename}_2d_plot.png'))
    plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
