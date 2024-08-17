import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from env import TwoBodyEnv, ApproxTwoBodyEnv
from model import DNN

G = 6.67430e-11
M = 5.972e24
m = 7.348e22
r = 38440000

v_orbital = np.sqrt(G * M / r)
initial_obs = np.array([r, 0, 0, 0, v_orbital, 0])
dt = 10

env = TwoBodyEnv(initial_obs, G=G, M=M, m=m, dt=dt)

outputs = [initial_obs]

done = False
state = env.reset()
while not done:
    state, done = env.step()
    outputs.append(state)
outputs = np.vstack(outputs)

inputs = [initial_obs] * len(outputs)
inputs = np.vstack(inputs)
t = (np.arange(len(outputs)) * dt).reshape(-1, 1)

scaler = MinMaxScaler()
outputs = scaler.fit_transform(outputs)
inputs = scaler.transform(inputs)
t_max = max(t).item()
t = t / t_max
inputs = np.concatenate((inputs, t), axis=1)

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, shuffle=True)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

model = DNN(7, 32, 16, 6)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []
for epoch in tqdm(range(10000)):
    optimizer.zero_grad()

    preds = model(x_train) 
    loss = criterion(preds, y_train)
    loss.backward()

    optimizer.step()

    loss = loss.item()
    train_losses.append(loss)

    with torch.no_grad():
        preds = model(x_test)
        loss = criterion(preds, y_test)
        loss = loss.item()
        val_losses.append(loss)

with torch.no_grad():
    preds = model(x_test)
    loss = criterion(preds, y_test)
    loss = loss.item()

print(f"mae: {loss}")

approx_env = ApproxTwoBodyEnv(initial_obs, dt, model, scaler, t_max)
states = []
done = False
state = approx_env.reset()
while not done:
    state, done = approx_env.step()
    states.append(state)
states = np.vstack(states)

path = './figures'
os.makedirs(path, exist_ok=True)

plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig(os.path.join(path, 'train_loss.png'))
plt.clf()

plt.plot(val_losses, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.savefig(os.path.join(path, 'val_loss.png'))
plt.clf()

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
ax.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', label='Orbiting Body')

ax.set_zlim(-10, 10)

ax.set_xlabel('X position (m)')
ax.set_ylabel('Y position (m)')
ax.set_zlabel('Z position (m)')
ax.set_title('Two-Body Problem in 3D')

ax.legend()
ax.set_box_aspect([1,1,1])
ax.grid(True)

plt.savefig(os.path.join(path, '3d_plot.png'))
plt.clf()









