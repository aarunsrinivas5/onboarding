import os
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from env import TwoBodyEnv
from model import DNN

G = 6.67430e-11
M = 5.972e24
m = 7.348e22
r = 38440000

v_orbital = np.sqrt(G * M / r)
initial_obs = np.array([r, 0, 0, 0, v_orbital, 0])
dt = 10

env = TwoBodyEnv(G, M, m, initial_obs, dt)

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
t = t / max(t)
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










