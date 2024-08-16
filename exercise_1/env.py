import numpy as np
import matplotlib.pyplot as plt

class TwoBodyEnv:
    
    def __init__(self, G, M, m, initial_obs, dt):
        self.G = G
        self.M = M
        self.m = m
        self.initial_obs = initial_obs
        self.dt = dt
        self.obs = initial_obs
        
    def reset(self):
        self.obs = self.initial_obs
        return self.obs
    
    def _internal_dynamics(self):
        r = np.sqrt(np.sum(self.obs[:3] ** 2))
        F = self.G * self.M * self.m / r ** 2
        a = - F * (self.obs[:3] / r) / self.m
        return a
    
    def step(self):
        a = self._internal_dynamics()
        velocities = self.obs[3:] + a * self.dt
        coordinates = self.obs[:3] + velocities * self.dt
        self.obs = np.concatenate((coordinates, velocities))
        return self.obs