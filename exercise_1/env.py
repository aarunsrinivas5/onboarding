import numpy as np


class TwoBodyEnv():
    
    def __init__(self, G, M, m, initial_obs, dt):
        self.G = G
        self.M = M
        self.m = m
        self.initial_obs = initial_obs
        self.dt = dt
        self.obs = initial_obs
        self.last_obs = initial_obs
        self.total_angle = 0
        
    def reset(self):
        self.obs = self.initial_obs
        self.last_obs = self.initial_obs
        self.total_angle = 0
        return self.obs
    
    def _angle_between_vectors(self, a, b):
        a = np.array(a)
        b = np.array(b)
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle numerical errors
        return np.degrees(angle_radians)
    
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
        self.total_angle += self._angle_between_vectors(self.obs, self.last_obs)
        self.last_obs = self.obs
        return self.obs, self.total_angle >= 360