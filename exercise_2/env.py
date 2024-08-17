import torch
import numpy as np


class TwoBodyEnv():
    
    def __init__(self, initial_obs, G=6.67430e-11, M=5.972e24, m=7.348e22, dt=10):
        self.G = G
        self.M = M
        self.m = m
        self.initial_obs = initial_obs
        self.dt = dt
        self.obs = initial_obs
        self.last_obs = initial_obs
        self.total_angle = 0
        self.count = 0
        
    def reset(self):
        self.obs = self.initial_obs
        self.last_obs = self.initial_obs
        self.total_angle = 0
        self.count = 0
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
    
    def _get_obs(self):
        a = self._internal_dynamics()
        velocities = self.obs[3:] + a * self.dt
        coordinates = self.obs[:3] + velocities * self.dt
        obs = np.concatenate((coordinates, velocities))
        return obs
    
    def step(self):
        self.obs = self._get_obs()
        self.total_angle += self._angle_between_vectors(self.obs, self.last_obs)
        self.last_obs = self.obs
        self.count += 1
        return self.obs, self.total_angle >= 360
    

class ApproxTwoBodyEnv(TwoBodyEnv):

    def __init__(self, initial_obs, dt, model, scaler, t_max):
        super().__init__(initial_obs, dt)
        self.model = model
        self.scaler = scaler
        self.t_max = t_max

    def _get_obs(self):
        obs = self.scaler.transform(self.initial_obs.reshape(1, -1))
        obs = np.concatenate((obs, np.array([[self.count * self.dt / self.t_max]])), axis=1)
        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():   
            obs = self.model(obs).numpy()
        obs = self.scaler.inverse_transform(obs)[0]
        return obs



