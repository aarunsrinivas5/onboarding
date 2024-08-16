import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
import itertools

class TwoBodyProblemDiscreteEnv(gym.Env):
    
    def __init__(self, G, M, m, initial_obs, dt, dv, low, high, target_altitude, v_max, max_steps=1000):
        self.G = G
        self.M = M
        self.m = m
        self.initial_obs = initial_obs
        self.dt = dt
        self.dv = dv
        self.low = low
        self.high = high
        self.obs = initial_obs
        self.last_obs = initial_obs
        
        self.target_altitude = target_altitude
        self.target_v_orbital = np.sqrt(G * M / target_altitude)
        
        self.max_steps = max_steps
        self.step_count = 0
        
        self.v_max = v_max
        self.scaler = MinMaxScaler().fit(np.array([
            [-high, -high, -high, -v_max, -v_max, -v_max],
            [high, high, high, v_max, v_max, v_max]
        ]))
        
        self.obs_norm = self.scaler.transform(self.obs.reshape(1, -1))[0]
        self.last_obs_norm = self.scaler.transform(self.last_obs.reshape(1, -1))[0]
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        )
        
        directions = list(itertools.product([0, 1], repeat=3)) + list(itertools.product([0, 1], repeat=3))[1:]
        self._action_to_acceleration = {i: np.array(v) if i < 1 + len(directions) // 2 else -np.array(v) for i, v in enumerate(directions)}
        
        self.action_space = gym.spaces.Discrete(len(self._action_to_acceleration))
        
    def reset(self, seed=0):
        self.obs = self.initial_obs
        self.last_obs = self.initial_obs
        self.obs_norm = self.scaler.transform(self.obs.reshape(1, -1))[0]
        self.last_obs_norm = self.scaler.transform(self.last_obs.reshape(1, -1))[0]
        self.step_count = 0

        return self.obs_norm, {'pos': self.obs}
    
    def _internal_dynamics(self):
        r = np.sqrt(np.sum(self.obs[:3] ** 2))
        F = self.G * self.M * self.m / r ** 2
        a = - F * (self.obs[:3] / r) / self.m
        return a
    
    def _compute_next_obs(self, action):
        a = self._internal_dynamics()
        a += self._action_to_acceleration[int(action)] * self.dv
        
        velocities = self.obs[3:] + a * self.dt
        coordinates = self.obs[:3] + velocities * self.dt
        obs = np.concatenate((coordinates, velocities))
        return obs
    
    def step(self, action):
        self.obs = self._compute_next_obs(action)
        
        next_obs = np.expand_dims(self.obs, axis=0)
        next_obs = self.scaler.transform(next_obs.reshape(1, -1))[0]
        self.obs_norm = next_obs
        
        truncated = self.step_count > self.max_steps
        
        if truncated:
            return self.obs_norm, 0, True, True, {'pos': self.obs}
        
        r = np.sqrt(np.sum(self.obs[:3] ** 2))
        v = np.sqrt(np.sum(self.obs[3:] ** 2))
        
        if r < self.low:
            return self.obs_norm, -1000 / self.step_count , True, False, {'pos': self.obs}
        if r > self.high:
            return self.obs_norm, -1000 / self.step_count, True, False, {'pos': self.obs}
        if v > self.v_max:
            return self.obs_norm, -1000 / self.step_count, True, False, {'pos': self.obs}
        
        orbital_penalty = - abs(v - self.v_orbit) / self.v_orbit
        altitude_penalty = - abs(r - self.target_altitude) / self.target_altitude
        reward = 0.1 * orbital_penalty + altitude_penalty

        self.last_obs = self.obs
        self.last_obs_norm = self.obs_norm
        self.step_count += 1
        
        return self.obs_norm, reward, False, truncated, {'pos': self.obs}