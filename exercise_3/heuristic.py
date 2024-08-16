import numpy as np

class HeuristicPolicy:
    
    def __init__(self, env):
        self.env = env
        
    def predict(self, obs):
        values = []

        for action in range(self.env.action_space.n):
            next_obs = self.env._compute_next_obs(action)
            r_next = np.sqrt(np.sum(next_obs[:3] ** 2))
            v_next = np.sqrt(np.sum(next_obs[3:] ** 2))
            
            orbital_penalty = abs(v_next - self.env.target_v_orbital) / self.env.target_v_orbital
            altitude_penalty = abs(r_next - self.env.target_altitude) / self.env.target_altitude
            if v_next > self.env.v_max or r_next > self.env.high:
                total_penalty = np.inf
            else:
                total_penalty = orbital_penalty + 100 * altitude_penalty
            values.append(total_penalty)
            
        action = np.argmin(values)
        return action