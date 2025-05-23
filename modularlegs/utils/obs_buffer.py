import pdb
import numpy as np


class ObservationBuffer():
    def __init__(self, num_obs, include_history_steps):

        self.num_obs = num_obs
        self.include_history_steps = include_history_steps

        self.num_obs_total = num_obs * include_history_steps

        self.obs_buf = np.zeros(self.num_obs_total)

    def reset(self, new_obs):
        self.obs_buf = np.tile(new_obs, self.include_history_steps)

    def insert(self, new_obs):
        # Shift observations back.
        self.obs_buf[: self.num_obs * (self.include_history_steps - 1)] = self.obs_buf[self.num_obs : self.num_obs * self.include_history_steps]

        # Add new observation.
        self.obs_buf[-self.num_obs:] = new_obs

    def get_obs_vec(self, obs_ids=None):
        """Gets history of observations indexed by obs_ids.
        
        Arguments:
            obs_ids: An array of integers with which to index the desired
                observations, where 0 is the latest observation and
                include_history_steps - 1 is the oldest observation.
        """
        if obs_ids is None:
            obs_ids = np.arange(self.include_history_steps)
        obs = []
        for obs_id in reversed(sorted(obs_ids)):
            slice_idx = self.include_history_steps - obs_id - 1
            obs.append(self.obs_buf[slice_idx * self.num_obs : (slice_idx + 1) * self.num_obs])
        return np.concatenate(obs)
    

if __name__ == "__main__":
    obs_buf = ObservationBuffer(3, 5)
    obs_buf.reset(np.array([1,2,3]))
    obs_buf.insert(np.array([0,0,0]))
    obs_buf.insert(np.array([1,1,1]))
    print(obs_buf.get_obs_vec())