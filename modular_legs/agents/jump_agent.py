


import numpy as np


class Jumper():
    # def __init__(self):

    def get_test_action(self, obs):
        projected_z = obs[5] # TODO
        if projected_z < 0:
            target = -1.5 # TODO
        else:
            target = 1.5
        action = np.array([target])
        return action