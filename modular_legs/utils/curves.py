import numpy as np


def forward_reward_curve2_1(vel, cmd, cmd_scale):
    # cmd_scale used for the case of cmd=0
    if cmd > 0:
        if vel >= cmd and vel <= cmd*2:
            r = 1
        elif vel <= 0 or vel >= 4*cmd:
            r = 0
        elif vel < cmd and vel > 0:
            r = 1/cmd * vel
        elif vel > 2*cmd and vel < 4*cmd:
            r = -1/(2*cmd) * vel + 2
    elif cmd < 0:
        if vel <= cmd and vel >= cmd*2:
            r = 1
        elif vel >= 0 or vel <= 4*cmd:
            r = 0
        elif vel > cmd and vel < 0:
            r = 1/cmd * vel
        elif vel < 2*cmd and vel > 4*cmd:
            r = -1/(2*cmd) * vel + 2
    elif cmd == 0:
        if vel <= cmd_scale/2 and vel >= -cmd_scale/2:
            r = 1
        elif vel >= cmd_scale*3/2 or vel <= -cmd_scale*3/2:
            r = 0
        elif vel > cmd_scale/2 and vel < cmd_scale*3/2:
            r = (vel-3/2*cmd_scale) / (1-3/2*cmd_scale)
        elif vel < -cmd_scale/2 and vel > -cmd_scale*3/2:
            r = (vel+3/2*cmd_scale) / (-1+3/2*cmd_scale)
    return r


def plateau(vel, max_desired_vel):
    if max_desired_vel > 0:
        if vel > 0 and vel <= max_desired_vel:
            r = vel / max_desired_vel
        elif vel > max_desired_vel:
            r = 1
        else:
            r = 0
    elif max_desired_vel < 0:
        if vel < 0 and vel >= max_desired_vel:
            r = vel / max_desired_vel
        elif vel < max_desired_vel:
            r = 1
        else:
            r = 0
    else:
        r = 0

    return r

def isaac_reward(desired_value, current_value, tracking_sigma = 0.25):
    lin_vel_error = np.sum(np.square(desired_value - current_value))
    lin_vel_reward = np.exp(-lin_vel_error/tracking_sigma)
    return lin_vel_reward