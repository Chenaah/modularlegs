from collections import defaultdict
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from tqdm.rich import tqdm
from rich.progress import track
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, ProgressBarCallback, BaseCallback
from modular_legs.utils.files import generate_unique_filename, get_cfg_name, get_cfg_path, get_curriculum_cfg_paths, update_cfg, load_cfg, get_latest_model


class EpisodicRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodicRewardCallback, self).__init__(verbose)
        # Store episode rewards here for aggregation.
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # 'infos' is provided by vectorized environments as a list, one per sub-environment.
        infos = self.locals.get("infos", [])
        for info in infos:
            # The Monitor wrapper adds an "episode" key when an episode is done.
            if "episode" in info:
                # Append the episode reward to our list.
                self.episode_rewards.append(info["episode"]["r"])
                # pdb.set_trace()
            # else:
            #     print("No episode info found in info")
            # print("info: ", info)

        # If at least one episode finished during this step, log the mean reward.
        if self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards)
            # Record the mean reward using the logger, e.g. for TensorBoard.
            self.logger.record("custom/ep_rew_mean", mean_reward)

        # Returning True lets training continue.
        return True

    def _on_rollout_end(self) -> None:
        # Optionally, clear the stored rewards after each rollout to avoid accumulating indefinitely.
        self.episode_rewards = []


class ProgressBarCallbackName(ProgressBarCallback):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps, desc=f"[deep_pink1]{self.name}")



def multiplex_obs(obs, mux_type, n_histories=3):
    obs_hist = obs.reshape(1,n_histories,-1)
    if mux_type == "4+1":
        # shape: (1,135)
        # 135: 5*9*3

         # (1,3,45)

        obs1_addr = [0,1,2,3,4,5,
                     6,15,24,42, # pos
                     7,16,25,43, # vel
                     8,17,26,44  # action
                     ]

        obs1 = obs_hist[:,:,obs1_addr].reshape(1,-1)

        obs2_addr = [27,28,29,30,31,32,33,34,35]

        obs2 = obs_hist[:,:,obs2_addr].reshape(1,-1)

        return obs1, obs2


    if mux_type == "5":
        obs_addr = [0,1,2,3,4,5,
                     6,15,24,33,42, # pos
                     7,16,25,34,43, # vel
                     8,17,26,35,44  # action
                     ]
        obs = obs_hist[:,:,obs_addr].reshape(1,-1)
        return obs
    
    if mux_type == "3+1+1":
        obs1_addr = [0,1,2,3,4,5,
                     6,15,24, # pos
                     7,16,25, # vel
                     8,17,26  # action
                     ]

        obs1 = obs_hist[:,:,obs1_addr].reshape(1,-1)

        obs2_addr = [27,28,29,30,31,32,33,34,35]

        obs2 = obs_hist[:,:,obs2_addr].reshape(1,-1)

        obs3_addr = [36,37,38,39,40,41,42,43,44]

        obs3 = obs_hist[:,:,obs3_addr].reshape(1,-1)

        return obs1, obs2, obs3
    

def load_model(load_run, env, alg, device="auto", info=None, alg_kwargs={}):
    if load_run is None:
        model = alg("MlpPolicy", env, verbose=1, **alg_kwargs)
    elif load_run.endswith(".zip"):
        model = alg.load(load_run, env=env, device=device, **alg_kwargs)
    elif load_run.endswith(".d3"):
        raise ValueError("D3 model not supported")
    elif load_run == "RANDOM":
        raise ValueError("RANDOM model not supported")
    elif load_run == "CPG":
        raise ValueError("CPG model not supported")
    else:
        raise ValueError("Invalid model file: ", load_run)
    return model


def save_rollout(rollout, save_dir):
    # Convert (T,B,D) to (T*B, D)
    # The observation returned for the i-th environment when done[i] is true will in fact be the first observation of the next episode
    print("Processing rollout...")
    batch_size = np.shape(rollout["observations"])[1]
    rollout_reshaped = defaultdict(list)
    for b in track(range(batch_size), description="Processing rollout..."):

        # for the b-th environment
        obs = np.array(rollout["observations"])[:,b,:].tolist()
        act = np.array(rollout["actions"])[:,b,:].tolist()
        rew = np.array(rollout["rewards"])[:,b].tolist()
        done = np.array(rollout["dones"])[:,b].tolist()

        # Set the first done to True (for the previous episode) and keep the last one as False
        if b != 0:
            done[0] = True
        # Remove False values from the end of the list
        while done and not done[-1]:  # Check if the list is not empty and the last element is False
            done.pop()
            obs.pop()
            act.pop()
            rew.pop()
        # also delete the last step
        if done and b != batch_size-1:
            done.pop()
            obs.pop()
            act.pop()
            rew.pop()

        rollout_reshaped["observations"].extend(obs)
        rollout_reshaped["actions"].extend(act)
        rollout_reshaped["rewards"].extend(rew)
        rollout_reshaped["dones"].extend(done)

    np.savez_compressed(generate_unique_filename(os.path.join(save_dir, f"rolloutN.npz")), **rollout_reshaped)
