
import csv
import pickle
import datetime
import itertools
from multiprocessing import Process
import os
import pdb
import re
from pytorch_lightning.callbacks import Callback
import wandb
import numpy as np
import torch
from modular_legs import LEG_ROOT_DIR
from rich import print, get_console
from rich.console import Console

from modular_legs.sim.evolution.encoding_wrapper import decode_onehot, polish_asym, to_onehot
from modular_legs.utils.others import is_list_like
from modular_legs.utils.visualization import take_photo


def _save_data(log_dir, saved_data, file_name="data"):
    np.savez_compressed((os.path.join(log_dir, f'{file_name}.npz')), **saved_data)

class Logger():
    def __init__(self, alg="RL", log_dir=None, note="", data_file_name="data"):
        
        if log_dir is None:
            fileid = int(datetime.datetime.today().strftime('%m%d')+"00")
            log_dir = os.path.join(LEG_ROOT_DIR, "logs", f"{alg}_0{fileid}{note}")
            while os.path.exists(log_dir):
                fileid += 1
                log_dir = os.path.join(LEG_ROOT_DIR, "logs", f"{alg}_0{fileid}{note}")
    
        # assert not os.path.exists(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        self.data_file_name = data_file_name
        self.log_dir = log_dir
        self.key_list = []

        self.tag_colors = {"[ERROR]": "bright_red",
                           "[WARN]": "bright_yellow",
                           "[Motor]": "dark_goldenrod",
                           "[Client Transceiver]": "dark_orange3",
                           "[Client Receiver]": "dark_orange3",
                           "[IMU]": "khaki3",
                           "[Server]": "dark_violet",
                           "[Debugger]": "wheat4",
                           "else": "green3"}
        
        self.i_log = 0

        self.save_interval = 100
        self.console = Console(highlight=False)

    def log_data(self, info: dict):
        for k in info.keys():
            if not hasattr(self, k):
                setattr(self, k, [])
            if not k in self.key_list:
                self.key_list.append(k)

            getattr(self, k).append(info[k])
        self.i_log += 1

        if (self.i_log % self.save_interval) == 0:
            saved_data = {}
            for k in self.key_list:
                saved_data[k] = np.array(getattr(self, k))
            save_process = Process(target=_save_data, args=(self.log_dir, saved_data, self.data_file_name))
            save_process.start()



    def _color_tag(self, msg):
        new_msg = msg
        for k, v in self.tag_colors.items():
            new_msg = new_msg.replace(k, f"[{v}]{k}[/{v}]")
        if new_msg == msg:
            pattern = r"\[([^]]*)\]"
            matches = re.findall(pattern, msg)
            # Extracted text within square brackets will be stored in matches
            if matches:
                k = f"[{matches[0]}]"
                v = self.tag_colors["else"]
                new_msg = new_msg.replace(k, f"[{v}]{k}[/{v}]")

        return new_msg

    def log_text(self, msg):
        formatted_now = datetime.datetime.now().strftime("%H:%M:%S")
        printed_msg = self._color_tag(msg)
        get_console().print(f"[bright_magenta]{formatted_now}[/bright_magenta] {printed_msg}", highlight=False)
        msg = formatted_now + " " + msg
        filename = os.path.join(self.log_dir, 'log.txt')
        # Check if the file exists
        if os.path.isfile(filename):
            # Open the file in append mode
            with open(filename, 'a') as file:
                # Write content to a new line
                file.write('\n'+msg)
        else:
            # Open the file in write mode (create a new file)
            with open(filename, 'w') as file:
                # Write content without adding a newline
                file.write(msg)



class LoggerCallback(Callback):

    def on_validation_epoch_end(self, trainer, pl_module):
        # Visualize latent space
        # self.visualize_latent_space(pl_module, 20) # Buggy
        pass

    def visualize_latent_space(self, model, nrow: int) -> torch.Tensor:

        # Currently only support 2D manifold visualization
        assert model.latent_dim >= 2

        # Create latent manifold
        unit_line = np.linspace(-4, 4, nrow)
        latent_grid = list(itertools.product(unit_line, repeat=2))
        latent_grid = np.array(latent_grid, dtype=np.float64)

        # Sample robots from the latent space
        num_sampled = 6
        r1, r2 = -4, 4
        sampled_latent = (r1 - r2) * torch.rand(num_sampled, model.latent_dim, device=model.device, dtype=model.dtype) + r2
        one_hot = model.decoder(sampled_latent)
        design = decode_onehot(one_hot, max_idx=model.max_idx)
        sampled_designs = [polish_asym(d) for d in design]
        sampled_imgs_l = [take_photo(pipeline=pipe) for pipe in sampled_designs if pipe]

        # Sample robots from the original dataset
        sampled_original_designs = model.sampled_original_data[:1]
        sampled_imgs_o_before = [take_photo(pipeline=pipe) for pipe in sampled_original_designs]
        onehoted_samples = to_onehot(sampled_original_designs, model.max_idx, model.max_length).to(device=model.device, dtype=model.dtype)
        mu, _ = model.encode_to_params(onehoted_samples)
        # robot_code_decoded  = model.decode_deterministic(mu).detach().cpu().numpy()[0]
        one_hot = model.decoder(mu)
        design = decode_onehot(one_hot, max_idx=model.max_idx)
        sampled_designs = [polish_asym(d) for d in design]
        sampled_imgs_o_after = [take_photo(pipeline=pipe) for pipe in sampled_designs]

        # print(sampled_imgs_l)
        # print(sampled_imgs_o_before)
        # print(sampled_imgs_o_after)


        if wandb.run is not None:
            wandb.log({
                        "Sampled Robots (z)": [wandb.Image(sampled_imgs_l[i], caption=f"sampled robots {i}") for i in range(len(sampled_imgs_l))],
                        "Sampled Robots (x)": [wandb.Image(sampled_imgs_o_before[0], caption="before VAE"), 
                                               wandb.Image(sampled_imgs_o_after[0], caption="after VAE")]
                        })
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "rec/train" in trainer.logged_metrics and wandb.run is not None:
            wandb.log({"rec/train": trainer.logged_metrics["rec/train"],
                       "kl/train": trainer.logged_metrics["kl/train"],
                       "loss/train": trainer.logged_metrics["loss/train"]})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "rec/val" in trainer.logged_metrics and wandb.run is not None:
            wandb.log({"rec/val": trainer.logged_metrics["rec/val"],
                       "kl/val": trainer.logged_metrics["kl/val"],
                       "loss/val": trainer.logged_metrics["loss/val"]})




def load_cached_pings(recent_threshold=10):
    file_path = os.path.join(LEG_ROOT_DIR, "exp", "cached_data", "cached_pings.pickle")

    # Define the recent threshold (e.g., files modified in the last 1 hour)
    recent_threshold = datetime.datetime.now() - datetime.timedelta(minutes=recent_threshold)

    # Check if the file was modified recently
    if os.path.exists(file_path):
        file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        if file_mod_time > recent_threshold:
            # Load the pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print("Pings file loaded successfully:", data)
        else:
            print("File is not modified recently.")
            data = {}
    else:
        print("File does not exist.")
        data = {}

    return data


def cache_pings(data):
    file_path = os.path.join(LEG_ROOT_DIR, "exp", "cached_data", "cached_pings.pickle")

    # Save the data to the pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print("Pings file saved successfully.")


def get_running_header():
    import ntplib
    import platform
    import socket
    import json

    def get_current_time():
        try:
            client = ntplib.NTPClient()
            response = client.request('pool.ntp.org')
            return f"[NTP]{datetime.datetime.fromtimestamp(response.tx_time)}"
        except:
            return f"[SYS]{datetime.datetime.now()}"  # Fallback to system time

    def get_cuda_info():
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                devices.append({
                    "Device Name": torch.cuda.get_device_name(i),
                    "Capability": torch.cuda.get_device_capability(i),
                    "Total Memory (MB)": torch.cuda.get_device_properties(i).total_memory // (1024 ** 2),
                })
            return {
                "CUDA Available": True,
                "CUDA Version": torch.version.cuda,
                "Devices": devices,
            }
        else:
            return {"CUDA Available": False}

    def get_system_info():
        system_info = {
            "OS": platform.system(),
            "OS Version": platform.version(),
            "Processor": platform.processor(),
            "Machine": platform.machine(),
            "Hostname": socket.gethostname(),  # Added hostname
        }
        return system_info

    # Collect all information
    experiment_info = {
        "Current Time": get_current_time(),
        "System Info": get_system_info(),
        "CUDA Info": get_cuda_info(),
    }

    # Print the collected information
    info = json.dumps(experiment_info)

    return info


def plot_learning_curve(csv_files, save_file=None, legends=None, max_length=-1):
    if not is_list_like(csv_files):
        csv_files = [csv_files]
    if not is_list_like(csv_files[0]):
        csv_files = [[i] for i in csv_files]
    
    flatten_csv_files = [item for sublist in csv_files for item in sublist]

    yss = []
    tss = []
    for csv_file in flatten_csv_files:
        ep_rew_mean_values = []
        ts = []
        with open(csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                ep_rew_mean_values.append(float(row["rollout/ep_rew_mean"]))
                ts.append(int(row["time/total_timesteps"]))

        if max_length > 0:
            ep_rew_mean_values = ep_rew_mean_values[:max_length]
            ts = ts[:max_length]
        yss.append(ep_rew_mean_values)
        tss.append(ts)

    # Reshape back
    lengths = [len(sublist) for sublist in csv_files]
    reshaped_yss = []
    start = 0
    for length in lengths:
        reshaped_yss.append(yss[start:start + length])
        start += length
    reshaped_tss = []
    start = 0
    for length in lengths:
        reshaped_tss.append(tss[start:start + length])
        start += length

    yss = np.mean(np.array(reshaped_yss), axis=1)  # Average across runs
    stdss = np.std(np.array(reshaped_yss), axis=1)  # Standard deviation
    tss = np.array(reshaped_tss)[:,0] 

    import matplotlib.pyplot as plt
    import seaborn as sns
    # sns.set(style="darkgrid")
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 6))  
    i_plot = 0
    for ys, ts, stds, label in zip(yss, tss, stdss, legends if legends is not None else [f"Run {i+1}" for i in range(len(yss))]):
        linestyle = '--' if i_plot > 10 else '-'
        plt.plot(ts, ys, label=label, linestyle=linestyle)
        plt.fill_between(ts, ys - stds, ys + stds, 
                    alpha=0.2)
        i_plot += 1

    # plt.plot(ts, ep_rew_mean_values)
    plt.xlabel('Total Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.title('Learning Curve')
    plt.legend()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, dpi=400)

#%%
if __name__ == "__main__":
    csv_file = "exp/sim_sbx/m3air1s-rp3-cmmg-0319022910/progress.csv"
    plot_learning_curve(csv_file)
#%%
