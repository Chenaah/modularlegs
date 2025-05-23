import os
import subprocess
import time

from modular_legs import LEG_ROOT_DIR


# Function to check if a screen session exists
def session_exists(session_name):
    result = subprocess.run(['screen', '-list'], stdout=subprocess.PIPE, text=True)
    return f".{session_name}\t" in result.stdout

# Function to terminate a screen session
def terminate_session(session_name):
    subprocess.run(['screen', '-S', session_name, '-X', 'quit'])
    time.sleep(5)



# def session_exists(session_name):
#     """
#     Check if a tmux session with the given name exists.
#     :param session_name: Name of the tmux session to check.
#     :return: True if the session exists, False otherwise.
#     """
#     result = subprocess.run(['tmux', 'list-sessions'], stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL)
#     return any(session_name in line.split(":")[0] for line in result.stdout.splitlines())


# def terminate_session(session_name):
#     """
#     Terminate a tmux session with the given name.
#     :param session_name: Name of the tmux session to terminate.
#     """
#     subprocess.run(['tmux', 'kill-session', '-t', session_name], stderr=subprocess.DEVNULL)
#     time.sleep(5)  # Optional: Wait to ensure the session is terminated

# Function to wait until a session is terminated
def wait_until_terminated(session_name, timeout=10):
    start_time = time.time()
    while session_exists(session_name):
        if time.time() - start_time > timeout:
            print(f"Warning: Timeout waiting for session {session_name} to terminate")
            break
        time.sleep(0.5)
        
def start_servers(n_servers=18):

    # Loop through each pair and run the command in a new screen session
    for i in range(n_servers):

        port = f"{5555 + i}"
        cuda = f"{i % 3 + 1}"

        screen_name = f"zmq{port}"
        
        # Terminate the existing session if it exists
        if session_exists(screen_name):
            print(f"Terminating existing session: {screen_name}")
            terminate_session(screen_name)
            wait_until_terminated(screen_name, timeout=100)
        
        print(f"Running in screen session: {screen_name}")
        command = f"screen -dmS {screen_name} zsh -c 'source $HOME/anaconda3/etc/profile.d/conda.sh && conda activate jax && CUDA_VISIBLE_DEVICES={cuda} python {os.path.join(LEG_ROOT_DIR, 'modular_legs/sim/train_server.py')} {port}; exec zsh'"
        subprocess.run(command, shell=True)

def start_server(port, cuda, conda="jax"):

    # cuda = ""
    
    screen_name = f"zmq{port}"
    
    # Terminate the existing session if it exists
    if session_exists(screen_name):
        print(f"Terminating existing session: {screen_name}")
        terminate_session(screen_name)
        wait_until_terminated(screen_name, timeout=100)
    
    print(f"Running in screen session: {screen_name}")

    # command = f"screen -dmS {screen_name} 'source $HOME/anaconda3/etc/profile.d/conda.sh && conda activate {conda} && CUDA_VISIBLE_DEVICES={cuda} MUJOCO_GL=egl python {os.path.join(LEG_ROOT_DIR, 'modular_legs/sim/train_server.py')} {port}; exec zsh'"
    command = f"screen -dmS {screen_name} zsh -c 'source $HOME/anaconda3/etc/profile.d/conda.sh && conda activate {conda} && CUDA_VISIBLE_DEVICES={cuda} MUJOCO_GL=egl python {os.path.join(LEG_ROOT_DIR, 'modular_legs/sim/train_server.py')} {port}; exec zsh'"
    subprocess.run(command, shell=True)
    # os.spawnlp(os.P_NOWAIT, "screen", "screen", "-dmS", screen_name, "zsh", "-c", f"source $HOME/anaconda3/etc/profile.d/conda.sh && conda activate {conda} && CUDA_VISIBLE_DEVICES={cuda} MUJOCO_GL=egl python {os.path.join(LEG_ROOT_DIR, 'modular_legs/sim/train_server.py')} {port}; exec zsh")


    # TMUX command
    # command = f"""tmux new-session -d -s {screen_name} \
    # "source $HOME/anaconda3/etc/profile.d/conda.sh && \
    # conda activate {conda} && \
    # CUDA_VISIBLE_DEVICES={cuda} MUJOCO_GL=egl \
    # python {os.path.join(LEG_ROOT_DIR, 'modular_legs/sim/train_server.py')} {port}" """

    # Run the command
    # subprocess.Popen(command, shell=True)