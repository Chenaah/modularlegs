import time
import numpy as np
from modular_legs.embodied.interface import Interface
from modular_legs.utils.files import load_cfg


cfg = load_cfg()
interface = Interface(cfg)

while True:
    interface.update_ui() 
    interface.receive_module_data()
    interface.get_observable_data()
    interface.send_action(np.zeros(1))
    time.sleep(0.02)