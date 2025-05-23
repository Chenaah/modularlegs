



from functools import partial
import math
import os
import pdb
import pickle
import numpy as np
from modular_legs import LEG_ROOT_DIR
# from modular_legs.sim.evolution.mutation_meta import random_gen
# from modular_legs.sim.evolution.async_ga_meta import are_ok
from rich.progress import Progress

# from modular_legs.sim.evolution.utils import is_metapipeline_valid
import concurrent.futures

from modular_legs.sim.designer_utils import fast_self_collision_check
from modular_legs.sim.robot_metadesigner import MetaDesignerAsym
from modular_legs.sim.scripts.homemade_robots_asym import MESH_DICT_DRAFT_CYLINDER, ROBOT_CFG_AIR1S

def gen_dateset(min_n_modules=2, max_n_modules=6, data_name="designs", ave_speed_threshold=0.15):
    # These are the numbers of ADDITIONAL modules


    def gen(deisgn_list, g, max_depth=100, _depth=0):
        
        add_pipline_length = np.random.randint(min_n_modules, max_n_modules+1)
        random_design = random_gen(add_pipline_length, g, max_depth)

        if random_design not in deisgn_list:
            print("New design added: ", random_design)
            deisgn_list.append(random_design)

    max_num_designs = 1e5
    max_depth = 10000
    self_collision_threshold = 999
    ave_speed_threshold = ave_speed_threshold
    design_file = os.path.join(LEG_ROOT_DIR, f"modular_legs/sim/evolution/vae/{data_name}.pkl")
    if os.path.isfile(design_file):
        print("Loading designs from file...")
        with open(design_file, "rb") as f:
            designs = pickle.load(f)
    else:
        designs = []

    if ave_speed_threshold > 0:
        g = partial(are_ok, self_collision_threshold=self_collision_threshold, ave_speed_threshold=ave_speed_threshold)
    else:
        g = lambda x: [is_metapipeline_valid(x[0], level=0)] # Only check buildability
    with Progress() as progress:
        task = progress.add_task("[green]Generating designs...", total=max_num_designs)
        while len(designs) < max_num_designs:
            gen(designs, g, max_depth=max_depth)
            progress.update(task, advance=1)
            if len(designs) % 100 == 0:
                with open(design_file, "wb") as f:
                    pickle.dump(designs, f)
    with open(design_file, "wb") as f:
        pickle.dump(designs, f)


def gen_5xp1(min_n_modules, max_n_modules):
    add_pipline_length = np.random.randint(min_n_modules, max_n_modules+1)
    return random_gen(add_pipline_length, None)

def gen_4x(min_n_modules, max_n_modules):
    # A simplified version of encoding
    add_pipline_length = np.random.randint(min_n_modules, max_n_modules+1)
    return gen_random_4xmetadesign(add_pipline_length)

def gen_job(n_designs_per_worker, min_n_modules, max_n_modules, seed, gen_func):
    np.random.seed(seed)  
    sub_pop = []
    while len(sub_pop) < n_designs_per_worker:
        design = gen_func(min_n_modules, max_n_modules)
        if design not in sub_pop:
            sub_pop.append(design)
        print(f"[{len(sub_pop)}/{n_designs_per_worker}] New design added: ", design)
    return sub_pop


def fast_gen_dateset(min_n_modules=1, max_n_modules=3, data_name="designs_short", gen_func=gen_5xp1, seed_offset=0):

    max_num_designs = 5e5
    n_workers = 100

    n_designs_per_worker = math.ceil(max_num_designs*1.5 / n_workers)


    all_designs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(gen_job, n_designs_per_worker, min_n_modules, max_n_modules, seed+seed_offset, gen_func) 
                   for seed in range(n_workers)]
        for future in concurrent.futures.as_completed(futures):
            try:
                data = future.result()
                all_designs.extend(data)
            except Exception as exc:
                print(f'Job generated an exception: {exc}')


    unique_designs = list(set(tuple(d) for d in all_designs))
    unique_designs = [list(t) for t in unique_designs]

    print(f"Total designs: {len(all_designs)}, Unique designs: {len(unique_designs)}")

    unique_designs = unique_designs[:int(max_num_designs)]
    print(f"Total designs: {len(all_designs)}, Unique designs: {len(unique_designs)}")

    design_file = os.path.join(LEG_ROOT_DIR, f"modular_legs/sim/evolution/vae/{data_name}.pkl")
    with open(design_file, "wb") as f:
        pickle.dump(unique_designs, f)

def fix_tuple(design_file):
    with open(design_file, "rb") as f:
        designs = pickle.load(f)
    list_designs = [list(t) for t in designs]
    with open(design_file, "wb") as f:
        pickle.dump(list_designs, f)



def gen_random_4xmetadesign(add_pipline_length):
    """
    Generate a random design pipeline (4x Version)
    """
    robot_designer = MetaDesigner() # used for checking the validity of the design
    # print("ini_pipline", ini_pipline)

    ini_pose = 0
    robot_designer.reset(ini_pose)
    new_pipline = []


    for _ in range(add_pipline_length):
        module = np.random.choice([0, 1]) # what's the next module
        parent = np.random.choice(robot_designer.node_ids) # which part to connect to
        pos_list = robot_designer.get_pos_id_list(module, parent)
        if not pos_list:
            break
        pos = np.random.choice(pos_list) # where to connect
        orientation = np.random.choice(robot_designer.get_rotation_id_list(module, parent, pos)) # how to connect
        # sibling_pos = np.random.choice(robot_designer.get_sibling_id_list(module)) # which sibling to connect
        pipe = [module, parent, pos, orientation]
        robot_designer.step(pipe+[0])
        new_pipline += pipe
        # print(f"Step: {pipe}")

    assert len(new_pipline) % 4 == 0, "Invalid design pipeline length!"

    return new_pipline


def gen_asym4x(min_n_modules, max_n_modules):
    add_pipeline_length = np.random.randint(min_n_modules, max_n_modules+1)

    pipe = []
    robot_designer = MetaDesignerAsym()
    robot_designer.reset()
    print("=====")
    for _ in range(add_pipeline_length):
        module = np.random.choice(robot_designer.get_available_module_ids())
        posa = np.random.choice(robot_designer.get_available_posa_ids(module))
        posb = np.random.choice(robot_designer.get_available_posb_ids())
        rotation = np.random.choice(robot_designer.get_available_rotation_ids(posa, posb))
        step = [module, posa, posb, rotation]
        print("Step: ", step)
        robot_designer.step(step)
        pipe += step

    assert len(pipe) % 4 == 0, "Invalid design pipeline length!"
    return pipe


def gen_asym4x_filtered(min_n_modules, max_n_modules):
    # assert min_n_modules == 1 and max_n_modules == 4, "Invalid min/max modules for gen_asym4x_filtered"
    # The number of total modules should be 2-5
    add_pipeline_length = np.random.randint(min_n_modules, max_n_modules+1)
    # add_pipeline_length = np.random.randint(1, 5)

    pipe = []
    robot_designer = MetaDesignerAsym(robot_cfg=ROBOT_CFG_AIR1S, mesh_dict=MESH_DICT_DRAFT_CYLINDER)
    robot_designer.reset()
    print("=====")
    for _ in range(add_pipeline_length):
        module = np.random.choice(robot_designer.get_available_module_ids())
        posa = np.random.choice(robot_designer.get_available_posa_ids(module))
        posb = np.random.choice(robot_designer.get_available_posb_ids())
        rotation = np.random.choice(robot_designer.get_available_rotation_ids(posa, posb))
        step = [module, posa, posb, rotation]
        print("Step: ", step)
        robot_designer.step(step)
        pipe += step

    if fast_self_collision_check(pipe) > 0:
        print("Self collision detected, retrying...")
        return gen_asym4x_filtered(min_n_modules, max_n_modules)

    assert len(pipe) % 4 == 0, "Invalid design pipeline length!"
    return pipe


def gen_asym4x_extended(min_n_modules, max_n_modules):
    # assert min_n_modules == 1 and max_n_modules == 4, "Invalid min/max modules for gen_asym4x_filtered"
    # The number of total modules should be 2-5
    add_pipeline_length = np.random.randint(min_n_modules, max_n_modules+1)
    # add_pipeline_length = np.random.randint(1, 5)
    # pdb.set_trace()

    pipe = [0,1,0,0,0,3,0,0]
    robot_designer = MetaDesignerAsym(robot_cfg=ROBOT_CFG_AIR1S, mesh_dict=MESH_DICT_DRAFT_CYLINDER)
    robot_designer.reset()
    print("=====")
    for _ in range(add_pipeline_length):
        # module = np.random.choice(robot_designer.get_available_module_ids())
        module, posb = 0, 0
        posa = np.random.choice(robot_designer.get_available_posa_ids(module))
        posb = np.random.choice(robot_designer.get_available_posb_ids())
        rotation = np.random.choice(robot_designer.get_available_rotation_ids(posa, posb))
        step = [module, posa, posb, rotation]
        print("Step: ", step)
        robot_designer.step(step)
        pipe += step

    if fast_self_collision_check(pipe) > 0:
        print("Self collision detected, retrying...")
        return gen_asym4x_filtered(min_n_modules, max_n_modules)

    assert len(pipe) % 4 == 0, "Invalid design pipeline length!"
    return pipe


def test_gen_asym4x():
    for _ in range(100):
        print(gen_asym4x(1, 4))


def test_gen_asym4x_filtered():
    # filter self collision
    for _ in range(100):
        print(gen_asym4x_filtered())


if __name__ == "__main__":
    # gen_dateset(1, 3, "designs_short", ave_speed_threshold=-1)
    # fast_gen_dateset()
    # fast_gen_dateset(data_name="designs_short_lite", gen_func=gen_4x)
    # fast_gen_dateset(1,4,data_name="designs_asym", gen_func=gen_asym4x)
    # test_gen_asym4x()
    # fix_tuple(os.path.join(LEG_ROOT_DIR, f"modular_legs/sim/evolution/vae/designs_short.pkl"))
    # fast_gen_dateset(1,4,data_name="designs_asym_v2", gen_func=gen_asym4x, seed_offset=300)
    # test_gen_asym4x_filtered()
    # fast_gen_dateset(1,4,data_name="designs_asym_filtered", gen_func=gen_asym4x_filtered, seed_offset=0)
    # fast_gen_dateset(1,2,data_name="designs_asym_3m_filtered", gen_func=gen_asym4x_filtered, seed_offset=0) # 2-3 modules
    # fast_gen_dateset(1,3,data_name="designs_asym_4m_filtered", gen_func=gen_asym4x_filtered, seed_offset=0) # 2-4 modules
    fast_gen_dateset(1,2,data_name="designs_asym_5m_extended", gen_func=gen_asym4x_extended, seed_offset=0) # 2-4 modules