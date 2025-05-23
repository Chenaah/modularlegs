
import pdb
import random
import numpy as np
# from modular_legs.sim.evolution.blackbox import BlackBox
from modular_legs.sim.evolution.utils import is_metapipeline_valid
# from modular_legs.sim.robot_metadesigner import MetaDesigner




def extend_random_design(add_pipline_length, ini_pipline=[]):
    """
    Generate a random design pipeline (5x+1 Version)
    """
    new_pipline = ini_pipline.copy()
    robot_designer = MetaDesigner() # used for checking the validity of the design
    # print("ini_pipline", ini_pipline)

    if not ini_pipline:
        ini_pose = random.choice([0, 1])
        robot_designer.reset(ini_pose)
        new_pipline += [ini_pose]
    else:
        robot_designer.reset(ini_pipline[0])

        for pipline in np.reshape(new_pipline[1:], (-1, 5)):
            robot_designer.step(pipline)

    for _ in range(add_pipline_length):
        module = np.random.choice([0, 1]) # what's the next module
        parent = np.random.choice(robot_designer.node_ids) # which part to connect to
        pos_list = robot_designer.get_pos_id_list(module, parent)
        if not pos_list:
            return extend_random_design(add_pipline_length, ini_pipline)
        pos = np.random.choice(pos_list) # where to connect
        orientation = np.random.choice(robot_designer.get_rotation_id_list(module, parent, pos)) # how to connect
        sibling_pos = np.random.choice(robot_designer.get_sibling_id_list(module)) # which sibling to connect
        pipe = [module, parent, pos, orientation, sibling_pos]
        robot_designer.step(pipe)
        new_pipline += pipe
        # print(f"Step: {pipe}")

    assert len(new_pipline) % 5 == 1, "Invalid design pipeline length!"

    return new_pipline


def find_closest_value(value, value_list):
    return min(value_list, key=lambda x:abs(x-value))


def _mutate_limb(design_pipeline, step_idx, sub_step_idx):
    """
    Mutate a random limb in the design pipeline
    step_idx: 0, 1, 5, 9, 13... 0 means changing the initial pose, in which case sub_step_idx is ignored
    sub_step_idx: 0, 1, 2, 3, 4 corresponding to module, parent, pos, orientation
    """
    new_design_pipeline = design_pipeline.copy()
    assert step_idx in [i for i in range(1, len(design_pipeline) - 4, 5)] + [0], "Invalid step index"
    assert sub_step_idx in [0, 1, 2, 3, 4], "Invalid sub-step index"
    # Mutate the desinated slice in the design pipeline

    if step_idx == 0:
        # Mutate the initial pose
        # This should not affect the buildability of the design [x]
        ini_pose = 0 if new_design_pipeline[0] == 1 else 1
        new_design_pipeline[0] = ini_pose
        designer = MetaDesigner()
        designer.reset(ini_pose)
        post_steps = new_design_pipeline[1:]
        pre_steps = [ini_pose]
        mutated_step = []

    else:
        pre_steps = new_design_pipeline[:step_idx]
        post_steps = new_design_pipeline[step_idx+5:]
        designer = MetaDesigner(init_pipeline=pre_steps)

        choosen_step = new_design_pipeline[step_idx:step_idx+5]
        module, parent, pos, orientation, sibling_pose = choosen_step
        old_module = module
        if sub_step_idx == 0:
            # Mutate the module type, keeping all others if possible
            module = 0 if module == 1 else 1
            available_pos = designer.get_pos_id_list(module, parent)
            if not available_pos:
                return None
            if pos not in available_pos:
                pos = find_closest_value(pos, available_pos)
            available_orientation = designer.get_rotation_id_list(module, parent, pos)
            if orientation not in available_orientation:
                orientation = find_closest_value(orientation, available_orientation)
            available_sibling_poses = designer.get_sibling_id_list(module)
            if sibling_pose not in available_sibling_poses:
                sibling_pose = find_closest_value(sibling_pose, available_sibling_poses)
        elif sub_step_idx == 1:
            # Mutate the parent id, keeping all others if possible
            candidate_parent = [x for x in designer.node_ids if x != parent]
            if candidate_parent:
                # Choose a different parent if possible
                parent = np.random.choice(candidate_parent)
            available_pos = designer.get_pos_id_list(module, parent)
            if not available_pos:
                return None
            if pos not in available_pos:
                pos = find_closest_value(pos, available_pos)
            available_orientation = designer.get_rotation_id_list(module, parent, pos)
            if orientation not in available_orientation:
                orientation = find_closest_value(orientation, available_orientation)
        elif sub_step_idx == 2:
            # Mutate the position id, keeping all others if possible
            available_pos = [x for x in designer.get_pos_id_list(module, parent) if x != parent]
            if not available_pos:
                return None
            pos = np.random.choice(available_pos)
            available_orientation = designer.get_rotation_id_list(module, parent, pos)
            if orientation not in available_orientation:
                orientation = find_closest_value(orientation, available_orientation)
        elif sub_step_idx == 3:
            # Mutate the orientation id, keeping all others if possible
            available_orientation = [x for x in designer.get_rotation_id_list(module, parent, pos) if x != orientation]
            if not available_orientation:
                return None
            orientation = np.random.choice(available_orientation)
        elif sub_step_idx == 4:
            # Mutate the sibling pose id, keeping all others if possible
            available_sibling_poses = [x for x in designer.get_sibling_id_list(module) if x != sibling_pose]
            sibling_pose = np.random.choice(available_sibling_poses)

        mutated_step = [module, parent, pos, orientation, sibling_pose]
        designer.step(mutated_step)

    new_post_steps = []
    for step in np.reshape(post_steps, (-1, 5)):
        module, parent, pos, orientation, sibling_pose = step
        assert module in [0, 1], "Something wrong: Invalid module in the original design pipeline!"

        if parent not in designer.node_ids:
            pdb.set_trace()
        assert parent in designer.node_ids, "Something wrong: This step should still find the parent!"

        pos_id_list = designer.get_pos_id_list(module, parent)

        if not pos_id_list:
            # This can happen when the module is changed from stick to ball and the available positions become fewer
            break
        
        if pos not in pos_id_list:
            # The position is not available
            # Find the closest position in the available positions
            pos = find_closest_value(pos, pos_id_list)
        orientation_id_list = designer.get_rotation_id_list(module, parent, pos)
        if orientation not in orientation_id_list:
            # The orientation is not available
            # Find the closest orientation in the available orientations
            orientation = find_closest_value(orientation, orientation_id_list)
        sibling_pose_id_list = designer.get_sibling_id_list(module)
        if sibling_pose not in sibling_pose_id_list:
            # The sibling pose is not available
            # Find the closest sibling pose in the available sibling poses
            sibling_pose = find_closest_value(sibling_pose, sibling_pose_id_list)

        new_step = [module, parent, pos, orientation, sibling_pose]
        designer.step(new_step)
        new_post_steps += new_step

    return pre_steps + mutated_step + new_post_steps


def random_gen(add_pipline_length, constraint_func, max_depth=20, _depth=0):

    # Maximum depth is not considered here

    new_design_pipeline = extend_random_design(add_pipline_length, ini_pipline=[])

    if not is_metapipeline_valid(new_design_pipeline, 0):
        # The new design pipeline should be at least buildable
        pdb.set_trace()

    if constraint_func is not None:
        if not constraint_func([new_design_pipeline])[0]:
            print(f"[Random Generation] Invalid random design ! Retry...", _depth)
            return random_gen(add_pipline_length, constraint_func, max_depth, _depth + 1)
        
    assert len(new_design_pipeline) % 5 == 1, "Invalid design pipeline length!"
        
    return new_design_pipeline
    

def mutate(design_pipeline, mutate_type, constraint_func, max_depth=20, _depth=0):
    """
    Mutate a single design pipeline;
    Input and output of constraint_func is assumed to be batch-based
    """
    design_pipeline = list(design_pipeline).copy()

    if _depth > max_depth:
        return design_pipeline

    if mutate_type is None:
        op = np.random.choice(["grow_limb", "mutate_limb", "delete_limb"], p=[0.2, 0.6, 0.2])
    else:
        op = mutate_type

    print(f"{op} -> {design_pipeline}")

    if op == "grow_limb":
        new_design_pipeline = extend_random_design(1, design_pipeline)
    elif op == "delete_limb":
        step_idx = np.random.choice([i for i in range(1, len(design_pipeline) - 4, 5)])
        new_design_pipeline = _delete_limb(design_pipeline, step_idx )
    elif op == "mutate_limb":
        normal_idx_list = [i for i in range(1, len(design_pipeline) - 4, 5)]
        step_idx = np.random.choice(normal_idx_list + [0], p=[5/(5*len(normal_idx_list)+1)]*len(normal_idx_list) + [1/(5*len(normal_idx_list)+1)]) #[1/len(normal_idx_list)*5] + [(1-1/len(normal_idx_list)*5)/len(normal_idx_list)]*len(normal_idx_list)
        sub_step_idx = np.random.choice([0, 1, 2, 3, 4])
        new_design_pipeline = _mutate_limb(design_pipeline, step_idx, sub_step_idx)

    if new_design_pipeline is None:
        print(f"[Mutate] Invalid mutation ({op}) ! Retry...", _depth)
        return mutate(design_pipeline, mutate_type, constraint_func, max_depth, _depth + 1)

    if not is_metapipeline_valid(new_design_pipeline, 0):
        # The new design pipeline should be at least buildable
        pdb.set_trace()

    if constraint_func is not None:
        if not constraint_func([new_design_pipeline])[0]:
            print(f"[Mutate] Invalid mutation ({op}) ! Retry...", _depth)
            return mutate(design_pipeline, mutate_type, constraint_func, max_depth, _depth + 1)
        
    return new_design_pipeline



def _crossover_step(design_pipeline0, design_pipeline1, cut_idx0, cut_idx1):
    """
    Crossover two design pipelines
    """
    design_pipeline0 = design_pipeline0.copy()
    design_pipeline1 = design_pipeline1.copy()

    # Crossover the design pipelines
    

    design0left, design0right = design_pipeline0[:cut_idx0], design_pipeline0[cut_idx0:]
    design1left, design1right = design_pipeline1[:cut_idx1], design_pipeline1[cut_idx1:]
    if not (len(design0left)%5 ==1 and len(design1left)%5==1 and len(design0right)%5==0 and len(design1right)%5==0):
        pdb.set_trace()

    designer0 = MetaDesigner(init_pipeline=design0left)
    designer1 = MetaDesigner(init_pipeline=design1left)
    last_id0 = designer0.node_ids[-1]
    last_id1 = designer1.node_ids[-1]

    # Design 0 left + Design 1 right

    new_design0right = []
    for step in np.reshape(design1right, (-1, 5)):
        module, parent, pos, orientation, sibling_pose = step
        assert module in [0, 1], "Something wrong: Invalid module in the original design pipeline!"

        if parent > last_id1:
            parent -= (last_id1 - last_id0)

        available_parent_ids = designer0.node_ids
        if parent not in available_parent_ids:
            parent = find_closest_value(parent, available_parent_ids)
        pos_id_list = designer0.get_pos_id_list(module, parent)
        if pos not in pos_id_list:
            if not pos_id_list:
                print("No available position!")
                # pdb.set_trace()
                break
            pos = find_closest_value(pos, pos_id_list)
        orientation_id_list = designer0.get_rotation_id_list(module, parent, pos)
        if orientation not in orientation_id_list:
            orientation = find_closest_value(orientation, orientation_id_list)
        sibling_pose_id_list = designer0.get_sibling_id_list(module)
        if sibling_pose not in sibling_pose_id_list:
            sibling_pose = find_closest_value(sibling_pose, sibling_pose_id_list)

        new_step = [module, parent, pos, orientation, sibling_pose]
        designer0.step(new_step)
        new_design0right += new_step

    # Design 1 left + Design 0 right

    new_design1right = []
    for step in np.reshape(design0right, (-1, 5)):
        module, parent, pos, orientation, sibling_pose = step
        assert module in [0, 1], "Something wrong: Invalid module in the original design pipeline!"

        if parent > last_id0:
            parent -= (last_id0 - last_id1)

        available_parent_ids = designer1.node_ids
        if parent not in available_parent_ids:
            parent = find_closest_value(parent, available_parent_ids)
        pos_id_list = designer1.get_pos_id_list(module, parent)
        if pos not in pos_id_list:
            if not pos_id_list:
                print("No available position!")
                # pdb.set_trace()
                break
            pos = find_closest_value(pos, pos_id_list)
        orientation_id_list = designer1.get_rotation_id_list(module, parent, pos)
        if orientation not in orientation_id_list:
            orientation = find_closest_value(orientation, orientation_id_list)
        sibling_pose_id_list = designer1.get_sibling_id_list(module)
        if sibling_pose not in sibling_pose_id_list:
            sibling_pose = find_closest_value(sibling_pose, sibling_pose_id_list)

        new_step = [module, parent, pos, orientation, sibling_pose]
        designer1.step(new_step)
        new_design1right += new_step

    if not is_metapipeline_valid(design0left + new_design0right, 0):
        # The new design pipeline should be at least buildable
        pdb.set_trace()
    if not is_metapipeline_valid(design1left + new_design1right, 0):
        pdb.set_trace()
    # if not (design0left + new_design0right):
    #     pdb.set_trace()
    # if not (design1left + new_design1right):
    #     pdb.set_trace()

    return design0left + new_design0right, design1left + new_design1right


def _delete_limb(design_pipeline, step_idx):
    '''
    Delete a limb from the design pipeline
    '''
    new_design_pipeline = design_pipeline.copy()
    new_design, _ = _crossover_step(new_design_pipeline, new_design_pipeline, step_idx, step_idx+5)
    return new_design
    



# def validity_check(design_pipeline):
#     blackbox = BlackBox("evolution_debug", connect_to_server=False)
#     blackbox.reset([design_pipeline])
#     return all(blackbox.is_builable())

def crossover(design_pipeline0, design_pipeline1, constraint_func, return_single=False, max_depth=20, _depth=0):
    """
    Crossover two design pipelines;
    Input and output of constraint_func is assumed to be batch-based
    """
    design_pipeline0 = list(design_pipeline0).copy()
    design_pipeline1 = list(design_pipeline1).copy()

    if _depth > max_depth:
        return design_pipeline0, design_pipeline1
    
    cut_idx0 = np.random.choice([i for i in range(1, len(design_pipeline0) - 4, 5)])
    cut_idx1 = np.random.choice([i for i in range(1, len(design_pipeline1) - 4, 5)])
    new_design_pipeline0, new_design_pipeline1, = _crossover_step(design_pipeline0, design_pipeline1, cut_idx0, cut_idx1)

    # assert validity_check(new_design_pipeline0) and validity_check(new_design_pipeline1), "Invalid design pipeline after crossover!"
    # assert new_design_pipeline0 and new_design_pipeline1, "Empty design pipeline after crossover!"

    if constraint_func is not None:
        valid = constraint_func([new_design_pipeline0, new_design_pipeline1])
        if not return_single:
            if not all(valid):
                print("[Crossover] Constraint not met! Retry...", _depth)
                return crossover(design_pipeline0, design_pipeline1, constraint_func, return_single, max_depth, _depth + 1)
        else:
            if not any(valid):
                print("[Crossover] Constraint not met! Retry...", _depth)
                return crossover(design_pipeline0, design_pipeline1, constraint_func, return_single, max_depth, _depth + 1)

    if return_single:
        if valid[0]:
            return new_design_pipeline0
        elif valid[1]:
            return new_design_pipeline1
    
    return new_design_pipeline0, new_design_pipeline1
    

if __name__ == "__main__":
    # n = delete_limb([1, 0, 1, 14, 1, 0, 2, 3, 1, 3, 4, 6], 4)
    # print(n)
    # for i in range(4):
    #     n = mutate_limb([1, 0, 1, 14, 1, 0, 2, 3, 1, 3, 4, 6], 0, i)
    #     print(n)
    # a,b = crossover([1, 0, 1, 14, 1, 0, 2, 3, 1, 3, 4, 6], [0, 0, 2, 0, 0, 0, 1, 1, 1, 4, 2, 9], None)
    # print(a)
    # print(b)

    print("Random Generation")
    p = random_gen(3, None)
    print(p)
    print("Mutation - Grow Limb")
    p = mutate(p, "grow_limb", None)
    print(p)
    print("Mutation - Mutate Limb")
    p = mutate(p, "mutate_limb", None)
    print(p)
    print("Mutation - Delete Limb")
    p = mutate(p, "delete_limb", None)
    print(p)
    print("Crossover")
    p1 = random_gen(3, None)
    p2 = random_gen(3, None)
    print(p1)
    print(p2)
    p1, p2 = crossover(p1, p2, None)
    print("-->")
    print(p1)
    print(p2)
    print("Mutate initial pose")
    p = _mutate_limb(p2, 0, 0)
    print(p)
