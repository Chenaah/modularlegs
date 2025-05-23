
import pdb
import random
import numpy as np
from modular_legs.sim.evolution.blackbox import BlackBox, is_pipeline_valid
from modular_legs.sim.robot_designer import RobotDesigner


def extend_random_design(add_pipline_length, ini_pipline=[]):
    """
    Generate a random design pipeline
    """
    new_pipline = ini_pipline.copy()
    robot_designer = RobotDesigner() # used for checking the validity of the design
    robot_designer.reset()
    # print("ini_pipline", ini_pipline)
    for pipline in np.reshape(new_pipline, (-1, 4)):
        robot_designer.step(pipline)

    for _ in range(add_pipline_length):
        module = np.random.choice([0, 1]) # what's the next module
        parent = np.random.choice(robot_designer.node_ids) # which part to connect to
        pos_list = robot_designer.get_pos_id_list(module, parent)
        if not pos_list:
            return extend_random_design(add_pipline_length, ini_pipline)
        pos = np.random.choice(pos_list) # where to connect
        orientation_list = robot_designer.get_rotation_id_list(module, parent, pos)
        orientation = np.random.choice(orientation_list) # how to connect
        pipe = [module, parent, pos, orientation]
        new_pipline += pipe
        robot_designer.step(pipe)
        # print(f"Step: {pipe}")
    return new_pipline




# def _delete_limb(design_pipeline, step_idx):
#     """
#     Delete a random limb from the design pipeline
#     """
#     new_design_pipeline = design_pipeline.copy()
#     assert step_idx in [i for i in range(0, len(new_design_pipeline) - 3, 4)], "Invalid step index"
#     # Delete the desinated slice in the design pipeline
#     deleted_module_type = new_design_pipeline[step_idx]
#     del new_design_pipeline[step_idx:step_idx+4]

#     # Sanitize the design pipeline
#     pre_steps = new_design_pipeline[:step_idx]
#     designer = RobotDesigner(init_pipeline=pre_steps)
#     dummy_node_ids = designer.add_dummy_node(deleted_module_type) # To keep the parent id consistent

#     post_steps = new_design_pipeline[step_idx:]
#     new_post_steps = []
#     for step in np.reshape(post_steps, (-1, 4)):
#         module, parent, pos, orientation = step
#         assert module in [0, 1], "Something wrong: Invalid module in the original design pipeline"
        
#         available_parent_ids = [x for x in designer.node_ids if x not in dummy_node_ids]
#         pdb.set_trace()
#         if parent not in available_parent_ids:
#             # Its parent has been deleted
#             # Find the closest parent in the existing nodes
#             parent = min(available_parent_ids, key=lambda x:abs(x-parent))
#         pos_id_list = designer.get_pos_id_list(module, parent)
#         if pos not in pos_id_list:
#             # The position is not available
#             # Find the closest position in the available positions
#             pos = min(pos_id_list, key=lambda x:abs(x-pos))
#         orientation_id_list = designer.get_rotation_id_list(module, parent, pos)
#         if orientation not in orientation_id_list:
#             # The orientation is not available
#             # Find the closest orientation in the available orientations
#             orientation = min(orientation_id_list, key=lambda x:abs(x-orientation))

#         new_step = [module, parent, pos, orientation]
#         designer.step(new_step)
#         new_post_steps += new_step

#     return pre_steps + new_post_steps


def _mutate_limb(design_pipeline, step_idx, sub_step_idx):
    """
    Mutate a random limb in the design pipeline
    sub_step_idx: 0, 1, 2, 3, corresponding to module, parent, pos, orientation
    """
    new_design_pipeline = design_pipeline.copy()
    assert step_idx in [i for i in range(0, len(new_design_pipeline) - 3, 4)], "Invalid step index"
    assert sub_step_idx in [0, 1, 2, 3], "Invalid sub-step index"
    # Mutate the desinated slice in the design pipeline

    pre_steps = new_design_pipeline[:step_idx]
    post_steps = new_design_pipeline[step_idx+4:]
    designer = RobotDesigner(init_pipeline=pre_steps)

    choosen_step = new_design_pipeline[step_idx:step_idx+4]
    module, parent, pos, orientation = choosen_step
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
    mutated_step = [module, parent, pos, orientation]
    designer.step(mutated_step)
    if module == 0:
        changed_node_ids = designer.node_ids[-2]
    elif module == 1:
        changed_node_ids = designer.node_ids[-1]

    new_post_steps = []
    for step in np.reshape(post_steps, (-1, 4)):
        module, parent, pos, orientation = step
        assert module in [0, 1], "Something wrong: Invalid module in the original design pipeline!"

        if sub_step_idx == 0:
            if old_module == 0:
                # ball -> stick; parent_id after this step should -1
                if parent > changed_node_ids:
                    parent -= 1
            elif old_module == 1:
                # stick -> ball; parent_id after this step should +1
                if parent > changed_node_ids:
                    parent += 1
        
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

        new_step = [module, parent, pos, orientation]
        designer.step(new_step)
        new_post_steps += new_step

    return pre_steps + mutated_step + new_post_steps



def mutate(design_pipeline, mutate_type, constraint_func, max_depth=20, _depth=0):
    """
    Mutate a single design pipeline;
    Input and output of constraint_func is assumed to be batch-based
    """
    design_pipeline = list(design_pipeline).copy()

    if _depth > max_depth:
        return design_pipeline

    if mutate_type is None:
        op = np.random.choice(["grow_limb", "mutate_limb", "delete_limb"])
    else:
        op = mutate_type

    print(op)

    if op == "grow_limb":
        new_design_pipeline = extend_random_design(1, design_pipeline)
    elif op == "delete_limb":
        try:
            step_idx = np.random.choice([i for i in range(0, len(design_pipeline) - 3, 4)])
        except ValueError:
            pdb.set_trace()
        new_design_pipeline = _delete_limb(design_pipeline, step_idx )
    elif op == "mutate_limb":
        try:
            step_idx = np.random.choice([i for i in range(0, len(design_pipeline) - 3, 4)])
        except ValueError:
            pdb.set_trace()
        sub_step_idx = np.random.choice([0, 1, 2, 3])
        new_design_pipeline = _mutate_limb(design_pipeline, step_idx, sub_step_idx)

    if new_design_pipeline is None:
        print(f"[Mutate] Invalid mutation ({op}) ! Retry...", _depth)
        return mutate(design_pipeline, mutate_type, constraint_func, max_depth, _depth + 1)

    if not is_pipeline_valid(new_design_pipeline, 0):
        pdb.set_trace()

    # assert is_pipeline_valid(new_design_pipeline, 0), "Invalid design pipeline after mutation!"
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
    if not (len(design0left)%4 ==0 and len(design1left)%4==0 and len(design0right)%4==0 and len(design1right)%4==0):
        pdb.set_trace()

    designer0 = RobotDesigner(init_pipeline=design0left)
    designer1 = RobotDesigner(init_pipeline=design1left)
    last_id0 = designer0.node_ids[-1]
    last_id1 = designer1.node_ids[-1]

    # Design 0 left + Design 1 right

    new_design0right = []
    for step in np.reshape(design1right, (-1, 4)):
        module, parent, pos, orientation = step
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

        new_step = [module, parent, pos, orientation]
        designer0.step(new_step)
        new_design0right += new_step

    # Design 1 left + Design 0 right

    new_design1right = []
    for step in np.reshape(design0right, (-1, 4)):
        module, parent, pos, orientation = step
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

        new_step = [module, parent, pos, orientation]
        designer1.step(new_step)
        new_design1right += new_step

    if not validity_check(design0left + new_design0right):
        pdb.set_trace()
    if not validity_check(design1left + new_design1right):
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
    new_design, _ = _crossover_step(new_design_pipeline, new_design_pipeline, step_idx, step_idx+4)
    return new_design
    



def validity_check(design_pipeline):
    blackbox = BlackBox("evolution_debug", connect_to_server=False)
    blackbox.reset([design_pipeline])
    return all(blackbox.is_builable())

def crossover(design_pipeline0, design_pipeline1, constraint_func, return_single=False, max_depth=20, _depth=0):
    """
    Crossover two design pipelines;
    Input and output of constraint_func is assumed to be batch-based
    """
    design_pipeline0 = list(design_pipeline0).copy()
    design_pipeline1 = list(design_pipeline1).copy()

    if _depth > max_depth:
        return design_pipeline0, design_pipeline1
    
    cut_idx0 = np.random.choice([i for i in range(0, len(design_pipeline0) - 3, 4)])
    cut_idx1 = np.random.choice([i for i in range(0, len(design_pipeline1) - 3, 4)])
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
    a,b = crossover([1, 0, 1, 14, 1, 0, 2, 3, 1, 3, 4, 6], [0, 0, 2, 0, 0, 0, 1, 1, 1, 4, 2, 9], None)
    print(a)
    print(b)