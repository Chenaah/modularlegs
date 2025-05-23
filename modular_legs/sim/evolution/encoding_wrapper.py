
import pdb
import numpy as np
import torch
from torch.nn import functional as F

# from modular_legs.sim.evolution.mutation import find_closest_value
from modular_legs.sim.evolution.utils import is_metapipeline_valid, find_closest_value
from modular_legs.sim.robot_metadesigner import MetaDesignerAsym
from modular_legs.sim.scripts.homemade_robots_asym import MESH_DICT_FINE, ROBOT_CFG_AIR1S


##### 

# from modular_legs.sim.robot_metadesigner import MetaDesigner


# class GraphParser():

#     def step(self, pipeline):
#         # [init module pose] -> [child module pose] -> [parent module id] -> [child module pose]  -> [parent module id] -> [child module pose] ...



def to_onehot(pipelines, max_idx, max_length=None):
    assert max_idx >= max(max(sublist) for sublist in pipelines), "max_idx must be greater than the maximum index in the pipelines"

    if max_length is None:
        max_length = max(len(sublist) for sublist in pipelines)
    # Pad each sublist with zeros (or any other value) to make them the same length
    padded_list_of_lists = [sublist + [-1] * (max_length - len(sublist)) for sublist in pipelines]
    padded_array = torch.tensor(padded_list_of_lists) + 1
    onehot = F.one_hot(padded_array, num_classes=max_idx + 2)  # Add 1 to max_idx to account for padding

    return onehot.flatten(start_dim=1, end_dim=2).float()

def decode_onehot(onehot_encoded, max_idx=23):
    # Convert one-hot encoded tensor back to indices by finding the index of the max value in each position
    indices = torch.argmax(onehot_encoded.reshape(onehot_encoded.shape[0], -1, max_idx + 2), dim=2) - 1  # Subtract 1 to adjust for the earlier +1 operation
    # Convert the tensor of indices back to a list of lists, removing padding in the process
    decoded = []
    for sublist in indices:
        decoded_sublist = []
        for item in sublist:
            if item.item() == -1:  # Stop adding to the sublist once padding (-1) is encountered
                break
            decoded_sublist.append(item.item())
        decoded.append(decoded_sublist)
    
    return decoded


def _cut_list_to_mod_5_eq_1(l):
    current_length = len(l)
    remainder = current_length % 5

    # If the current length % 5 is already 1, no need to cut the list
    if remainder == 1:
        return l
    
    # Calculate how many elements to remove
    elements_to_remove = remainder - 1 if remainder > 1 else 4
    
    # Cut the list
    new_length = current_length - elements_to_remove
    return l[:new_length]

def _cut_list_to_mod_4_eq_0(l):
    current_length = len(l)
    remainder = current_length % 4

    # If the current length % 4 is already 0, no need to cut the list
    if remainder == 0:
        return l
    
    # Calculate how many elements to remove
    elements_to_remove = remainder if remainder > 0 else 0
    
    # Cut the list
    new_length = current_length - elements_to_remove
    return l[:new_length]


def polish(pipeline):
    pipeline = pipeline.copy()
    new_pipeline = []   
    # No batch dimension 
    # For the 5x+1 pipeline
    # Cut the pipeline to make it 5x+1
    pipeline = _cut_list_to_mod_5_eq_1(pipeline)
    init_pose = pipeline[0]
    if init_pose not in [0, 1]:
        init_pose = find_closest_value(init_pose, [0, 1])
    new_pipeline.append(init_pose)
        
    designer = MetaDesigner()
    designer.reset(init_pose)

    design_pipeline = np.reshape(pipeline[1:], (-1, 5))
    for step in design_pipeline:
        module, parent, pos, orientation, sibling_pose = step
        if module not in [0, 1]:
            module = find_closest_value(module, [0, 1])
        if parent not in designer.node_ids:
            parent = find_closest_value(parent, designer.node_ids)
        pos_list = designer.get_pos_id_list(module, parent)
        if not pos_list:
            break
        if pos not in pos_list:
            pos = find_closest_value(pos, pos_list)
        orientation_list = designer.get_rotation_id_list(module, parent, pos)
        if orientation not in orientation_list:
            orientation = find_closest_value(orientation, orientation_list)
        sibling_id_list = designer.get_sibling_id_list(module)
        if sibling_pose not in sibling_id_list:
            sibling_pose = find_closest_value(sibling_pose, sibling_id_list)
        new_step = [module, parent, pos, orientation, sibling_pose]
        new_pipeline.extend(new_step)
        designer.step(new_step)

    return new_pipeline


def polish_4x(pipeline):
    pipeline = pipeline.copy()
    new_pipeline = []   
    # No batch dimension 
    # For the 4x pipeline
    # Cut the pipeline to make it 4x
    pipeline = _cut_list_to_mod_4_eq_0(pipeline)
    init_pose = 0
        
    designer = MetaDesigner()
    designer.reset(init_pose)

    design_pipeline = np.reshape(pipeline, (-1, 4))
    for step in design_pipeline:
        module, parent, pos, orientation = step
        if module not in [0, 1]:
            module = find_closest_value(module, [0, 1])
        if parent not in designer.node_ids:
            parent = find_closest_value(parent, designer.node_ids)
        pos_list = designer.get_pos_id_list(module, parent)
        if not pos_list:
            break
        if pos not in pos_list:
            pos = find_closest_value(pos, pos_list)
        orientation_list = designer.get_rotation_id_list(module, parent, pos)
        if orientation not in orientation_list:
            orientation = find_closest_value(orientation, orientation_list)
        # sibling_id_list = designer.get_sibling_id_list(module)
        # if sibling_pose not in sibling_id_list:
        #     sibling_pose = find_closest_value(sibling_pose, sibling_id_list)
        new_pipeline.extend([module, parent, pos, orientation])
        designer.step([module, parent, pos, orientation, 0])

    return new_pipeline


def polish_asym(pipeline):
    pipeline = pipeline.copy()
    new_pipeline = []   
    # No batch dimension 
    # For the 4x ASYM pipeline
    # Cut the pipeline to make it 4x
    pipeline = _cut_list_to_mod_4_eq_0(pipeline)
        
    designer = MetaDesignerAsym(robot_cfg=ROBOT_CFG_AIR1S, mesh_dict=MESH_DICT_FINE)
    designer.reset()

    design_pipeline = np.reshape(pipeline, (-1, 4))
    for step in design_pipeline:
        module, posa, posb, rotation = step
        available_module_ids = designer.get_available_module_ids()
        if module not in available_module_ids:
            module = find_closest_value(module, available_module_ids)
        available_posa_ids = designer.get_available_posa_ids(module)
        if posa not in available_posa_ids:
            if not available_posa_ids:
                break
            posa = find_closest_value(posa, available_posa_ids)
        available_posb_ids = designer.get_available_posb_ids()
        if posb not in available_posb_ids:
            posb = find_closest_value(posb, available_posb_ids)
        available_rotation_ids = designer.get_available_rotation_ids(posa, posb)
        if rotation not in available_rotation_ids:
            rotation = find_closest_value(rotation, available_rotation_ids)
        new_step = [module, posa, posb, rotation]

        new_pipeline.extend(new_step)
        designer.step(new_step)

    return new_pipeline



def extent_to_5x1(pipeline):
    # Extent the 4x version of meta pipeline to 5x+1
    # 4x version is used when custom default joint position is allowed
    assert len(pipeline) % 4 == 0, "The length of the pipeline must be 4x"
    new_pipeline = [0]
    for step in np.reshape(pipeline, (-1, 4)):
        new_pipeline.extend(step)
        new_pipeline.append(0)

    return new_pipeline

def test():
    ini_xs = [[0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 13, 0, 6, 1, 2], [0, 0, 1, 0, 0, 3, 0, 2, 1, 4, 1, 2, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0, 2, 1, 0, 2, 1, 1, 0, 5, 0, 0], [1, 0, 1, 7, 0, 1, 0, 2, 0, 0, 2, 1, 1, 2, 5, 7], [1, 1, 2, 12, 0, 0, 2, 0, 1, 0, 1, 3], [1, 1, 1, 2, 0, 0, 2, 1, 0, 4, 0, 1, 0, 6, 2, 0], [0, 1, 1, 1, 0, 0, 2, 1, 1, 3, 0, 9], [1, 0, 1, 10, 0, 1, 2, 0, 1, 4, 1, 6], [0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 2, 1], [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 2, 2, 1, 2, 1, 6], [1, 1, 2, 0, 1, 0, 1, 10, 0, 2, 2, 0], [0, 1, 2, 0, 1, 2, 1, 7, 1, 0, 1, 8], [0, 0, 2, 1, 0, 2, 1, 1, 1, 1, 0, 10], [1, 0, 1, 7, 0, 2, 5, 0, 1, 3, 1, 9], [0, 1, 2, 1, 0, 3, 1, 2, 0, 5, 0, 0], [1, 0, 2, 9, 1, 0, 1, 10, 0, 1, 0, 0], [1, 1, 0, 9, 1, 2, 2, 0, 1, 0, 2, 1], [1, 0, 2, 6, 0, 1, 0, 1, 0, 4, 2, 1], [0, 1, 0, 2, 0, 0, 2, 0, 1, 2, 2, 7, 0, 1, 1, 0, 1, 2, 1, 5], [0, 0, 1, 0, 1, 3, 1, 5, 0, 2, 1, 2, 0, 1, 2, 0], [0, 1, 2, 1, 1, 3, 2, 2, 1, 0, 1, 14], [0, 1, 1, 0, 1, 3, 0, 14, 0, 0, 2, 2], [0, 1, 0, 1, 1, 3, 0, 1, 1, 4, 6, 3, 0, 5, 5, 0], [1, 0, 2, 11, 1, 2, 7, 8, 1, 2, 4, 3, 0, 0, 1, 2], [0, 1, 1, 2, 1, 2, 1, 13, 1, 0, 2, 9, 0, 5, 8, 1, 1, 4, 1, 2], [0, 1, 0, 2, 0, 0, 2, 2, 0, 5, 1, 0, 0, 1, 2, 2], [1, 1, 1, 13, 0, 1, 0, 1, 1, 4, 1, 8], [1, 0, 1, 3, 0, 1, 1, 1, 0, 0, 2, 1], [1, 1, 1, 14, 0, 1, 0, 0, 1, 4, 2, 5], [0, 0, 1, 1, 1, 2, 1, 6, 1, 4, 4, 1, 0, 4, 5, 0, 1, 1, 2, 10], [0, 1, 1, 2, 0, 2, 1, 2, 1, 5, 1, 0, 0, 4, 2, 1, 1, 2, 2, 13], [0, 1, 2, 1, 0, 0, 2, 0, 1, 4, 2, 6], [0, 0, 1, 0, 0, 1, 0, 0, 1, 4, 2, 7, 1, 1, 2, 0, 0, 0, 2, 0], [1, 1, 1, 2, 0, 2, 5, 2, 0, 4, 2, 1], [0, 1, 1, 2, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1], [0, 1, 2, 1, 0, 3, 1, 2, 0, 5, 2, 1]]

    onehot = to_onehot(ini_xs)
    decoded = decode_onehot(onehot)

    assert decoded == ini_xs

    junk_pipeline = np.random.randint(0, 20, (110))
    new_pipeline = polish(junk_pipeline)
    # pdb.set_trace()

    assert is_metapipeline_valid(new_pipeline, 0)


if __name__ == "__main__":
    # test()
    # print(polish_4x([9, 0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 13, 0, 6, 1, 2]))
    print(polish_asym([9, 0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 13, 0, 6, 1, 2]))