
import pdb
import numpy as np


def wave20_angle_to_elevation_z(angle):
    if angle == 15:
        return 0.266*2
    elif angle == 30:
        return 0.573*2
    elif angle == 45:
        return 0.994*2
    elif angle == 60:
        return 1.72*2
    else:
        raise NotImplementedError

def gen_wave(nt=50, x_offset=0, y_offset=0):
    size = 1000
    # Create x and y coordinates ranging from -1 to 1 (or any other range you prefer)
    x = np.linspace(-nt, nt, size)
    y = np.linspace(-nt, nt, size)

    # Create the meshgrid
    X, Y = np.meshgrid(x, y)

    # Calculate z = cos(sqrt(x^2 + y^2))
    Z = -np.cos(np.sqrt((X-x_offset)**2 + (Y-y_offset)**2))
    Z = Z/2 + 0.5  # Normalize the data between 0 and 1
    assert np.all(Z >= 0), "Height field data must be normalized between 0 and 1"
    return Z


def gen_uniform_discrete():

    # Create a 2D NumPy array filled with zeros
    size = 1000
    height = size
    width = size
    array = np.zeros((height, width), dtype=int)

    # Define the number of rectangles, border thickness, and the gap between borders
    num_rectangles = 24
    border_thickness = 10
    gap = 10  # Space between each two borders

    # Loop to draw each rectangle, starting from the outermost
    for i in range(num_rectangles):
        # Calculate the start and end coordinates for the current rectangle
        top = i * (border_thickness + gap)
        bottom = height - i * (border_thickness + gap)
        left = i * (border_thickness + gap)
        right = width - i * (border_thickness + gap)
        
        # Draw the top and bottom borders of the current rectangle
        array[top:top+border_thickness, left:right] = 1
        array[bottom-border_thickness:bottom, left:right] = 1
        
        # Draw the left and right borders of the current rectangle
        array[top:bottom, left:left+border_thickness] = 1
        array[top:bottom, right-border_thickness:right] = 1

    return array

def gen_uniform_gaps(gap_width=0.4):

    radius_x = 20

    if gap_width == 0.4:
        num_rectangles = 16 #24
        border_thickness = 10 # GAP WIDTH
        size = 1000
    elif gap_width == 0.2:
        num_rectangles = 32
        border_thickness = 5
        size = 1000
    elif gap_width == 0.1:
        num_rectangles = 18
        border_thickness = 2
        size = 800
    else:
        raise ValueError("Invalid gap width. Please choose 0.2 or 0.4.")
    
    x_spacing = 2 * radius_x / (size - 1)
    # print("Gap width: ", border_thickness*x_spacing)

    height = size
    width = size
    array = np.zeros((height, width), dtype=int)

    gap = 20  # Space between each two borders

    # Loop to draw each rectangle, starting from the outermost
    for i in range(num_rectangles):
        # Calculate the start and end coordinates for the current rectangle
        top = i * (border_thickness + gap)
        bottom = height - i * (border_thickness + gap)
        left = i * (border_thickness + gap)
        right = width - i * (border_thickness + gap)
        
        # Draw the top and bottom borders of the current rectangle
        array[top:top+border_thickness, left:right] = 1
        array[bottom-border_thickness:bottom, left:right] = 1
        
        # Draw the left and right borders of the current rectangle
        array[top:bottom, left:left+border_thickness] = 1
        array[top:bottom, right-border_thickness:right] = 1

    return -array + 1


def gen_grid():

    # Define the size of the grid
    rows = 1000  # number of rows
    cols = 1000  # number of columns
    spacing = 6  # spacing between grid lines
    thickness = 2  # thickness of the grid lines

    # Create a 2D array filled with zeros
    grid = np.zeros((rows, cols), dtype=int)

    # Place 1s to form thicker grid lines
    for i in range(thickness):
        grid[i::spacing, :] = 1  # horizontal lines
        grid[:, i::spacing] = 1  # vertical lines

    return grid


class Terrain():
    def __init__(self) -> None:
        self.reset_terrain_type = None
        self.reset_terrain_param = None

    def reset_terrain(self, xml_compiler, reset_terrain_type, reset_terrain_param):
        # Reset terrain to new type and parameters
        self.reset_terrain_type = reset_terrain_type
        self.reset_terrain_param = reset_terrain_param

        # Hfield-based terrain
        if reset_terrain_type == "set_slope":
            angle = reset_terrain_param
            elevation_z = wave20_angle_to_elevation_z(angle)
            xml_compiler.reset_hfield(20, 20, elevation_z, 0.2, size=1000)
        elif reset_terrain_type == "set_discrete":
            height = reset_terrain_param
            xml_compiler.reset_hfield(20, 20, height, 0.2, size=1000)
        elif reset_terrain_type == "set_gaps":
            self.gpa_width = reset_terrain_param
            xml_compiler.reset_hfield(20, 20, 1, 0.2, size = 800 if self.gpa_width == 0.1 else 1000)
        elif reset_terrain_type == "set_grid":
            xml_compiler.reset_hfield(20, 20, 1, 0.2, size = 1000)

        # Box-based terrain
        elif reset_terrain_type == "set_bumpy":
            highest_step = reset_terrain_param if reset_terrain_param is not None else 0.15
            terrain_params = {"num_bumps": 200, "height_range": (highest_step/2, highest_step), "width_range": (0.15, 0.25)} # TODO: Add to reset_terrain_param
            xml_compiler.reset_obstacles(terrain_params)

        else:
            raise NotImplementedError("Terrain type not implemented")
        

    def update_hfield(self, model):
        # Update hfield after new model is compiled
        assert self.reset_terrain_type is not None, "Terrain type not set"
        if self.reset_terrain_type == "set_slope":
            model.hfield_data = gen_wave(20, 0, 0).flatten()
        elif self.reset_terrain_type == "set_discrete":
            model.hfield_data = gen_uniform_discrete().flatten()
        elif self.reset_terrain_type == "set_gaps":
            model.hfield_data = gen_uniform_gaps(self.gpa_width).flatten()
        elif self.reset_terrain_type == "set_grid":
            model.hfield_data = gen_grid().flatten()