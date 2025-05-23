

import os
import pdb
from matplotlib import pyplot as plt
import numpy as np

from modular_legs import LEG_ROOT_DIR


size = 1000

radius_x = 20

def gen_random_discrete():

    map = np.zeros((size, size))

    for _ in range(500):
        x_start = np.random.randint(0, size)
        y_start = np.random.randint(0, size)
        if np.random.rand() < 0.5:
            x_length = np.random.randint(50, 200)
            y_length = 10
        else:
            x_length = 10
            y_length = np.random.randint(50, 200)
        for x in range(x_start, x_start + x_length):
            for y in range(y_start, y_start + y_length):
                if x < size and y < size:
                    map[x][y] = 1
    return map

def gen_uniform_gaps(gap_width=0.4):

    # map = np.zeros((size, size))

    # Create a 2D NumPy array filled with zeros
    

    # Define the number of rectangles, border thickness, and the gap between borders
    
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


def gen_gaps_hfield(gap_width):

    hfield_data = gen_uniform_gaps(gap_width)
    plt.imshow(hfield_data, cmap='gray', interpolation='nearest')  # Display the data as an image
    # hfield_data = np.round(hfield_data)

    plt.axis('off')  # Remove axes for saving
    f = os.path.join(LEG_ROOT_DIR, 'modular_legs', 'sim', 'assets', 'parts', 'gaps.png')
    plt.savefig(f, bbox_inches='tight', pad_inches=0)  # Save the image

if __name__ == "__main__":
    gen_gaps_hfield(0.1)