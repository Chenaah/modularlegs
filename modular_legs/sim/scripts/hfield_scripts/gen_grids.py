import os
import pdb
from matplotlib import pyplot as plt
import numpy as np

from modular_legs import LEG_ROOT_DIR

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

if __name__ == "__main__":
    grid = gen_grid()
    print(grid)
    plt.imshow(grid, cmap='gray', interpolation='nearest')  # Display the data as an image
    plt.axis('off')  # Remove axes for saving
    f = os.path.join(LEG_ROOT_DIR, 'modular_legs', 'sim', 'assets', 'parts', 'grid.png')
    plt.savefig(f, bbox_inches='tight', pad_inches=0)  # Save the image