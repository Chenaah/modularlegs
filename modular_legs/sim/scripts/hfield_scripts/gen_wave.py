


import os
import pdb
from matplotlib import pyplot as plt
import numpy as np
import numpy as np

from modular_legs import LEG_ROOT_DIR

# Define the size of the grid
size = 1000



def gen_wave(nt=50, x_offset=0, y_offset=0):
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




def calculate_slope(hfield_data):
    # Example hfield parameters (replace with your actual values)
    radius_x = 20  # X radius
    radius_y = 20  # Y radius
    elevation_z = 0.994*2  # Maximum elevation
    # elevation_z = 0.266*2  # Maximum elevation
    hfield_array = hfield_data  # Replace with your hfield data array
    nrows, ncols = hfield_array.shape

    # Convert normalized height field data to actual height values
    heights = hfield_array * elevation_z

    # Calculate grid spacing
    x_spacing = 2 * radius_x / (ncols - 1)
    y_spacing = 2 * radius_y / (nrows - 1)

    T = 2*np.pi*size/(2*nt)*x_spacing
    print("---->T: ", T) 

    # Calculate the differences in height
    dx = np.diff(heights, axis=1)  # Difference along the x-axis
    dy = np.diff(heights, axis=0)  # Difference along the y-axis

    # Calculate the gradient (slope)
    slope_x = dx / x_spacing
    slope_y = dy / y_spacing

    # Pad dx and dy to match the size of the original hfield
    slope_x = np.pad(slope_x, ((0, 0), (0, 1)), 'edge')
    slope_y = np.pad(slope_y, ((0, 1), (0, 0)), 'edge')

    # Compute the magnitude of the slope
    slope_magnitude = np.sqrt(slope_x**2 + slope_y**2)

    # Find the maximum slope magnitude
    sharpest_slope = np.max(slope_magnitude)

    print(f"The sharpest slope rate is: {sharpest_slope}")

    # Calculate the angle in radians
    slope_angle_radians = np.arctan(sharpest_slope)

    # Optionally, convert the angle to degrees
    slope_angle_degrees = np.degrees(slope_angle_radians)
    print(f"The sharpest slope angle is: {slope_angle_radians} radians or {slope_angle_degrees} degrees")


if __name__ == "__main__":

    # waves = []
    # for _ in range(10):
    nt = 20
    x_offset = 0
    y_offset = 0
    hfield_data = gen_wave(nt, x_offset, y_offset)
    # waves.append(z)

    # hfield_data = np.sum(waves, axis=0)  # Sum the waves to create the height field

    # plt.imshow(hfield_data, cmap='gray')  # Display the data as an image
    # plt.axis('off')  # Remove axes for saving
    # f = os.path.join(LEG_ROOT_DIR, 'modular_legs', 'sim', 'assets', 'parts', 'wave20.png')
    # plt.savefig(f, bbox_inches='tight', pad_inches=0)  # Save the image


    calculate_slope(hfield_data)


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
    
