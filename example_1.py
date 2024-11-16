


import lib
import matplotlib.pyplot as plt

def example_1():
    """
    Checks for the presence of a RealSense camera, captures RGB and depth images if present,
    and displays them using matplotlib.
    The function performs the following steps:
    1. Checks if a RealSense camera is connected and logs the result.
    2. If a RealSense camera is present, captures an RGB image and a depth image.
    3. Displays the captured images side by side using matplotlib.
    Returns:
        None
    Shows:
        matplotlib plot: RGB image and depth image side by side.
    """



    realsense_presence = lib.check_if_realsense_is_present(print_logs= True)
    print(realsense_presence)

    if realsense_presence:
        color_image, depth_image, realsense_params = lib.get_rgb_and_depth_image_from_realsense()
        print(realsense_params)

        fig = plt.figure(figsize=(14,7))

        fig.add_subplot(1,2,1)
        plt.imshow(color_image)
        fig.add_subplot(1,2,2)
        plt.imshow(depth_image)

        plt.show()


if __name__ == "__main__":
    example_1()
