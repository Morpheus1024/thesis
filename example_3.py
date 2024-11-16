# Example shows how to create 3D semantic map from an loaded image

import lib
import numpy as np
from PIL import Image


def example_3():

    #Load image
    image = Image.open("./example_3.png")

    #Segment image and get depth image
    segmented_image, _, _  = lib.use_BEiT_semantic(image, add_legend=False)
    depth_image, _ = lib.use_BEiT_depth(image)

    # create 3D semantic map
    semantic_3d_map = lib.create_semantic_3D_map(segmented_image, depth_image, fx = 385, fy = 385, print_logs=True, save_ply=True)

    # view 3D semantic map
    lib.view_cloude_point(semantic_3d_map)


if __name__ == "__main__":
    example_3()
