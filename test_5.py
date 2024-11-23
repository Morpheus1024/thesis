# Example shows how to create 3D semantic map from an loaded image

import lib
from PIL import Image
import matplotlib.pyplot as plt


def test_5():

    image = Image.open("./example_3.jpg")
    # plt.imshow(image)
    # plt.show()

    segmented_image, labels, _  = lib.use_mask2former(image,model = 'large',add_legend=False) # here can by used any other semantic segmentation model aviable in library
    depth_image, _ = lib.use_BEiT_depth(image) # here can by used any other depth estimation model aviable in library

    print(labels)
    plt.imshow(segmented_image)
    plt.show()

    semantic_3d_map1 = lib.create_semantic_3D_map(segmented_image, depth_image, fx = 100, fy = 100, print_logs=True, save_ply=False)

    lib.view_cloude_point(semantic_3d_map1)

    semantic_3d_map1 = lib.create_semantic_3D_map(segmented_image, depth_image, fx = 200, fy = 200, print_logs=True, save_ply=False)

    lib.view_cloude_point(semantic_3d_map1)

    semantic_3d_map1 = lib.create_semantic_3D_map(segmented_image, depth_image, fx = 385, fy = 385, print_logs=True, save_ply=False)

    lib.view_cloude_point(semantic_3d_map1)

    semantic_3d_map1 = lib.create_semantic_3D_map(segmented_image, depth_image, fx = 500, fy = 500, print_logs=True, save_ply=False)

    lib.view_cloude_point(semantic_3d_map1)

    semantic_3d_map1 = lib.create_semantic_3D_map(segmented_image, depth_image, fx = 600, fy = 600, print_logs=True, save_ply=False)

    lib.view_cloude_point(semantic_3d_map1)




if __name__ == "__main__":
    test_5()
