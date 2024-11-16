import lib
import numpy as np
import matplotlib.pyplot as plt


def example_2():

    camera_presense = lib.check_if_realsense_is_present(print_logs= False)

    if not camera_presense:
        print("Realsense camera is not present")
        return

    realsense_config = lib.get_realsense_camera_config()
    print(f"model: {realsense_config.model}")
    print(f"fx: {realsense_config.fx}, fy:{realsense_config.fy}")
    print(f"{realsense_config.width}x{realsense_config.height}")

    color_image, _ = lib.get_rgb_and_depth_image_from_realsense()

    segmented_image,labels,masks  = lib.use_BEiT_semantic(color_image, add_legend=True)

    depth_image_AI,_ = lib.use_BEiT_depth(color_image)

    print(f"Number of objects detected: {len(labels)}")
    print(f"Labels: {labels}")

    fig = plt.figure(figsize=(18,7))
    fig.add_subplot(1,2,1)
    plt.imshow(segmented_image)
    fig.add_subplot(1,2,2)
    plt.imshow(depth_image_AI, cmap='gray')
    plt.show()


if __name__ == "__main__":
    example_2()