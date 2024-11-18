import lib
import cv2
import matplotlib.pyplot as plt

def test_3():

    camera_present = lib.check_if_realsense_is_present()
    
    if not camera_present:
        print("Camera not found")
        return

    camera_config = lib.get_realsense_camera_config()
    color_image, depth_image = lib.get_rgb_and_depth_image_from_realsense()

    gaussian_blur_depth_11= cv2.GaussianBlur(depth_image, (3,3), 0)
    gaussian_blur_depth_12= cv2.GaussianBlur(depth_image, (3,3), 1)
    gaussian_blur_depth_13= cv2.GaussianBlur(depth_image, (3,3), 5)

    gaussian_blur_depth_21= cv2.GaussianBlur(depth_image, (5,5), 0)
    gaussian_blur_depth_22= cv2.GaussianBlur(depth_image, (5,5), 1)
    gaussian_blur_depth_23= cv2.GaussianBlur(depth_image, (5,5), 5)

    gaussian_blur_depth_31= cv2.GaussianBlur(depth_image, (7,7), 0)
    gaussian_blur_depth_32= cv2.GaussianBlur(depth_image, (7,7), 1)
    gaussian_blur_depth_33= cv2.GaussianBlur(depth_image, (7,7), 5)



    

    #display two depth images
    fig = plt.figure(figsize=(20, 17))
    fig.add_subplot(3,3,2)
    plt.imshow(depth_image)
    plt.title("Original Depth Image")
    fig.add_subplot(3,3,4)
    plt.imshow(gaussian_blur_depth_11)
    plt.title("Gaussian Blur 3x3, sigma=0")
    fig.add_subplot(4,3,5)
    plt.imshow(gaussian_blur_depth_12)
    plt.title("Gaussian Blur 3x3, sigma=1")
    fig.add_subplot(3,3,6)
    plt.imshow(gaussian_blur_depth_13)
    plt.title("Gaussian Blur 3x3, sigma=5")
    fig.add_subplot(3,3,7)
    plt.imshow(gaussian_blur_depth_31)
    plt.title("Gaussian Blur 7x7, sigma=0")
    fig.add_subplot(3,3,8)
    plt.imshow(gaussian_blur_depth_32)
    plt.title("Gaussian Blur 7x7, sigma=1")
    fig.add_subplot(3,3,9)
    plt.imshow(gaussian_blur_depth_33)
    plt.title("Gaussian Blur 7x7, sigma=5")

    plt.show()


    point_cloud1 = lib.create_semantic_3D_map(color_image, depth_image, fx = camera_config.fx, fy = camera_config.fy, z_scale=0.001)

    point_cloud2 = lib.create_semantic_3D_map(color_image, gaussian_blur_depth_11, fx = camera_config.fx, fy = camera_config.fy)
    point_cloud3 = lib.create_semantic_3D_map(color_image, gaussian_blur_depth_33, fx = camera_config.fx, fy = camera_config.fy)

    lib.view_cloude_point(point_cloud1)
    lib.view_cloude_point(point_cloud2)
    lib.view_cloude_point(point_cloud3)



if __name__ == '__main__':
    test_3()