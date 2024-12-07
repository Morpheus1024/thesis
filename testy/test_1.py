import lib
import matplotlib.pyplot as plt

def read_depth_measurement(depth_image):
    return depth_image[240, 320]

def test_1():

    camera_present = lib.check_if_realsense_is_present()
    
    if not camera_present:
        print("Camera not found")
        return
    
    print("Please enter the distance in cm to wait for the camera to focus on:")
    wait_30cm = input("30 cm: ")
    _, depth_image30 = lib.get_rgb_and_depth_image_from_realsense()
    print(read_depth_measurement(depth_image30))

    print("Please enter the distance in cm to wait for the camera to focus on:")
    wait_50cm = input("50 cm: ")
    _, depth_image50 = lib.get_rgb_and_depth_image_from_realsense()
    print(read_depth_measurement(depth_image50))

    print("Please enter the distance in cm to wait for the camera to focus on:")
    wait_70cm = input("70 cm: ")
    _, depth_image70 = lib.get_rgb_and_depth_image_from_realsense()
    print(read_depth_measurement(depth_image70))

    print("Please enter the distance in cm to wait for the camera to focus on:")
    wait_100cm = input("100 cm: ")
    _, depth_image100 = lib.get_rgb_and_depth_image_from_realsense()
    print(read_depth_measurement(depth_image100))


    fig = plt.figure(figsize=(20, 5))
    fig.add_subplot(1,4,1)
    plt.imshow(depth_image30)
    plt.title("Depth Image at 30 cm")
    fig.add_subplot(1,4,2)
    plt.imshow(depth_image50)
    plt.title("Depth Image at 50 cm")
    fig.add_subplot(1,4,3)
    plt.imshow(depth_image70)
    plt.title("Depth Image at 70 cm")
    fig.add_subplot(1,4,4)
    plt.imshow(depth_image100)
    plt.title("Depth Image at 100 cm")

    plt.show()

    plt.imsave("./testy/depth_image302.png", depth_image30)
    plt.imsave("./testy/depth_image502.png", depth_image50)
    plt.imsave("./testy/depth_image702.png", depth_image70)
    plt.imsave("./testy/depth_image1002.png", depth_image100)







if __name__ == "__main__":
    test_1()