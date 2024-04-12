# Stages of creating a semantic 3d map of the environment

# 1. data acquisition
# - [x] camera images
# - [x] depth map

# 2. preprocessing
# - [x] calibration - compensation for geometric error of sensor spacing - already done
# - [ ] image segmentation
# - [ ] depth map filtering

# 3. 3D reconstruction
# - [ ] cloud points from depth map
# - [ ] 3D mesh - a mesh representing surfaces is created from the point cloud

# 4. semantic segmentation
# - [ ] classification of each segment
# - [ ] assignment of labels to objects - apply multiple models

# 5. integration
# - [ ] combine data from segmentation, classification and 3D mesh to create a semantic map
# - [ ] visialization of data


import pyrealsense2 as rs
import numpy as np
import cv2

from matplotlib import pyplot as plt

from skimage import io, segmentation, color, feature
 
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:

    while True:

        ## 1. Acquire image and depth data from camera

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap ))

        ## 2. Preprocessing 

        ###  callibration - Intel Realsense is already callibrated
        
        ### image segmentation
        
        #### K-means

        image_2d_k_mean = color_image.reshape(-1, 3)
        image_2d_k_mean = np.float32(image_2d_k_mean)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        K = 3
        attempts = 10
        ret, label, center = cv2.kmeans(image_2d_k_mean, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        segmented_data = center[label.flatten()]
        segmented_image_k_means = segmented_data.reshape((color_image.shape))
        
        #### Color-based segmentation
        image_2d_color = color_image
        low = np.array([0, 0, 0])
        high = np.array([255, 255, 255])
        mask = cv2.inRange(image_2d_color, low, high)
        segmented_image_color = cv2.bitwise_and(color_image, color_image, mask=mask)
        

        #### Contour detection
        image_2d_contour = color_image
        gray = cv2.cvtColor(image_2d_contour, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(thresh, 0, 255)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        segmented_image_contour = cv2.drawContours(color_image, contours, -1, (0, 255, 0), 3)

        ### depth map filtering
        
        





        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
            # Wyświetlanie wyników
        # plt.imshow(segmented_image)
        # plt.show()


finally:

    # Stop streaming
    pipeline.stop()