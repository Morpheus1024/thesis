import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import pyrealsense2 as rs
import matplotlib.pyplot as plt
#import tensorflow_hub as tf_hub
#from transformers import AutoProcessor
import torchvision.transforms as transforms
from transformers import BeitForSemanticSegmentation
from transformers import pipeline, AutoModel, AutoImageProcessor
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def generate_color_palette(n):
    '''
        Function generates a color palette with n colors.
    '''
    palette = []
    for i in range(n):
        palette.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    return palette


def get_rgb_and_depth_image(print_logs = False):

    '''
        Functions is looking for RealSense camera.
        If present, returns color and depth image.
        If camera is not found, function returns None, None

    '''

    try:
        pc = rs.pointcloud()
        points = rs.points()
        pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        #device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                if print_logs: print("Camera found")
                break
        if not found_rgb:
            print("No RGB camera found")
            return None, None
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        profile = pipeline.get_active_profile()
        camera_params = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        colorizer = rs.colorizer()
        align_to = rs.stream.color
        align = rs.align(align_to)


        color_image = None
        depth_image = None
        if print_logs: print("Getting data...")

        while True:
            for i in range(50):

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                colorized = colorizer.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                depth_image = np.asanyarray(frames.get_depth_frame().get_data())

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


            break

        #ply = rs.save_to_ply('cloude_points.ply')
        pipeline.stop()
        return color_image, depth_image, camera_params

    except Exception as e:
        print(e)
        return None, None, None
    

def get_point_cloud() -> o3d.geometry.PointCloud:
    '''
        Function is looking for RealSense camera and returns point cloud.
        If camera is not found, function returns None
    '''

    try:
        pc = rs.pointcloud()
        points = rs.points()
    
        pipeline = rs.pipeline()
        config = rs.config()
    
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        #device_product_line = str(device.get_info(rs.camera_info.product_line))
    
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("No RGB camera found")
            return None
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        colorizer = rs.colorizer()
        align_to = rs.stream.color
        align = rs.align(align_to)
    
        print("Getting data...")

        # depth_image = None
        color_image = None
        
        for _ in range(20):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            #colorized = colorizer.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

        ply = rs.save_to_ply("./output.ply")
        print("done")
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        pipeline.stop()
        return rs.pointcloud()
    
    except Exception as e:
        print(e)
        return None
    
    
def save_ply_file_from_realsense(filename: str):
    '''
        Function is looking for RealSense camera and saves point cloud to .ply file.
    '''
    pc = rs.pointcloud()
    points = rs.points()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    colorizer = rs.colorizer()

    try:
        frames = pipeline.wait_for_frames()
        colorized = colorizer.process(frames)

        ply = rs.save_to_ply(f"{filename}.ply")
        ply.set_option(rs.save_to_ply.option_ply_binary, False)
        ply.set_option(rs.save_to_ply.option_ply_normals, True)

        print("Saving to {filename}.ply...")
        ply.process(colorized)
        print("Done")
    except Exception as e:
        print(e)

    finally:
        pipeline.stop()

def get_realsense_camera_config() -> rs.intrinsics:
    '''
        Function is looking for RealSense camera and returns its configuration.
    '''
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("No RGB camera found")
        return None
    
    pipeline.start(config)
    pipeline.stop()
    profile = pipeline.get_active_profile()
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    
    
    return depth_intrinsics
    
def create_semantic_3D_map(segmented_color_image, depth_image, fx: float, fy: float, z_scale = 0.001, print_logs = False, save_ply = False): #MARK: 3D semantic map
    """
    Create a 3D semantic map from the segmented color image and the depth image.

    Parameters:
    segmented_color_image (numpy.ndarray): Segmented RGB image.
    depth_image (numpy.ndarray): Depth image corresponding to the segmented RGB image.

    Returns:
    open3d.geometry.PointCloud: A 3D point cloud representing the semantic map.
    """

    if segmented_color_image.shape[:2] != depth_image.shape:
        raise ValueError("The segmented color image and the depth image must have the same dimensions.")

    points = []
    colors = []

    cx = segmented_color_image.shape[1] // 2
    cy = segmented_color_image.shape[0] // 2

    if print_logs: print(f"cx: {cx}, cy: {cy}")

    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            z = depth_image[v, u] * z_scale  
            if z == 0:
                continue  
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])
            color = segmented_color_image[v, u, :3] / 255.0  
            colors.append(color)

    print(colors[-1])
    if print_logs: 
        print("Przeanalizowano piksele i naniesiono na chmurę głębi")
        print(f"points len: {len(points)}")
        print(f"colors len: {len(colors)}")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))


    if save_ply: 
        o3d.io.write_point_cloud("semantic_map.ply", point_cloud)
        if print_logs: print("Ply file saved")

    return point_cloud
    
def segment_knn(photo, centroids_number: int):
    '''
        Function takes a photo and returns segmented photo using knn algorythm.
    '''
    # Convert the image to RGB
    #photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    # Reshape the image to be a list of pixels
    pixels = photo.reshape(-1, 3)
    # Convert to float
    pixels = np.float32(pixels)
    # Define criteria, number of clusters and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = centroids_number
    _, labels, (centers) = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert back to 8 bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    # Reshape back to the original image dimension
    segmented_image = segmented_data.reshape((photo.shape))
    return segmented_image, labels, centers

def segment_thresholding(photo, threshold: int):
    '''
        Function takes a photo and returns segmented photo using thresholding algorythm.
    '''
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    _, segmented_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return segmented_image

def segment_local_thresholding(photo):
    '''
        Function takes a photo and returns segmented photo using local thresholding algorythm.
    '''
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    segmented_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return segmented_image

def segment_canny(photo, lower_boundry=100, upper_boundry=200):
    '''
        Function takes a photo and returns segmented photo using canny algorythm.
    '''
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    segmented_image = cv2.Canny(gray, lower_boundry, upper_boundry)
    return segmented_image
    
def segment_sobel(photo, kernel_size=3, gray=True):
    '''
        Function takes a photo and returns segmented photo using sobel algorythm.
    '''
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    if gray:
        gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        segmented_image = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
    else:
        sobelx = cv2.Sobel(photo, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(photo, cv2.CV_64F, 0, 1, ksize=kernel_size)
        segmented_image = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
    return segmented_image

def segment_region_growing(image, seed_point: list, threshold=10):
    height, width, channels = image.shape
    segmented_region = np.zeros((height, width), np.bool_)

    pixels_to_check = [seed_point]
    seed_value = image[seed_point[0], seed_point[1], :]

    while len(pixels_to_check) > 0:
        # Pobieramy obecny piksel do sprawdzenia
        current_pixel = pixels_to_check.pop(0)
        x, y = current_pixel[0], current_pixel[1]

        if segmented_region[x, y]:
            continue

        if np.all(np.abs(image[x, y, :] - seed_value) <= threshold):
            segmented_region[x, y] = True

            if x > 0 and not segmented_region[x - 1, y]:
                pixels_to_check.append((x - 1, y))
            if x < height - 1 and not segmented_region[x + 1, y]:
                pixels_to_check.append((x + 1, y))
            if y > 0 and not segmented_region[x, y - 1]:
                pixels_to_check.append((x, y - 1))
            if y < width - 1 and not segmented_region[x, y + 1]:
                pixels_to_check.append((x, y + 1))

    return segmented_region

def segment_watershed(image): 

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]

    return image

def use_MiDaS(image, model_type = "MiDaS_small"): #DONE
    #model_type = "DPT_ Large"
    # model_type = "DPT_Hybrid"
    model_type = "MiDaS_small"

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid": transform = midas_transforms.dpt_transform
    else: transform = midas_transforms.small_transform

    image = np.array(image)
    input_batch = transform(image).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

def use_EVP(image): #TODO: do naprawy XDD

    depth_estimation = pipeline("feature-extraction", model="MykolaL/evp_depth", trust_remote_code=True)

    results = depth_estimation(image)
    print(results)

def use_ResNet50(image, add_legend = False): #DONE
    image = _cv2_to_pil(image)

    segmentation = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
    results = segmentation(image)

    colors = generate_color_palette(len(results))

    semantic_masks = []
    semantic_labels = []

    for i in range(len(results)):
        semantic_masks.append(results[i]['mask'])
        semantic_labels.append(results[i]['label'])

    masked_image = np.zeros_like(image)
    for i in range(len(results)):
        mask = np.array(semantic_masks[i])
        for j in range(3):
            masked_image[:,:,j] = masked_image[:,:,j] + mask * colors[i][j]
    
    if add_legend: masked_image = _add_legend_next_to_segmented_imega(masked_image, semantic_labels, colors)

    return masked_image, semantic_labels, semantic_masks
    

def use_DeepLabV3(image, add_legend = False): #DONE

    segmentation = pipeline("image-segmentation", model=f"apple/deeplabv3-mobilevit-small")
    results = segmentation(image)

    colors = generate_color_palette(len(results))

    semantic_masks = []
    semantic_labels = []

    for i in range(len(results)):
        semantic_masks.append(results[i]['mask'])
        semantic_labels.append(results[i]['label'])

    masked_image = np.zeros_like(image)
    for i in range(len(results)):
        mask = np.array(semantic_masks[i])
        for j in range(3):
            masked_image[:,:,j] = masked_image[:,:,j] + mask * colors[i][j]
    
    if add_legend: masked_image = _add_legend_next_to_segmented_imega(masked_image, semantic_labels, colors)

    return masked_image, semantic_labels, semantic_masks

def use_DeepLabV3_xx(image, add_legend = False): #DONE
    segmentation = pipeline("image-segmentation", model="apple/deeplabv3-mobilevit-xx-small")
    results = segmentation(image)

    colors = generate_color_palette(len(results))

    semantic_masks = []
    semantic_labels = []

    for i in range(len(results)):
        semantic_masks.append(results[i]['mask'])
        semantic_labels.append(results[i]['label'])

    masked_image = np.zeros_like(image)
    for i in range(len(results)):
        mask = np.array(semantic_masks[i])
        for j in range(3):
            masked_image[:,:,j] = masked_image[:,:,j] + mask * colors[i][j]
    
    if add_legend: masked_image = _add_legend_next_to_segmented_imega(masked_image, semantic_labels, colors)

    return masked_image, semantic_labels, semantic_masks

def use_DeepLabV3_by_Google(image, add_legend = False): #DONE
    segmentation = pipeline("image-segmentation", model="google/deeplabv3_mobilenet_v2_1.0_513")
    results = segmentation(image)

    colors = generate_color_palette(len(results))

    semantic_masks = []
    semantic_labels = []

    for i in range(len(results)):
        semantic_masks.append(results[i]['mask'])
        semantic_labels.append(results[i]['label'])

    masked_image = np.zeros_like(image)
    for i in range(len(results)):
        mask = np.array(semantic_masks[i])
        for j in range(3):
            masked_image[:,:,j] = masked_image[:,:,j] + mask * colors[i][j]
    
    if add_legend: masked_image = _add_legend_next_to_segmented_imega(masked_image, semantic_labels, colors)

    return masked_image, semantic_labels, semantic_masks

def use_OneFormer(image, task = 'semantic', model_size = 'large', add_legend = False): #DONE
    '''
        Function takes an image and returns segmented image using OneFormer model.
        :param image: image to segment
        :param task: 'semantic', 'instance' or 'panoptic'
    '''
    # https://huggingface.co/docs/transformers/main/en/model_doc/oneformer#transformers.OneFormerForUniversalSegmentation

    image = _cv2_to_pil(image)

    segmentation = pipeline("image-segmentation", model=f"shi-labs/oneformer_ade20k_swin_{model_size}")
    results = segmentation(image)

    colors = generate_color_palette(len(results))

    semantic_masks = []
    semantic_labels = []
    for i in range(len(results)):
        semantic_masks.append(results[i]['mask'])
        semantic_labels.append(results[i]['label'])

    masked_image = np.zeros_like(image)
    for i in range(len(results)):
        mask = np.array(semantic_masks[i])
        for j in range(3):
            masked_image[:,:,j] = masked_image[:,:,j] + mask * colors[i][j]

    if add_legend: masked_image = _add_legend_next_to_segmented_imega(masked_image, semantic_labels, colors)

    return masked_image, semantic_labels, semantic_masks

def use_BEiT(image):
    #https://huggingface.co/docs/transformers/main/en/model_doc/beit#transformers.BeitForImageClassification

    '''
    '''

    image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
    model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

    input = image_processor(images=image, return_tensors="pt")
    output = model(**input)

    logits = output.logits
    predicted_segmentation = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    return predicted_segmentation, logits

def use_SegFormer(image, add_legend = False): #DONE
    #https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512

    semantic_segmentation = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512")#, device=0)
    results = semantic_segmentation(image)
    
    colors = generate_color_palette(len(results))

    semantic_masks = []
    semantic_labels = []
    for i in range(len(results)):
        semantic_masks.append(results[i]['mask'])
        semantic_labels.append(results[i]['label'])

    masked_image = np.zeros_like(image)
    for i in range(len(results)):
        mask = np.array(semantic_masks[i])
        for j in range(3):
            masked_image[:,:,j] = masked_image[:,:,j] + mask * colors[i][j]

    if add_legend: masked_image_with_legend = _add_legend_next_to_segmented_imega(masked_image, semantic_labels, colors)

    return masked_image_with_legend, semantic_labels, semantic_masks
    
def _add_legend_next_to_segmented_imega(segmented_image, labels: list, colors: list):


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2

    image = np.array(segmented_image)
    legend_width = 250  # Fixed width for the legend area
    legend_image = np.ones((image.shape[0], legend_width, 3), dtype=np.uint8) * 255

    label_height = image.shape[0] // len(labels)

    for i, label in enumerate(labels): x_position = 10; y_position = (i % (len(labels))) * label_height + 50*font_scale; cv2.putText(legend_image, label, (x_position, y_position), font, font_scale, colors[i], font_thickness, cv2.LINE_AA)

    masked_image_with_legend = np.hstack((segmented_image, legend_image))

    return masked_image_with_legend

def _check_results_pipeline(results):
    for i in range(len(results)):
        label = results[i]['label']
        print(f"Label: {label}")
    for i in range(len(results)):
        mask = results[i]['mask']
        print(f"Mask: {mask}")

def _cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))