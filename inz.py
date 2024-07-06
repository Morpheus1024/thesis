import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

#model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
#model = torchvision.models.vgg16(pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'yolov8n', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'yolov8s', pretrained=True)
model
#model = model.cuda()


model.eval()

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
    print("There is no camera")
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

        for i in range(50): #kamera na początku pokazuje obraz w odcieniach niebieskich. Z czasem kolory poprawiają swoją barwę
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
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.033), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))


        #MARK: Segmentacja semantyczna

        preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        color_image_tensor = preprocess(color_image)
        color_image_tensor = color_image_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            color_image_tensor = color_image_tensor.to('cuda')
            model.to('cuda')


        with torch.no_grad():
            output = model(color_image_tensor)['out'][0]
            output_predictions = output.argmax(0)
        
        # Convert the output to a numpy array
        output_predictions = output_predictions.byte().cpu().numpy()

        # Define a color map
        colormap = np.random.randint(0, 255, size=(150, 3), dtype=np.uint8)

        segmented_image = colormap[output_predictions % len(colormap)]

        segmented_image = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        image_to_show = np.hstack((color_image, segmented_image))

        # Wyświetlenie obrazu za pomocą OpenCV
        #break

        # Utworzenie chmury punktów 3D
        intrinsics = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        point_cloud = []
        for y in range(depth_image.shape[0]):
            for x in range(depth_image.shape[1]):
                depth = depth_image[y, x]
                if depth > 0:
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                    class_color = colormap[output_predictions[y, x] % len(colormap)]
                    point_cloud.append((point[0], point[1], point[2], class_color[0], class_color[1], class_color[2]))

        point_cloud = np.array(point_cloud)

        #wizulizacja chmury punktów 3D przez matplotliba

        

        # Zapisanie chmury punktów do pliku PLY
        # with open("3d_map.ply", 'w') as file:
        #     file.write("ply\n")
        #     file.write("format ascii 1.0\n")
        #     file.write(f"element vertex {len(point_cloud)}\n")
        #     file.write("property float x\n")
        #     file.write("property float y\n")
        #     file.write("property float z\n")
        #     file.write("property uchar red\n")
        #     file.write("property uchar green\n")
        #     file.write("property uchar blue\n")
        #     file.write("end_header\n")
        #     for point in point_cloud:
        #         file.write(f"{point[0]} {point[1]} {point[2]} {int(point[3])} {int(point[4])} {int(point[5])}\n")
        # cv2.imshow('Segmented Image', image_to_show)
        # key = cv2.waitKey(0)

        break

finally:
    cv2.destroyAllWindows()
    # Zatrzymanie strumieniowania
    pipeline.stop()