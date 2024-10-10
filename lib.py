import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

def get_rgb_and_depth_image() -> (np.ndarray, np.ndarray):

    '''
        Functions is looking for RealSense camera and returns color and depth image.
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
                print("Camera found")
                break
        if not found_rgb:
            print("No RGB camera found")
            return None, None
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        colorizer = rs.colorizer()
        align_to = rs.stream.color
        align = rs.align(align_to)


        color_image = None
        depth_image = None
        print("Getting data...")

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

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.033), cv2.COLORMAP_JET)

                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

            break

        pipeline.stop()
        return color_image, depth_image

    except Exception as e:
        print(e)
        return None, None
    

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
        device_product_line = str(device.get_info(rs.camera_info.product_line))
    
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

        depth_image = None
        color_image = None
        
        for i in range(50):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            colorized = colorizer.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        vtx = np.asanyarray(points.get_vertices())
        tex = np.asanyarray(points.get_texture_coordinates())

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(vtx.view(np.float32).reshape(-1, 3))
        point_cloud.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)

        pipeline.stop()
        return point_cloud
    
    except Exception as e:
        print(e)
        return None