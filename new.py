import cv2
import lib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import open3d as o3d


color_image, depth_image, camera_params= lib.get_rgb_and_depth_image(print_logs=True)

#depth_from_model = lib.use_MiDaS(color_image)
# points_cloud = lib.get_point_cloud()

plt.imsave("images/color_image.png", color_image)
plt.imsave("images/depth_image.png", depth_image)
#plt.imsave("images/depth_from_model.png", depth_from_model)
#print(camera_params)

# print(camera_params.fx, camera_params.fy)
print("segmenting image")
color_image, _, _ = lib.use_OneFormer(color_image)
plt.imsave("images/seg_image.png", color_image)

point_cloud = lib.create_semantic_3D_map(color_image, depth_image, fx = camera_params.fx, fy = camera_params.fy, print_logs=True, save_ply=True)




# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points_cloud[:,0], points_cloud[:,1], points_cloud[:,2], s=10)

# ax.set_title("Chmura punktów")

# plt.show()

#seed_list = [(100, 100), (200, 200), (300, 300), (400, 400)]

#image = cv2.imread("./image.jpg")
#image = Image.open("./image.jpg")


# SegFormer
#semantic_image, _, _ = lib.use_SegFormer(image)
#plt.imsave("./seg_image.jpg", semantic_image)

#OneFormer
# semantic_image, _,_ = lib.use_OneFormer(image)
# plt.imsave("./seg_image.jpg", semantic_image)

#MiDaS
# depth_image = lib.use_MiDaS(image)
# plt.imsave("./depth_image.jpg", depth_image)

#EVP
#lib.use_EVP(image)
# plt.imsave("./depth_image.jpg", depth_image)

#DeepLabV3
# semantic_image, _, _ = lib.use_DeepLabV3(image)
# plt.imsave("./seg_image.jpg", semantic_image)

# #DeepLabV3_xx
# semantic_image, _, _ = lib.use_DeepLabV3_xx(image)
# plt.imsave("./seg_image.jpg", semantic_image)

#DeepLabV3_by_google
# semantic_image, _, _ = lib.use_DeepLabV3_by_Google(image)
# plt.imsave("./seg_image.jpg", semantic_image)

#ResNet50   
# semantic_image, _, _ = lib.use_ResNet50(image)
# plt.imsave("./seg_image.jpg", semantic_image)

#print(lib.get_realsense_camera_config())


print("## DONE ##")



