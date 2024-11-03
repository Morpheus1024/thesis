import cv2
import lib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import open3d as o3d


# color_image, depth_image = lib.get_rgb_and_depth_image()
# points_cloud = lib.get_point_cloud()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points_cloud[:,0], points_cloud[:,1], points_cloud[:,2], s=10)

# ax.set_title("Chmura punktów")

# plt.show()

seed_list = [(100, 100), (200, 200), (300, 300), (400, 400)]

#image = cv2.imread("./image.jpg")
image = Image.open("./image.jpg")


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

#DeepLabV3_xx
semantic_image, _, _ = lib.use_DeepLabV3_by_Google(image)
plt.imsave("./seg_image.jpg", semantic_image)

print("## DONE ##")



