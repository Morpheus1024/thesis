import cv2
import lib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d


print("## START ##")
#print(lib.check_if_realsense_is_present(print_logs=False))

# color_image, depth_image, camera_params= lib.get_rgb_and_depth_image_from_realsense(print_logs=True)

# plt.imsave("color.png", color_image)
# plt.imsave("depth.png", depth_image)

# points_cloud = lib.get_point_cloud()

#plt.imsave("images/color_image.png", color_image)
#plt.imsave("images/depth_image.png", depth_image)
#plt.imsave("images/depth_from_model.png", depth_from_model)
#print(camera_params)

# print(camera_params.fx, camera_params.fy)
#print("segmenting image")
#color_image, _, _ = lib.use_OneFormer(color_image)
#plt.imsave("images/seg_image.png", color_image)

#point_cloud = lib.create_semantic_3D_map(color_image, depth_image, fx = camera_params.fx, fy = camera_params.fy, print_logs=True, save_ply=True)




# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points_cloud[:,0], points_cloud[:,1], points_cloud[:,2], s=10)

# ax.set_title("Chmura punktów")

# plt.show()

#seed_list = [(100, 100), (200, 200), (300, 300), (400, 400)]



image = Image.open("./color_rs.png")
image = image.convert("RGB")
#image.save("./color_image.jpg")
depth,_ = lib.use_BEiT_depth(image)
plt.imshow(depth, cmap='gray')
plt.imsave("./depth_image.jpg", depth)
plt.show()
#plt.
# image = cv2.imread("./seg_image.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(image.shape)

# depth = cv2.imread("./depth_rs.png")
# depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
# print(len(depth.shape))
# depth = -depth

# ply = lib.create_semantic_3D_map(image, depth, fx = 385, fy = 385, print_logs=True, save_ply=True, z_scale=1)


# SegFormer
#semantic_image, _, _ = lib.use_SegFormer(image)
#plt.imsave("./seg_image.jpg", semantic_image)

#OneFormer
# semantic_image, _,_ = lib.use_OneFormer(image)use_mask2former

# plt.imsave("./seg_image.jpg", semantic_image)

#MiDaS
# depth_image = lib.use_MiDaS(image)
# #plt.imsave("./depth_image.jpg", depth_image)
# plt.imshow(depth_image)
# plt.show()


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
# masked_image_with_legend, panoptic_labels, panoptic_masks = lib.use_ResNet50(image, add_legend=True)

# # print(panoptic_labels)
# # plt.imsave("./seg_image.jpg", masked_image_with_legend)
# plt.imshow(masked_image_with_legend)
# plt.show()
#plt.imsave("./panoptic_labels.jpg", panoptic_labels)
#plt.imsave("./panoptic_masks.jpg", panoptic_masks)

# depth, depth_result = lib.EVP(image)
# plt.imshow(depth)
# plt.show()

# masked_image_large, labels, masks = lib.use_mask2former(image, add_legend=False)
# print(labels)

# plt.imsave("./seg_image.png", masked_image_large)
#masked_image_base, results, labels = lib.use_OneFormer(image, add_legend=True, dataset='cityscapes', model='large')#, model = 'small')
#masked_image_base,_,_ = lib.use_mask2former(image, add_legend=True)
# masked_image_base,_,_ = lib.use_maskformer(image, add_legend=True, model = 'large')
#masked_image_base.save("./_image.png")
# depth_image,_ = lib.use_MiDaS_Hybrid(image )
#masked_image_base.save("./masked_image.png")
#plt.imshow(masked_image_base)
# plt.imsave("./seg_image.jpg", masked_image_base)
#plt.show()

#segmented_image,_,_ = lib.use_maskformer(image, add_legend=False, model = 'large')

# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# axs[0].imshow(masked_image_large)
# axs[0].set_title('Masked Image Large')

# axs[1].imshow(masked_image_base)
# axs[1].set_title('Masked Image Base')

# masked_image_base = np.array(masked_image_base)
# image = np.array(image)
# image = np.array(image)
# shape = image.shape
# cloude_point = lib.create_semantic_3D_map(image,depth_image, fx = int(shape[1]/2), fy = int(shape[0]/2), print_logs=False, save_ply=False)

# o3d.visualization.draw_geometries([cloude_point])




print("## DONE ##")



