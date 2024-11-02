import cv2
import lib
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

image = cv2.imread("./image.jpg")


# image_seg, logs = lib.use_SegFormer(image)
# print("Wyświetlanie obrazu")
# plt.imshow(image_seg)#, cmap='gray')
# plt.show()
# plt.imsave("image_seg.jpg", image_seg)
# print("Wyświetlanie logów")
# print(logs)

depth = lib.use_MiDaS(image, model_type="DPT_Large")
print("Wyświetlanie obrazu")
plt.imsave('image_seg.jpg',depth)#, cmap='gray')
