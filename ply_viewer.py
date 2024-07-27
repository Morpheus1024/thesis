import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud("cloude_of_points.ply")
points = np.asarray(pcd.points)

mask_path = "/maski/mask.png"

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], s=10)

ax.set_title("Chmura punktów")

plt.show()