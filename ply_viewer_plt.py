import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie chmury punktów
pcd = o3d.io.read_point_cloud("semantic_map.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Utworzenie wykresu
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Wyświetlanie chmury punktów z kolorami
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)

ax.set_title("Chmura punktów")

plt.show()
