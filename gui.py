import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2
import open3d as o3d

# Zakładamy, że Twoje funkcje są zaimportowane z lib.py
from lib import (
    get_rgb_and_depth_image_from_realsense, 
    create_semantic_3D_map,
    segment_knn, 
    segment_thresholding,
    segment_sobel
)

class SemanticMapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Semantic 3D Map Generator")

        # UI Elements
        self.frame = tk.Frame(self.root)
        self.frame.pack(pady=10)

        self.load_button = tk.Button(self.frame, text="Capture Image from RealSense", command=self.capture_image)
        self.load_button.grid(row=0, column=0, padx=10)

        self.segment_button = tk.Button(self.frame, text="Segment Image (KNN)", command=self.segment_image_knn)
        self.segment_button.grid(row=0, column=1, padx=10)

        self.generate_map_button = tk.Button(self.frame, text="Generate 3D Map", command=self.generate_3d_map)
        self.generate_map_button.grid(row=0, column=2, padx=10)

        self.quit_button = tk.Button(self.frame, text="Quit", command=self.root.quit)
        self.quit_button.grid(row=0, column=3, padx=10)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Variables to hold images
        self.color_image = None
        self.depth_image = None
        self.segmented_image = None

    def capture_image(self):
        """Capture RGB and Depth image from RealSense camera."""
        color_image, depth_image, _ = get_rgb_and_depth_image_from_realsense()
        if color_image is None or depth_image is None:
            messagebox.showerror("Error", "Could not find RealSense camera.")
            return

        self.color_image = color_image
        self.depth_image = depth_image

        img = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

    def segment_image_knn(self):
        """Segment the captured image using KNN algorithm."""
        if self.color_image is None:
            messagebox.showerror("Error", "No image captured.")
            return

        segmented_image, _, _ = segment_knn(self.color_image, centroids_number=5)
        self.segmented_image = segmented_image

        img = Image.fromarray(segmented_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

    def generate_3d_map(self):
        """Generate a 3D semantic map from segmented image and depth image."""
        if self.segmented_image is None or self.depth_image is None:
            messagebox.showerror("Error", "No segmented image or depth image available.")
            return

        fx, fy = 500.0, 500.0  # Placeholder values for focal length; should be adjusted
        point_cloud = create_semantic_3D_map(self.segmented_image, self.depth_image, fx, fy, z_scale=0.001)

        # Visualize the point cloud using Open3D
        o3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    root = tk.Tk()
    app = SemanticMapApp(root)
    root.mainloop()