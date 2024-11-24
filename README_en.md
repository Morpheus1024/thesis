# Easy 3D Semantic Map

This repository is related to the thesis “Creating and operating on a 3D semantic map”.

The purpose of the repository is to simplify the process of creating semantic maps using the RealSense D435f camera.

## Environment

The repository was run on Python 3.9.18. For reasons of library compatibility, it is not recommended to use a Python version higher than 3.10.
In order to run, the repository should be cloned.
It is recommended to build a new virtual environment. The necessary libraries used in this project are included in [requirements.txt](/requirements.txt). To install, use the command
```
pip install -r requirements.txt
```

## Examples of use

3 programs have been prepared to demonstrate the operation of the library.

### [example_1.py](/example_1.py).

Example 1 shows how to operate the RealSense D435f camera. It shows how to check for the presence of the camera, get its configuration information, retrieve the color image and depth reading of the scene as well as display the point cloud.

### [example_2.py](/example_2.py).

Example 2 shows how to use the function with image segmentation models and depth estimation on the image obtained from the camera.

### [example_3.py](/example_3.py)

Example 3 shows how a 3D semantic map is created from any loaded image.

### [example_0.py](/example_0.py).

In addition, the so-called example 0 has been added, which is a script that operates exclusively on the functions of the `pyrealsesne2` library for the current preview of the color image as well as depth from the camera.
