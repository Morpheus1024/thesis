import lib
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_2():
    camera_present = lib.check_if_realsense_is_present()

    if not camera_present:
        print("Camera not found")
        return
    
    config_time = time.time()
    camera_config = lib.get_realsense_camera_config()
    lib.log_execution_time(config_time, "get_realsense_camera_config", print_log=True)
    start_time = time.time()
    color_image, depth_image = lib.get_rgb_and_depth_image_from_realsense()
    lib.log_execution_time(start_time, "get_rgb_and_depth_image_from_realsense", print_log=True)
    
    depth_max = max(depth_image.flatten())
    print("Depth at center: ", depth_image[depth_image.shape[0]//2,depth_image.shape[1]//2])
    depth_image1 = depth_max - depth_image

    BEiT_time = time.time()
    depth_image_AI1,result1 = lib.use_BEiT_depth(color_image)
    lib.log_execution_time(BEiT_time, "use_BEiT_depth", print_log=True)
    #print("BEiT max result: ",torch.max(result1['predicted_depth']).item())
    print("BEiT center result: ",result1['predicted_depth'][result1['predicted_depth'].shape[0]//2,result1['predicted_depth'].shape[1]//2].item())



    MiDaS_Hybrid_time = time.time()
    depth_image_AI2,result2 = lib.use_MiDaS_Hybrid(color_image)
    lib.log_execution_time(MiDaS_Hybrid_time, "use_MiDaS_Hybrid", print_log=True)
    #print("MiDaS_Hybrid max result: ",torch.max(result2['predicted_depth']).item())
    print("MiDaS_Hybrid center result: ",result2['predicted_depth'][result2['predicted_depth'].shape[0]//2,result2['predicted_depth'].shape[1]//2].item())

    Depth_anything_time = time.time()
    depth_image_AI3,result3 = lib.use_Depth_Anything(color_image)
    lib.log_execution_time(Depth_anything_time, "use_Depth_Anything", print_log=True)
    #print("MiDaS_Hybrid max result: ",torch.max(result3['predicted_depth']).item())
    print("Depth Anything small center result: ",result3['predicted_depth'][result3['predicted_depth'].shape[0]//2,result3['predicted_depth'].shape[1]//2].item())

    Depth_anything_time = time.time()
    depth_image_AI4,result4 = lib.use_Depth_Anything(color_image, model = "large")
    lib.log_execution_time(Depth_anything_time, "use_Depth_Anything", print_log=True)
    #print("MiDaS_Hybrid max result: ",torch.max(result4['predicted_depth']).item())
    print("Depth Anything large center result: ",result4['predicted_depth'][result4['predicted_depth'].shape[0]//2,result4['predicted_depth'].shape[1]//2].item())

    Depth_anything_time = time.time()
    depth_image_AI5,result5 = lib.use_Depth_Anything(color_image, model = "base")
    lib.log_execution_time(Depth_anything_time, "use_Depth_Anything", print_log=True)
    #print("MiDaS_Hybrid max result: ",torch.max(result5['predicted_depth']).item())
    print("Depth Anything base center result: ",result5['predicted_depth'][result5['predicted_depth'].shape[0]//2,result5['predicted_depth'].shape[1]//2].item())



    with open("./testy/depth_logs.txt", "a") as f:
        f.write("Depth at center: " + str(depth_image[depth_image.shape[0]//2,depth_image.shape[1]//2]) + "\n")
        f.write("BEiT center result: " + str(result1['predicted_depth'][result1['predicted_depth'].shape[0]//2,result1['predicted_depth'].shape[1]//2].item()) + "\n")
        f.write("MiDaS_Hybrid center result: " + str(result2['predicted_depth'][result2['predicted_depth'].shape[0]//2,result2['predicted_depth'].shape[1]//2].item()) + "\n")
        f.write("Depth Anything small center result: " + str(result3['predicted_depth'][result3['predicted_depth'].shape[0]//2,result3['predicted_depth'].shape[1]//2].item()) + "\n")
        f.write("Depth Anything large center result: " + str(result4['predicted_depth'][result4['predicted_depth'].shape[0]//2,result4['predicted_depth'].shape[1]//2].item()) + "\n")
        f.write("Depth Anything base center result: " + str(result5['predicted_depth'][result5['predicted_depth'].shape[0]//2,result5['predicted_depth'].shape[1]//2].item()) + "\n")
        f.write("\n")
        f.write("\n")



    cmap = 'jet'



    print()


if __name__ == "__main__":
    test_2()

