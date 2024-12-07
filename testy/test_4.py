
import lib
import time
import numpy as np
from PIL import Image
# wykaz czasu wykoanania danych funkcji

def test4_rs_time():
    camera = lib.check_if_realsense_is_present()

    if not camera: 
        print("No camera")
        return 

    start_time = time.time()
    rgb_image, depth_image = lib.get_rgb_and_depth_image_from_realsense()
    lib.rs_log_execution_time(time.time()-start_time, "rs" ,True)        

def test_4_map_creation():

    #Load image
    image = Image.open("./example_3.png")

    segmented_image, _, _  = lib.use_BEiT_semantic(image, add_legend=False) # here can by used any other semantic segmentation model aviable in library
    depth_image, _ = lib.use_MiDaS_Hybrid(image) # here can by used any other depth estimation model aviable in library

    start_time = time.time()
    semantic_3d_map = lib.create_semantic_3D_map(segmented_image, depth_image, fx = 385, fy = 385, print_logs=False, save_ply=True)
    lib.rs_log_execution_time(time.time()-start_time, "semantic_3d_map", True)

    print("DONE")


def test4():
    time_log_file = open("execution_time_log.txt", "r")

    time_list = []

    for line in time_log_file:
        name = line.split()[0]
        time = float(line.split()[1])

        #print(name, time)
        log_dict = dict(name =name, time = time)
        time_list.append(log_dict)
    
    time_log_file.close()

    #merge dict with the same name, and calculate average time
    time_dict = {}
    for log in time_list:
        if log['name'] in time_dict:
            time_dict[log['name']].append(log['time'])
        else:
            time_dict[log['name']] = [log['time']]

    for key in time_dict:
        time_dict[key] = np.mean(time_dict[key])
        time_dict[key] = round(time_dict[key], 4)

    for key in time_dict:
        print(key, time_dict[key])





if __name__ == '__main__':
    # test4()
    # test4_rs_time()
    test_4_map_creation()
