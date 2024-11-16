import lib
import numpy as np
import matplotlib.pyplot as plt
def example_2():


    rs_present = lib.check_if_realsense_is_present(print_logs=False)

    if rs_present:
        
        rs_point_cloude = lib.get_point_cloud_from_realsense()
        print(rs_point_cloude)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(rs_point_cloude[:,0], rs_point_cloude[:,1], rs_point_cloude[:,2], s=10)
        ax.set_title("Point cloud")
        plt.show()







if __name__ == "__main__":
    example_2()