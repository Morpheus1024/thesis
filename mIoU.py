import numpy as np
from PIL import Image

model_name = "ResNet_101"
gt_image = Image.open("./map_test/GT_image.png")
gt_image = gt_image.convert("RGB")
gt_image = np.array(gt_image)
segmented_image = Image.open(f"./map_test/{model_name}.png")
segmented_image = segmented_image.convert("RGB")
segmented_image = np.array(segmented_image)

colors = [
    (70,70,70), #1. building
    (70,130,180), #2 sky
    (107,142,35), #3 tree
    (128, 64, 128), #4 road, route
    (244, 35,232), #5 sidewalk, pavement
    (220,20,60), #6 person
    (0,0,142), #7 car
    (220,220,0) #9 signboard
]

C = len(colors)
sum_IoU = 0

for color in colors:
    mask_gt = np.all(gt_image == color, axis=-1)
    mask_segmented = np.all(segmented_image == color, axis=-1)

    # plt.imshow(mask_gt)
    # plt.show()
    # plt.imshow(mask_segmented)
    # plt.show()
    intersection = np.logical_and(mask_gt, mask_segmented)
    union = np.logical_or(mask_gt, mask_segmented)
    iou_score = np.sum(intersection) / np.sum(union)
    sum_IoU += iou_score

print(f"mIoU: {sum_IoU/C}")

with open("mIoU.txt", "a") as f:
    f.write(f"{model_name}: {sum_IoU/C} \n")

