import cv2
import lib
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_segmentation_comparison(model1, model2, label, IoU):
    with open("./testy/segmentation_comparison.txt", "a") as f:
        f.write(f"{model1} {model2} {label} {IoU}\n")

def compare_segmentations(masks1, labels1, masks2, labels2, model1, model2, save_comparison = False):

    for i in range(len(labels1)):
        for j in range(len(labels2)):
            if labels1[i] == labels2[j]:
                intersection = np.logical_and(masks1[i], masks2[j])
                union = np.logical_or(masks1[i], masks2[j])
                iou_score = np.sum(intersection) / np.sum(union)
                print(f"IoU for {labels1[i]}: {iou_score}")
                if save_comparison:
                    save_segmentation_comparison(model1, model2, labels1[i], iou_score)

def save_label(model_name, labels):
    label = ""
    for l in labels:
        label += l + ", "
    
    with open("./testy/etykiety.txt", "a") as f:
        f.write(f"{model_name}: {label}\n")

    # Calculate percentage of each label in the segmented images
def calculate_label_percentage(segmented_image, labels):
    label_percentages = {}
    total_pixels = segmented_image.size
    for label in labels:
        label_mask = (segmented_image == label)
        label_pixels = np.sum(label_mask)
        label_percentages[label] = (label_pixels / total_pixels) * 100
    return label_percentages
               

def test_3():   

    image = Image.open("./example_3.jpg")

    #deeplabv3

    start = time.time()
    segmented_image1,labels1,masks1 = lib.use_BEiT_semantic(image, model = 'base', test_colors = True)
    end = time.time() - start
    print("Time: ", end)
    lib.log_execution_time(end, "use_BEiT-base")
    print("BEiT-base")
    print("Labels: ", labels1)

    start = time.time()
    segmented_image2,labels2,masks2 = lib.use_BEiT_semantic(image, model = 'large', test_colors = True)
    end = time.time() - start
    print("Time: ", end)
    lib.log_execution_time(end, "use_BEiT-large")
    print("BEiT-large")
    print("Labels: ", labels2)



    # Display segmented images
    plt.figure(figsize=(15, 5))

    plt.subplot(2, 4, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Obraz źródłowy")

    plt.subplot(2, 4, 2)
    plt.imshow(segmented_image1)
    plt.axis('off')
    plt.title("BEiT-base")
    plt.xlabel(f"Labels: {', '.join(labels1)}")

    save_label("BEiT-base", labels1)

    plt.subplot(2, 4, 3)
    plt.imshow(segmented_image2)
    plt.axis('off')
    plt.title("BEiT-large")
    plt.xlabel(f"Labels: {', '.join(labels2)}")

    save_label("BEiT-large", labels2)

    # display difference between segmented images
    segmented_image1 = np.array(segmented_image1)
    segmented_image2 = np.array(segmented_image2)

    diff1 = cv2.absdiff(segmented_image1, segmented_image2)

    plt.subplot(2, 4, 6)
    plt.imshow(diff1)
    plt.axis('off')
    plt.title("Różnice między segmentacjami")

    #plt.show()

    label_percentages1 = calculate_label_percentage(segmented_image1, labels1)
    label_percentages2 = calculate_label_percentage(segmented_image2, labels2)

    print("Label percentages for deeplabv3-apple: ", label_percentages1)
    print("Label percentages for deeplabv3-apple-xx: ", label_percentages2)

    print()
    print("Comparing segmentations:")
    print("BEiT-base vs BEiT-large")
    compare_segmentations(masks1, labels1, masks2, labels2, model1 = "BEiT-base", model2 = "BEiT-large", save_comparison=False)



if __name__ == "__main__":
    test_3()