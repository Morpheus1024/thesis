import cv2
import lib
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_segmentation_comparison(model1, model2, label, IoU):
    with open("./testy/segmentation_comparison.txt", "a") as f:
        f.write(f"{model1} {model2} {label} {IoU}\n")

def save_label(model_name, labels):
    label = ""
    for l in labels:
        label += l + ", "
    
    with open("./testy/etykiety.txt", "a") as f:
        f.write(f"{model_name}: {label}\n")

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
    segmented_image1,labels1,masks1 = lib.use_DeepLabV3(image, model = 'apple', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "deeplabv3-apple")
    print("Labels: ", labels1)
    save_label("deeplabv3-apple", labels1)

    start = time.time()
    segmented_image2,labels2,masks2 = lib.use_DeepLabV3(image, model = 'apple-xx', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "deeplabv3-apple-xx")
    print("Labels: ", labels2)
    save_label("deeplabv3-apple-xx", labels2)

    start = time.time()
    segmented_image3,labels3,masks3 = lib.use_DeepLabV3(image, model = 'google', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "deeplabv3-google")
    print("Labels: ", labels3)
    save_label("deeplabv3-google", labels3)


    # Display segmented images
    plt.figure(figsize=(15, 5))

    plt.subplot(2, 4, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Obraz źródłowy")

    plt.subplot(2, 4, 2)
    plt.imshow(segmented_image1)
    plt.axis('off')
    plt.title("DeepLabV3 - Apple")
    plt.xlabel(f"Labels: {', '.join(labels1)}")

    plt.subplot(2, 4, 3)
    plt.imshow(segmented_image2)
    plt.axis('off')
    plt.title("DeepLabV3 - Apple-XX")
    plt.xlabel(f"Labels: {', '.join(labels2)}")

    plt.subplot(2, 4, 4)
    plt.imshow(segmented_image3)
    plt.axis('off')
    plt.title("DeepLabV3 - Google")
    plt.xlabel(f"Labels: {', '.join(labels3)}")

    # display difference between segmented images
    segmented_image1 = np.array(segmented_image1)
    segmented_image2 = np.array(segmented_image2)
    segmented_image3 = np.array(segmented_image3)

    diff1 = cv2.absdiff(segmented_image1, segmented_image2)
    diff2 = cv2.absdiff(segmented_image1, segmented_image3)
    diff3 = cv2.absdiff(segmented_image2, segmented_image3)

    plt.subplot(2, 4, 6)
    plt.imshow(diff1)
    plt.axis('off')
    plt.title("Apple - Apple-XX")

    plt.subplot(2, 4, 7)
    plt.imshow(diff2)
    plt.axis('off')
    plt.title("Apple - Google")

    plt.subplot(2, 4, 8)
    plt.imshow(diff3)
    plt.axis('off')
    plt.title("Apple-XX - Google")

    plt.show()

    label_percentages1 = calculate_label_percentage(segmented_image1, labels1)
    label_percentages2 = calculate_label_percentage(segmented_image2, labels2)
    label_percentages3 = calculate_label_percentage(segmented_image3, labels3)

    print("Label percentages for deeplabv3-apple: ", label_percentages1)
    print("Label percentages for deeplabv3-apple-xx: ", label_percentages2)
    print("Label percentages for deeplabv3-google: ", label_percentages3)


    print()
    print("Comparing segmentations:")
    print("Apple - Apple-XX:")
    compare_segmentations(masks1, labels1, masks2, labels2, model1 = "deeplabv3-apple", model2 = "deeplabv3-apple-xx", save_comparison=False)
    print("Apple - Google:")
    compare_segmentations(masks1, labels1, masks3, labels3, model1 = "deeplabv3-apple", model2 = "deeplabv3-google", save_comparison=False)
    print("Apple-XX - Google:")
    compare_segmentations(masks2, labels2, masks3, labels3, model1="deeplabv3-apple-xx", model2="deeplabv3-google", save_comparison=False)



if __name__ == "__main__":
    test_3()