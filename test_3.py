import cv2
import lib
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#from sklearn.metrics import jaccard_score

def IoU(image1, image2):
    # Convert images to numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    # Ensure the images have the same shape
    assert image1.shape == image2.shape, "Images must have the same shape to compute IoU"
    
    # Calculate intersection and union
    intersection = np.logical_and(image1, image2)
    intersection = np.sum(intersection)
    union = np.logical_or(image1, image2)
    union = np.sum(union)
    
    # Compute IoU
    iou = intersection / union
    return iou

def test_3():   

    image = Image.open("./example_3.jpg")

    #deeplabv3

    start = time.time()
    segmented_image1,labels1,masks1 = lib.use_DeepLabV3(image, model = 'apple', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "deeplabv3-apple")
    print("Labels: ", labels1)

    start = time.time()
    segmented_image2,labels2,masks2 = lib.use_DeepLabV3(image, model = 'apple-xx', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "deeplabv3-apple-xx")
    print("Labels: ", labels2)

    start = time.time()
    segmented_image3,labels3,masks3 = lib.use_DeepLabV3(image, model = 'google', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "deeplabv3-google")
    print("Labels: ", labels3)


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

    # Calculate percentage of each label in the segmented images
    def calculate_label_percentage(segmented_image, labels):
        label_percentages = {}
        total_pixels = segmented_image.size
        for label in labels:
            label_mask = (segmented_image == label)
            label_pixels = np.sum(label_mask)
            label_percentages[label] = (label_pixels / total_pixels) * 100
        return label_percentages

    label_percentages1 = calculate_label_percentage(segmented_image1, labels1)
    label_percentages2 = calculate_label_percentage(segmented_image2, labels2)
    label_percentages3 = calculate_label_percentage(segmented_image3, labels3)

    print("Label percentages for deeplabv3-apple: ", label_percentages1)
    print("Label percentages for deeplabv3-apple-xx: ", label_percentages2)
    print("Label percentages for deeplabv3-google: ", label_percentages3)




    # wyniki zgodności etykiet poszczegołnych modeli:

    common_labels_1_2 = set(labels1).intersection(labels2)
    common_labels_1_3 = set(labels1).intersection(labels3)
    common_labels_2_3 = set(labels2).intersection(labels3)
    print("Common labels between deeplabv3-apple and deeplabv3-apple-xx: ", common_labels_1_2, len(common_labels_1_2)/len(labels1))
    #print(f"Iou: {IoU(masks1, masks2)}")
    #print(jaccard_score(np.array(segmented_image1).flatten()), np.array(segmented_image2).flatten())
    print("Common labels between deeplabv3-apple and deeplabv3-google: ", common_labels_1_3, len(common_labels_1_3)/len(labels1))
    #print(f"Iou: {IoU(segmented_image1, segmented_image3)}")
    print("Common labels between deeplabv3-apple-xx and deeplabv3-google: ", common_labels_2_3, len(common_labels_2_3)/len(labels2))
    #print(f"Iou: {IoU(segmented_image2, segmented_image3)}")



if __name__ == "__main__":
    test_3()