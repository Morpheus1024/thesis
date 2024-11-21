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
    segmented_image1,labels1,masks1 = lib.use_OneFormer(image, model = 'large', dataset="ade20k", test_colors = True)
    end = time.time() - start
    print("Time: ", end)
    lib.log_execution_time(end, "use_oneformer-large-ade20k")
    print("Labels: ", labels1)
    save_label("oneformer-large-ade20k", labels1)

    start = time.time()
    segmented_image2,labels2,masks2 = lib.use_OneFormer(image, model = 'large', dataset="coco", test_colors = True)
    end = time.time() - start
    print("Time: ", end)
    lib.log_execution_time(end, "use_oneformer-large-coco")
    print("Labels: ", labels2)
    save_label("oneformer-large-coco", labels2)

    start = time.time()
    segmented_image3,labels3,masks3 = lib.use_OneFormer(image, model = 'large', dataset="cityscapes", test_colors = True)
    end = time.time() - start
    print("Time: ", end)
    lib.log_execution_time(end, "use_oneformer-large-cityscapes")
    print("Labels: ", labels3)
    save_label("oneformer-large-cityscapes", labels3)

    start = time.time()
    segmented_image4,labels4,masks4 = lib.use_OneFormer(image, model = 'tiny', dataset="ade20k", test_colors = True)
    end = time.time() - start
    print("Time: ", end)
    lib.log_execution_time(end, "use_oneformer-tiny-ade20k")
    print("Labels: ", labels4)
    save_label("oneformer-tiny-ade20k", labels4)


    # Display segmented images
    plt.figure(figsize=(20, 5))

    plt.subplot(2, 5, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Obraz źródłowy")

    plt.subplot(2, 5, 2)
    plt.imshow(segmented_image1)
    plt.axis('off')
    plt.title("OneFormer - Large - ADE20K")
    plt.xlabel(f"Labels: {', '.join(labels1)}")

    plt.subplot(2, 5, 3)
    plt.imshow(segmented_image2)
    plt.axis('off')
    plt.title("OneFormer - Large - COCO")
    plt.xlabel(f"Labels: {', '.join(labels2)}")

    plt.subplot(2, 5, 4)
    plt.imshow(segmented_image3)
    plt.axis('off')
    plt.title("OneFormer - Large - Cityscapes")
    plt.xlabel(f"Labels: {', '.join(labels3)}")

    plt.subplot(2, 5, 5)
    plt.imshow(segmented_image4)
    plt.axis('off')
    plt.title("OneFormer - Tiny - ADE20K")
    plt.xlabel(f"Labels: {', '.join(labels4)}")

    # display difference between segmented images
    # segmented_image1 = np.array(segmented_image1)
    # segmented_image2 = np.array(segmented_image2)
    # segmented_image3 = np.array(segmented_image3)
    # segmented_image4 = np.array(segmented_image4)

    plt.show()

    label_percentages1 = calculate_label_percentage(segmented_image1, labels1)
    label_percentages2 = calculate_label_percentage(segmented_image2, labels2)
    label_percentages3 = calculate_label_percentage(segmented_image3, labels3)
    label_percentages4 = calculate_label_percentage(segmented_image4, labels4)

    print("Label percentages for oneformer_large_ade20k: ", label_percentages1)
    print("Label percentages for oneformer_large_coco: ", label_percentages2)
    print("Label percentages for oneformer_large_cityscapes: ", label_percentages3)
    print("Label percentages for oneformer_tiny_ade20k: ", label_percentages4)

    save_comparisone = True
    print()
    print("Comparing segmentations:")
    print("OneFormer - Large - ADE20K vs OneFormer - Large - COCO")
    compare_segmentations(masks1, labels1, masks2, labels2, model1 = "oneformer-large-ade20k", model2 = "oneformer-large-coco", save_comparison=save_comparisone)
    print("OneFormer - Large - ADE20K vs OneFormer - Large - Cityscapes")
    compare_segmentations(masks1, labels1, masks3, labels3, model1 = "oneformer-large-ade20k", model2 = "oneformer-large-cityscapes", save_comparison=save_comparisone)
    print("OneFormer - Large - ADE20K vs OneFormer - Tiny - ADE20K")
    compare_segmentations(masks1, labels1, masks4, labels4, model1="oneformer-large-ade20k", model2="oneformer-tiny-ade20k", save_comparison=save_comparisone)
    print("OneFormer - Large - COCO vs OneFormer - Large - Cityscapes")
    compare_segmentations(masks2, labels2, masks3, labels3, model1="oneformer-large-coco", model2="oneformer-large-cityscapes", save_comparison=save_comparisone)
    print("OneFormer - Large - COCO vs OneFormer - Tiny - ADE20K")
    compare_segmentations(masks2, labels2, masks4, labels4, model1="oneformer-large-coco", model2="oneformer-tiny-ade20k", save_comparison=save_comparisone)
    print("OneFormer - Large - Cityscapes vs OneFormer - Tiny - ADE20K")
    compare_segmentations(masks3, labels3, masks4, labels4, model1="oneformer-large-cityscapes", model2="oneformer-tiny-ade20k", save_comparison=save_comparisone)



if __name__ == "__main__":
    test_3()