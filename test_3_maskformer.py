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

    # Calculate percentage of each label in the segmented images
def calculate_label_percentage(segmented_image, labels):
    label_percentages = {}
    total_pixels = segmented_image.size
    for label in labels:
        label_mask = (segmented_image == label)
        label_pixels = np.sum(label_mask)
        label_percentages[label] = (label_pixels / total_pixels) * 100
    return label_percentages

def save_label(model_name, labels):
    label = ""
    for l in labels:
        label += l + ", "
    
    with open("./testy/etykiety.txt", "a") as f:
        f.write(f"{model_name}: {label}\n")
               

def test_3():   

    image = Image.open("./example_3.jpg")

    #deeplabv3

    start = time.time()
    segmented_image1,labels1,masks1 = lib.use_MaskFormer(image, model = 'tiny', dataset='coco', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "use_maskformer-tiny-coco")
    print("Labels: ", labels1)
    save_label("use_maskformer-tiny-coco", labels1)

    start = time.time()
    segmented_image2,labels2,masks2 = lib.use_MaskFormer(image, model = 'tiny', dataset='ade', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "use_maskformer-tiny-ade")
    print("Labels: ", labels2)
    save_label("use_maskformer-tiny-ade", labels2)

    start = time.time()
    segmented_image3,labels3,masks3 = lib.use_MaskFormer(image, model = 'small', dataset='coco', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "use_maskformer-small-coco")
    print("Labels: ", labels3)
    save_label("use_maskformer-small-coco", labels3)

    start = time.time()
    segmented_image4,labels4,masks4 = lib.use_MaskFormer(image, model = 'small', dataset='ade', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "use_maskformer-small-ade")
    print("Labels: ", labels4)
    save_label("use_maskformer-small-ade", labels4)

    start = time.time()
    segmented_image5,labels5,masks5 = lib.use_MaskFormer(image, model = 'base', dataset='coco', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "use_maskformer-base-coco")
    print("Labels: ", labels5)
    save_label("use_maskformer-base-coco", labels5)

    start = time.time()
    segmented_image6,labels6,masks6 = lib.use_MaskFormer(image, model = 'base', dataset='ade', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "use_maskformer-base-ade")
    print("Labels: ", labels6)
    save_label("use_maskformer-base-ade", labels6)

    start = time.time()
    segmented_image7,labels7,masks7 = lib.use_MaskFormer(image, model = 'large', dataset='coco', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "use_maskformer-large-coco")
    print("Labels: ", labels7)
    save_label("use_maskformer-large-coco", labels7)

    start = time.time()
    segmented_image8,labels8,masks8 = lib.use_MaskFormer(image, model = 'large', dataset='ade', test_colors = True)
    start = time.time() - start
    print("Time: ", start)
    lib.log_execution_time(start, "use_maskformer-large-ade")
    print("Labels: ", labels8)
    save_label("use_maskformer-large-ade", labels8)


    # Display segmented images
    plt.figure(figsize=(15, 5))

    plt.subplot(2, 5, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Obraz źródłowy")

    plt.subplot(2, 5, 2)
    plt.imshow(segmented_image1)
    plt.axis('off')
    plt.title("MaskFormer - Tiny - COCO")
    plt.xlabel(f"Labels: {', '.join(labels1)}")

    plt.subplot(2, 5, 3)
    plt.imshow(segmented_image2)
    plt.axis('off')
    plt.title("MaskFormer - Tiny - ADE")
    plt.xlabel(f"Labels: {', '.join(labels2)}")

    plt.subplot(2, 5, 4)
    plt.imshow(segmented_image3)
    plt.axis('off')
    plt.title("MaskFormer - Small - COCO")
    plt.xlabel(f"Labels: {', '.join(labels3)}")

    plt.subplot(2, 5, 5)
    plt.imshow(segmented_image4)
    plt.axis('off')
    plt.title("MaskFormer - Small - ADE")
    plt.xlabel(f"Labels: {', '.join(labels4)}")

    plt.subplot(2, 5, 6)
    plt.imshow(segmented_image5)
    plt.axis('off')
    plt.title("MaskFormer - Base - COCO")
    plt.xlabel(f"Labels: {', '.join(labels5)}")

    plt.subplot(2, 5, 7)
    plt.imshow(segmented_image6)
    plt.axis('off')
    plt.title("MaskFormer - Base - ADE")
    plt.xlabel(f"Labels: {', '.join(labels6)}")

    plt.subplot(2, 5, 8)
    plt.imshow(segmented_image7)
    plt.axis('off')
    plt.title("MaskFormer - Large - COCO")
    plt.xlabel(f"Labels: {', '.join(labels7)}")

    plt.subplot(2, 5, 9)
    plt.imshow(segmented_image8)
    plt.axis('off')
    plt.title("MaskFormer - Large - ADE")
    plt.xlabel(f"Labels: {', '.join(labels8)}")


    #plt.show()

    save_comparisone = True

    print()
    print("Comparing segmentations:")
    #comere every segmented image with each other
    print("MaskFormer - Tiny - COCO vs MaskFormer - Tiny - ADE")
    compare_segmentations(masks1, labels1, masks2, labels2, model1 = "maskformer-tiny-coco", model2 = "maskformer-tiny-ade", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - COCO vs MaskFormer - Small - COCO")
    compare_segmentations(masks1, labels1, masks3, labels3, model1 = "maskformer-tiny-coco", model2 = "maskformer-small-coco", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - COCO vs MaskFormer - Small - ADE")
    compare_segmentations(masks1, labels1, masks4, labels4, model1 = "maskformer-tiny-coco", model2 = "maskformer-small-ade", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - COCO vs MaskFormer - Base - COCO")
    compare_segmentations(masks1, labels1, masks5, labels5, model1 = "maskformer-tiny-coco", model2 = "maskformer-base-coco", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - COCO vs MaskFormer - Base - ADE")
    compare_segmentations(masks1, labels1, masks6, labels6, model1 = "maskformer-tiny-coco", model2 = "maskformer-base-ade", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - COCO vs MaskFormer - Large - COCO")
    compare_segmentations(masks1, labels1, masks7, labels7, model1 = "maskformer-tiny-coco", model2 = "maskformer-large-coco", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - COCO vs MaskFormer - Large - ADE")
    compare_segmentations(masks1, labels1, masks8, labels8, model1 = "maskformer-tiny-coco", model2 = "maskformer-large-ade", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - ADE vs MaskFormer - Small - COCO")
    compare_segmentations(masks2, labels2, masks3, labels3, model1 = "maskformer-tiny-ade", model2 = "maskformer-small-coco", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - ADE vs MaskFormer - Small - ADE")
    compare_segmentations(masks2, labels2, masks4, labels4, model1 = "maskformer-tiny-ade", model2 = "maskformer-small-ade", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - ADE vs MaskFormer - Base - COCO")
    compare_segmentations(masks2, labels2, masks5, labels5, model1 = "maskformer-tiny-ade", model2 = "maskformer-base-coco", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - ADE vs MaskFormer - Base - ADE")
    compare_segmentations(masks2, labels2, masks6, labels6, model1 = "maskformer-tiny-ade", model2 = "maskformer-base-ade", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - ADE vs MaskFormer - Large - COCO")
    compare_segmentations(masks2, labels2, masks7, labels7, model1 = "maskformer-tiny-ade", model2 = "maskformer-large-coco", save_comparison=save_comparisone)
    print("MaskFormer - Tiny - ADE vs MaskFormer - Large - ADE")
    compare_segmentations(masks2, labels2, masks8, labels8, model1 = "maskformer-tiny-ade", model2 = "maskformer-large-ade", save_comparison=save_comparisone)
    print("MaskFormer - Small - COCO vs MaskFormer - Small - ADE")
    compare_segmentations(masks3, labels3, masks4, labels4, model1 = "maskformer-small-coco", model2 = "maskformer-small-ade", save_comparison=save_comparisone)
    print("MaskFormer - Small - COCO vs MaskFormer - Base - COCO")
    compare_segmentations(masks3, labels3, masks5, labels5, model1 = "maskformer-small-coco", model2 = "maskformer-base-coco", save_comparison=save_comparisone)
    print("MaskFormer - Small - COCO vs MaskFormer - Base - ADE")
    compare_segmentations(masks3, labels3, masks6, labels6, model1 = "maskformer-small-coco", model2 = "maskformer-base-ade", save_comparison=save_comparisone)
    print("MaskFormer - Small - COCO vs MaskFormer - Large - COCO")
    compare_segmentations(masks3, labels3, masks7, labels7, model1 = "maskformer-small-coco", model2 = "maskformer-large-coco", save_comparison=save_comparisone)
    print("MaskFormer - Small - COCO vs MaskFormer - Large - ADE")
    compare_segmentations(masks3, labels3, masks8, labels8, model1 = "maskformer-small-coco", model2 = "maskformer-large-ade", save_comparison=save_comparisone)
    print("MaskFormer - Small - ADE vs MaskFormer - Base - COCO")
    compare_segmentations(masks4, labels4, masks5, labels5, model1 = "maskformer-small-ade", model2 = "maskformer-base-coco", save_comparison=save_comparisone)
    print("MaskFormer - Small - ADE vs MaskFormer - Base - ADE")
    compare_segmentations(masks4, labels4, masks6, labels6, model1 = "maskformer-small-ade", model2 = "maskformer-base-ade", save_comparison=save_comparisone)
    print("MaskFormer - Small - ADE vs MaskFormer - Large - COCO")
    compare_segmentations(masks4, labels4, masks7, labels7, model1 = "maskformer-small-ade", model2 = "maskformer-large-coco", save_comparison=save_comparisone)
    print("MaskFormer - Small - ADE vs MaskFormer - Large - ADE")
    compare_segmentations(masks4, labels4, masks8, labels8, model1 = "maskformer-small-ade", model2 = "maskformer-large-ade", save_comparison=save_comparisone)
    print("MaskFormer - Base - COCO vs MaskFormer - Base - ADE")
    compare_segmentations(masks5, labels5, masks6, labels6, model1 = "maskformer-base-coco", model2 = "maskformer-base-ade", save_comparison=save_comparisone)
    print("MaskFormer - Base - COCO vs MaskFormer - Large - COCO")
    compare_segmentations(masks5, labels5, masks7, labels7, model1 = "maskformer-base-coco", model2 = "maskformer-large-coco", save_comparison=save_comparisone)
    print("MaskFormer - Base - COCO vs MaskFormer - Large - ADE")
    compare_segmentations(masks5, labels5, masks8, labels8, model1 = "maskformer-base-coco", model2 = "maskformer-large-ade", save_comparison=save_comparisone)
    print("MaskFormer - Base - ADE vs MaskFormer - Large - COCO")
    compare_segmentations(masks6, labels6, masks7, labels7, model1 = "maskformer-base-ade", model2 = "maskformer-large-coco", save_comparison=save_comparisone)
    print("MaskFormer - Base - ADE vs MaskFormer - Large - ADE")
    compare_segmentations(masks6, labels6, masks8, labels8, model1 = "maskformer-base-ade", model2 = "maskformer-large-ade", save_comparison=save_comparisone)
    print("MaskFormer - Large - COCO vs MaskFormer - Large - ADE")
    compare_segmentations(masks7, labels7, masks8, labels8, model1 = "maskformer-large-coco", model2 = "maskformer-large-ade", save_comparison=save_comparisone)


if __name__ == "__main__":
    test_3()