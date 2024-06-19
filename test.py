import os
import argparse
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch

def laplacian_of_gaussian(image, sigma):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), sigma)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian

def detect_cracks(image, sigma=1.5, threshold=30):
    log_image = laplacian_of_gaussian(image, sigma)
    _, binary_image = cv2.threshold(log_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def superimpose_cracks(original_image, crack_image):
    crack_image_3channel = cv2.merge((crack_image, crack_image, crack_image))
    enhanced_original = cv2.addWeighted(original_image, 0.7, crack_image_3channel, 0.3, 0)
    enhanced_original = cv2.convertScaleAbs(enhanced_original)
    gray_enhanced = cv2.cvtColor(enhanced_original, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray_enhanced)
    enhanced_result = cv2.merge((enhanced_gray, enhanced_gray, enhanced_gray))
    return enhanced_result

def split_image_into_patches(original_image, patch_rows, patch_cols):
    height, width = original_image.shape[:2]
    patch_height = height // patch_rows
    patch_width = width // patch_cols
    patches = []
    for i in range(patch_rows):
        for j in range(patch_cols):
            patch = original_image[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width]
            patches.append(patch)
    return patches

def reconstruct_image_from_patches(patch_folder, patch_rows, patch_cols):
    # Initialize an empty list to store patches
    patches = []

    # Iterate over the patches in row-major order
    for i in range(patch_rows):
        for j in range(patch_cols):
            # Load each patch
            patch_path = os.path.join(patch_folder, f"image{i*patch_cols + j}.jpg")
            patch = cv2.imread(patch_path)
            if patch is None:
                print(f"Error loading patch {patch_path}")
                return None
            patches.append(patch)

    # Concatenate patches row by row
    reconstructed_image = np.vstack([np.hstack(patches[j*patch_cols:(j+1)*patch_cols]) for j in range(patch_rows)])

    return reconstructed_image

def parse_args():
    parser = argparse.ArgumentParser(description="Run multi_patch_crack notebook as a script")
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--patch_size', type=int, required=True, help='Number of patches in rows and columns')
    parser.add_argument('--output_image_path', type=str, required=True, help='Path to save the reconstructed image')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    img_path = args.img_path
    model_path = args.model_path
    patch_size = args.patch_size  # This represents both rows and columns
    output_image_path = args.output_image_path

    # Set up YOLO model to use CPU
    model = YOLO(model_path)

    original_image = cv2.imread(img_path)
    image_shape = original_image.shape

    crack_image = detect_cracks(original_image)
    superimposed_image = superimpose_cracks(original_image, crack_image)
    patches = split_image_into_patches(superimposed_image, patch_size, patch_size)

    patch_results = model.predict(patches, imgsz=320, conf=0.001, device='cpu', iou=0.1, line_width=1, save=True)
    for r in patch_results[0:1]:
        print(r.save_dir)

    # Use the directory where patch_results are saved
    patch_images_folder = r.save_dir

    # Reconstruct the original image from patches stored in memory
    reconstructed_image = reconstruct_image_from_patches(patch_images_folder, patch_size, patch_size)

    if reconstructed_image is not None:
        print("Reconstructed image shape:", reconstructed_image.shape)
        reconstructed_image_rgb = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)
        plt.imshow(reconstructed_image_rgb)
        plt.axis('off')
        plt.show()

        cv2.imwrite(output_image_path, reconstructed_image)
    else:
        print("Failed to reconstruct the image.")