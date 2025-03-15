import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import time

def compute_energy(image):
    """Compute energy of an image using absolute gradients in x and y directions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)

    dx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    dy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2])

    return dx + dy

def find_vertical_seam(energy):
    """Find the optimal vertical seam using dynamic programming."""
    h, w = energy.shape
    cost = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int32)

    for i in range(1, h):
        left = np.pad(cost[i - 1, :-1], (1, 0), constant_values=np.inf)
        up = cost[i - 1]
        right = np.pad(cost[i - 1, 1:], (0, 1), constant_values=np.inf)

        min_idx = np.argmin([left, up, right], axis=0) - 1
        cost[i] += np.minimum.reduce([left, up, right])
        backtrack[i] = np.clip(np.arange(w) + min_idx, 0, w - 1)

    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = np.argmin(cost[-1])
    for i in range(h - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]

    return seam

def remove_vertical_seam(image, seam):
    """Remove a vertical seam efficiently using NumPy."""
    h, w, _ = image.shape
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return image[mask].reshape(h, w - 1, 3)

def find_horizontal_seam(energy):
    """Find the optimal horizontal seam."""
    return find_vertical_seam(energy.T)

def remove_horizontal_seam(image, seam):
    """Remove a horizontal seam efficiently using NumPy."""
    h, w, _ = image.shape
    mask = np.ones((h, w), dtype=bool)
    mask[seam, np.arange(w)] = False
    return image[mask].reshape(h - 1, w, 3)

def seam_carve(image, new_width, new_height):
    """Perform seam carving to resize the image."""
    image = image.copy()
    seam_visual = image.copy()
    energy_map = compute_energy(image)

    for _ in range(image.shape[1] - new_width):
        energy = compute_energy(image)
        seam = find_vertical_seam(energy)
        seam_visual[np.arange(image.shape[0]), seam] = [0, 0, 255]
        image = remove_vertical_seam(image, seam)

    for _ in range(image.shape[0] - new_height):
        energy = compute_energy(image)
        seam = find_horizontal_seam(energy)
        seam_visual[seam, np.arange(image.shape[1])] = [0, 255, 0]
        image = remove_horizontal_seam(image, seam)

    return image, seam_visual, energy_map

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Seam Carving for Image Resizing')
    parser.add_argument('-in', '--input_image', required=True, help='Path to the input image')
    parser.add_argument('-wf', '--width_factor', type=int, required=True, help='Width reduction factor')
    parser.add_argument('-hf', '--height_factor', type=int, required=True, help='Height reduction factor')
    parser.add_argument('-out', '--output_dir', required=True, help='Directory to save the output images')

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.input_image)
    if image is None:
        print(f"Error: Could not open or find the image at {args.input_image}")
        return

    h, w, _ = image.shape

    # Calculate new dimensions
    new_width = w // args.width_factor
    new_height = h // args.height_factor

    # Perform seam carving
    t0 = time.time()
    resized_image, seam_visual, energy_map = seam_carve(image, new_width, new_height)
    tf = time.time()
    print(f"time: {tf - t0:.2f} seconds")
    
    # Normalize energy map for visualization
    energy_map = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output file paths
    resized_image_path = os.path.join(args.output_dir, 'resized.jpg')
    seams_visual_path = os.path.join(args.output_dir, 'seams_on_original.jpg')
    energy_map_path = os.path.join(args.output_dir, 'energy_map.jpg')

    # Save results
    cv2.imwrite(resized_image_path, resized_image)
    cv2.imwrite(seams_visual_path, seam_visual)
    cv2.imwrite(energy_map_path, energy_map)

    # Display results
    plt.figure(figsize=(10, 7))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title("Resized Image")

    plt.subplot(2, 2, 3)
    plt.imshow(energy_map, cmap='gray')
    plt.title("Energy Map")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(seam_visual, cv2.COLOR_BGR2RGB))
    plt.title("Seams Visualization")

    plt.show()

if __name__ == '__main__':
    main()
