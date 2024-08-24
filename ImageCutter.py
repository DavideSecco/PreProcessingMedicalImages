import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from PIL import Image


def visualize_image(image, name=None):
    # Display the image
    plt.imshow(image, cmap='gray')  # You can change 'gray' to other colormaps as needed
    plt.title(name)
    plt.axis('on')  # Hide the axis
    plt.show()

def visualize_images(images, names=None):
    rows = 2
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # Incompresible line to pass from a 1-D array to a 2-D array
    images = [images[i * cols:(i + 1) * cols] for i in range(rows)]
    names = [names[i * cols:(i + 1) * cols] for i in range(rows)]

    for row in range(0, rows):
        for col in range(0, cols):
            axes[row, col].imshow(images[row][col], cmap='gray')
            axes[row, col].set_title(names[row][col])
            axes[row, col].axis('on')

    plt.tight_layout()
    plt.show()

def find_centroid(image):
    # Convert the image to grayscale (optional, depending on the use case)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask (you can adjust the threshold value)
    _, binary_mask = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)

    # Find the indices of the non-zero pixels in the binary mask
    indices = np.argwhere(binary_mask == 0)

    # Calculate the centroid (mean of the indices)

    cX, cY = map(int,np.mean(indices, axis=0))
    # print("Centroid : cX", cX, ", cY:", cY)

    ###### Visualizzo centroid e true image (DA COMMENTARE IN PRODUZIONE) ######

    # cv2.circle(image, (cY, cX), 70, (255, 0, 0), thickness=-1)  # Draw red circle
    # plt.figure(figsize=(6, 6))
    # plt.title("True Image")
    # plt.imshow(image, cmap='gray') # cmap='gray'
    # plt.axis('on')  # Hide axes for a cleaner look
    # plt.show()

    ###### Visualizzo centroid e binary_mask image ######

    cv2.circle(binary_mask, (cY, cX), 70, (255, 0, 0), thickness=-1)  # Draw red circle
    plt.figure(figsize=(6, 6))
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap='gray')  # cmap='gray'
    plt.axis('on')  # Hide axes for a cleaner look
    plt.show()

    # Print the centroid coordinates
    # print("Centroid (y, x):", centroid)
    return cX, cY

def divide_4_pieces(image, cX=None, cY=None):
    """
    :param image:
    :param cX: X coordinate of the centroid
    :param cY: Y coordinate of the centroid
    :return:

    The function divide an image in 4 part, if the point of the centroid are provided
    the image is divided using that point, otherwise in cutted in 4 equal part
    """
    if cX == None or cY == None:
        # Get the dimensions of the image
        height, width, levels = image.shape

        # Calculate the coordinates for the 4 pieces
        cX = width // 2
        cY = height // 2

    # Slice the image into four parts
    upper_left = image[:cX, :cY]
    upper_right = image[:cX, cY:]
    bottom_left = image[cX:, :cY]
    bottom_right = image[cX:, cY:]

    images = [upper_left, upper_right, bottom_left, bottom_right]
    names = ['upper_left.tif', 'upper_right.tif', 'bottom_left.tif', 'bottom_right.tif']

    return images, names


def convert_rgba_to_rgb(rgba_image, background=(255, 255, 255)):
    """Convert an RGBA image to RGB by blending it with a background color."""
    # Split the RGBA image into its components
    r, g, b, a = np.split(rgba_image, 4, axis=-1)

    # Normalize the alpha channel to be in the range [0, 1]
    alpha = a / 255.0

    # Blend the RGB channels with the background color using the alpha channel
    background = np.array(background).reshape(1, 1, 3)
    rgb_image = (1 - alpha) * background + alpha * np.concatenate([r, g, b], axis=-1)

    # Convert back to uint8 data type
    rgb_image = rgb_image.astype(np.uint8)

    return rgb_image


def is_rgb(image):
    """Check if the image is in RGB format."""
    return image.shape[-1] == 3 and image.dtype == np.uint8

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Parse the argument passed
    parser = argparse.ArgumentParser(description="Script to process a file path")
    parser.add_argument('path', type=str, help='The path to the file')
    args = parser.parse_args()

    ####### Visualizzazione e eventuale conversione dell'immagine #####
    image = tiff.imread(args.path)
    print(image.shape)
    visualize_image(image)

    if not is_rgb(image):
        print("Not an RGB image: conversion from RGBA to RGB")
        image = convert_rgba_to_rgb(image)
        visualize_image(image)

    print("Dimensione dell'immagine RGB:", image.shape)

    ###### Trovo centroid #######
    cX, cY = find_centroid(image)
    print("Cordinate centroid: cX: ", cX, "cY: ", cY)

    ###### Divisione dell'immagine ######
    images, names = divide_4_pieces(image, cX, cY)

    ###### Aggiunta padding alle immagini #####
    padded_images = [None] * 4

    # Per tutti i nuovi frammenti dell'immagine
    for index in range(0, len(images)):
        padded_images[index] = np.pad(images[index], ((100, 100), (100, 100), (0,0)), mode='constant', constant_values=np.iinfo(image.dtype).max)
        print("Dimensione frammento: ", index, ": ", padded_images[index].shape)
        # visualize_image(padded_images[index])

        # Save images with padding
        tiff.imwrite(names[index], padded_images[index])

    print(type(padded_images[0]))

    visualize_images(padded_images, names)





