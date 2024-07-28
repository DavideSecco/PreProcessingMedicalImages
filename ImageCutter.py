import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import cv2


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
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to segment the tissue
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the tissue
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the centroid of the largest contour
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Draw the largest contour and centroid on the image
    output_image = cv2.drawContours(image.copy(), [largest_contour], -1, (0, 255, 0), 2)
    cv2.circle(output_image, (cX, cY), 70, (255, 0, 0), thickness=-1)
    cv2.putText(output_image, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    visualize = True
    if visualize:
        # Display the results
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 2)
        plt.title("Binary Image")
        plt.imshow(binary_image, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Contour and Centroid")
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

        plt.show()


def divide_4_pieces(image):
    # Get the dimensions of the image
    height, width, levels = image.shape

    # Calculate the coordinates for the 4 pieces
    mid_height = height // 2
    mid_width = width // 2

    # Slice the image into four parts
    top_left = image[:mid_height, :mid_width]
    top_right = image[:mid_height, mid_width:]
    bottom_left = image[mid_height:, :mid_width]
    bottom_right = image[mid_height:, mid_width:]

    images = [top_left, top_right, bottom_left, bottom_right]
    names = ['top_left.tif', 'top_right.tif', 'bottom_left.tif', 'bottom_right.tif']

    return images, names


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Open the TIFF file
    file_path = '/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/Subset1_Train_30.tif'
    image = tiff.imread(file_path)

    find_centroid(image)

    visualize_image(image)

    images, names = divide_4_pieces(image)
    padded_images = [None] * 4

    for index in range(0,len(images)):
        # Aggiungo il padding
        padded_images[index] = np.pad(images[index], ((200, 200), (200, 200), (0,0)), mode='constant', constant_values=np.iinfo(image.dtype).max)

        # Save images with padding
        tiff.imwrite(names[index], padded_images[index])

    visualize_images(padded_images, names)




