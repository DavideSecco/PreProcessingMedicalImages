import matplotlib.pyplot as plt
import numpy as np
import argparse
import tifffile
from PIL import Image
import os

OPENSLIDE_PATH = r'C:\Program Files\openslide-bin-4.0.0.3-windows-x64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide as o
else:
    import openslide as o


# PATH3 = r'/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A2S1/MultudisciplinaryProject/pythostitcher/data/TCGA-A2-A3XY/TCGA-A2-A3XY-01Z-00-DX1.E57FC9BF-411E-4028-AC10-8BCA5D0C8472.svs'

def read_image(image_path):
    obj = o.OpenSlide(image_path)
    size_image = round(os.path.getsize(image_path) / 1024 ** 2, 2)
    print('full resolution image size: ', obj.dimensions)
    print('number of downsampling levels: ', obj.level_count)
    print('downsampling level sizes: ', obj.level_dimensions)
    print(f"file size: {size_image} MB")
    return obj, size_image


def downsample_image(image_object, level, region=None, size=None, show=True):
    level_dimensions = image_object.level_dimensions[level]

    # check if a region central point is specified
    if region is not None:
        r = region
    else:
        width, height = level_dimensions[0], level_dimensions[1]
        r = (width // 2, height // 2)

    # check if a size of the tile is specified
    if size is not None:
        s = size
    else:
        s = level_dimensions

    # get the image matrix
    downsampled_img = image_object.read_region(location=r, level=level, size=s)
    print(type(downsampled_img))
    if show:
        # plt.figure(figsize=(20, 20))
        plt.title("Downsampled image before saving")
        plt.imshow(downsampled_img)
        plt.show()

    # return np.array(downsampled_img)
    return downsampled_img


def convert_rgba_to_rgb(rgba_image, background=(255, 255, 255)):
    """Convert an RGBA image to RGB by blending it with a background color."""
    rgba_image = np.array(rgba_image)
    r, g, b, a = np.split(rgba_image, 4, axis=-1)
    alpha = a / 255.0
    background = np.array(background).reshape(1, 1, 3)
    rgb_image = (1 - alpha) * background + alpha * np.concatenate([r, g, b], axis=-1)
    rgb_image = rgb_image.astype(np.uint8)
    return Image.fromarray(rgb_image, 'RGB')

def process_image(image_path, save_dir):
    level = 2
    stichpro_compatible = True
    good_image = False

    # Read the path of the image
    original_img, _ = read_image(image_path)

    while (not good_image):
        original_img_name = os.path.basename(image_path)
        downsampled_img_name = os.path.splitext(original_img_name)[0] + "_downsampled.tif"


        # image is return in RGBA format for now:
        # Last dimesion is cutted because of stitchpro that doesn't accept the trasparency channel
        # downsampled_image = downsample_image(original_img, level=level)[..., 0:3]
        downsampled_image = downsample_image(original_img, level=level)

        width, height = downsampled_image.size

        # Get the number of channels
        mode = downsampled_image.mode
        if mode == 'L':
            channels = 1
        elif mode == 'RGB':
            channels = 3
        elif mode == 'RGBA':
            channels = 4
        else:
            channels = len(mode)  # For other modes, this is a general approach

        # Print the shape
        print(f"Image shape: ({height}, {width}, {channels})")

        plt.title("Downsampled image before saving")
        plt.imshow(downsampled_image)
        plt.show()

        # Save the image
        downsampled_image_path = os.path.join(save_dir, downsampled_img_name)
        # print(downsampled_image.mode)
        # print(np.array(downsampled_image))
        # pixel = (2500,5000)
        # print(downsampled_image.getpixel(pixel))
        if downsampled_image.mode != 'RGB':
            downsampled_image = convert_rgba_to_rgb(downsampled_image)

        downsampled_image.save(downsampled_image_path, format='TIFF')

        plt.title("Downsampled image AFTER SAVING")
        plt.imshow(downsampled_image)
        plt.show()

        # print(downsampled_image.getpixel(pixel))

        # try:
        #     tifffile.imwrite(downsampled_image_path, downsampled_image)
        # except:
        #    print(f"Errore nella creazione della cartella: {e}")


        downsampled_image_size = round(os.path.getsize(downsampled_image_path) / 1024 ** 2, 2)

        if stichpro_compatible and downsampled_image_size > 200:
            print(f"You set stichpro_compatible to {stichpro_compatible}, but the image create at level {level} is {downsampled_image_size}.")
            print("Procedure must be repeated going down of one level")
            good_image = False
            level = level + 1
        else:
            good_image = True

if __name__ == '__main__':
    # Parse the argument passed
    parser = argparse.ArgumentParser(description="Script to process a file path")
    parser.add_argument('path', type=str, help='The path to the file')
    args = parser.parse_args()

    if os.path.isfile(args.path):
        save_dir = os.path.curdir

        process_image(args.path, os.path.curdir)

    if os.path.isdir(args.path):
        save_dir = os.path.join(os.getcwd(), os.path.basename(os.path.normpath(args.path)))

        try:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Cartella '{save_dir}' creata con successo")
        except OSError as e:
            print(f"Errore nella creazione della cartella: {e}")

        for file in os.listdir(args.path):
            process_image(os.path.join(args.path, file), save_dir)





