import matplotlib.pyplot as plt
import numpy as np
import argparse
import tifffile

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
    print('full resolution image size: ', obj.dimensions)
    print('number of downsampling levels: ', obj.level_count)
    print('downsampling level sizes: ', obj.level_dimensions)
    print(f"file size: {round(os.path.getsize(image_path) / 1024 ** 2, 2)} MB")
    return obj


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

    if show:
        plt.figure(figsize=(20, 20))
        plt.imshow(downsampled_img)
        plt.show()


    return np.array(downsampled_img)


if __name__ == '__main__':
    # Parse the argument passed
    parser = argparse.ArgumentParser(description="Script to process a file path")
    parser.add_argument('path', type=str, help='The path to the file')
    args = parser.parse_args()

    # Read the path of the image
    img = read_image(args.path)

    # image is return in RGBA format for now
    # the last dimesione is cutted because of stitchpro
    # that doesn't accept the trasparency channel
    new_img = downsample_image(img, level=3)[..., 0:3]

    # Save the image
    tifffile.imwrite('output_file.tiff', new_img)






