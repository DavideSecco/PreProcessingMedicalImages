import matplotlib.pyplot as plt
import numpy as np

OPENSLIDE_PATH = r'C:\Program Files\openslide-bin-4.0.0.3-windows-x64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide as o
else:
    import openslide as o


PATH3 = r'C:\Users\dicia\NL2_project\StitchPro_NL2\datasets\ultimi-isabella\TCGA-A7-A0DA-01Z-00-DX1.5F087009-16E9-4A07-BA24-62340E108B17.svs'


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

    img = read_image(PATH3)

    # image is return in RGBA format for now
    new_img = downsample_image(img, level=3)
    # print(new_img.shape)






