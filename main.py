from PIL import Image, ImageOps, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import cv2
img_path = "./images/2024-11-20 18.36.12.jpg"
img_path2 = "./images/grayscale.jpg"

def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def classify():

def use_pil():
    image = Image.open(img_path)
    # im = ImageOps.flip(image)
    # im.show()
    # image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

    sample = [ [190, 250,38, 68, 19, 28], [94,29,50, 19, 23, 94], [68,97,125, 82, 98, 182],
               [190, 269,38, 68, 19, 28], [94,29,50, 19, 23, 94], [68,97,125, 82, 98, 182],
               [190, 269, 38, 68, 19, 28], [94, 29, 50, 19, 23, 94], [68, 97, 125, 82, 98, 182],
               [190, 269, 38, 68, 19, 28], [94, 29, 50, 19, 23, 94], [68, 97, 125, 82, 98, 182]
,[190, 269,38, 68, 19, 28], [94,29,50, 19, 23, 94], [68,97,125, 82, 98, 182],
               [190, 269,38, 68, 19, 28], [94,29,50, 19, 23, 94], [68,97,125, 82, 98, 182],[190, 269,38, 68, 19, 28], [94,29,50, 19, 23, 94], [68,97,125, 82, 98, 182],
               [190, 269,38, 68, 19, 28], [94,29,50, 19, 23, 94], [68,97,125, 82, 98, 182]               ]
    # image = cv2.GaussianBlur(image, (3,3), sigmaX=1, sigmaY=1)
    # image = np.array(image)
    # image = image+20
    # image = image.fi
    image = image.filter(ImageFilter.MedianFilter)
    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap='gray')
    plt.show()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # use_pil()
    classify()
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
