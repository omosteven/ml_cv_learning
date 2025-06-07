import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_image(image_1, image_2, title_1='Original',title_2='New Image'):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(image_1, cmap='gray')
    plt.title(title_1)
    plt.subplot(1,2,2)
    plt.imshow(image_2, cmap='gray')
    plt.title(title_2)
    plt.show()

def plot_hist(old_image, new_image, title_old='Original', title_new="New Image"):
    intensity_values = np.array([x for x in range(256)])
    plt.subplot(1,2,1)
    plt.bar(intensity_values, cv2.calcHist([old_image, [0], None, [256], [0,256]])[:,0], width=5)
    plt.title(title_old)
    plt.xlabel('intensity')
    plt.subplot(1,2,2)
    plt.bar(intensity_values, cv2, cv2.calcHist([new_image, [0], None, [256], [0,256]])[:,0], width=5)
    plt.title(title_new)
    plt.xlabel('intensity')
    plt.show()