"""
Fabrice Normandin

Computer Vision [30 points]

This assignment will give you experience with an image corpus.
For most of the questions in this assingment, you need to write a python script.

"""
from typing import *

image_path = "assignment_3_data/lena.jpg"

"""
1- Read image: Write a python code to read the image.
You can find the image of this assingment on the webpage.
"""
import cv2
import numpy as np

def load_bgr(path: str) -> np.ndarray:
    return cv2.imread(path)
image = load_bgr(image_path)
"""
2- Pre-processing: data augmentation:
Write a python code to resize the image and make it 20% smaller, and save the image as greyscale image.
"""

def rescale(image: np.ndarray, scale: float) -> np.ndarray:
    w, h = image.shape[0], image.shape[1]
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h))

image = rescale(image, 0.2)
grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
"""
3- Filter: Write a python code that load the image in RGB
and generate three images where in each one the colors of one channel is inversed.
Show the generated images.
"""

def invert_channel(image: np.ndarray, channel: int):
    inverted = np.copy(image)
    inverted[..., channel] = cv2.bitwise_not(image[..., channel])
    return inverted

def weird_filter(image: np.ndarray):
    for channel in range(3):
        inverted_image = invert_channel(image, channel)
        cv2.imshow(f"Inverted channel {channel}", inverted_image)
        cv2.waitKey()

