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

"""
4- Suggest a filter that make the image 50% lighter and 50% darker.
Write a python function that applies the filter on the image. 
"""

def change_brightness(image: np.ndarray, ratio: float) -> np.ndarray:
	image = np.copy(image)
	image = image * ratio
	return image.clip(0, 255).astype(np.uint8)

def lightening_filter(image: np.ndarray) -> np.ndarray:
	"""make the image 50% lighter"""
	return change_brightness(image, 1.5)
	
def darkening_filter(image: np.ndarray) -> np.ndarray: 
	"""make the image 50% darker"""
	return change_brightness(image, 0.5)

brighter = lightening_filter(image)
cv2.imshow("Brighter", brighter)
cv2.waitKey()

darker = darkening_filter(image)
cv2.imshow("Darker", darker)
cv2.waitKey()
"""
5- Write two python functions that apply a 3 x 3 median and 3 x 3 mean filter on the image.
Mean filter is a simple sliding window that replace the center value with the average of all pixel
values in the window.
While median filter is a simple sliding window that replace the center value with the Median of all pixel values in the window.
Note that the border pixels remain unchanged.
"""

def three_by_three_filter(image: np.ndarray, func: Callable) -> np.ndarray:
	padded = np.pad(image, 1, mode="edge")
	row, col = np.indices(image.shape[:-1])
	row = row + 1
	col = col + 1
	return func(np.dstack((
		padded[row-1, col-1],
		padded[row-1, col],
		padded[row-1, col+1],
		padded[row, col-1],
		padded[row, col],
		padded[row, col+1],
		padded[row+1, col-1],
		padded[row+1, col],
		padded[row+1, col+1]
	)), axis=-1).astype(np.uint8)

def mean_filter(image: np.ndarray) -> np.ndarray:
	"""
	apply 3 x  3 mean filter
	"""	
	return three_by_three_filter(image, np.mean)

filtered_image = mean_filter(image)
cv2.imshow("filtered image (MEAN)", filtered_image)
cv2.waitKey()

def median_filter(image: np.ndarray) -> np.ndarray:
	"""
	apply 3 x  3 median filter
	"""
	return three_by_three_filter(image, np.median)


filtered_image = median_filter(image)
cv2.imshow("filtered image (MEDIAN)", filtered_image)
cv2.waitKey()

"""
6- A mean filter is a linear filter, but a median filter is not. Why?

Because 
"""
