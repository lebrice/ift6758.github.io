
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import *

# 1. (1 point) Search MNIST dataset at OpenML, it is called "mnist_784". Download it using sklearn function fetch_openml. Get features and targets.

# I will be using keras directly instead of sklearn.

def part_one() -> Tuple[np.ndarray, np.ndarray]:    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    mnist_samples = np.concatenate([x_train, x_test], axis=0)
    mnist_targets = np.concatenate([y_train, y_test], axis=0)
    return mnist_samples, mnist_targets

# 2. (3 points) Reshape it back to 28x28 and visualize a couple of images with matplotlib.

# Reshaping is already done for us above.

def visualize_a_couple_of_images(samples: np.ndarray, targets: np.ndarray):
    for i in range(3):
        plt.imshow(samples[i])
        plt.title(f"An example of a {targets[i]}")
        plt.show()

mnist_samples, mnist_targets = part_one()
# visualize_a_couple_of_images(mnist_samples, mnist_targets)


# 3. (3 points) Add a channel dimension:
mnist_samples = tf.expand_dims(mnist_samples, axis=-1)


# 4. (3 point) Filter data leaving only classes 1, 3, 7. Transform features and targets. How many data points left after filtering?
def filter_examples(features: np.ndarray, labels: np.ndarray, labels_to_keep: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    filter_mask = np.any(np.stack([labels == value for value in labels_to_keep], axis=-1), axis=-1)
    return features[filter_mask], labels[filter_mask]

remaining_x, remaining_y = filter_examples(mnist_samples, mnist_targets, [1, 3, 7])
print(f"There are {remaining_x.shape[0]} data points left after removing everything but classes 1, 3, and 7.")

# 5. (5 point) Convert targets to one-hot representation. Complete the following template.
def to_categorical(array: np.ndarray, classes: List[int]) -> np.ndarray:
    """
    array -- array of targets
    classes -- list of classes
    """
    return np.stack([array == target for target in classes], axis=-1)

mnist_targets = to_categorical(mnist_targets, classes=list(range(10)))


# 6. (5 point) Split the dataset into train, validataion, and test.
# Take first 16,000 images and targets as the train, then next 3,000 as validation, then the rest as the test subset.
def split(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return x[:16000], x[16000:19000], x[19000:]

train_x, valid_x, test_x = split(mnist_samples)
train_y, valid_y, test_y = split(mnist_targets)
