"""Datascience assignment 4, Fabrice Normandin. ID 20142128
"""

# Training.
# Use Keras (https://keras.io/) to create a neural network model. Use a sequential layer to combine following layers in this order:

#     Convolution with 6 feature maps 5x5
#     Rectified linear unit activation
#     Max-pooling by factor of 2 each spacial dimension
#     Convolution with 16 feature maps 5x5
#     Rectified linear unit activation
#     Max-pooling by factor of 2 each spacial dimension
#     Flatten layer
#     Dense layer with 128 output units
#     Rectified linear unit activation
#     Dense layer. Same size as the target.
#     Softmax activation

from typing import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, MaxPool2D, ReLU,
                                     Softmax)

from mnist import test_x, test_y, train_x, train_y, valid_x, valid_y

# 1. (10 points) Complete the following template.

# 2. (5 point) Create a stochastic gradient optimizer optimizer with learning rate of 10−4.
# Compile the model with the categorical crossentropy loss.
# Set the model to report accuracy metric. Complete the template.

def get_model(num_classes: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[28, 28, 1]),
        Conv2D(filters=6, kernel_size=5),
        ReLU(),
        MaxPool2D(),
        Conv2D(filters=16, kernel_size=5),
        ReLU(),
        MaxPool2D(),
        Flatten(),
        Dense(128),
        ReLU(),
        Dense(num_classes),
        Softmax()
    ])
    optimizer = tf.keras.optimizers.SGD(lr=1e-4) # create stochastic gradient optimizer
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'],
                )
    # model.summary()
    return model

 
# 3. (15 points) Train the model on the training set for at least 5 epochs. Perform validation after every epoch.
# HINT Find a method that performs training in the Keras documentation.
# Study the documentation paying attention to all arguments and the return value of the method.
# The model should have at least 95% accuracy on the training set.
# It might happen that the training gets stuck. In this case, go to the step before prevoious, recreate and rerun the model.
# WARNING This step might take several minutes to compute on a laptop.
import os
from datetime import datetime


def train(model: tf.keras.Model, save_path: str) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs"), profile_batch=0),
        tf.keras.callbacks.EarlyStopping(),
        tf.keras.callbacks.ModelCheckpoint(save_path),
    ]
    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data=(valid_x, valid_y),
        epochs=100,
        callbacks=callbacks,
    )
    return model, history


def make_plot(history: tf.keras.callbacks.History, model_name="default"):
    import matplotlib.pyplot as plt
    plt.title(f"Training and Validation Accuracy of the {model_name} model over {len(history.epoch)} epochs.")
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="valid acc")
    plt.legend()
    plt.savefig(f"{model_name}_model_loss.png")
    plt.show()

weights_path = os.path.join("logs", "model")
if not os.path.exists(weights_path):
    model = get_model(10)
    print("Training a new Model.")
    model, history = train(model, weights_path)
    model.save(weights_path)
    make_plot(history)
else:
    print("Loading a previously trained Model.")
    model = tf.keras.models.load_model(weights_path)

# Evaluation (5 points)

# 1. (1 point) Make prediction of the model on the test set

if isinstance(test_x, tf.Tensor):
    test_x = test_x.numpy().astype(float)

predictions = model.predict(x=test_x, verbose=0)



# 2. (4 points) Compute the confusion matrix and the accuracy. Which classes confused most often?
# The model should have at least 90% accuracy.

def compute_confusion_matrix(predictions: np.ndarray, onehot_targets: np.ndarray) -> Tuple[float, np.ndarray]:
    assert predictions.shape == onehot_targets.shape
    num_classes = predictions.shape[-1]

    pred_labels = np.argmax(predictions, axis=-1)
    true_labels = np.argmax(onehot_targets, axis=-1)

    accuracy = np.sum(pred_labels == true_labels) / pred_labels.shape[0]

    confusion_matrix = np.zeros([num_classes, num_classes], dtype=int)
    for pred_label, true_label in zip(pred_labels, true_labels):
        confusion_matrix[true_label][pred_label] += 1

    return accuracy, confusion_matrix

accuracy, confusion_matrix = compute_confusion_matrix(predictions, test_y)
print(f"Test accuracy: {accuracy:.3%}")

def most_confused_classes(confusion_matrix: np.ndarray):
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
    num_classes = confusion_matrix.shape[0]

    # sum the non-diagonal entries in the confusion matrix:
    # (as the diagonal entries are classified correctly)
    classified_correctly = np.identity(num_classes, dtype=bool)
    misclassified = confusion_matrix ^ ~classified_correctly
    misclassifications = np.sum(confusion_matrix, axis=0)
    misclassifications_dict = dict(enumerate(misclassifications))
    most_misclassified = sorted(misclassifications_dict.items(), key=lambda kv: kv[1], reverse=True)
    return list(most_misclassified)

# print(confusion_matrix)

most_misclassified = most_confused_classes(confusion_matrix)
print(most_misclassified)


# Bonus point (10 points)
# Can you suggest an improvement to the model? Implement it and compare to the one above. How to do robust comparison of the performance?

# - Use Dropout after the first dense layer to further reduce overfitting:
from tensorflow.keras.layers import Dropout


def get_improved_model(num_classes: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[28, 28, 1]),
        Conv2D(filters=6, kernel_size=5),
        ReLU(),
        MaxPool2D(),
        Conv2D(filters=16, kernel_size=5),
        ReLU(),
        MaxPool2D(),
        Flatten(),
        Dense(128),
        Dropout(0.1),
        ReLU(),
        Dense(num_classes),
        Softmax()
    ])
    optimizer = tf.keras.optimizers.Adam(lr=1e-4) # create stochastic gradient optimizer
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'],
                )
    # model.summary()
    return model

weights_path = os.path.join("logs", "improved_model")
if not os.path.exists(weights_path):
    improved_model = get_improved_model(10)
    print("Training a new (improved) Model.")
    improved_model, history = train(improved_model, weights_path)
    improved_model.save(weights_path)
    make_plot(history, model_name="improved")
else:
    print("Loading a previously trained (improved) Model.")
    improved_model = tf.keras.models.load_model(weights_path)
    # improved_model.summary()

predictions = improved_model.predict(x=test_x, verbose=0)
improved_accuracy, confusion_matrix = compute_confusion_matrix(predictions, test_y)
print(f"(improved) Test accuracy: {improved_accuracy:.3%}")
most_misclassified = most_confused_classes(confusion_matrix)
print(most_misclassified)