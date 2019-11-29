#!/usr/bin/env python3.7


import argparse
import json
import os
from collections import namedtuple
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import *

import numpy as np
import pandas as pd
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorboard.plugins.hparams import api as hp

from collections import OrderedDict

from model import HyperParameters, get_model
import utils
from preprocessing_pipeline import preprocess_train

today_str = (datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
from utils import DEBUG

print("DEBUGGING: ", DEBUG)

tf.random.set_seed(123123)

baseline_metrics = {
    "gender_binary_accuracy":         0.591,
    "age_group_categorical_accuracy": 0.594,
    "ope_root_mean_squared_error":    0.652,
    "neu_root_mean_squared_error":    0.798,
    "ext_root_mean_squared_error":    0.788,
    "agr_root_mean_squared_error":    0.665,
    "con_root_mean_squared_error":    0.734,
}



@dataclass()
class TrainConfig():
    experiment_name: str = "debug" if DEBUG else "default_experiment"
    """
    Name of the experiment
    """

    log_dir: str = ""
    """
    The directory where the model checkpoints, as well as logs and event files should be saved at.
    """

    validation_data_fraction: float = 0.2
    """
    The fraction of all data corresponding to the validation set.
    """

    epochs: int = 50
    """Number of passes through the dataset"""   


    early_stopping_patience: int = 5
    """Interrupt training if `val_loss` doesn't improving for over `early_stopping_patience` epochs."""
    
    def __post_init__(self):
        if not self.log_dir:
            self.log_dir = os.path.join(os.path.curdir, "checkpoints", self.experiment_name , today_str)
        os.makedirs(self.log_dir, exist_ok=True)




@dataclass
class TrainingResults:
    total_epochs: int = -1
    metrics_dict: Dict[str, float] = field(default_factory=dict)
    model_param_count: int = -1
    training_time_secs: float = -1


@dataclass()
class TrainData():    
    train_features: pd.DataFrame
    """vectorized features scaled between 0 and 1
    for each user id in the training set, concatenated for all modalities
    (order = text + image + relation), with userid as DataFrame index.
    """
    
    train_likes_list: np.ndarray
    """
    For each user, a list of the index of the corresponding liked page id in `likes_kept` 
    """


    features_min_max: Tuple[pd.DataFrame, pd.DataFrame]
    """series of min and max values of
    text + image features from train dataset, to be used to scale test data.
    Note that the multihot relation features do not necessitate scaling.
    """
    image_means: List[float]
    """
    means from oxford dataset to replace missing entries in oxford test set
    """
    likes_kept: pd.Index
    """ordered likes_ids to serve as columns for test set relation features matrix
    """
    train_labels: pd.DataFrame
    """labels ordered by userid (alphabetically)
    for the training set, with userids as index.
    """



    def write_training_data_config(self, log_dir: str):
        mins, maxes = self.features_min_max
        image_means = self.image_means
        likes = self.likes_kept
        with open(os.path.join(log_dir, "train_features_max.csv"), "w") as f:    
            f.write(",".join(str(v) for v in maxes))
        with open(os.path.join(log_dir, "train_features_min.csv"), "w") as f:
            f.write(",".join(str(v) for v in mins))
        with open(os.path.join(log_dir, "train_features_image_means.csv"), "w") as f:
            f.write(",".join(str(v) for v in image_means))
        with open(os.path.join(log_dir, "train_features_likes.csv"), "w") as f:
            f.write(",".join(likes))

def train_input_pipeline(data_dir: str, hparams: HyperParameters, train_config: TrainConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_data = TrainData(*preprocess_train(data_dir, hparams.num_like_pages, max_num_likes=hparams.max_number_of_likes))

    features = train_data.train_features
    labels = train_data.train_labels
    if DEBUG:
        train_data.likes_kept = [str(i) for i in range(hparams.num_like_pages)]

    # shuffle the features and labels
    features, labels = shuffle(features, labels)

    column_names = list(features.columns)
    print("Total number of columns:", len(column_names))
    
    assert "faceID" not in column_names
    assert "userId" not in column_names
    assert "userid" not in column_names
    column_names = features.columns
    # The names of each column for each type of feature. Could be useful for debugging.
    text_columns_names, image_columns_names, likes_columns_names = split_features(column_names, hparams)

    expected_num_columns = hparams.num_text_features + hparams.num_image_features + hparams.num_like_pages
    assert features.shape[1] == expected_num_columns


    train_data.write_training_data_config(train_config.log_dir)
    (train_features, train_labels), (valid_features, valid_labels) = train_valid_split(features, labels, train_config)

    train_dataset = make_dataset(train_features, train_labels, hparams)
    if train_config.validation_data_fraction != 0:
        valid_dataset = make_dataset(valid_features, valid_labels, hparams)
        return train_dataset, valid_dataset
    else:
        return train_dataset, None

def train_valid_split(features: pd.DataFrame, labels: pd.DataFrame, train_config: TrainConfig) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    all_features = features.values
    validation_data_fraction = train_config.validation_data_fraction
    if validation_data_fraction == 0:
        print("USING NO VALIDATION SET.")
        
    cutoff = int(all_features.shape[0] * validation_data_fraction)

    # perform the train-valid split.
    valid_features, valid_labels = features.values[:cutoff], labels[:cutoff]
    train_features, train_labels = features.values[cutoff:], labels[cutoff:]
    return (train_features, train_labels), (valid_features, valid_labels)


def make_dataset(features: np.ndarray, labels: np.ndarray, hparams: HyperParameters) -> tf.data.Dataset:
    text_features, image_features, likes_features = split_features(features, hparams)

    # print(text_features.shape)
    # print(image_features.shape)
    # print(likes_features.shape)
    features_dataset = tf.data.Dataset.from_tensor_slices(
        {
            "text_features": text_features.astype("float32"),
            "image_features": image_features.astype("float32"),
            "likes_features": likes_features.astype("bool"),
        }
    )
    labels_dataset = tf.data.Dataset.from_tensor_slices({
        "userid": labels.index,
        "gender": labels.gender.astype("bool"),
        "age_group": labels.age_group,
        "ope": labels['ope'].astype("float32"),
        "con": labels['con'].astype("float32"),
        "ext": labels['ext'].astype("float32"),
        "agr": labels['agr'].astype("float32"),
        "neu": labels['neu'].astype("float32"),
    })
    return (tf.data.Dataset.zip((features_dataset, labels_dataset))
        .cache()
        .shuffle(5 * hparams.batch_size)
        .batch(hparams.batch_size)
    )


def split_features(features: np.ndarray, hparams: HyperParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the ndarray with the three types of features into three ndarrays.
    
    Args:
        features (np.ndarray): [description]
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: [description]
    """
    
    text_features_start_index = 0
    text_features_end_index = text_features_start_index + hparams.num_text_features

    image_features_start_index = text_features_end_index
    image_features_end_index = image_features_start_index + hparams.num_image_features
    
    likes_features_start_index = image_features_end_index
    likes_features_end_index = likes_features_start_index + hparams.num_like_pages 

    text_features   = features[..., text_features_start_index:text_features_end_index]
    image_features  = features[..., image_features_start_index:image_features_end_index]
    likes_features  = features[..., likes_features_start_index:likes_features_end_index]

    return text_features, image_features, likes_features


def shuffle(features: pd.DataFrame, labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffles the features and labels (in the same exact way.)
    
    Args:
        features (pd.DataFrame): [description]
        labels (pd.DataFrame): [description]
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: [description]
    """
    import random
    random_state = random.randint(0, 1000)
    return (
        features.sample(frac=1, random_state=random_state),
        labels.sample(frac=1, random_state=random_state)
    )

# if __name__ == "__main__":
#     p1 = pd.DataFrame(np.arange(10), index=[f"user_{v}" for v in np.arange(10)])
#     p2 = pd.DataFrame(np.arange(10)+100, index=[f"user_{v}" for v in np.arange(10)])
#     print(p1)
#     print(p2)
#     features, labels = shuffle(p1, p2)
#     print(features)
#     print(labels)
#     # for feat, label in zip(features.values, labels.values):
#     #     print(f"feat: '{feat}', label: '{label}'")
#     exit()




def train(train_data_dir: str, hparams: HyperParameters, train_config: TrainConfig) -> TrainingResults:
    
    print("Hyperparameters:", hparams)
    print("Train_config:", train_config)

    start_time = time.time()


    # save the hyperparameter config to a file.
    with open(os.path.join(train_config.log_dir, "hyperparameters.json"), "w") as f:
        json.dump(asdict(hparams), f, indent=4)
    
    with open(os.path.join(train_config.log_dir, "train_config.json"), "w") as f:
        json.dump(asdict(train_config), f, indent=4)

    model = get_model(hparams)
    model.summary()

    train_dataset, valid_dataset = train_input_pipeline(train_data_dir, hparams, train_config)
    if DEBUG:
        train_dataset = train_dataset.repeat(100)
        if valid_dataset:
            valid_dataset = valid_dataset.repeat(100)

    using_validation_set = valid_dataset is not None

    training_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir = train_config.log_dir, profile_batch=0),
        hp.KerasCallback(train_config.log_dir, asdict(hparams)),
        tf.keras.callbacks.TerminateOnNaN(),
        utils.EarlyStoppingWhenValueExplodes(monitor="loss", check_every_batch=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(train_config.log_dir, "model.h5"),
            monitor = "val_loss" if using_validation_set else "loss",
            verbose=1,
            save_best_only=True,
            mode = 'auto'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=train_config.early_stopping_patience,
            monitor='val_loss' if using_validation_set else "loss"
        ),
    ]
    history = None
    try:
        history = model.fit(
            train_dataset,
            validation_data=valid_dataset if using_validation_set else None,
            epochs=train_config.epochs,
            callbacks=training_callbacks,
            # steps_per_epoch=int(train_samples / hparams.batch_size),
        )
        end_time = time.time()
        training_time_secs = end_time - start_time
        
        loss_metric = "val_loss" if using_validation_set else "loss"
        best_loss_value = min(history.history[loss_metric])
        num_epochs = len(history.history[loss_metric])

        print(f"BEST {'VALIDATION' if using_validation_set else 'TRAIN'} LOSS:", best_loss_value)
        
        results = TrainingResults()
        results.total_epochs = num_epochs
        results.model_param_count = model.count_params()
        results.training_time_secs = training_time_secs
        
        if using_validation_set:
            metrics = model.evaluate(valid_dataset)
            results.metrics_dict = OrderedDict(zip(model.metrics_names, metrics))
        else:
            results.metrics_dict = {metric_name: values[-1] for metric_name, values in history.history.items()}
        
        return results

    except Exception as e:
        print(f"\n\n {e} \n\n")
        return TrainingResults()


def log_results(results: TrainingResults):
    metrics_dict = results.metrics_dict
    total_epochs = results.total_epochs    
    from io import StringIO
    f = StringIO()
    def relative_improvement(metric_name) -> float:
        value_to_beat = baseline_metrics[metric_name]
        result_value = metrics_dict[metric_name]
        if "accuracy" in metric_name:
            return result_value - value_to_beat
        else:
            # the term is an error or loss.
            return value_to_beat - result_value
    
    
    if DEBUG:
        f.write("(DEBUG)\t")
    f.write(f"Total epochs: {total_epochs:04d}, log_dir: {train_config.log_dir}, ")
    if not metrics_dict:
        f.write("TRAINING DIVERGED!")
        print("TRAINING DIVERGED!")
        return

    relevant_metrics = {
        metric_name: result for metric_name, result in metrics_dict.items() if metric_name in baseline_metrics
    }
    other_metrics = {
        metric_name: result for metric_name, result in metrics_dict.items() if metric_name not in baseline_metrics
    }
    improvement_from_baseline = {
        metric_name: relative_improvement(metric_name) for metric_name in relevant_metrics
    }
    beating_the_baseline = {
        metric_name: improvement for metric_name, improvement in improvement_from_baseline.items() if improvement > 0 
    }
    total_improvement = sum(improvement_from_baseline.values())
    f.write(f"total improvement: {total_improvement}\n\t")
    
    for metric_name, result_value in relevant_metrics.items(): 
        f.write(f"{metric_name}: {result_value:.3f} ")
    f.write("\n\t")

    for metric_name, result_value in other_metrics.items(): 
        f.write(f"{metric_name}: {result_value:.3f} ")
    f.write("\n\t")

    for metric_name, improvement in beating_the_baseline.items():
        result = metrics_dict[metric_name]
        target = baseline_metrics[metric_name]
        f.write(f"BEATING THE BASELINE AT '{metric_name}': (Baseline: {target:.4f}, Ours: {result:.4f}, improvement of {improvement:.4f})\n\t")

    f.write(f"Hparams: {hparams}\n")
    f.write("\n")

    f.seek(0)
    print(f.read())
    f.seek(0)

    os.makedirs("logs", exist_ok=True)
    experiment_results_file = os.path.join("logs", train_config.experiment_name +"-results.txt")
    with open(experiment_results_file, "a") as runs_results_file:
        runs_results_file.write(f.read())


def main(hparams: HyperParameters, train_config: TrainConfig):
    print("Experiment name:", train_config.experiment_name)
    print("Hyperparameters:", hparams)
    print("Train_config:", train_config)

    # create the results path so its directly possible to call the "tail" progam to stream the results as they come in.
    experiment_results_file = os.path.join("logs", train_config.experiment_name +"-results.txt")
    with open(experiment_results_file, "a") as runs_results_file:
        pass

    train_data_dir = os.path.join(os.path.curdir, "debug_data") if DEBUG else "~/Train"
    # Create the required directories if not present.
    os.makedirs(train_config.log_dir, exist_ok=True)
    
    print("Training directory:", train_config.log_dir)

    with utils.log_to_file(os.path.join(train_config.log_dir, "train_log.txt")):
        results = train(train_data_dir, hparams, train_config)
        
        print(f"Saved model weights are located at '{train_config.log_dir}'")
    
    log_results(results)

    using_validation_set = train_config.validation_data_fraction != 0.0
    if using_validation_set:
        from orion.client import report_results    
        report_results([dict(
            name='validation_loss',
            type='objective',
            value=results.metrics_dict.get("loss", np.Inf),
        )])

    print("TRAINING COMPLETE")


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_arguments(HyperParameters, "hparams")
    parser.add_arguments(TrainConfig, "train_config")

    args = parser.parse_args()
    
    hparams: HyperParameters = args.hparams
    train_config: TrainConfig = args.train_config
    main(hparams, train_config)
    
