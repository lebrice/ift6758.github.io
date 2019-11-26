#!/usr/bin/python


import argparse
import json
import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import *

import numpy as np
import pandas as pd
import tensorflow as tf

from model import HyperParameters, get_model, best_model_so_far
from preprocessing_pipeline import preprocess_test
from train import TrainConfig
from utils import DEBUG


@dataclass
class TestConfig:
    i: str # Input directory
    o: str # Output directory
    trained_model_dir: str # Directory containing the best trained model to use

    min_max_train: Tuple[np.ndarray, np.ndarray] = field(init=False, repr=False)
    likes_kept_train: List[int] = field(init=False, repr=False)
    image_means_train: np.ndarray = field(init=False, repr=False)

    test_features: pd.DataFrame = field(init=False)
    training_config: TrainConfig = field(init=False)

    def __post_init__(self):
        trained_model_config_path=os.path.join(trained_model_dir, "train_config.json")
        with open(trained_model_config_path) as f:
            self.training_config = TrainConfig(**json.load(f))

        def to_np_float_array(csv_file_line: str) -> np.ndarray:
            return np.asarray([float(v) for v in csv_file_line.split(",") if v.strip() != ""])

        with open(os.path.join(self.trained_model_dir, "train_features_max.csv")) as f:
            maxes = to_np_float_array(f.readline())
        with open(os.path.join(self.trained_model_dir, "train_features_min.csv")) as f:
            mins = to_np_float_array(f.readline())
        self.min_max_train = (mins, maxes)

        with open(os.path.join(self.trained_model_dir, "train_features_likes.csv")) as f:
            self.likes_kept_train = [int(like) for like in f.readline().split(",") if like.strip() != ""]
        with open(os.path.join(self.trained_model_dir, "train_features_image_means.csv")) as f:
            self.image_means_train = to_np_float_array(f.readline())
        
        self.test_features = preprocess_test(self.i, self.min_max_train, self.image_means_train, self.likes_kept_train)






def test_input_pipeline(data_dir: str, test_config: TestConfig, hparams: HyperParameters):
    test_features = test_config.test_features

    # TODO: save the information that will be used in the testing phase to a file or something.
    column_names = list(test_features.columns)
    print("number of columns:", len(column_names))

    assert "faceID" not in column_names
    assert "userId" not in column_names
    assert "userid" not in column_names
    # expected_num_columns= hparams.num_text_features + hparams.num_image_features + hparams.num_like_pages

    # message = f"columnds present in train set but not in test set: {set(train_columns) ^ set(column_names)}"
    # assert len(column_names) == expected_num_columns, message

    all_features=test_features.values
    text_features, image_features, likes_features = split_features(all_features, hparams)
    
    print(text_features.shape, text_features.dtype)
    print(image_features.shape, image_features.dtype)
    print(likes_features.shape, likes_features.dtype)
    features_dataset= tf.data.Dataset.from_tensor_slices(
        {
            "userid": test_features.index.astype(str),
            "text_features": text_features.astype(float),
            "image_features": image_features.astype(float),
            "likes_features": likes_features.astype(bool),
        }
    )
    return features_dataset.batch(hparams.batch_size)

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

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--trained_model_dir",
        type = str,
        default = ("server_" if DEBUG else "") + best_model_so_far,
        help = "directory of the trained model to use for inference."
    )
    parser.add_argument(
        "-i", type = str, default = "./debug_data", help = "Input directory")
    parser.add_argument(
        "-o", type = str, default = "./debug_output", help = "Output directory")
    args=parser.parse_args()

    input_dir:  str=args.i
    output_dir: str=args.o
    trained_model_dir: str=args.trained_model_dir
    
    print("Args given:", args)
    test_config = TestConfig(
        i=args.i,
        o=args.o,
        trained_model_dir = args.trained_model_dir,
    )

    trained_model_weights_path=os.path.join(trained_model_dir, "model.h5")
    trained_model_hparams_path=os.path.join(trained_model_dir, "hyperparameters.json")
    trained_model_config_path=os.path.join(trained_model_dir, "train_config.json")
    
    import json
    import os

    with open(trained_model_hparams_path) as f:
        hparams=HyperParameters(**json.load(f))
        print("Hyperparameters:", hparams)
        print("number of text features:", hparams.num_text_features)
        print("number of image features:", hparams.num_image_features)
        print("number of like features:", hparams.num_like_pages)

    with open(trained_model_config_path) as f:
        train_config=TrainConfig(**json.load(f))

    model=get_model(hparams)
    model.load_weights(trained_model_weights_path)
    test_dataset=test_input_pipeline(input_dir, test_config, hparams)

    pred_labels = ["age_group", "gender", "ext", "ope", "agr", "neu", "con"]

    predictions=model.predict(test_dataset)
    print(len(predictions), "predictions")
    from user import User
    
    for i, user in enumerate(test_dataset.unbatch()):
        pred_dict = dict(zip(pred_labels, [p[i] for p in predictions]))
        userid = user["userid"].numpy().decode("utf-8")
        pred_dict["userid"] = userid
        pred_dict["age_group_id"] = np.argmax(pred_dict.pop("age_group"))
        pred_dict["is_female"] = np.round(pred_dict.pop("gender")) == 1
        
        user = User(**pred_dict)
        print(user)
        with open(os.path.join(output_dir, f"{userid}.xml"), "w") as xml_file:
            xml_file.write(user.to_xml())
