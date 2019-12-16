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
from old_model import HyperParameters as OldHyperParameters, get_model as old_get_model, best_model_so_far as old_best_model_so_far

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
    training_config: TrainConfig = field(init=False)

    using_old_model: bool = False

    train_hparams: Union[OldHyperParameters, HyperParameters] = field(init=False)
    
    likes_multihot_matrix: np.ndarray = field(init=False)

    def __post_init__(self):

        
        trained_model_hparams_path=os.path.join(self.trained_model_dir, "hyperparameters.json")
        try:
            self.train_hparams = HyperParameters.load_json(trained_model_hparams_path)
        except KeyError:
            # using OLD model:
            self.using_old_model = True
            self.train_hparams = OldHyperParameters.load_json(trained_model_hparams_path)

        trained_model_config_path=os.path.join(self.trained_model_dir, "train_config.json")
        self.training_config = TrainConfig.load_json(trained_model_config_path)

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
            
            if len(self.image_means_train) != self.train_hparams.num_image_features:
                print("WARNING, hparams has different number of image features than model!")
    
    def get_test_features(self) -> pd.DataFrame:
        if not self.using_old_model:
            hparams = cast(HyperParameters, self.train_hparams)
            # new model:
            test_features = preprocess_test(
                self.i,
                self.min_max_train,
                self.image_means_train,
                self.likes_kept_train,
                max_num_likes=hparams.max_number_of_likes,
            )
            return test_features
        else:
            old_hparams = cast(OldHyperParameters, self.train_hparams)
            test_features, likes_data = preprocess_test(
                self.i,
                self.min_max_train,
                self.image_means_train,
                self.likes_kept_train,
                max_num_likes=2000,
                output_mhot=True
            )
            self.likes_multihot_matrix = likes_data.values
            return test_features

def test_input_pipeline(data_dir: str, test_config: TestConfig, hparams: Union[OldHyperParameters, HyperParameters], use_old_model = False):
    test_features = test_config.get_test_features()
    likes_multihot =  test_config.likes_multihot_matrix
    # TODO: save the information that will be used in the testing phase to a file or something.
    column_names = list(test_features.columns)
    print("number of columns:", len(column_names))
    assert "faceID" not in column_names
    assert "userId" not in column_names
    assert "userid" not in column_names
    # expected_num_columns= hparams.num_text_features + hparams.num_image_features + hparams.num_like_pages

    # message = f"columnds present in train set but not in test set: {set(train_columns) ^ set(column_names)}"
    # assert len(column_names) == expected_num_columns, message
    
    if test_config.using_old_model:
        old_hparams = cast(OldHyperParameters, test_config.train_hparams)

        old_features = test_features.drop(["noface", "multiface"], axis=1)
        all_old_features = old_features.values
        old_text_features, old_image_features, old_likes_features = split_features(all_old_features, old_hparams)
        features_dataset= tf.data.Dataset.from_tensor_slices(
            {
                "userid": old_features.index.astype(str),
                "text_features": old_text_features.astype(float),
                "image_features": old_image_features.astype(float),
                "likes_features": likes_multihot.astype(bool),
            }
        )
        
        all_features=test_features.values
        hparams.num_image_features += 2
        # hparams.max_number_of_likes = hparams.num_like_pages
    else:
        all_features = test_features.values
        text_features, image_features, likes_features = split_features(all_features, hparams)
    
    text_features, image_features, likes_features = split_features(all_features, hparams)
    print(text_features.shape, text_features.dtype)
    print(image_features.shape, image_features.dtype)
    print(likes_features.shape, likes_features.dtype)
    
    if not test_config.using_old_model:
        features_dataset= tf.data.Dataset.from_tensor_slices(
            {
                "userid": test_features.index.astype(str),
                "text_features": text_features.astype(float),
                "image_features": image_features.astype(float),
                "likes_features": likes_features,
            }
        )
    age_group_model_dataset = tf.data.Dataset.from_tensor_slices(
        {
            "userid": np.copy(test_features.index.astype(str)),
            "text_features": np.copy(text_features.astype(float)),
            "image_features": np.copy(image_features.astype(float)),
            "likes_features": likes_features,
        }
    )
    return features_dataset.batch(hparams.batch_size), age_group_model_dataset.batch(hparams.batch_size)

def split_features(features: np.ndarray, hparams: Union[OldHyperParameters, HyperParameters]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    text_features   = features[..., text_features_start_index:text_features_end_index]
    image_features  = features[..., image_features_start_index:image_features_end_index]
    likes_features  = features[..., likes_features_start_index:]

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
    
    hparams = test_config.train_hparams
    print("Hyperparameters:", hparams)
    print("number of text features:", hparams.num_text_features)
    print("number of image features:", hparams.num_image_features)
    print("number of like features:", hparams.num_like_pages)

    model = old_get_model(hparams) if test_config.using_old_model else get_model(hparams)
    model.load_weights(trained_model_weights_path)
    test_dataset, age_group_dataset = test_input_pipeline(input_dir, test_config, hparams, use_old_model=test_config.using_old_model)

    pred_labels = ["age_group", "gender", "ext", "ope", "agr", "neu", "con"]


    predictions=model.predict(test_dataset)
    print(len(predictions), "predictions")
    from user import User
    
    from age_group import get_age_model
    age_group_model = get_age_model()
    age_group_predictions = age_group_model.predict(age_group_dataset)
    
    
    age_group_ids = np.argmax(predictions[0], axis=-1)
    # print("previous age group ids (older model)", age_group_ids)
    age_group_ids = np.argmax(age_group_predictions, axis=-1)
    # print("New age group ids (specific model)", age_group_ids)

    for i, user in enumerate(test_dataset.unbatch()):
        
        pred_dict = {
            "age_group_id" : np.asscalar(age_group_ids[i]),
            "gender": np.asscalar(predictions[1][i]),
            "ext": np.asscalar(predictions[2][i]),
            "ope": np.asscalar(predictions[3][i]),
            "agr": np.asscalar(predictions[4][i]),
            "neu": np.asscalar(predictions[5][i]),
            "con": np.asscalar(predictions[6][i]),
        }
        userid = user["userid"].numpy().decode("utf-8")
        pred_dict["userid"] = userid
        pred_dict["is_female"] = np.round(pred_dict.pop("gender")) == 1
        
        user = User(**pred_dict)
        print(user)
        with open(os.path.join(output_dir, f"{userid}.xml"), "w") as xml_file:
            xml_file.write(user.to_xml())
