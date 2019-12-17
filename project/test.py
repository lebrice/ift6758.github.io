#!/usr/bin/python


import argparse
import json
import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import *
try:
        
    import numpy as np
    import pandas as pd
    import tensorflow as tf

except ImportError as e:
    print("ERROR:", e)
    print("Make sure to first activate the 'datascience' conda environment (which can be created from the 'environment.yml' file found at 'ift6758.github.io/project/environment.yml'.)")
    exit()

from model import HyperParameters, get_model, best_model_so_far
from model_old import HyperParameters as OldHyperParameters, get_model as old_get_model

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
        # Load the hyperparameters used during Training.
        trained_model_hparams_path=os.path.join(self.trained_model_dir, "hyperparameters.json")
        try:
            self.train_hparams = HyperParameters.load_json(trained_model_hparams_path)
        except Exception:
            # using OLD model:
            self.using_old_model = True
            self.train_hparams = OldHyperParameters.load_json(trained_model_hparams_path)

        trained_model_config_path=os.path.join(self.trained_model_dir, "train_config.json")
        self.training_config = TrainConfig.load_json(trained_model_config_path)

        def read_floats(file_name: str) -> np.ndarray:
            with open(os.path.join(self.trained_model_dir, file_name)) as f:
                line = f.readline()
                line_parts = line.split(",")
                array = np.array([float(v) for v in line_parts], dtype=float)
                return array

        maxes = read_floats("train_features_max.csv")
        mins = read_floats("train_features_min.csv")

        self.min_max_train = (mins, maxes)

        self.likes_kept_train = read_floats("train_features_likes.csv").astype(int)

        self.image_means_train = read_floats("train_features_image_means.csv")
            
        if len(self.image_means_train) != self.train_hparams.num_image_features:
            print("WARNING, hparams uses a different number of image features than the saved model!")
    
    def get_test_features(self) -> pd.DataFrame:
        if self.using_old_model:
            from task_specific_models.age_group import max_len
            max_num_likes = max_len # usually equal to 2000.
        else:
            hparams = cast(HyperParameters, self.train_hparams)
            max_num_likes = hparams.max_number_of_likes
        
        test_features, likes_multihot_matrix = preprocess_test(
            self.i,
            self.min_max_train,
            self.image_means_train,
            self.likes_kept_train,
            max_num_likes=max_num_likes,
        )
        self.likes_multihot_matrix = likes_multihot_matrix.values
        return test_features

def test_input_pipeline(data_dir: str, test_config: TestConfig):
    test_features = test_config.get_test_features()

    likes_multihot =  test_config.likes_multihot_matrix
    # TODO: save the information that will be used in the testing phase to a file or something.
    column_names = list(test_features.columns)
    print("number of columns:", len(column_names))
    assert "faceID" not in column_names
    assert "userId" not in column_names
    assert "userid" not in column_names
        
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
        OldHyperParameters.num_image_features += 2
    
    all_features = test_features.values
    text_features, image_features, likes_features = split_features(all_features, test_config.train_hparams)
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
    test_dataset, age_group_dataset = test_input_pipeline(input_dir, test_config)

    pred_labels = ["age_group", "gender", "ext", "ope", "agr", "neu", "con"]

    predictions=model.predict(test_dataset)
    print(len(predictions), "predictions")
    from user import User
    
    age_group_ids = np.argmax(predictions[0], axis=-1)
    print("previous age group ids (general model)", age_group_ids)


    # TODO: add option to use this or not:
    use_backup_age_model = True

    if use_backup_age_model:
        from task_specific_models.age_group import get_age_model, preprocess_test_agemodel
        age_group_model = get_age_model()
        age_group_userids, x_test_txt, x_test_img, x_test_lik = preprocess_test_agemodel(input_dir)
        age_group_predictions = age_group_model.predict([x_test_txt, x_test_img, x_test_lik])
        
        age_group_logits = dict(zip(age_group_userids, age_group_predictions))
        age_group_ids_dict: Dict[str, int] = {
            userid: np.argmax(age_group_predictions) for userid, age_group_predictions in age_group_logits.items()
        }

        


        print("New age group ids (specific model)", age_group_ids_dict)

    for i, user in enumerate(test_dataset.unbatch()):
        userid = user["userid"].numpy().decode("utf-8")
        pred_dict = {
            "age_group_id" : np.asscalar(age_group_ids[i]),           # normal (general) model
            "gender": np.asscalar(predictions[1][i]),
            "ext": np.asscalar(predictions[2][i]),
            "ope": np.asscalar(predictions[3][i]),
            "agr": np.asscalar(predictions[4][i]),
            "neu": np.asscalar(predictions[5][i]),
            "con": np.asscalar(predictions[6][i]),
        }

        if use_backup_age_model:
            # Overwrite the age group ID using the backup model:
            age_group_id = np.asscalar(age_group_ids_dict[userid])
            pred_dict["age_group_id"] = age_group_id

        pred_dict["userid"] = userid
        pred_dict["is_female"] = np.round(pred_dict.pop("gender")) == 1
        
        user = User(**pred_dict)
        print(user)
        with open(os.path.join(output_dir, f"{userid}.xml"), "w") as xml_file:
            xml_file.write(user.to_xml())

    if use_backup_age_model:
        from collections import Counter
        print("age group value counts:")
        print(Counter(age_group_ids_dict.values()))

