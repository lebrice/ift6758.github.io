"""
@Fabrice: Here's an example of the kind of Model we could potentially use. (given what we know about the inputs we're gonna receive.)
"""

import tensorflow as tf
import dataclasses
from dataclasses import dataclass, field
from tensorboard.plugins.hparams import api as hp
from typing import *

from utils import JsonSerializable

# Best model so far:
# best_model_hparams = HyperParameters(batch_size=128, num_layers=1, dense_units=32, activation='tanh', optimizer='ADAM', learning_rate=0.01, l1_reg=0.0, l2_reg=0.005, num_like_pages=10000, use_dropout=False, dropout_rate=0.1, use_batchnorm=False, gender_loss_weight=1.0, age_loss_weight=1.0, gender_num_layers=1, gender_num_units=32, gender_use_batchnorm=False, gender_use_dropout=False, gender_dropout_rate=0.1, gender_use_likes=False, gender_likes_condensing_layers=0, gender_likes_condensing_units=0, age_group_num_layers=2, age_group_num_units=64, age_group_use_batchnorm=False, age_group_use_dropout=False, age_group_dropout_rate=0.1, age_group_use_likes=True, age_group_likes_condensing_layers=1, age_group_likes_condensing_units=16, personality_num_layers=1, personality_num_units=8, personality_use_batchnorm=False, personality_use_dropout=False, personality_dropout_rate=0.1, personality_use_likes=False, personality_likes_condensing_layers=0, personality_likes_condensing_units=0)
best_model_so_far = "checkpoints/one-model-each-marie-2/2019-11-25_21-14-40"

@dataclass
class HyperParameters(JsonSerializable):
    """Hyperparameters of our model."""
    # the batch size
    batch_size: int = 128
    # the number of dense layers in our model.
    num_layers: int = 1
    # the number of units in each dense layer.
    dense_units: int = 32
    
    # the activation function used after each dense layer
    activation: str = "tanh"
    # Which optimizer to use during training.
    optimizer: str = "sgd"
    # Learning Rate
    learning_rate: float = 0.005

    # L1 regularization coefficient
    l1_reg: float = 0.005
    # L2 regularization coefficient
    l2_reg: float = 0.005

    # number of individual 'pages' that were kept during preprocessing of the 'likes'.
    # This corresponds to the number of entries in the multi-hot like vector.
    num_like_pages: int = 5_000
    # wether or not Dropout layers should be used
    use_dropout: bool = True
    # the dropout rate
    dropout_rate: float = 0.1
    # wether or not Batch Normalization should be applied after each dense layer.
    use_batchnorm: bool = False

    gender_loss_weight: float = 1.0
    age_loss_weight: float = 1.0


    num_text_features: ClassVar[int] = 91
    num_image_features: ClassVar[int] = 63


    # Gender model settings:
    gender_num_layers: int = 1
    gender_num_units: int = 32
    gender_use_batchnorm: bool = False
    gender_use_dropout: bool = False
    gender_dropout_rate: float = 0.1
    gender_use_likes: bool = False
    gender_likes_condensing_layers: int = 0
    gender_likes_condensing_units: int = 0

    # Age Group Model settings:
    age_group_num_layers: int = 2
    age_group_num_units: int = 64
    age_group_use_batchnorm: bool = False
    age_group_use_dropout: bool = False
    age_group_dropout_rate: float = 0.1
    age_group_use_likes: bool = True
    age_group_likes_condensing_layers: int = 1
    age_group_likes_condensing_units: int = 16

    # Personality Model(s) settings:
    personality_num_layers: int = 1
    personality_num_units: int = 8
    personality_use_batchnorm: bool = False
    personality_use_dropout: bool = False
    personality_dropout_rate: float = 0.1
    personality_use_likes: bool = False
    personality_likes_condensing_layers: int = 0
    personality_likes_condensing_units: int = 0


def sequential_block(name: str, hparams: HyperParameters) -> tf.keras.Sequential:
    """Series of dense layers
    
    Arguments:
        name {str} -- The name to give to this series of layers.
        hparams {HyperParameters} -- Hyperparameters
    
    Returns:
        tf.keras.Sequential -- a Sequential model
    """
    dense_layers = tf.keras.Sequential(name=name)
    for i in range(hparams.num_layers):
        dense_layers.add(tf.keras.layers.Dense(
            units=hparams.dense_units,
            activation=hparams.activation,
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=hparams.l1_reg, l2=hparams.l2_reg),
        ))
        
        if hparams.use_batchnorm:
            dense_layers.add(tf.keras.layers.BatchNormalization())
        
        if hparams.use_dropout:
            dense_layers.add(tf.keras.layers.Dropout(hparams.dropout_rate))
    return dense_layers

def gender_model(hparams: HyperParameters, image_features: tf.Tensor, text_features: tf.Tensor, likes_features: tf.Tensor) -> tf.keras.Sequential:
    # Likes Condensing, if required:
    if hparams.gender_use_likes:
        likes_condensing = tf.keras.Sequential(name="gender_likes_condensing")
        for i in range(hparams.gender_likes_condensing_layers):
            likes_condensing.add(tf.keras.layers.Dense(
                units=hparams.gender_likes_condensing_units,
                activation=hparams.activation,
            ))
        likes_features = likes_condensing(likes_features)

    # Model:
    model = tf.keras.Sequential(name="gender")
    model.add(tf.keras.layers.Concatenate())
    for i in range(hparams.gender_num_layers):
        model.add(tf.keras.layers.Dense(
            units=hparams.gender_num_units,
            activation=hparams.activation,
        ))

        if hparams.gender_use_batchnorm:
            model.add(tf.keras.layers.BatchNormalization())

        if hparams.gender_use_dropout:
            model.add(tf.keras.layers.Dropout(hparams.gender_dropout_rate))
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid", name="gender_out"))

    # Calling the model to create the outputs:
    model_inputs = [text_features, image_features]
    if hparams.gender_use_likes:
        model_inputs.append(likes_features)
    model_output = model(model_inputs)
    return model_output


def age_group_model(hparams: HyperParameters, image_features: tf.Tensor, text_features: tf.Tensor, likes_features: tf.Tensor) -> tf.keras.Sequential:
    # Likes Condensing, if required:
    if hparams.age_group_use_likes:
        likes_condensing = tf.keras.Sequential(name="age_group_likes_condensing")
        for i in range(hparams.age_group_likes_condensing_layers):
            likes_condensing.add(tf.keras.layers.Dense(
                units=hparams.age_group_likes_condensing_units,
                activation=hparams.activation,
            ))
        likes_features = likes_condensing(likes_features)

    # Model:
    model = tf.keras.Sequential(name="age_group")
    model.add(tf.keras.layers.Concatenate())
    for i in range(hparams.age_group_num_layers):
        model.add(tf.keras.layers.Dense(
            units=hparams.age_group_num_units,
            activation=hparams.activation,
        ))

        if hparams.age_group_use_batchnorm:
            model.add(tf.keras.layers.BatchNormalization())

        if hparams.age_group_use_dropout:
            model.add(tf.keras.layers.Dropout(hparams.age_group_dropout_rate))
    model.add(tf.keras.layers.Dense(units=4, activation="softmax", name="age_group_out"))

    # Calling the model to create the outputs:
    model_inputs = [text_features, image_features]
    if hparams.age_group_use_likes:
        model_inputs.append(likes_features)
    model_output = model(model_inputs)
    return model_output




def personality_model(personality_trait: str, hparams: HyperParameters, image_features: tf.Tensor, text_features: tf.Tensor, likes_features: tf.Tensor) -> tf.keras.Sequential:
    # Likes Condensing, if required:
    if hparams.personality_use_likes:
        likes_condensing = tf.keras.Sequential(name=f"{personality_trait}_likes_condensing")
        for i in range(hparams.personality_likes_condensing_layers):
            likes_condensing.add(tf.keras.layers.Dense(
                units=hparams.personality_likes_condensing_units,
                activation=hparams.activation,
            ))
        likes_features = likes_condensing(likes_features)

    # Model:
    model = tf.keras.Sequential(name=f"{personality_trait}")
    model.add(tf.keras.layers.Concatenate())
    for i in range(hparams.personality_num_layers):
        model.add(tf.keras.layers.Dense(
            units=hparams.personality_num_units,
            activation=hparams.activation,
        ))

        if hparams.personality_use_batchnorm:
            model.add(tf.keras.layers.BatchNormalization())

        if hparams.personality_use_dropout:
            model.add(tf.keras.layers.Dropout(hparams.personality_dropout_rate))
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid", name=f"{personality_trait}_sigmoid"))
    model.add(personality_scaling(f"{personality_trait}_out"))

    # Calling the model to create the outputs:
    model_inputs = [text_features, image_features]
    if hparams.personality_use_likes:
        model_inputs.append(likes_features)
    model_output = model(model_inputs)
    return model_output


def personality_scaling(name: str) -> tf.keras.layers.Layer:
    """Returns a layer that scales a sigmoid output [0, 1) output to the desired 'personality' range of [1, 5)
    
    Arguments:
        name {str} -- the name to give to the layer.
    
    Returns:
        tf.keras.layers.Layer -- the layer to use.
    """
    return tf.keras.layers.Lambda(lambda x: x * 4.0 + 1.0, name=name)


def get_model(hparams: HyperParameters) -> tf.keras.Model:
    # INPUTS: genderate content: (e.g., text, image and relations)
    # Inputs: user id (str), text + image info (157 floats), 
    # OUTPUTS:  Gender (ACC), Age (ACC), EXT (RMSE), ope (RMSE), AGR (RMSE), NEU (RMSE), CON (RMSE)
    # Texte: 82 (LIWC) + 11 (NRC)
    # Image
    
    # defining the inputs:
    # userid         =    tf.keras.Input([], dtype=tf.string, name="userid")
    image_features =    tf.keras.Input([hparams.num_image_features], dtype=tf.float32, name="image_features")
    text_features  =    tf.keras.Input([hparams.num_text_features], dtype=tf.float32, name="text_features")
    likes_features =    tf.keras.Input([hparams.num_like_pages], dtype=tf.bool, name="likes_features")
    
    # MODEL OUTPUTS:
    gender = gender_model(hparams, image_features, text_features, likes_features)
    age_group = age_group_model(hparams, image_features, text_features, likes_features)
   
    personality_outputs: List[tf.Tensor] = []
    for personality_trait in ["ext", "ope", "agr", "neu", "con"]:
        output_tensor = personality_model(personality_trait, hparams, text_features, image_features, likes_features)
        personality_outputs.append(output_tensor)

    model = tf.keras.Model(
        inputs=[text_features, image_features, likes_features],
        outputs=[age_group, gender, *personality_outputs]
    )
    model.compile(
        optimizer=tf.keras.optimizers.get({"class_name": hparams.optimizer,
                               "config": {"learning_rate": hparams.learning_rate}}),
        loss={
            # TODO: use weights for the different age groups, depending on their frequency
            "age_group": tf.keras.losses.CategoricalCrossentropy(),
            # TODO: same for gender
            "gender": "binary_crossentropy",
            "ext": "mse",
            "ope": "mse",
            "agr": "mse",
            "neu": "mse",
            "con": "mse",
        },
        #TODO: We can use this to change the importance of each output in the loss calculation, if need be.
        loss_weights={
            "age_group": hparams.age_loss_weight,
            "gender": hparams.gender_loss_weight,
            "ext": 1,
            "ope": 1,
            "agr": 1,
            "neu": 1,
            "con": 1,
        },
        metrics={
            "age_group": tf.keras.metrics.CategoricalAccuracy(),
            "gender": [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall()],
            "ext": tf.keras.metrics.RootMeanSquaredError(),
            "ope": tf.keras.metrics.RootMeanSquaredError(),
            "agr": tf.keras.metrics.RootMeanSquaredError(),
            "neu": tf.keras.metrics.RootMeanSquaredError(),
            "con": tf.keras.metrics.RootMeanSquaredError(),
        },
    )
    # model.summary()
    return model
