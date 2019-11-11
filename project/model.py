"""
@Fabrice: Here's an example of the kind of Model we could potentially use. (given what we know about the inputs we're gonna receive.)
"""

import tensorflow as tf
import dataclasses
from dataclasses import dataclass, field
from tensorboard.plugins.hparams import api as hp
from typing import *

# Best model so far:

@dataclass
class HyperParameters():
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

    # factor used as the stride and kernel size in the likes compressing block.
    likes_condensing_factor: int = 5
    # Final (reduced) length for the likes features vector.
    likes_condensed_vector_max_size: int = 512

    num_text_features: ClassVar[int] = 91
    num_image_features: ClassVar[int] = 63

best_model_hparams = HyperParameters(batch_size=128, num_layers=1, dense_units=32, activation='tanh', optimizer='sgd', learning_rate=0.005, l1_reg=0.005, l2_reg=0.005, num_like_pages=5000, use_dropout=True, dropout_rate=0.1, use_batchnorm=False, gender_loss_weight=5.0, age_loss_weight=5.0)


def likes_condensing(hparams: HyperParameters, use_conv = False) -> tf.keras.Sequential:
    """Condenses a `hparams.num_like_pages`-long vector down to something more manageable.
    The transformation itself will be trained by backpropagation from all the outputs.
    
    Arguments:
        hparams {HyperParameters} -- the model hyperparameters
    
    Returns:
        tf.keras.Sequential -- A Sequential block that takes in a like multi-hot binary (bool) tensor, and returns a condensed float tensor.
    """
    block = tf.keras.Sequential(name="likes_condensing_block")
    block.add(tf.keras.layers.Lambda(lambda likes_onehot: tf.cast(likes_onehot, tf.bfloat16), input_shape=[hparams.num_like_pages]))

    if use_conv:
        block.add(tf.keras.layers.Reshape((hparams.num_like_pages, 1)))
        # while the output tensor is greater than the maximum size:
        while block.output_shape[-2] > hparams.likes_condensed_vector_max_size:
            print("adding output to reduce dimension of like vector:", block.output_shape)
            block.add(tf.keras.layers.Conv1D(
                filters=1,
                strides=hparams.likes_condensing_factor,
                kernel_size=hparams.likes_condensing_factor,
            ))
        block.add(tf.keras.layers.Flatten())
        return block
    else:
        for units in [256, 128, 64]:
            block.add(tf.keras.layers.Dense(
                units=units,
                activation=hparams.activation,
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=hparams.l1_reg, l2=hparams.l2_reg),
            ))
        return block


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
    
    # NOTE: trying dense vs conv at the moment.
    likes_condensing_block = likes_condensing(hparams)
    condensed_likes = likes_condensing_block(likes_features)

    feature_vector = tf.keras.layers.Concatenate(name="feature_vector")([text_features, image_features, condensed_likes])

    # MODEL OUTPUTS:
    gender_block = sequential_block("gender", hparams)
    gender_block.add(tf.keras.layers.Dense(units=1, activation="sigmoid", name="gender_out"))
    gender = gender_block(feature_vector)

    age_group_block = sequential_block("age_group", hparams)
    age_group_block.add(tf.keras.layers.Dense(units=4, activation="softmax", name="age_group_out"))
    age_group = age_group_block(feature_vector)
   
    personality_outputs: List[tf.Tensor] = []
    for personality_trait in ["ext", "ope", "agr", "neu", "con"]:
        block = sequential_block(personality_trait, hparams)
        block.add(tf.keras.layers.Dense(units=1, activation="sigmoid", name=f"{personality_trait}_sigmoid"))
        block.add(personality_scaling(f"{personality_trait}_scaling"))

        output_tensor = block(feature_vector)
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
