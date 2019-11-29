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
    
    # the activation function used after each dense layer
    activation: str = "tanh"
    # Which optimizer to use during training.
    optimizer: str = "sgd"
    # Learning Rate
    learning_rate: float = 0.001

    # L1 regularization coefficient
    l1_reg: float = 0.005
    # L2 regularization coefficient
    l2_reg: float = 0.005

    # number of individual 'pages' that were kept during preprocessing of the 'likes'.
    # This corresponds to the number of entries in the multi-hot like vector.
    num_like_pages: int = 10_000
    # wether or not Dropout layers should be used
    use_dropout: bool = True
    # the dropout rate
    dropout_rate: float = 0.1
    # wether or not Batch Normalization should be applied after each dense layer.
    use_batchnorm: bool = False

    gender_loss_weight: float   = 1.0
    age_loss_weight: float      = 10.0

    num_text_features: ClassVar[int] = 91
    num_image_features: ClassVar[int] = 65


    # Gender model settings:
    gender_num_layers: int = 1
    gender_num_units: int = 32
    gender_use_batchnorm: bool = False
    gender_use_dropout: bool = True
    gender_dropout_rate: float = 0.1
    gender_use_likes: bool = True
    gender_use_image_features: bool = True

    # Age Group Model settings:
    age_group_num_layers: int = 2
    age_group_num_units: int = 64
    age_group_use_batchnorm: bool = False
    age_group_use_dropout: bool = True
    age_group_dropout_rate: float = 0.1
    age_group_use_likes: bool = True
    age_group_use_image_features: bool = True

    # Personality Model(s) settings:
    personality_num_layers: int = 1
    personality_num_units: int = 8
    personality_use_batchnorm: bool = False
    personality_use_dropout: bool = True
    personality_dropout_rate: float = 0.1
    personality_use_image_features: bool = False
    personality_use_likes: bool = False

    # settings related to like embeddings:
    max_number_of_likes: int = 2000 # The maximum number of a user's likes that will be considered
    embedding_dim: int = 8 # the output size of the like embedding layer


best_model_so_far = "checkpoints/embedding/2019-11-29_15-45-36"


@dataclass
class TaskHyperParameters:
    name: str
    num_layers: int = 1
    num_units: int = 8
    activation: str = "tanh"
    use_batchnorm: bool = False
    use_dropout: bool = True
    dropout_rate: float = 0.1
    use_image_features: bool = True
    use_likes: bool = True
    # L1 regularization coefficient
    l1_reg: float = 0.005
    # L2 regularization coefficient
    l2_reg: float = 0.005

def likes_embedding(name: str, num_page_likes: int, max_number_of_likes: int, embedding_dim: int = 8) -> tf.keras.Sequential:
    likes_condensing = tf.keras.Sequential(name=f"{name}_likes_condensing")
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=num_page_likes,
        output_dim=embedding_dim,
        input_length=max_number_of_likes,
        mask_zero=True,
        name=f"{name}_likes_embedding",
    )
    likes_condensing.add(embedding_layer)
    likes_condensing.add(tf.keras.layers.Flatten())
    return likes_condensing


def task_model(hparams: HyperParameters, task_params: TaskHyperParameters) -> tf.keras.Sequential:
    # Model:
    model = tf.keras.Sequential(name=task_params.name)

    if task_params.use_image_features or task_params.use_likes:
        model.add(tf.keras.layers.Concatenate())

    for i in range(task_params.num_layers):
        model.add(tf.keras.layers.Dense(
            units=task_params.num_units,
            activation=task_params.activation,
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=task_params.l1_reg, l2=task_params.l2_reg),
        ))
        if task_params.use_batchnorm:
            model.add(tf.keras.layers.BatchNormalization())
        if task_params.use_dropout:
            model.add(tf.keras.layers.Dropout(task_params.dropout_rate))
    
    return model


def gender_model(hparams: HyperParameters, image_features: tf.Tensor, text_features: tf.Tensor, likes_features: tf.Tensor) -> tf.keras.Sequential:
    task_model_params = TaskHyperParameters(
        name="gender",
        # Gender model settings:
        num_layers          = hparams.gender_num_layers,
        num_units           = hparams.gender_num_units,
        use_batchnorm       = hparams.gender_use_batchnorm,
        use_dropout         = hparams.gender_use_dropout,
        dropout_rate        = hparams.gender_dropout_rate,
        use_likes           = hparams.gender_use_likes,
        use_image_features  = hparams.gender_use_image_features,
    )
    model = task_model(hparams, task_model_params)
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid", name="gender_out"))
    
    model_inputs = [text_features]
    if task_model_params.use_image_features:
        model_inputs.append(image_features)
    
    if task_model_params.use_likes:    
        likes_embedding_model = likes_embedding(
            name                = task_model_params.name,
            num_page_likes      = hparams.num_like_pages,
            max_number_of_likes = hparams.max_number_of_likes,
        )
        likes_embeddings = likes_embedding_model(likes_features)
        model_inputs.append(likes_embeddings)
    if len(model_inputs) == 1:
        model_output = model(model_inputs[0])
    else:
        model_output = model(model_inputs)
    return model_output   


def age_group_model(hparams: HyperParameters, image_features: tf.Tensor, text_features: tf.Tensor, likes_features: tf.Tensor) -> tf.keras.Sequential:
    task_model_params = TaskHyperParameters(
        name="age_group",
        # Gender model settings:
        num_layers          = hparams.age_group_num_layers,
        num_units           = hparams.age_group_num_units,
        use_batchnorm       = hparams.age_group_use_batchnorm,
        use_dropout         = hparams.age_group_use_dropout,
        dropout_rate        = hparams.age_group_dropout_rate,
        use_likes           = hparams.age_group_use_likes,
        use_image_features  = hparams.age_group_use_image_features,
    )
    model = task_model(hparams, task_model_params)
    model.add(tf.keras.layers.Dense(units=4, activation="softmax", name="age_group_out"))
    
    model_inputs = [text_features]
    if task_model_params.use_image_features:
        model_inputs.append(image_features)
    
    if task_model_params.use_likes:    
        likes_embedding_model = likes_embedding(
            name                = task_model_params.name,
            num_page_likes      = hparams.num_like_pages,
            max_number_of_likes = hparams.max_number_of_likes,
        )
        likes_embeddings = likes_embedding_model(likes_features)
        model_inputs.append(likes_embeddings)
    if len(model_inputs) == 1:
        model_output = model(model_inputs[0])
    else:
        model_output = model(model_inputs)
    return model_output   


def personality_model(personality_trait: str, hparams: HyperParameters, image_features: tf.Tensor, text_features: tf.Tensor, likes_features: tf.Tensor) -> tf.keras.Sequential:
    task_model_params = TaskHyperParameters(
        name=personality_trait,
        # Gender model settings:
        num_layers          = hparams.personality_num_layers,
        num_units           = hparams.personality_num_units,
        use_batchnorm       = hparams.personality_use_batchnorm,
        use_dropout         = hparams.personality_use_dropout,
        dropout_rate        = hparams.personality_dropout_rate,
        use_likes           = hparams.personality_use_likes,
        use_image_features  = hparams.personality_use_image_features,
    )
    model = task_model(hparams, task_model_params)
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid", name=f"{personality_trait}_out"))
    model.add(personality_scaling(f"{personality_trait}_out"))
    
    model_inputs = [text_features]
    if task_model_params.use_image_features:
        model_inputs.append(image_features)
    if task_model_params.use_likes:    
        likes_embedding_model = likes_embedding(
            name                = task_model_params.name,
            num_page_likes      = hparams.num_like_pages,
            max_number_of_likes = hparams.max_number_of_likes,
        )
        likes_embeddings = likes_embedding_model(likes_features)
        model_inputs.append(likes_embeddings)
    if len(model_inputs) == 1:
        model_output = model(model_inputs[0])
    else:
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
    likes_features =    tf.keras.Input([hparams.max_number_of_likes], dtype=tf.int32, name="likes_features")
    
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
