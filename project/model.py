"""
@Fabrice: Here's an example of the kind of Model we could potentially use. (given what we know about the inputs we're gonna receive.)
"""

import tensorflow as tf
import dataclasses
from dataclasses import dataclass, field
from tensorboard.plugins.hparams import api as hp
from typing import Callable
from typing import *
from utils import JsonSerializable

best_model_so_far = "checkpoints/shared-embedding-02/2019-12-14_03-59-18"
""" New Model:

Total epochs: 0064, log_dir: ./checkpoints/shared-embedding-02/2019-12-14_03-59-18, total improvement: -0.299338481903076
	age_group_categorical_accuracy: 0.000 gender_binary_accuracy: 0.800 ext_root_mean_squared_error: 0.785 ope_root_mean_squared_error: 0.612 agr_root_mean_squared_error: 0.654 neu_root_mean_squared_error: 0.794 con_root_mean_squared_error: 0.706 
	loss: 7.232 age_group_loss: 3.312 gender_loss: 0.440 ext_loss: 0.615 ope_loss: 0.373 agr_loss: 0.428 neu_loss: 0.632 con_loss: 0.499 gender_recall: 0.898 
	BEATING THE BASELINE AT 'gender_binary_accuracy': (Baseline: 0.5910, Ours: 0.8000, improvement of 0.2090)
	BEATING THE BASELINE AT 'ext_root_mean_squared_error': (Baseline: 0.7880, Ours: 0.7850, improvement of 0.0030)
	BEATING THE BASELINE AT 'ope_root_mean_squared_error': (Baseline: 0.6520, Ours: 0.6120, improvement of 0.0400)
	BEATING THE BASELINE AT 'agr_root_mean_squared_error': (Baseline: 0.6650, Ours: 0.6538, improvement of 0.0112)
	BEATING THE BASELINE AT 'neu_root_mean_squared_error': (Baseline: 0.7980, Ours: 0.7944, improvement of 0.0036)
	BEATING THE BASELINE AT 'con_root_mean_squared_error': (Baseline: 0.7340, Ours: 0.7062, improvement of 0.0278)
	Hparams: HyperParameters(batch_size=64, optimizer='ADAM', learning_rate=0.001, num_like_pages=10000, gender_loss_weight=1.0, age_loss_weight=1.0, max_number_of_likes=2000, embedding_dim=8, shared_likes_embedding=True, use_custom_likes=True, gender=TaskHyperParameters(name='gender', num_layers=1, num_units=32, activation='tanh', use_batchnorm=False, use_dropout=True, dropout_rate=0.1, use_image_features=True, use_likes=True, l1_reg=0.005, l2_reg=0.005, embed_likes=False), age_group=TaskHyperParameters(name='age_group', num_layers=2, num_units=64, activation='tanh', use_batchnorm=False, use_dropout=True, dropout_rate=0.1, use_image_features=True, use_likes=True, l1_reg=0.005, l2_reg=0.005, embed_likes=False), personality=TaskHyperParameters(name='personality', num_layers=1, num_units=8, activation='tanh', use_batchnorm=False, use_dropout=True, dropout_rate=0.1, use_image_features=False, use_likes=False, l1_reg=0.005, l2_reg=0.005, embed_likes=False))

"""

# best_model_so_far = "checkpoints/one-model-each-marie-2/2019-11-25_21-14-40"
"""

Total epochs: 0012, log_dir: ./checkpoints/one-model-each-marie-2/2019-11-25_21-14-40, total improvement: -0.3390645436495541
	age_group_categorical_accuracy: 0.011 gender_binary_accuracy: 0.783 ext_root_mean_squared_error: 0.795 ope_root_mean_squared_error: 0.617 agr_root_mean_squared_error: 0.656 neu_root_mean_squared_error: 0.790 con_root_mean_squared_error: 0.726 
	loss: 6.400 age_group_loss: 3.323 gender_loss: 0.478 ext_loss: 0.631 ope_loss: 0.382 agr_loss: 0.432 neu_loss: 0.626 con_loss: 0.527 gender_recall: 0.937 
	BEATING THE BASELINE AT 'gender_binary_accuracy': (Baseline: 0.5910, Ours: 0.7832, improvement of 0.1922)
	BEATING THE BASELINE AT 'ope_root_mean_squared_error': (Baseline: 0.6520, Ours: 0.6174, improvement of 0.0346)
	BEATING THE BASELINE AT 'agr_root_mean_squared_error': (Baseline: 0.6650, Ours: 0.6564, improvement of 0.0086)
	BEATING THE BASELINE AT 'neu_root_mean_squared_error': (Baseline: 0.7980, Ours: 0.7902, improvement of 0.0078)
	BEATING THE BASELINE AT 'con_root_mean_squared_error': (Baseline: 0.7340, Ours: 0.7259, improvement of 0.0081)
	Hparams: HyperParameters(batch_size=128, num_layers=1, dense_units=32, activation='tanh', optimizer='ADAM', learning_rate=0.01, l1_reg=0.0, l2_reg=0.005, num_like_pages=10000, use_dropout=False, dropout_rate=0.1, use_batchnorm=False, gender_loss_weight=1.0, age_loss_weight=1.0, gender_num_layers=1, gender_num_units=32, gender_use_batchnorm=False, gender_use_dropout=False, gender_dropout_rate=0.1, gender_use_likes=False, gender_likes_condensing_layers=0, gender_likes_condensing_units=0, age_group_num_layers=2, age_group_num_units=64, age_group_use_batchnorm=False, age_group_use_dropout=False, age_group_dropout_rate=0.1, age_group_use_likes=True, age_group_likes_condensing_layers=1, age_group_likes_condensing_units=16, personality_num_layers=1, personality_num_units=8, personality_use_batchnorm=False, personality_use_dropout=False, personality_dropout_rate=0.1, personality_use_likes=False, personality_likes_condensing_layers=0, personality_likes_condensing_units=0)

"""


@dataclass
class TaskHyperParameters(JsonSerializable):
    """
    HyperParameters for a task-specific model
    """
    # name of the task
    name: str
    # number of dense layers
    num_layers: int = 1
    # units per layer
    num_units: int = 8
    # activation function
    activation: str = "tanh"
    # wether or not to use batch normalization after each dense layer
    use_batchnorm: bool = False
    # wether or not to use dropout after each dense layer
    use_dropout: bool = True
    # the dropout rate
    dropout_rate: float = 0.1
    # wether or not image features should be used as input
    use_image_features: bool = True
    # wether or not 'likes' features should be used as input
    use_likes: bool = True
    # L1 regularization coefficient
    l1_reg: float = 0.005
    # L2 regularization coefficient
    l2_reg: float = 0.005
    # Wether or not a task-specific Embedding layer should be used on the 'likes' features.
    # When set to 'True', it is expected that there no shared embedding is used.
    embed_likes: bool = False

@dataclass
class HyperParameters(JsonSerializable):
    """Hyperparameters of our model."""
    # the batch size
    batch_size: int = 128

    # the activation function used after each dense layer
    activation: str = "tanh"
    # Which optimizer to use during training.
    optimizer: str = "sgd"
    # Learning Rate
    learning_rate: float = 0.001

    # number of individual 'pages' that were kept during preprocessing of the 'likes'.
    # This corresponds to the number of entries in the multi-hot like vector.
    num_like_pages: int = 10_000

    gender_loss_weight: float   = 1.0
    age_loss_weight: float      = 1.0

    num_text_features: ClassVar[int] = 91
    num_image_features: ClassVar[int] = 65

    max_number_of_likes: int = 2000
    embedding_dim: int = 8

    shared_likes_embedding: bool = True

    # Wether or not to use Rémi's better kept like pages
    use_custom_likes: bool = True

    # Gender model settings:
    gender: TaskHyperParameters = TaskHyperParameters(
        "gender",
        num_layers=1,
        num_units=32,
        use_batchnorm=False,
        use_dropout=True,
        dropout_rate=0.1,
        use_image_features=True,
        use_likes=True,
    )

    # Age Group Model settings:
    age_group: TaskHyperParameters = TaskHyperParameters(
        "age_group",
        num_layers=2,
        num_units=64,
        use_batchnorm=False,
        use_dropout=True,
        dropout_rate=0.1,
        use_image_features=True,
        use_likes=True,
    )

    # Personality Model(s) settings:
    personality: TaskHyperParameters = TaskHyperParameters(
        "personality",
        num_layers=1,
        num_units=8,
        use_batchnorm=False,
        use_dropout=True,
        dropout_rate=0.1,
        use_image_features=False,
        use_likes=False,
    )


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


# define a Model type, for simplicity
Model = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]

def task_model(task_params: TaskHyperParameters, name: str = None) -> tf.keras.Sequential:
    # Model:
    model = tf.keras.Sequential(name=task_params.name if name is None else name)

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


def apply_model(model: tf.keras.Sequential, hparams: HyperParameters, task_params: TaskHyperParameters,  name: str = None) -> Model:
    """Given a task-specific model and the related HyperParameters, this
    function returns a Callable which will apply the task-specific model
    to the selected inputs and return the corresponding output Tensor. 
    
    Args:
        model (tf.keras.Sequential): The Task-specific model which will be applied to the inputs and which returns the output tensor.
        hparams (HyperParameters): The global hyperparameters
        task_params (TaskHyperParameters): The task-specific hyperparameters
        name (str, optional): A custom name to use for the components of this model. Defaults to None, in which case the value of `task_params.name` is used.
    
    Returns:
        Model: A Callable which accepts the Text, Image and Relation tensors and returns an output Tensor.
    """

    
    def apply(text_features: tf.Tensor, image_features: tf.Tensor, likes_features: tf.Tensor) -> tf.Tensor:
        model_inputs = [text_features]
        if task_params.use_image_features:
            model_inputs.append(image_features)
        if task_params.use_likes:
            if task_params.embed_likes:
                assert not hparams.shared_likes_embedding
                likes_embedding_model = likes_embedding(
                    name                = task_params.name if name is None else name,
                    num_page_likes      = hparams.num_like_pages,
                    max_number_of_likes = hparams.max_number_of_likes,
                )
                likes_embeddings = likes_embedding_model(likes_features)
                model_inputs.append(likes_embeddings)
            else:
                assert hparams.shared_likes_embedding
                model_inputs.append(likes_features)

        if len(model_inputs) == 1:
            model_output = model(model_inputs[0])
        else:
            model_output = model(model_inputs)
        return model_output 
    return apply


def gender_model(hparams: HyperParameters) -> Model:
    model = task_model(hparams.gender)
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid", name="gender_out"))
    model = apply_model(model, hparams, hparams.gender)
    return model

def age_group_model(hparams: HyperParameters) -> Model:
    model = task_model(hparams.age_group)
    model.add(tf.keras.layers.Dense(units=4, activation="softmax", name="age_group_out"))
    model = apply_model(model, hparams, hparams.age_group)
    return model

def personality_model(personality_trait: str, hparams: HyperParameters) -> Model:
    model = task_model(hparams.personality, name=personality_trait)
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid", name=f"{personality_trait}_out"))
    model.add(personality_scaling(f"{personality_trait}_out"))
    model = apply_model(model, hparams, hparams.personality, name=personality_trait)
    return model


def personality_scaling(name: str) -> tf.keras.layers.Layer:
    """Returns a layer that scales a sigmoid output [0, 1) output to the desired big-5 'personality' range of [1, 5)"""
    return tf.keras.layers.Lambda(lambda x: x * 4.0 + 1.0, name=name)


def get_model(hparams: HyperParameters) -> tf.keras.Model:
    # INPUTS: genderate content: (e.g., text, image and relations)
    # OUTPUTS:  Gender (ACC), Age (ACC), EXT (RMSE), ope (RMSE), AGR (RMSE), NEU (RMSE), CON (RMSE)
    
    # defining the inputs:
    # userid         =    tf.keras.Input([], dtype=tf.string, name="userid")
    text_features  =    tf.keras.Input([hparams.num_text_features], dtype=tf.float32, name="text_features")
    image_features =    tf.keras.Input([hparams.num_image_features], dtype=tf.float32, name="image_features")
    likes_features =    tf.keras.Input([hparams.max_number_of_likes], dtype=tf.int32, name="likes_features")

    if hparams.shared_likes_embedding:
        # single (shared) embedding model:
        likes_embedding_model = likes_embedding(
            name                = "shared_likes_embedding",
            num_page_likes      = hparams.num_like_pages,
            max_number_of_likes = hparams.max_number_of_likes,
        )
        likes_embeddings = likes_embedding_model(likes_features)
        # likes_embedding_model.summary()
    
    input_features = [text_features, image_features, likes_features]
    if hparams.shared_likes_embedding:
        task_model_inputs = [text_features, image_features, likes_embeddings]
    else:
        task_model_inputs = input_features

    gender_predictor = gender_model(hparams)
    gender = gender_predictor(*task_model_inputs)

    age_group_predictor = age_group_model(hparams)
    age_group = age_group_predictor(*task_model_inputs)

    personality_outputs: List[tf.Tensor] = []
    for personality_trait in ["ext", "ope", "agr", "neu", "con"]:
        personality_trait_predictor = personality_model(personality_trait, hparams)
        output_tensor = personality_trait_predictor(*task_model_inputs)
        personality_outputs.append(output_tensor)

    # MODEL OUTPUTS:
    model_outputs = [age_group, gender, *personality_outputs]

    model = tf.keras.Model(
        inputs=input_features,
        outputs=model_outputs,
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
        #NOTE: We can use this to change the importance of each output in the loss calculation, if need be.
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
