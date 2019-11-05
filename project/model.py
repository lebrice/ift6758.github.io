"""
@Fabrice: Here's an example of the kind of Model we could potentially use. (given what we know about the inputs we're gonna receive.)
"""

import tensorflow as tf
import dataclasses
from dataclasses import dataclass, field
from tensorboard.plugins.hparams import api as hp
from typing import *

# Best model so far:

"""
checkpoints/slim_models/2019-11-03_21:02:19
Total epochs: 0133,
val_loss: 7.832,
log_dir: checkpoints/slim_models/2019-11-03_21:02:19,
    age_group_loss: 3.2845
    gender_loss: 0.6823
    ext_loss: 0.6500
    ope_loss: 0.3907
    agr_loss: 0.4342
    neu_loss: 0.6248
    con_loss: 0.5157
    age_group_categorical_accuracy: 0.2895
    gender_binary_accuracy: 0.5738
    gender_recall: 1.0000
    ext_root_mean_squared_error: 0.8064
    ope_root_mean_squared_error: 0.6251
    agr_root_mean_squared_error: 0.6590
    neu_root_mean_squared_error: 0.7904
    con_root_mean_squared_error: 0.7181
    val_loss: 7.8410
    val_age_group_loss: 3.3402
    val_gender_loss: 0.6772
    val_ext_loss: 0.6843
    val_ope_loss: 0.4380
    val_agr_loss: 0.4615
    val_neu_loss: 0.6744
    val_con_loss: 0.5460
    val_age_group_categorical_accuracy: 0.0058
    val_gender_binary_accuracy: 0.5905
    val_gender_recall: 1.0000
    val_ext_root_mean_squared_error: 0.8277
    val_ope_root_mean_squared_error: 0.6621
    val_agr_root_mean_squared_error: 0.6797
    val_neu_root_mean_squared_error: 0.8223
    val_con_root_mean_squared_error: 0.7408
    BEST VALIDATION LOSS: 7.831837256749471
hparams: HyperParameters(
    batch_size=64,
    num_layers=1,
    dense_units=32,
    activation='tanh',
    optimizer='sgd',
    learning_rate=0.01,
    l1_reg=0.005,
    l2_reg=0.005,
    num_like_pages=5000,
    use_dropout=True,
    dropout_rate=0.1,
    use_batchnorm=False
)

Total epochs: 0125,
val_loss: 7.970,
log_dir: checkpoints/slim_models/2019-11-03_21:06:43,
hparams: HyperParameters(
    batch_size=64,
    num_layers=1,
    dense_units=32,
    activation='tanh',
    optimizer='sgd',
    learning_rate=0.005,
    l1_reg=0.005,
    l2_reg=0.005,
    num_like_pages=5000,
    use_dropout=True,
    dropout_rate=0.1,
    use_batchnorm=False
)
"""

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
    num_like_pages: int = 10_000
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

    def likes_condensing_block(hparams: HyperParameters) -> tf.keras.Sequential:    
        # TODO: maybe use some kind of binary neural network here to condense a [`num_like_pages`] bool vector down to something more manageable (ex: [128] floats)
        block = tf.keras.Sequential(name="likes_condensing_block")
        block.add(tf.keras.layers.Lambda(lambda likes_onehot: tf.cast(likes_onehot, tf.bfloat16), input_shape=[hparams.num_like_pages]))
        block.add(tf.keras.layers.Reshape((hparams.num_like_pages, 1)))
        while block.output_shape[-2] > hparams.likes_condensed_vector_max_size:
            print("adding output to reduce dimension of like vector:", block.output_shape)
            block.add(tf.keras.layers.Conv1D(
                filters=1,
                strides=hparams.likes_condensing_factor,
                kernel_size=hparams.likes_condensing_factor,
            ))
        block.add(tf.keras.layers.Flatten())
        return block

    likes_condensing = likes_condensing_block(hparams)
    condensed_likes = likes_condensing(likes_features)

    feature_vector = tf.keras.layers.Concatenate(name="feature_vector")([text_features, image_features, condensed_likes])
    
    def sequential_block(name: str, hparams: HyperParameters) -> tf.keras.Sequential:
        """Series of dense layers
        
        Arguments:
            name {str} -- The name to give to this series of layers.
            hparams {HyperParameters} -- [description]
        
        Returns:
            [type] -- [description]
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
    
    # MODEL OUTPUTS:

    gender_block = sequential_block("gender", hparams)
    gender_block.add(tf.keras.layers.Dense(units=1, activation="sigmoid", name="gender_out"))
    gender = gender_block(feature_vector)

    age_group_block = sequential_block("age_group", hparams)
    age_group_block.add(tf.keras.layers.Dense(units=4, activation="softmax", name="age_group_out"))
    age_group = age_group_block(feature_vector)
    
    def personality_scaling(name: str) -> tf.keras.layers.Layer:
        """Returns a layer that scales a sigmoid output [0, 1) output to the desired 'personality' range of [1, 5)
        
        Arguments:
            name {str} -- the name to give to the layer.
        
        Returns:
            tf.keras.layers.Layer -- the layer to use.
        """
        return tf.keras.layers.Lambda(lambda x: x * 4.0 + 1.0, name=name)
    
    
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
