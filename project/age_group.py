# function

import tensorflow as tf

def get_age_model() -> tf.keras.Model:
    age_model_path = 'saved_models/age_model_embedding_2000.h5'

    num_layers=2
    dense_units=64
    learning_rate=0.00005
    l1_reg=0.0025
    l2_reg=0.005
    dropout_rate=0.1
    num_text_features = 91
    num_image_features = 65
    max_len = 2000

    image_features = tf.keras.Input([num_image_features], dtype=tf.float32, name="image_features")
    text_features  = tf.keras.Input([num_text_features], dtype=tf.float32, name="text_features")
    likes_features = tf.keras.Input([max_len], dtype=tf.int32, name="likes_features")

    likes_embedding_block = tf.keras.Sequential(name="likes_embedding_block")
    likes_embedding_block.add(tf.keras.layers.Embedding(10000, 8, input_length=max_len))
    likes_embedding_block.add(tf.keras.layers.Flatten())

    condensed_likes = likes_embedding_block(likes_features)

    dense_layers = tf.keras.Sequential(name="dense_layers")
    dense_layers.add(tf.keras.layers.Concatenate())
    for i in range(num_layers):
        dense_layers.add(tf.keras.layers.Dense(
            units=dense_units,
            activation= 'tanh', #'tanh',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
            ))

        dense_layers.add(tf.keras.layers.Dropout(dropout_rate))

    features = dense_layers([text_features, image_features, condensed_likes])

    age_group = tf.keras.layers.Dense(units=4, activation="softmax", name="age_group")(features)

    model_age = tf.keras.Model(
        inputs=[text_features, image_features, likes_features],
        outputs= age_group
    )

    model_age.compile(
        optimizer = tf.keras.optimizers.get({"class_name": 'ADAM',
                                   "config": {"learning_rate": 0.0005}}),
        loss = 'categorical_crossentropy',
        metrics = ['acc', 'categorical_accuracy']
    )

    model_age.load_weights(age_model_path)

    return model_age
