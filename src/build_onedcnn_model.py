import tensorflow as tf


def build_cnn1d_model(
    input_shape=(71, 13),
    num_classes=9
):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=input_shape
        ),

        # --------------------------------------------
        # Initial 1D convolution
        # --------------------------------------------

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            use_bias=False
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        # --------------------------------------------
        # Convolutional block 1
        # --------------------------------------------

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            use_bias=False
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        # --------------------------------------------
        # Convolutional block 2
        # --------------------------------------------

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            use_bias=False
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        # --------------------------------------------
        # Classification head
        # --------------------------------------------

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model