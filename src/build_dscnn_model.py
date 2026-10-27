import tensorflow as tf


def build_dscnn_model(
    input_shape=(71, 13, 1),
    num_classes=9
):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=input_shape
        ),

        # --------------------------------------------
        # Initial standard convolution
        # --------------------------------------------

        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        # --------------------------------------------
        # Depthwise separable block 1
        # --------------------------------------------

        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            padding="same",
            use_bias=False
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(1, 1),
            padding="same",
            use_bias=False
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        # --------------------------------------------
        # Depthwise separable block 2
        # --------------------------------------------

        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            padding="same",
            use_bias=False
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(1, 1),
            padding="same",
            use_bias=False
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        # --------------------------------------------
        # Classification head
        # --------------------------------------------

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model