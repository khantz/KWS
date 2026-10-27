import tensorflow as tf


def build_mobile_net_v2_model(
    input_shape=(96, 96, 3),
    num_classes=9,
    alpha=0.35,
    weights="imagenet",
    freeze_base=True,
):
    """
    Build a MobileNetV2-based keyword spotting model.
    """

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights=weights,
    )

    base_model.trainable = not freeze_base

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=input_shape
            ),

            base_model,

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dropout(
                0.3
            ),

            tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
            ),
        ],
        name="MobileNetV2_keyword_spotting",
    )

    return model