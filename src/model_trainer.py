import json

import tensorflow as tf


class ModelTrainer:

    def __init__(
        self,
        learning_rate=0.001,
        epochs=30,
        batch_size=64
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def compile_model(self, model):

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def train(
        self,
        model,
        x_train,
        y_train,
        x_val,
        y_val
    ):

        history = model.fit(
            x_train,
            y_train,

            validation_data=(
                x_val,
                y_val
            ),

            epochs=self.epochs,

            batch_size=self.batch_size,

            shuffle=True
        )

        return history

    @staticmethod
    def save_history(history, output_path):

        history_data = {
            key: [
                float(value)
                for value in values
            ]
            for key, values in history.history.items()
        }

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history_data,
                file,
                indent=4
            )