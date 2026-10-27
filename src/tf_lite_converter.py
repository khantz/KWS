import numpy as np
import tensorflow as tf


class TFLiteConverter:

    @staticmethod
    def convert_fp32(model, output_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        tflite_model = converter.convert()

        with open(output_path, "wb") as file:
            file.write(tflite_model)

        print(f"FP32 TFLite saved: {output_path}")

    @staticmethod
    def convert_int8(model, representative_features, output_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [
            tf.lite.Optimize.DEFAULT
        ]

        def representative_dataset():
            for feature in representative_features:
                sample = feature[np.newaxis, ...].astype(np.float32)
                yield [sample]

        converter.representative_dataset = representative_dataset

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]

        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        with open(output_path, "wb") as file:
            file.write(tflite_model)

        print(f"INT8 TFLite saved: {output_path}")