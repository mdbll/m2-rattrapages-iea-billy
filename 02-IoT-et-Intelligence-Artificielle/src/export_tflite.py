import os

import numpy as np
import tensorflow as tf

from data import load_dataset
from train import MODEL_PATH, MODELS_DIR

TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "har_cnn.tflite")


def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def tflite_accuracy(tflite_model, x_test, y_test):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    correct = 0
    for window, label in zip(x_test, y_test):
        interpreter.set_tensor(input_index, window[np.newaxis].astype(np.float32))
        interpreter.invoke()
        prediction = np.argmax(interpreter.get_tensor(output_index)[0])
        correct += int(prediction == label)

    return correct / len(y_test)


if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)
    tflite_model = convert_to_tflite(model)

    with open(TFLITE_MODEL_PATH, "wb") as file:
        file.write(tflite_model)

    _, _, x_test, y_test, _, _ = load_dataset()
    keras_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    lite_accuracy = tflite_accuracy(tflite_model, x_test, y_test)

    print()
    print("TFLite model saved in", TFLITE_MODEL_PATH)
    print(f"Keras model size:  {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")
    print(f"TFLite model size: {os.path.getsize(TFLITE_MODEL_PATH) / 1024:.1f} KB")
    print(f"Keras accuracy on test:  {keras_accuracy:.3f}")
    print(f"TFLite accuracy on test: {lite_accuracy:.3f}")
