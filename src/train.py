import json
import os

import tensorflow as tf

from data import ACTIVITIES, load_dataset

MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODELS_DIR, "har_cnn.keras")
NORM_STATS_PATH = os.path.join(MODELS_DIR, "norm_stats.json")

EPOCHS = 30
BATCH_SIZE = 32


def build_model(input_shape, nb_classes):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv1D(16, kernel_size=5, activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=5, activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(nb_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_norm_stats(mean, std):
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    with open(NORM_STATS_PATH, "w") as file:
        json.dump(stats, file, indent=2)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)

    x_train, y_train, x_test, y_test, mean, std = load_dataset()

    model = build_model(input_shape=x_train.shape[1:], nb_classes=len(ACTIVITIES))
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    save_norm_stats(mean, std)

    print("Model saved in", MODEL_PATH)
    print("Normalization stats saved in", NORM_STATS_PATH)
