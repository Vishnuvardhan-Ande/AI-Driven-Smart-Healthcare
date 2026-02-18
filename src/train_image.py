import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

DATA_DIR = "data/chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def fine_tune_model():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint
    import os

    global DATA_DIR, IMG_SIZE, BATCH_SIZE

    print("Loading model for fine-tuning...")
    model = tf.keras.models.load_model("models/dense_best.h5")

    print("Unfreezing last DenseNet convolution blocks...")

    set_trainable = False
    for layer in model.layers:
        name = layer.name.lower()
        if "conv5_block" in name:
            set_trainable = True
        layer.trainable = set_trainable

    print("Layers successfully unfrozen from conv5_block onward.")

    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    ckpt = ModelCheckpoint(
        "models/dense_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        callbacks=[ckpt]
    )

    print("Fine-tuning complete! Updated model saved.")
