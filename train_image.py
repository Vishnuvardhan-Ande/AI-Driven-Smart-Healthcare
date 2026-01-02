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
    """
    Fine‑tune DenseNet with stronger augmentation and callbacks to improve accuracy.
    This will overwrite models/dense_best.h5 used by the Flask app.
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    import os

    global DATA_DIR, IMG_SIZE, BATCH_SIZE

    print("Loading model for fine-tuning...")
    model = tf.keras.models.load_model("models/dense_best.h5")

    print("Unfreezing deeper DenseNet convolution blocks...")
    set_trainable = False
    for layer in model.layers:
        name = layer.name.lower()
        if "conv4_block" in name or "conv5_block" in name:
            set_trainable = True
        layer.trainable = set_trainable

    print("Layers successfully unfrozen from conv4_block onward.")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.15,
        shear_range=0.08,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    val_gen = val_datagen.flow_from_directory(
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
    lr_sched = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )
    early = EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[ckpt, lr_sched, early]
    )

    print("Fine-tuning complete! Updated model saved to models/dense_best.h5")
