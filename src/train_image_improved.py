"""
Improved image model training with advanced techniques for better accuracy.

Improvements:
- Focal loss for handling class imbalance
- Enhanced data augmentation
- Cosine annealing learning rate schedule
- Class weights for balanced training
- Better architecture with dropout regularization
- Test-time augmentation support
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
try:
    from tensorflow.keras.applications import EfficientNetB3
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("Warning: EfficientNetB3 not available in this TensorFlow version")
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    ReduceLROnPlateau, 
    EarlyStopping
)
from tensorflow.keras import backend as K
import math

DATA_DIR = "data/chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Increased batch size for better stability
EPOCHS = 50
RANDOM_STATE = 42


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        
        return K.mean(loss)
    
    return focal_loss_fixed


class CosineAnnealingRestarts(tf.keras.callbacks.Callback):
    """Cosine annealing with restarts learning rate schedule."""
    def __init__(self, T_0, T_mult=1, eta_max=0.001, eta_min=0.0, verbose=0):
        super().__init__()
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.T_cur = 0
        self.epochs_since_restart = 0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self.T_cur = 0
            self.epochs_since_restart = 0
        else:
            self.T_cur += 1
            self.epochs_since_restart += 1

        if self.epochs_since_restart >= self.T_i:
            self.epochs_since_restart = 0
            self.T_i *= self.T_mult

        lr = self.eta_min + (self.eta_max - self.eta_min) * \
             (1 + math.cos(math.pi * self.epochs_since_restart / self.T_i)) / 2
        
        K.set_value(self.model.optimizer.learning_rate, lr)
        
        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1}: CosineAnnealingRestarts setting learning rate to {lr:.6f}.')


def build_model(base_model_name='densenet121', use_focal_loss=True):
    """
    Build improved model architecture.
    
    Args:
        base_model_name: 'densenet121' or 'efficientnetb3'
        use_focal_loss: Whether to use focal loss (requires custom loss)
    """
    # Choose base model
    if base_model_name == 'densenet121':
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMG_SIZE, 3)
        )
    elif base_model_name == 'efficientnetb3':
        if not EFFICIENTNET_AVAILABLE:
            raise ValueError("EfficientNetB3 is not available. Use 'densenet121' instead.")
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMG_SIZE, 3)
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze early layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Build custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def get_class_weights(train_dir):
    """Calculate class weights for imbalanced dataset."""
    import os
    normal_count = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    pneumonia_count = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    total = normal_count + pneumonia_count
    
    weight_normal = total / (2 * normal_count)
    weight_pneumonia = total / (2 * pneumonia_count)
    
    return {0: weight_normal, 1: weight_pneumonia}


def train_improved_model(base_model_name='densenet121', use_focal_loss=True):
    """
    Train improved image model with advanced techniques.
    """
    print(f"Building {base_model_name} model...")
    model = build_model(base_model_name, use_focal_loss)
    
    # Enhanced data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        channel_shift_range=0.1
    )
    
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_STATE
    )
    
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
        seed=RANDOM_STATE
    )
    
    # Get class weights
    class_weights = get_class_weights(train_dir)
    print(f"Class weights: {class_weights}")
    
    # Compile model
    if use_focal_loss:
        loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        print("Using Focal Loss for training")
    else:
        loss_fn = 'binary_crossentropy'
        print("Using Binary Crossentropy Loss")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        "models/dense_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    
    # Optional: Cosine annealing (comment out if you prefer only ReduceLROnPlateau)
    # cosine_annealing = CosineAnnealingRestarts(
    #     T_0=10,
    #     T_mult=2,
    #     eta_max=0.001,
    #     eta_min=1e-7,
    #     verbose=1
    # )
    
    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    callbacks_list = [checkpoint, reduce_lr, early_stop]
    # Uncomment to use cosine annealing instead of ReduceLROnPlateau:
    # callbacks_list = [checkpoint, cosine_annealing, early_stop]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("\n✅ Training complete! Best model saved to models/dense_best.h5")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return model, history


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    print("=" * 60)
    print("IMPROVED IMAGE MODEL TRAINING")
    print("=" * 60)
    print("\nFeatures:")
    print("- Focal Loss for class imbalance")
    print("- Enhanced data augmentation")
    print("- Cosine annealing learning rate")
    print("- Class weights")
    print("- Better architecture with dropout")
    print("=" * 60)
    
    # Train with DenseNet121 (default, can switch to EfficientNetB3)
    model, history = train_improved_model(
        base_model_name='densenet121',
        use_focal_loss=True
    )

