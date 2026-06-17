import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflowjs as tfjs

# ============================================================================
# Council AI: DenseNet-121 Classification Training Pipeline
# ============================================================================
# This script trains a DenseNet-121 model on an actual dataset (e.g. Brain MRI)
# and exports the weights to the TensorFlow.js format used by the CouncilMed UI.
# ============================================================================

# Configuration
DATASET_DIR = './data/brain_tumor_mri'  # Path to your actual medical dataset
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 4 # [glioma, meningioma, notumor, pituitary]
EXPORT_DIR = '../public/models/classifier'

def build_model():
    # Load pretrained DenseNet121 (pretrained on ImageNet for transfer learning)
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    
    # Freeze base model for initial training
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    return model, base_model

def train():
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' not found. Please download a dataset and place it here.")
        print("Example: Kaggle Brain Tumor MRI dataset")
        return

    # Load and augment data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, 'Training'),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, 'Testing'),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    model, base_model = build_model()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('densenet_best.h5', save_best_only=True)
    ]

    print("--- Phase 1: Transfer Learning ---")
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    print("--- Phase 2: Fine-Tuning ---")
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
                  
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Export to TensorFlow.js format
    print(f"Exporting model to {EXPORT_DIR}...")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    tfjs.converters.save_keras_model(model, EXPORT_DIR)
    print("Export complete. The web application will now load these actual weights.")

if __name__ == '__main__':
    train()
