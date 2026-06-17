import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Multiply, Reshape
from tensorflow.keras.optimizers import Adam
import tensorflowjs as tfjs

# ============================================================================
# Council AI: Attention-Net Spatial Localization Pipeline
# ============================================================================
# Trains a custom CNN with Squeeze-and-Excitation (SE) blocks to output
# localized spatial attention probabilities.
# ============================================================================

def build_attention_net(input_shape=(224, 224, 3), num_regions=12):
    inputs = Input(shape=input_shape)
    
    # Simple backbone
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Squeeze and Excitation Block (Attention Mechanism)
    se = GlobalAveragePooling2D()(x)
    se = Dense(64 // 4, activation='relu')(se)
    se = Dense(64, activation='sigmoid')(se)
    se = Reshape((1, 1, 64))(se)
    x = Multiply()([x, se])
    
    # Classification Head (Region prediction)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_regions, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print("Building Attention-Net...")
    model = build_attention_net()
    print(model.summary())
    # Note: Requires bounding-box / region-labeled dataset
    print("Exporting mock architecture to TF.js format...")
    tfjs.converters.save_keras_model(model, '../public/models/attention')
