import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

# Define the U-Net model architecture
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Downsampling through the model
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Expanding path
    u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c2)
    u1 = concatenate([u1, c1])
    c3 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u1)
    c3 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c3)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Generate synthetic data
def generate_simple_data(num_samples=10):
    # Generate simple square shapes in images
    images = np.zeros((num_samples, 256, 256, 3), dtype=np.float32)
    labels = np.zeros((num_samples, 256, 256, 1), dtype=np.float32)

    for i in range(num_samples):
        cx, cy = np.random.randint(64, 192, 2)
        w, h = np.random.randint(40, 100, 2)
        images[i, cx-w//2:cx+w//2, cy-h//2:cy+h//2, :] = 1.0
        labels[i, cx-w//2:cx+w//2, cy-h//2:cy+h//2, 0] = 1.0

    return images, labels

# Create and train the model
train_images, train_labels = generate_simple_data(600)
test_images, test_labels = generate_simple_data(100)

model = unet_model()
model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# Make predictions
predicted_masks = model.predict(test_images)

# Visualize the results
plt.figure(figsize=(10, 8))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(test_images[i])
    plt.imshow(predicted_masks[i, :, :, 0] > 0.5, alpha=0.5)
plt.show()
