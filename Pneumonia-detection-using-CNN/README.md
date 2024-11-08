# Import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import cv2
from google.colab import drive
from keras.models import load_model

# Mount Google Drive
drive.mount('/content/drive')

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load training data from directory
train_images = "/content/drive/MyDrive/chest_xray/train"
train_generator = train_datagen.flow_from_directory(
    train_images,
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

# Data augmentation for validation images
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/chest_xray/val',
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

# Display some training images
plt.figure(figsize=(10, 10))
for i, idx in enumerate([41, 176, 354, 267, 710, 1090]):
    plt.subplot(3, 2, i + 1)
    img = plt.imread(train_generator.filepaths[idx])
    plt.imshow(img)
    plt.axis('off')
plt.show()

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Display model architecture
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=50, validation_data=validation_generator)

# Plot training & validation loss
plt.figure(figsize=(15, 10))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(fontsize=16)
plt.title("Loss Vs Epochs", fontsize=18)
plt.xlabel("Num. of Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.show()

# Plot training & validation accuracy
plt.figure(figsize=(15, 10))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(fontsize=16)
plt.title("Accuracy Vs Epochs", fontsize=18)
plt.xlabel("Num. of Epochs", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.show()

# Save the trained model
model_path = "/content/drive/MyDrive/trained_pneumonia_detection.h5"
model.save(model_path)

# Load the trained model
model = load_model(model_path)

# Evaluate the model
test_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/chest_xray/test',
    target_size=(300, 300),
    batch_size=128, 
    class_mode='binary'
)
eval_result = model.evaluate(test_generator)
print('Test Loss:', eval_result[0])
print('Test Accuracy:', eval_result[1])

# Predicting on new images
def predict_image(filepath):
    img = cv2.imread(filepath)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for plt display
        tempimg = img.copy()
        img = cv2.resize(img, (300, 300))
        img = img / 255.0
        img = img.reshape(1, 300, 300, 3)
        prediction = model.predict(img)[0][0]
        result = "Pneumonia" if prediction >= 0.5 else "Normal"
        plt.imshow(tempimg)
        plt.title(f"Prediction: {result}", fontsize=14)
        plt.axis('off')
        plt.show()
        print("Prediction:", result)
    else:
        print(f"Error loading image: {filepath}")

# Example prediction for Normal case
predict_image('/content/drive/MyDrive/chest_xray/test/NORMAL/IM-0017-0001.jpeg')

# Example prediction for Pneumonia case
predict_image('/content/drive/MyDrive/chest_xray/test/PNEUMONIA/person104_bacteria_492.jpeg')
