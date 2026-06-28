#Lab 4: Implementing a Deep Neural Network (DNN) for Digit Classification
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from sklearn.metrics import confusion_matrix

# 1. Load and Preprocess Data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values (0-255) to be between 0 and 1
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# 2. Build the DNN Architecture
# Add two dense hidden layers with ReLU activation
digit_classifier = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. Compile the Model
# Specify the optimizer, loss function, and metrics
digit_classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
training_history = digit_classifier.fit(train_images, train_labels, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = digit_classifier.evaluate(test_images, test_labels, verbose=0)

print("\nTest Accuracy:", test_accuracy)

# Generate predictions
predicted_digits = np.argmax(digit_classifier.predict(test_images), axis=1)

# Compute Confusion Matrix
confusion_matrix_result = confusion_matrix(test_labels, predicted_digits)

print("\nConfusion Matrix:")
print(confusion_matrix_result)

# Find Misclassified Images
misclassified_indices = np.where(test_labels != predicted_digits)[0]

print("First 10 Misclassified Image IDs:")
print(misclassified_indices[:10])

# Visualize misclassified samples
for idx in misclassified_indices[:3]:
    plt.imshow(test_images[idx], cmap='gray')
    plt.title(f"True: {test_labels[idx]}, Predicted: {predicted_digits[idx]}")
    plt.show()