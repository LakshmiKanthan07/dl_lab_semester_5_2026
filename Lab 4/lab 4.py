#Lab 4: Implementing a Deep Neural Network (DNN) for Digit Classification
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from sklearn.metrics import confusion_matrix
import numpy as np

#1. Load and Preprocess Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#TODO: Normalize the pixel values (0-255) to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 2. Build the DNN Architecture
#TODO: Add two dense hidden layers with ReLU activation
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. Compile the Model
#TODO: Specify the optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(  x_test, y_test, verbose=0)

print("\nTest Accuracy:", test_acc)

# Predictions
y_pred = np.argmax( model.predict(x_test), axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

#To Find Real Misclassified Images
misclassified = np.where(y_test != y_pred)[0]

print("First 10 Misclassified Image IDs:")
print(misclassified[:10])

import matplotlib.pyplot as plt

for i in misclassified[:3]:
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"True: {y_test[i]}, Predicted: {y_pred[i]}")
    plt.show()