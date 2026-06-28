#Lab 2: Multilayer Perceptron (MLP)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# XOR Dataset
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,1,1,0])

model = Sequential([
    Input(shape=(2,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X,
    y,
    epochs=20,
    verbose=1
)

loss, acc = model.evaluate(X, y, verbose=0)

print("\nFinal Loss:", loss)
print("Final Accuracy:", acc)

predictions = (model.predict(X) > 0.5).astype(int)

print("\nPredictions:")
for i in range(len(X)):
    print(X[i], "->", predictions[i][0])
    
#Lab 2 Hyperparameter Comparison
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

experiments = [
    ("sigmoid", 0.01),
    ("sigmoid", 0.1),
    ("relu", 0.01),
    ("relu", 0.1)
]

for activation, lr in experiments:

    model = Sequential([
        Input(shape=(2,)),
        Dense(8, activation=activation),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X,
        y,
        epochs=20,
        verbose=0
    )

    loss, acc = model.evaluate(X, y, verbose=0)

    print(
        f"Activation={activation}, "
        f"LR={lr}, "
        f"Loss={loss:.4f}, "
        f"Accuracy={acc:.4f}"
    )