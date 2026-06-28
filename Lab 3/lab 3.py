#Lab 3: Advanced Hyperparameter Optimization
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

trials = [
    (0.001, 16, 0.0),
    (0.001, 32, 0.2),
    (0.01, 32, 0.2),
    (0.01, 64, 0.3),
    (0.01, 64, 0.5)
]

for lr, hidden, dropout in trials:

    model = Sequential([
        Input(shape=(4,)),
        Dense(hidden, activation='relu'),
        Dropout(dropout),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        verbose=0
    )

    train_loss = history.history['loss'][-1]
    val_acc = history.history['val_accuracy'][-1]

    print(
        f"LR={lr}, Hidden={hidden}, "
        f"Dropout={dropout}, "
        f"Train Loss={train_loss:.4f}, "
        f"Val Acc={val_acc:.4f}"
    )
    
