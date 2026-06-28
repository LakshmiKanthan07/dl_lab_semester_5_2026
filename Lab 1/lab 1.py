#Lab 1: Perceptron Learning Implementation
import numpy as np

def train_perceptron(X, y, lr=0.1, epochs=20):
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        errors = 0

        for i in range(len(X)):
            linear_output = np.dot(X[i], weights) + bias

            y_pred = 1 if linear_output >= 0 else 0

            update = lr * (y[i] - y_pred)

            weights += update * X[i]
            bias += update

            if update != 0:
                errors += 1

        if errors == 0:
            print(f"Converged at epoch {epoch+1}")
            break

    return weights, bias


# AND Gate Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

weights, bias = train_perceptron(X, y)

print("Weights:", weights)
print("Bias:", bias)

print("\nPredictions:")
for x in X:
    prediction = 1 if np.dot(x, weights) + bias >= 0 else 0
    print(x, "->", prediction)
    
#Decision boundary
import matplotlib.pyplot as plt
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i,0], X[i,1],
                    color='red',
                    marker='o',
                    s=100,
                    label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i,0], X[i,1],
                    color='blue',
                    marker='s',
                    s=100,
                    label='Class 1')

# Decision Boundary
x_values = np.linspace(-0.5, 1.5, 100)

if weights[1] != 0:
    y_values = -(weights[0] * x_values + bias) / weights[1]

    plt.plot(
        x_values,
        y_values,
        color='green',
        linewidth=2,
        label='Decision Boundary'
    )

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Decision Boundary (AND Gate)")
plt.grid(True)
plt.legend()

plt.show()