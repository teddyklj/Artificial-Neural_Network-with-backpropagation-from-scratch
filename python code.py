import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import time

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode the targets
encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Simple ANN Implementation
class SimpleANN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = y.shape[0]
        dz2 = (output - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_deriv(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Training loop
epochs = 5
ann = SimpleANN(input_dim=4, hidden_dim=8, output_dim=3, lr=0.1)

train_acc = []
epoch_times = []

for epoch in range(epochs):
    start = time.time()
    output = ann.forward(X_train)
    ann.backward(X_train, y_train, output)
    preds = ann.predict(X_train)
    acc = np.mean(np.argmax(y_train, axis=1) == preds)
    train_acc.append(acc)
    epoch_times.append(time.time() - start)
    print(f"Epoch {epoch+1}/{epochs} - Accuracy: {acc:.4f} - Time: {epoch_times[-1]:.4f}s")

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_acc, marker='o', color='teal')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy over Epochs')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), epoch_times, marker='o', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.title('Training Time per Epoch')
plt.grid(True)

plt.tight_layout()
plt.show()
