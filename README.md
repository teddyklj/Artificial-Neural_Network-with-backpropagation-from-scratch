🧠 Artificial Neural Network with Backpropagation from Scratch
This repository contains a foundational deep learning project: a simple Artificial Neural Network (ANN) built entirely from scratch using only NumPy. The goal is to provide a hands-on understanding of the core mechanics of neural networks, particularly the backpropagation algorithm.

🌟 Objective
The primary goal is to implement an ANN that can effectively classify the Iris flower species dataset. This project serves as an educational tool to:

🔁 Understand the forward pass: how data flows through the network to generate predictions.
🔄 Master the backward pass: how backpropagation calculates gradients to update weights.
📈 Visualize the learning process: track how accuracy improves and training time evolves over epochs.
🚀 Key Concepts & Features
Custom SimpleANN Class: A complete, self-contained Python class encapsulating the neural network logic. No TensorFlow or PyTorch.
Backpropagation Explained: Manually implements backpropagation and uses gradient descent to minimize the cross-entropy loss.
Fundamental Building Blocks:
🧮 Activation Functions: ReLU (hidden layers), Softmax (output layer)
📉 Loss Function: Cross-Entropy Loss for multi-class classification
🔢 One-Hot Encoding: Transforms class labels for compatibility
📏 Feature Scaling: StandardScaler from scikit-learn for normalization
📊 Results & Output
The model was trained for just 5 epochs and showed rapid, effective learning, demonstrating the power of backpropagation.

Training Progress

Training Accuracy (Left): The model's accuracy quickly climbs to nearly 80%, showing it is learning the underlying patterns in the data effectively.
Training Time (Right): Each epoch is computationally inexpensive, taking only milliseconds to complete.
🛠️ How to Run
Clone the Repository

git clone https://github.com/Ayush03A/Artificial-Neural-Network-with-Backpropagation-from-Scratch.git
cd Artificial-Neural-Network-with-Backpropagation-from-Scratch
Install Dependencies

pip install numpy scikit-learn matplotlib
Run the Script This will train the model and save the output plots to your project directory.

✅ Conclusion
This project is a testament to the efficient and elegant design of the backpropagation algorithm. By building an ANN from scratch, we gain a deep, intuitive understanding of how neural networks learn and adapt—achieving high accuracy on a real-world dataset with only a few lines of NumPy-powered code.
