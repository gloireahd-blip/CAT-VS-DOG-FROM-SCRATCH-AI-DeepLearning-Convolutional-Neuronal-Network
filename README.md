#  🐾 Cat vs Dog CNN from Scratch AI DeepLearning Convolutional Neuronal Network
A custom-built Convolutional Neural Network (CNN) implemented from scratch using NumPy and SciPy to classify cats and dogs without deep learning frameworks. It features manual implementation of forward/backward propagation, Max-Pooling, and ReLU/Sigmoid activations to achieve 80% accuracy.

## Framework-Free Convolutional Neural Network Implementation

This project presents a Deep Learning model capable of classifying images of dogs and cats. The unique feature of this project is that it was coded entirely "from scratch", using only matrix mathematics, without relying on high-level libraries such as TensorFlow, PyTorch, or Keras.

## 🎯 Objective

To demonstrate a fundamental understanding of the internal mechanisms of a CNN (Convolutional Neural Network) by manually implementing forward propagation, backpropagation, and gradient optimization.

## 🛠️ Technical Stack

- **Matrix Calculus**: NumPy
- **Image Processing**: PIL (Pillow)
- **Mathematical Tools**: SciPy (used for basic 2D convolutions)
- **Visualization**: Matplotlib

## 🏗️ Model Pipeline

- **Preprocessing**: Loading, grayscale conversion, and resizing to 28x28 pixels.
- **Initialization**: Using the Xavier/Glorot method for weight initialization.
- **Convolutional Layer**: Feature extraction via learned filters.
- **Pooling Layer**: Dimensionality reduction using Max Pooling (stride of 2).
- **Dense Layers**: Calculation of the weighted sum  

  $$Z = W \cdot X + B$$

- **Activations**: ReLU for hidden layers and Sigmoid for the final classification.

## 📉 The Mathematics Behind the Model

### 1. Convolution & Pooling

The core of pattern extraction. The convolution applies a filter $M$ to the image $A$:

$$C_p = \sum A_i  \cdot M_{i+p}$$

Max Pooling reduces resolution while preserving the dominant information:

$$P_{ij} = \max(E[i:i+p, j:j+p])$$

### 2. Activation Functions

**ReLU**: Introduces non-linearity by blocking negative values ($\max(0, x)$).  

**Sigmoid**: Maps the output to a probability $a \in [0, 1]$:

$$\sigma(x)=\frac{1}{1+e^{-x}}$$

### 3. Training & Backpropagation

The model minimizes a cost function (Log-Loss/Bernoulli) via Gradient Descent. Parameters $W$ and $B$ are updated at each iteration:

$$W = W - \alpha \cdot \frac{\partial L}{\partial W}$$

$$B = B - \alpha \cdot \frac{\partial L}{\partial B}$$

## 📊 Results

- **Dataset**: 1,000 images (500 for training / 500 for testing).
- **Performance**: After 50 epochs, the model achieves **80% accuracy** on entirely unseen images.

<h3 align="center">1. Learning Curve</h3>
<p align="center">
  <img src="https://github.com/user-attachments/assets/16205fa4-0f40-4d8d-bdff-f038115d57c5" width="500" alt="Training Loss Curve">
</p>

<br> <h3 align="center">2. Model Predictions</h3>
<table align="center">
  <tr>
    <td align="center" valign="bottom">
      <img src="https://github.com/user-attachments/assets/4ad18018-9959-49d9-9d08-6de7b62001fe" width="350" alt="Dog Prediction Success">
      <br><em>Dog Prediction</em>
    </td>
    <td align="center" valign="bottom">
      <img src="https://github.com/user-attachments/assets/10eb77b5-522f-4c10-a795-d91301056eb7" width="350" alt="Cat Prediction Success">
      <br><em>Cat Prediction</em>
    </td>
  </tr>
</table>



## 🚀 Usage

The model weights are saved in `.npy` format for instant use without the need for retraining.

## 📁 Dataset
The dataset used for training (500 cats and 500 dogs) is available here:  
[Download Dataset from Google Drive](https://drive.google.com/file/d/1dnztCkhoU4CU4-TYglUTHHdCh_8D2-ut/view?usp=drive_link)
