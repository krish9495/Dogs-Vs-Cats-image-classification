# 🐱🐶 Cats vs Dogs Image Classification

This project builds a Convolutional Neural Network (CNN) model to classify images of cats and dogs using TensorFlow and Keras. It involves loading image data, preprocessing, model building, training, and evaluation.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Improvements](#future-improvements)

---

## 📖 Overview

This deep learning project focuses on binary image classification using a custom Convolutional Neural Network. The goal is to accurately distinguish between images of cats and dogs.

---

## 📊 Dataset

- **Source:** [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- The dataset contains **25,000 labeled images** of cats and dogs (12,500 each).
- Images are loaded, resized, normalized, and split into training and validation sets.

---

## 💻 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib** for visualization
- **Google Colab / Jupyter Notebook** for execution

---

## 🧠 Model Architecture

The CNN model includes the following layers:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dropout layers to prevent overfitting
- Flatten and Dense layers
- Final sigmoid layer for binary classification

Example architecture:
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
📈 Training and Evaluation
Loss Function: Binary Crossentropy

Optimizer: Adam

Metrics: Accuracy

Model is trained using model.fit() with real-time data augmentation using ImageDataGenerator.

## ▶️ How to Run
### Clone the repository:
git clone https://github.com/yourusername/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier

### Download and extract the dataset to the appropriate folder (/data/train).

### Run the notebook:
jupyter notebook Cats_v_Dogs_classification.ipynb
✅ Results
Achieved >90% accuracy on validation set.

The model performs well in distinguishing between cat and dog images.

Loss and accuracy graphs show good convergence and minimal overfitting.

🔍 Conclusion
This project successfully demonstrates the use of CNNs for image classification tasks. It uses standard deep learning practices and achieves high accuracy on a well-known binary classification problem.

🚀 Future Improvements
Use pre-trained models like VGG16, ResNet (Transfer Learning)

Tune hyperparameters (learning rate, batch size)

Deploy as a web app using Flask or Streamlit
