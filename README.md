# ✍️ Signature Recognition using CNN

A deep learning-based **offline signature recognition system** that uses a **Convolutional Neural Network (CNN)** to classify handwritten signatures into different classes.

The system is trained using the **CEDAR signature dataset** and performs image preprocessing, model training, and signature classification. It also generates **prediction confidence scores** for each class and saves the results in a CSV file for further analysis.

## ✨ Features

* ✍️ Recognizes handwritten signatures using CNN
* 🧠 Uses deep learning for signature classification
* 🖼️ Preprocesses signature images before training
* 📚 Trained on the **CEDAR signature dataset**
* 📊 Generates prediction confidence scores for each class
* 📁 Saves prediction results to a CSV file
* 🔬 Supports further analysis of model predictions

## 🛠️ Tech Stack

**Programming:** Python
**Deep Learning:** TensorFlow, Keras
**Data Processing:** NumPy, Pandas
**Machine Learning:** Scikit-learn
**Image Processing:** Pillow
**Dataset:** CEDAR Signature Dataset

## 🔄 Project Workflow

```text
Signature Images
       ↓
Image Preprocessing
       ↓
Training Dataset
       ↓
CNN Model
       ↓
Model Training
       ↓
Signature Classification
       ↓
Prediction Confidence Scores
       ↓
CSV Results
```

## 🎯 Objective

The objective of this project is to build an offline deep learning system capable of learning signature patterns from handwritten signature images and classifying them into their respective classes.

## 📊 Output

The trained CNN model produces:

* Predicted signature class
* Prediction confidence score
* Results stored in CSV format for analysis
