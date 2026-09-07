# PRODIGY_ML_03
# 🐱🐶 Cats vs Dogs Image Classification using SVM

## 📌 Project Overview

This project implements a **Support Vector Machine (SVM)** to classify images of cats and dogs using the Kaggle Dogs vs Cats dataset.

The images are resized and converted into numerical features before training the SVM classifier.

## 📊 Dataset

**Dataset:** Dogs vs Cats  
**Source:** Kaggle Competition

The dataset contains images belonging to two classes:
- 🐱 Cat
- 🐶 Dog

For this project, a balanced subset of **5,000 images** was used:
- 2,500 Cat images
- 2,500 Dog images

## 🛠️ Technologies Used

- Python
- Google Colab
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PIL
- Joblib

## ⚙️ Methodology

1. Downloaded the Dogs vs Cats dataset from Kaggle.
2. Selected 2,500 cat and 2,500 dog images.
3. Resized images to **64 × 64 pixels**.
4. Converted images into numerical feature vectors.
5. Normalized pixel values between 0 and 1.
6. Split the dataset into:
   - 80% Training
   - 20% Testing
7. Trained an **SVM classifier with RBF kernel**.
8. Evaluated the model using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

## 🤖 Model

**Algorithm:** Support Vector Machine (SVM)

```python
SVC(kernel='rbf', C=10, gamma='scale')
