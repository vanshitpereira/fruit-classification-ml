# Fruit Classification using Multiple Classifiers

An image-based fruit classification system using multiple machine
learning classifiers and ensemble learning with majority voting.

## Project Objective

The objective of this project is to classify fruit images using multiple
machine learning algorithms and combine their predictions using ensemble
voting to obtain a robust final prediction.

## Dataset

The project uses the Fruits-360 dataset.

The complete dataset is not stored in this repository because of its
large size. The dataset will be accessed during development using
Google Colab.

## Machine Learning Models

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

## Ensemble Learning

The project implements:

- Hard Majority Voting
- Soft Voting

The final system compares individual classifiers with ensemble methods.

## Image Processing

The planned image processing pipeline includes:

1. Image resizing
2. Image normalization
3. Color processing
4. Feature extraction
5. Feature scaling

## Feature Extraction

Features will be investigated using:

- Color features
- HOG features
- Texture features such as LBP
- Combined feature representations

The final feature combination will be selected based on experimental
results.

## Evaluation Metrics

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Technologies

- Python
- Google Colab
- OpenCV
- Scikit-learn
- Scikit-image
- Flask
- React
- GitHub

## System Architecture

```text
Fruit Image
     |
     v
Image Preprocessing
     |
     v
Feature Extraction
     |
     +-----------------------------+
     |             |               |
     v             v               v
    SVM           KNN       Decision Tree
     |             |               |
     +-------------+---------------+
                   |
                   v
             Random Forest
                   |
                   v
          Ensemble Voting
                   |
                   v
          Final Fruit Class