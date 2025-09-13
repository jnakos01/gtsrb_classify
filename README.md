# GTSRB Traffic Sign Classification

This repository contains a pipeline for classifying traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The workflow covers data preprocessing, feature extraction, model training, evaluation, and generating predictions.

## Overview

- **Preprocessing:** Prepare and augment the training and test datasets.
- **Model Training:** Build and train a CNN, SVM and RF, create a final stacking metamodel for traffic sign classification.
- **Evaluation:** Assess performance and generate predictions.

## Prerequisites

Before running any code, make sure you have the following Python packages installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- opencv-python
- pillow
- tensorflow

You can install all dependencies at once:

```bash
pip install pandas numpy scikit-learn matplotlib opencv-python pillow tensorflow
```

## Usage

Workflow organised using Jupyter Notebooks:

1. **Preprocess Training Data:**
   - Open `preprocessing.ipynb` and run all cells.

2. **Preprocess Test Data:**
   - Open `preprocessing_test.ipynb` and run all cells.

3. **Train and Test Model:**
   - Open `train_test.ipynb` and run all cells.
   - This will train the model and generate predictions.

4. **Submission File:**
   - The predictions will be saved as `submission.csv` for competition submission.

## Repository Structure

- `preprocessing.ipynb` — Preprocessing steps for training data
- `preprocessing_test.ipynb` — Preprocessing steps for test data
- `train_test.ipynb` — Model training, evaluation, and prediction
- `submission.csv` — Output file for kaggle test submission

Accuracy on hidden Kaggle test set: 0.97

