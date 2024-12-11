# Sehatin Machine Learning Models

This repository contains the machine learning models used in the Sehatin app. The models are organized into two main folders: **daily_step_recommendation** (regression model) and **food_detection** (transfer learning model). Follow the steps below to replicate or deploy the models.

---

## 1. Daily Step Recommendation (Regression Model)

### Description
This model predicts daily step recommendations based on user data.

### Files and Structure
- **Dataset**: `daily_step_recommendation/dataset/final_df.csv`
- **Notebook**: `daily_step_recommendation/regression_model.ipynb`
- **Saved Model**: `daily_step_recommendation/model1.h5`

### Steps to Replicate
1. Navigate to the `daily_step_recommendation` folder.
2. Use the dataset file located at `dataset/final_df.csv`.
3. Open the Jupyter Notebook: `regression_model.ipynb`.
4. Train the model using the notebook or load the pre-trained model (`model1.h5`).
5. Deploy the saved model to the cloud.

---

## 2. Food Detection (Transfer Learning Model)

### Description
This model classifies food images using transfer learning techniques.

### Files and Structure
- **Dataset**: `food_detection/food_dataset/`
- **Notebook**: `food_detection/food_model.ipynb`
- **Saved Model**: `food_detection/best_model.h5`

### Steps to Replicate
1. Navigate to the `food_detection` folder.
2. Use the image dataset located in `food_dataset/`.
3. Open the Jupyter Notebook: `food_model.ipynb`.
4. Train the model using the notebook or load the pre-trained model (`best_model.h5`).
5. Deploy the saved model to the cloud.

---

## Prerequisites
- Python 3.8+
- TensorFlow 2.18.0
- Jupyter Notebook
