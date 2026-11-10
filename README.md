# E-Commerce Data Classification Using K-Nearest Neighbors (KNN)

## Project Overview
This project implements a supervised machine learning classification model using the **K-Nearest Neighbors (KNN)** algorithm in Python. The objective is to analyze an e-commerce transactions dataset (`Dataset for Data Analytics.xlsx`) and categorize orders based on order characteristics.

## Key Workflow & Steps
1. **Data Loading & Inspection:** Read and processed transactional data using `pandas`.
2. **Feature Engineering:** Defined a target variable (`HighValueOrder`) by comparing order totals against the median price point.
3. **Data Splitting:** Partitioned the data into training and testing sets (80/20 split) using scikit-learn.
4. **Feature Scaling:** Applied `StandardScaler` to normalize numerical features, ensuring equal distance weighting for the KNN algorithm.
5. **Model Training & Evaluation:** Trained a `KNeighborsClassifier` ($k=5$) and evaluated performance using Accuracy, Classification Reports (Precision, Recall, F1-Score), and Confusion Matrices.

## Tech Stack
* **Python**
* **Pandas** & **Openpyxl** (Data manipulation and Excel handling)
* **Scikit-Learn** (Machine learning modeling, scaling, and evaluation)

## How to Run
1. Ensure the required libraries are installed:
   ```bash
   pip install pandas openpyxl scikit-learn
