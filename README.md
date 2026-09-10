🛳️# Titanic Data Cleaning and Exploratory Data Analysis

This repository contains my internship project completed in two consecutive weeks using the Titanic public dataset.

## Week 1 – Data Acquisition, Cleaning and Preprocessing

### Objectives

* Load the Titanic dataset.
* Explore data structure and missing values.
* Handle missing values and duplicates.
* Detect and treat outliers.
* Perform feature engineering and preprocessing.
* Save the cleaned dataset.

### Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## Week 2 – Exploratory Data Analysis (EDA) and Visualization

### Objectives

* Perform exploratory data analysis on the cleaned Titanic dataset.
* Generate visualizations to identify patterns and relationships.
* Interpret trends using statistical summaries and charts.

---

# Week 3 – K-Means Clustering

## Objective

The objective of Week 3 was to apply unsupervised learning techniques to the Titanic dataset and identify groups of passengers with similar characteristics.

## Dataset

The analysis was performed using the cleaned Titanic dataset prepared during Week 1.

## Features Used

The following features were used for clustering:

- Age
- Fare
- FamilySize

## Method Used

K-Means Clustering was applied to group passengers into different clusters based on their characteristics.

## Steps Performed

1. Loaded the cleaned Titanic dataset.
2. Selected relevant numerical features.
3. Prepared the data for clustering.
4. Applied K-Means clustering.
5. Assigned passengers to clusters.
6. Analysed the characteristics of each cluster.
7. Visualized the resulting clusters.


## Python File

```text
clustering_analysis.py

```
---

# Week 4 – Supervised Learning

## Objective

The objective of Week 4 was to build supervised machine learning models to predict whether a Titanic passenger survived.

## Dataset

The project continued using the cleaned Titanic dataset from the previous weeks.

## Models Used

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model for predicting passenger survival.

### 2. Random Forest

Random Forest was used to model more complex relationships between passenger features and survival.

## Steps Performed

1. Loaded the cleaned Titanic dataset.
2. Selected the required features.
3. Prepared the target variable.
4. Split the data into training and testing sets.
5. Trained Logistic Regression.
6. Trained Random Forest.
7. Evaluated both models.
8. Compared their performance.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Python File

```text
supervised_learning.py

```


---

# Week 5 – Deep Learning

## Objective

The objective of Week 5 was to extend the Titanic survival prediction project by implementing a Deep Learning model.

## Dataset

The same cleaned Titanic dataset used in the previous weeks was used for the Deep Learning model.

## Model Used

A PyTorch Multi-Layer Perceptron (MLP) neural network was developed for binary classification.

## Model Architecture

```text
Input Features
      ↓
Linear Layer (16 → 32)
      ↓
ReLU
      ↓
Dropout
      ↓
Linear Layer (32 → 16)
      ↓
ReLU
      ↓
Dropout
      ↓
Output Layer

```


---

# Week 6 – Integrative Capstone

## Objective

The objective of Week 6 was to integrate the work completed during Weeks 1–5 into one complete Data Science project.

## Project Workflow

```text
Data
 ↓
Data Cleaning & Preprocessing
 ↓
Exploratory Data Analysis
 ↓
K-Means Clustering
 ↓
Supervised Learning
 ↓
Deep Learning
 ↓
Model Evaluation
 ↓
Insights & Recommendations

```

## Author

**Shourya Raj**

B.Tech CSE (AI & ML)

Dr. B. C. Roy Engineering College
