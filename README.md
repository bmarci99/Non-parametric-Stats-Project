# Stroke Prediction Analysis

## Overview
This study aims to develop a predictive model for identifying individuals at high risk of stroke based on demographic, medical, and lifestyle factors.

## Dataset Description
The dataset consists of 5,110 patient records, each containing 12 attributes:

### **Target Variable:**
- **Stroke**: Binary (Yes/No)

### **Demographic Characteristics:**
- **ID**: Unique identifier
- **Gender**: Male/Female
- **Age**: In years
- **Residence Type**: Rural/Urban
- **Marital Status (Ever Married)**: Yes/No

### **Lifestyle Factors:**
- **Smoking Status**: Formerly smoked, never smoked, smokes, or unknown
- **Work Type**: Children, Govt_job, Never_worked, Private, Self-employed

### **Medical History:**
- **Hypertension**: Binary (1 = Yes, 0 = No)
- **Heart Disease**: Binary (1 = Yes, 0 = No)

### **Health Indicators:**
- **Average Glucose Level**: mg/dL
- **BMI**: Body Mass Index

## Exploratory Data Analysis (EDA)
### **Feature Analysis**
- Stroke risk increases with age.
- Higher glucose levels correlate with stroke incidence.
- BMI distribution varies for stroke and non-stroke patients.

## Statistical Tests
- **Bootstrap Confidence Interval** for glucose levels.
- **Wilcoxon Rank-Sum Test & KS Test** for BMI, glucose level, and age.
- **Kruskal-Wallis Test** for hypertension, heart disease, and work type.

## Feature Engineering
- **Cosine Similarity** for patient grouping.
- **Louvain Community Detection** for subgroup identification.

## Outlier Detection
Methods used:
- **Mahalanobis Distance**
- **Local Outlier Factor (LOF)**
- **Isolation Forest (IF)**

## Modeling Approach
- **Logistic Regression** for classification.
- **SMOTE & ADASYN** for data balancing.

## Experimental Results
- **Logistic Regression with SMOTE**: Balanced accuracy of 0.74, AUC 0.8207.
- **Logistic Regression with ADASYN**: Balanced accuracy of 0.68, AUC 0.8149.
- **Community-based models improved performance**, especially for high-risk groups.

## References
- Harvard Health Publishing. *What is a silent stroke?* Harvard Medical School, 2023.
- Gaetanlopez. *How to Make Clean Visualizations*. Kaggle, 2021.
