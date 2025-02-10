# Electricity Market Bid Data

This dataset provides **hourly bid data** from the Italian electricity market, retrieved from the MercatoElettrico FTP server. Each XML file represents a single day's market activity, capturing information from market bids, though only demand bids (`BID` types) are retained in this processed dataset. The dataset focuses on specific market zones for **time-series analysis** and is provided as a **pickled DataFrame file**.

## Data Overview

Each orginal XML file contains hourly records for a single day, represented by the following fields:

- **Day**: Date in `YYYYMMDD` format.
- **Hour**: The hoStroke Prediction Analysis

Overview

This study aims to develop a predictive model for identifying individuals at high risk of stroke based on demographic, medical, and lifestyle factors.

Dataset Description

The dataset consists of 5,110 patient records, each containing 12 attributes:

Target Variable:

Stroke: Binary (Yes/No)

Demographic Characteristics:

ID: Unique identifier

Gender: Male/Female

Age: In years

Residence Type: Rural/Urban

Marital Status (Ever Married): Yes/No

Lifestyle Factors:

Smoking Status: Formerly smoked, never smoked, smokes, or unknown

Work Type: Children, Govt_job, Never_worked, Private, Self-employed

Medical History:

Hypertension: Binary (1 = Yes, 0 = No)

Heart Disease: Binary (1 = Yes, 0 = No)

Health Indicators:

Average Glucose Level: mg/dL

BMI: Body Mass Index

Exploratory Data Analysis (EDA)

Feature Analysis

Stroke risk increases with age.

Higher glucose levels correlate with stroke incidence.

BMI distribution varies for stroke and non-stroke patients.

Statistical Tests

Bootstrap Confidence Interval for glucose levels.

Wilcoxon Rank-Sum Test & KS Test for BMI, glucose level, and age.

Kruskal-Wallis Test for hypertension, heart disease, and work type.

Feature Engineering

Cosine Similarity for patient grouping.

Louvain Community Detection for subgroup identification.

Outlier Detection

Methods used:

Mahalanobis Distance

Local Outlier Factor (LOF)

Isolation Forest (IF)

Modeling Approach

Logistic Regression for classification.

SMOTE & ADASYN for data balancing.

Experimental Results

Logistic Regression with SMOTE: Balanced accuracy of 0.74, AUC 0.8207.

Logistic Regression with ADASYN: Balanced accuracy of 0.68, AUC 0.8149.

Community-based models improved performance, especially for high-risk groups.

References

Harvard Health Publishing. What is a silent stroke? Harvard Medical School, 2023.

Gaetanlopez. How to Make Clean Visualizations. Kaggle, 2021.

ur associated with each bid `(0–23)`.
- **Quantity**: Bid quantity in `MWh`.
- **Price**: The bid price in `Euros` per `MWh`.
- **Market (Mercato)**: Constant field equal to `MGP` (Mercato del Giorno Prima).
- **Market Zones (ZonaMercato)**: The full list of market zones participating in the auction. Only full markets are retained in the final dataset.
- **Bid Type (Tipo)**: Indicates the type of bid, either `OFF` (offer) or `BID` (demand). Only demand bids are retained in the final dataset.

Each instance in the dataset corresponds to an individual bid, not a daily or hourly aggregate. This level of granularity allows for a detailed examination of bidding patterns and pricing.

## Preprocessing Steps

1. **FTP Access**: XML files are downloaded daily over a defined date range.
2. **Data Parsing**: Each XML file is processed, and relevant bid data is parsed into a DataFrame.
3. **Filtering**: Only demand bids (`Type "BID"`) across all participating market zones (`ZonaMercato`) are retained. The dataset is then reduced to three main columns: `timestamp`, `Quantity`, and `Price`.
4. **Timestamp Conversion**: Each record’s date and hour fields are combined into a single `timestamp` column, facilitating easier indexing and analysis.

## Nonparametric Statistics Project Objective

### Tentative Goals: Functional Analysis of Hourly Bidding Patterns in the Electricity Market

- **Objective**: To treat daily bid profiles as functional data, analyzing continuous variations in prices and quantities over hourly intervals to understand evolving trends.

### Possible Research Questions

1. How do daily profiles of bid prices and quantities evolve over time, and can we detect significant functional differences across days or zones?
2. Using survival analysis, can we estimate the likelihood that prices will remain within a certain standard deviation over time?

### Possible Methods from Class

- **Functional Data Analysis (FDA)**: Model daily bidding profiles as functional curves, applying techniques like Functional Principal Component Analysis (FPCA) to capture dominant patterns and cycles.
- **Functional ANOVA (FANOVA)**: Test for significant differences in functional profiles across days, zones, or other factors.
- **Survival Analysis**: Examine the persistence of price levels within set thresholds, using techniques like Kaplan-Meier estimates or Cox proportional hazards models for "survival" within defined price limits.
- **Nonparametric Hypothesis Tests**: Apply tests such as the Mann-Whitney U-test or Kruskal-Wallis test to assess the statistical significance of differences in bid prices and quantities across zones or times.


