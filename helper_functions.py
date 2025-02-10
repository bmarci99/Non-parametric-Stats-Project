from tabulate import tabulate
from scipy.stats import chi2
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums, ks_2samp
from sklearn.neighbors import KernelDensity
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from pingouin import multivariate_normality
from pygam import LogisticGAM, s, f
from imblearn.over_sampling import ADASYN
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split

def bootstrap_ci(data, group_col, value_col, group1, group2, stat_fn=np.median, n_iterations=1000, alpha=0.05):
    group1_values = data[data[group_col] == group1][value_col].dropna()
    group2_values = data[data[group_col] == group2][value_col].dropna()

    stat_differences = []
    for _ in range(n_iterations):
        g1_sample = np.random.choice(group1_values, size=len(group1_values), replace=True)
        g2_sample = np.random.choice(group2_values, size=len(group2_values), replace=True)
        stat_differences.append(stat_fn(g1_sample) - stat_fn(g2_sample))

    ci_lower = np.percentile(stat_differences, 100 * (alpha / 2))
    ci_upper = np.percentile(stat_differences, 100 * (1 - alpha / 2))
    return stat_differences, ci_lower, ci_upper

# Kernel Density Estimation (KDE)
def kde_density(data, col, group_col, group, bandwidth=1.0):
    subset = data[data[group_col] == group][col].dropna().values[:, None]
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(subset)
    x_grid = np.linspace(subset.min(), subset.max(), 1000)[:, None]
    log_density = kde.score_samples(x_grid)
    return x_grid, np.exp(log_density)


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")


def calculateClasswiseTopNAccuracy(discreteActualLabels, predictionsProbs, TOP_N):
    """
    TOP_N is the top n% predictions you want to use for each class
    """
    #discreteActualLabels = [1 if item[1] > item[0] else 0 for item in actualLabels]
    discretePredictions = [1 if item[1] > item[0] else 0 for item in predictionsProbs]
    #discretePredictions = [1 if item >= .5 else 0 for item in predictionsProbs]

    predictionProbsTopNHealthy, predictionProbsTopNPneumonia = [item[0] for item in predictionsProbs], [item[1] for item in predictionsProbs]
    predictionProbsTopNHealthy = list(reversed(sorted(predictionProbsTopNHealthy)))[:int(len(predictionProbsTopNHealthy) * TOP_N / 100)][-1]
    predictionProbsTopNPneumonia = list(reversed(sorted(predictionProbsTopNPneumonia)))[:int(len(predictionProbsTopNPneumonia) * TOP_N / 100)][-1]

    # Calculate accuracy for both classes
    accuracyHealthy = []
    accuracyPneumonia = []
    for i in range(0, len(discretePredictions)):
        if discretePredictions[i] == 1:
            # Tilted
            if predictionsProbs[i][1] > predictionProbsTopNPneumonia:
                accuracyPneumonia.append(int(discreteActualLabels[i]) == 1)
        else:
            # Normal
            if predictionsProbs[i][0] > predictionProbsTopNHealthy:
                accuracyHealthy.append(int(discreteActualLabels[i]) == 0)

    accuracyHealthy = round((accuracyHealthy.count(True) * 100) / len(accuracyHealthy), 2)
    accuracyPneumonia = round((accuracyPneumonia.count(True) * 100) / len(accuracyPneumonia), 2)
    return accuracyHealthy, accuracyPneumonia


def calculateMetrics(predictions, predictionsProbabilities, actualLabels):
    # Convert label format from [0,1](label 1) and [1,0](label 0) into single integers: 1 and 0.
    #actualLabels = [item[1] for item in actualLabels]
    
    # Get probabilities for the class with label 1. That is all we need to compute AUCs. We don't need probabilities for class 0.
    predictionsProbabilities = [item[1] for item in predictionsProbabilities]
    
    # Calculate metrics using scikit-learn functions. The round function is used to round the numbers up to 2 decimal points.
    try:
        accuracy = round(accuracy_score(actualLabels, predictions) * 100, 2)
        precisionNegative = round(precision_score(actualLabels, predictions, average = None)[0] * 100, 2)
        precisionPositive = round(precision_score(actualLabels, predictions, average = None)[1] * 100, 2)
        recallNegative = round(recall_score(actualLabels, predictions, average = None)[0] * 100, 2)
        recallPositive = round(recall_score(actualLabels, predictions, average = None)[1] * 100, 2)
    except:
        print("An exception occurred but was caught.")
    auc = round(roc_auc_score(actualLabels, predictionsProbabilities) * 100, 2)
    
    return auc
    plt.show()

def calculateMetricsAndPrint(predictions, predictionsProbabilities, actualLabels):
    actualLabels = [item[1] for item in actualLabels] # Convert label format from [0,1](label 1) and [1,0](label 0) into single integers: 1 and 0.

    predictionsProbabilities = [item[1] for item in predictionsProbabilities] # Get probabilities for the class with label 1. That is all we need to compute AUCs. We don't need probabilities for class 0.

    
    # Calculate metrics using scikit-learn functions. The round function is used to round the numbers up to 2 decimal points.
    accuracy = round(accuracy_score(actualLabels, predictions) * 100, 2)
    precisionNegative = round(precision_score(actualLabels, predictions, average = None)[0] * 100, 2)
    precisionPositive = round(precision_score(actualLabels, predictions, average = None)[1] * 100, 2)
    recallNegative = round(recall_score(actualLabels, predictions, average = None)[0] * 100, 2)
    recallPositive = round(recall_score(actualLabels, predictions, average = None)[1] * 100, 2)
    auc = round(roc_auc_score(actualLabels, predictionsProbabilities) * 100, 2)
    confusionMatrix = confusion_matrix(actualLabels, predictions)
    
    # Print metrics. .%2f prints a number upto 2 decimal points only.
    print("------------------------------------------------------------------------")
    print("Accuracy: %.2f\nPrecisionNegative: %.2f\nPrecisionPositive: %.2f\nRecallNegative: %.2f\nRecallPositive: %.2f\nAUC Score: %.2f" % 
          (accuracy, precisionNegative, precisionPositive, recallNegative, recallPositive, auc))
    print("------------------------------------------------------------------------")
    
    print("+ Printing confusion matrix...\n")
    # Display confusion matrix
    displayConfusionMatrix(confusionMatrix, precisionNegative, precisionPositive, recallNegative, recallPositive, "Confusion Matrix")
    
    print("+ Printing ROC curve...\n")
    # ROC Curve
    plt.rcParams['figure.figsize'] = [16, 8]
    FONT_SIZE = 16
    falsePositiveRateDt, truePositiveRateDt, _ = roc_curve(actualLabels, predictionsProbabilities)
    plt.plot(falsePositiveRateDt, truePositiveRateDt, linewidth = 5, color='black')
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xlabel("False Positive Rate", fontsize=FONT_SIZE)
    plt.ylabel("True Positive Rate", fontsize=FONT_SIZE)
    plt.show()
    
    return auc

def cool_plotting(y_test, y_pred_probs,topNValues):
    accuraciesHealthy, accuraciesPneumonia = [], []
    for topn in topNValues:
        accuracyHealthy, accuracyPneumonia = calculateClasswiseTopNAccuracy(y_test, y_pred_probs, topn)
        accuraciesHealthy.append(accuracyHealthy)
        accuraciesPneumonia.append(accuracyPneumonia)
        
        print("+ Accuracy for top %d percent predictions for Healthy: %.2f, Stroke: %.2f" % (topn, accuracyHealthy, accuracyPneumonia))
        
    # Plot results
    x = np.arange(len(accuraciesHealthy))
    plt.plot(x, accuraciesHealthy, linewidth = 3, color = '#e01111')
    scatterHealthy = plt.scatter(x, accuraciesHealthy, marker = 's', s = 100, color = '#e01111')
    plt.plot(x, accuraciesPneumonia, linewidth = 3, color = '#0072ff')
    scatterPneumonia = plt.scatter(x, accuraciesPneumonia, marker = 'o', s = 100, color = '#0072ff')
    plt.xticks(x, topNValues, fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel("Top N%", fontsize = 15)
    plt.ylabel("Accuracy", fontsize = 15)
    plt.legend([scatterHealthy, scatterPneumonia], ["Accuracy for Healthy", "Accuracy for Stroke"], fontsize = 17)
    plt.ylim(0, 110)
    plt.show()





# Compute Mahalanobis distance for each observation
def mahalanobis_distance(x, mean_vector, inv_cov_matrix):
    x_minus_mu = x - mean_vector
    left_term = np.dot(x_minus_mu, inv_cov_matrix)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

def fit_and_predict(model, data, scaling = None, plots = True, sampling= None, topNValues = [10,20,30,40,50,60,70,80,90], ensemble=None, cv_folds=5,  use_cv=False):
    print("\n" + "="*50)
    print(f"{'Fit and Predict Summary':^50}")
    print("="*50)

    # Separate features and target
    X = data.drop(columns=['stroke_1'])
    y = data['stroke_1']

    if use_cv:
            # (Optional) Apply scaling if provided
        if scaling is not None:
            scaler = scaling
            X = scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, scoring="balanced_accuracy", cv=cv)
        print("Cross-validated balanced accuracy: {:.3f} ± {:.3f}".format(np.mean(scores), np.std(scores)))

    else:
        # Perform splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_train.reset_index(drop = True, inplace = True)
        X_test.reset_index(drop = True, inplace = True)
        y_train.reset_index(drop = True, inplace = True)
        y_test.reset_index(drop = True, inplace = True)

        # apply scaling if needed
        if scaling != None:
            scaler = scaling
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # apply sampling if needed
        if sampling == "ADASYN":
            from imblearn.over_sampling import ADASYN
            adasyn = ADASYN(random_state=42)
            X_train, y_train = adasyn.fit_resample(X_train, y_train)
        elif sampling == "SMOTE":
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        elif sampling =="RandomUnderSampler":
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)

       # ensemble
        if ensemble:
            model = VotingClassifier(ensemble, voting="soft") # we are using only SOFT (sum the probs), HARD: argmax


        # Fit
        model.fit(X_train, y_train)
        # predict
        y_pred = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)

        # Calculate metrics
        print("\nEvaluation Metrics:")
        print("-" * 50)

        # Class-wise accuracy and Balanced
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        metrics_table = [[f"Class {i}", f"{acc:.2f}"] for i, acc in enumerate(class_accuracies)]
        metrics_table.append(["Balanced Accuracy", f"{balanced_accuracy_score(y_test, y_pred):.2f}"])
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

        # plot if asked
        if plots == True: cool_plotting(y_test, y_pred_probs,topNValues)
        plot_confusion_matrix(y_test, y_pred)

        #Permutation Test

        metrics = ["accuracy", "log_loss", "auc"]
        results = []

        for metric in metrics:
            p_value, observed_stat = permutation_test_logistic(model, X_train, X_test, y_train, y_test, y_pred, y_pred_probs, metric=metric)
            results.append([metric, f"{observed_stat:.4f}", p_value])

        # Pretty-print the results
        print("\nPermutation Test Results:")
        print(tabulate(results, headers=["Metric", "Observed Statistic", "p-value"], tablefmt="grid"))

        # Plot Importance
        if hasattr(model, "coef_"):
            feature_importance = np.abs(model.coef_).flatten()  # Absolute values of coefficients
            sorted_idx = np.argsort(feature_importance)[::-1]  # Sort in descending order
            important_features = [X.columns[i] for i in sorted_idx]
            
            # Plot Feature Importance
            plt.figure(figsize=(10, 5))
            plt.barh(important_features[::-1], feature_importance[sorted_idx][::-1])  # Invert order for best display
            plt.xlabel("Absolute Coefficient Value")
            plt.ylabel("Feature")
            plt.title("Feature Importance (Logistic Regression)")
            plt.show()


def permutation_test_logistic(model, X_train, X_test, y_train, y_test, y_pred, y_prob, metric="log_loss", n_permutations=1000):

    # Compute the observed test statistic
    if metric == "accuracy":
        observed_stat = accuracy_score(y_test, y_pred)
    elif metric == "log_loss":
        observed_stat = -log_loss(y_test, y_prob[:, 1])  # Negative log-loss (higher is better)
    elif metric == "auc":
        observed_stat = roc_auc_score(y_test, y_prob[:, 1])
    else:
        raise ValueError("Invalid metric! Choose 'accuracy', 'log_loss', or 'auc'.")

    # Permutation testing
    permuted_stats = []

    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y_train)  # Shuffle y
        model.fit(X_train, y_permuted)  # Refit model
        y_permuted_pred = model.predict(X_test)
        y_permuted_prob = model.predict_proba(X_test)[:, 1]

        if metric == "accuracy":
            permuted_stat = accuracy_score(y_test, y_permuted_pred)
        elif metric == "log_loss":
            permuted_stat = -log_loss(y_test, y_permuted_prob)
        elif metric == "auc":
            permuted_stat = roc_auc_score(y_test, y_permuted_prob)

        permuted_stats.append(permuted_stat)

    # Compute p-value (proportion of permuted stats greater than or equal to observed stat)
    p_value = np.mean(np.array(permuted_stats) >= observed_stat)
    return p_value, observed_stat


def find_best_hyperparameters(model, param_grid, data, scoring='f1'):
    print("\n" + "=" * 50)
    print(f"{'Hyperparameter Tuning':^50}")
    print("=" * 50)

    # Separate features and target
    X = data.drop(columns=['stroke_1'])
    y = data['stroke_1']

    # Perform splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Grid Search
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:", grid_search.best_params_)
    print("\nBest Score:", grid_search.best_score_)

    # Return the best model
    return grid_search.best_estimator_
