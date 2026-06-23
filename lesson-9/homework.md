# **Classification Metrics: Evaluating a Binary Classifier (Breast Cancer Dataset)**

**Objective:**
In this homework, you will train a binary classification model to predict whether a tumor is **malignant** or **benign** using the **Breast Cancer dataset** from scikit-learn.
Your main goal is to understand, compute, and interpret common **classification metrics**, including:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* AUC Score
* Classification Report

---

# **Dataset (Sklearn Built-in)**

Load using:

python
from sklearn.datasets import load_breast_cancer


Features include 30 numeric predictors such as:

* radius_mean
* texture_mean
* perimeter_mean
* area_mean
* smoothness_mean
* … (and many more)

**Target**:

* 0 → malignant
* 1 → benign

---

# **Homework Steps**

## **1. Data Loading and Exploration**

1. Load the dataset and convert it to a pandas DataFrame.
2. Display:

   * First 5 rows
   * Shape of the dataset
   * Feature names
   * Target distribution (value counts)
3. Add a short discussion:

   * Is the dataset balanced or imbalanced?

---

## **2. Train–Test Split & Scaling**

1. Split the dataset:

   * 80% training
   * 20% testing

2. Scale the features using:

   python
   from sklearn.preprocessing import StandardScaler
   

3. Transform both train and test sets.

---

## **3. Model Training**

Train **two classifiers**:

### 1️⃣ Logistic Regression

### 2️⃣ Random Forest Classifier

Use:

python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


Train both models and generate predictions for the test set.

---

# **4. Compute Classification Metrics**

For **each model**, compute the following:

### **1. Accuracy**

### **2. Precision, Recall, F1-score**

Use:

python
from sklearn.metrics import precision_score, recall_score, f1_score


### **3. Confusion Matrix**

Plot it using `seaborn.heatmap`.

### **4. Classification Report**

Use:

python
from sklearn.metrics import classification_report


### **5. ROC Curve & AUC Score**

Generate:

* FPR (false positive rate)
* TPR (true positive rate)
* Plot ROC curve
* Compute AUC for both models

Use:

python
from sklearn.metrics import roc_curve, auc


---

# **5. Visualizations**

Create the following plots for **each model**:

### **Plot 1: Confusion Matrix**

* Annotated heatmap
* Labels: TP, FP, FN, TN

### **Plot 2: ROC Curve**

* Plot curves of *both models* on the same graph
* Label them clearly
* Add diagonal “random guess” line

### **Plot 3: Bar Chart of Metrics**

For each model:

* Accuracy
* Precision
* Recall
* F1

Put them on a grouped bar chart.

---

# **6. Comparison & Interpretation**

Write short answers to the following questions:

### **1. Which model performed better overall? Why?**

### **2. Which model had better Recall?**

(Important for cancer detection: fewer false negatives)

### **3. Which model had higher Precision?**

(Useful when false positives are costly)

### **4. Which model had better AUC?**

### **5. Does accuracy alone tell the full story?**

Explain using confusion matrix insights.

---

# **7. Bonus (Optional, but recommended)**

### **Perform Threshold Tuning**

1. For Logistic Regression, collect prediction probabilities.

2. Test different classification thresholds: 0.3, 0.4, 0.5, 0.6, 0.7

3. For each threshold, compute:

   * Precision
   * Recall
   * F1-score

4. Plot: **Threshold vs Precision/Recall/F1**

5. Explain:

   * How changing the threshold shifts the tradeoff between precision and recall.