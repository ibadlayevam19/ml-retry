# **Multiclass Classification: Predicting Handwritten Digits (Sklearn Digits Dataset)**

**Objective:**
In this homework, you will work with the **Digits dataset** from scikit-learn to build, evaluate, and analyze multiclass classification models.
You will learn:

* How multiclass classification works
* One-vs-Rest (OvR) and One-vs-One (OvO) strategies
* Evaluation metrics for multiclass problems
* Confusion matrix interpretation
* Visualization of predictions
* Performance comparison between models

---

# **Dataset (Sklearn Built-in)**

Load using:

python
from sklearn.datasets import load_digits


This dataset contains:

* **1797 grayscale images** of handwritten digits (0–9)
* Each image is **8×8 pixels**
* **64 numerical features** (pixel intensities)
* **Target classes:** digits from **0** to **9**

This is a classic multiclass classification task.

---

# **Homework Steps**

## **1. Data Loading and Exploration**

1. Load the dataset and convert to:

   * `X` → features
   * `y` → target labels

2. Display:

   * Dataset shape
   * Number of classes
   * Unique class labels
   * One example image using `plt.matshow()`

3. Print basic statistics:

   python
   import numpy as np
   print("Min pixel:", np.min(X))
   print("Max pixel:", np.max(X))
   

---

## **2. Train–Test Split & Scaling**

1. Split into:

   * **80% training**
   * **20% testing**

2. Scale the features using:

   python
   from sklearn.preprocessing import StandardScaler
   

> Although pixel data is already normalized, scaling improves some classifiers (especially SVM).

---

## **3. Train Two Multiclass Models**

Train the following classifiers:

### **1️⃣ Logistic Regression (multiclass='ovr')**

python
from sklearn.linear_model import LogisticRegression


### **2️⃣ Support Vector Machine (SVC, OvO approach)**

python
from sklearn.svm import SVC


### **3️⃣ Optional (Harder): Random Forest**

Train both models on the scaled data.

---

# **4. Predictions and Basic Evaluation**

Generate predictions for the test set.

For **each model**, compute:

* **Accuracy**
* **Confusion Matrix** (plot using `seaborn.heatmap`)
* **Classification Report** with:

  * precision
  * recall
  * f1-score
  * support

---

# **5. Multiclass Evaluation Metrics**

Compute the following metrics:

### ✔ Macro-average Precision

### ✔ Macro-average Recall

### ✔ Macro-average F1 Score

### ✔ Weighted-average Precision

### ✔ Weighted-average Recall

### ✔ Weighted-average F1 Score

Use:

python
from sklearn.metrics import precision_score, recall_score, f1_score


Compare metrics for both models:

* Which performs better on average?
* Is there a large gap between macro and weighted scores?
* What does this gap tell you about class imbalance?

---

# **6. Visualization Tasks**

## **1. Confusion Matrix Heatmap**

For each model:

* Plot confusion matrix
* Annotate counts
* Add titles and labels

## **2. ROC Curve (One-vs-Rest)**

For **each class**, plot ROC curves using:

python
from sklearn.metrics import roc_curve, auc


This requires:

* Binarizing the target using `LabelBinarizer`
* Getting probability scores from each model (Logistic Regression required; SVC must use `probability=True`)

Plot all 10 curves on a single graph per model.

> This step gives deeper intuition about multiclass ROC analysis.

## **3. Sample Predictions Visualization**

* Display **10 random test images**
* Annotate each with:

  * True label
  * Predicted label
  * Whether prediction was correct (green/red border)

---

# **7. Comparison & Interpretation**

Write short answers:

### **1. Which model has highest accuracy?**

### **2. Compare precision/recall between Logistic Regression and SVM.**

### **3. Which digits are hardest to classify?**

(Inspect confusion matrix)

### **4. Which model generalizes better? Why might SVM outperform Logistic Regression?**

### **5. Based on ROC curves, which class is easiest/hardest?**