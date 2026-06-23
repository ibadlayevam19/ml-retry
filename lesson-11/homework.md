# **Cross Validation: Model Selection with the Wine Dataset**

**Objective:**
In this homework, you will use **Cross Validation (CV)** to evaluate and compare multiple classification models on the **Wine dataset** from scikit-learn.
You will learn:

* Why cross validation is important
* How to use K-Fold and Stratified K-Fold CV
* How to compare models fairly
* How to tune hyperparameters using cross validation
* How to interpret variance in CV scores

---

# **Dataset (Sklearn Built-in)**

Load using:

python
from sklearn.datasets import load_wine


The dataset contains:

* **178 wine samples**
* **13 numerical features** such as:

  * alcohol
  * malic_acid
  * ash
  * magnesium
  * phenols
  * proanthocyanins
  * color_intensity
  * hue
  * proline
* **3 target classes** (Wine Type: 0, 1, 2)

This is a **multiclass classification** task.

---

# **Homework Steps**

## **1. Data Loading and Exploration**

1. Load the dataset and convert it into a DataFrame (`X` and `y`).
2. Display:

   * First 5 rows
   * Shape of the dataset
   * Summary statistics
   * Value counts of each class
3. Short discussion:

   * Is the dataset balanced?
   * Are there any obvious outliers?

---

# **2. Baseline Model (Train-Test Split)**

Before using cross validation:

1. Perform an 80/20 train-test split.
2. Train a simple classifier (e.g., Logistic Regression).
3. Compute the test accuracy.

This baseline gives you something to compare against.

---

# **3. K-Fold Cross Validation**

Use **KFold** with `k = 5`.

1. Import:

   python
   from sklearn.model_selection import KFold, cross_val_score
   
2. Use `cross_val_score()` with:

   * Logistic Regression
   * Number of folds = 5
3. Print:

   * All 5 scores
   * Mean CV score
   * Standard deviation

Interpretation questions:

* Why is the CV mean a more reliable measure?
* What does standard deviation tell you about model stability?

---

# **4. Stratified K-Fold Cross Validation**

Wine dataset is not perfectly balanced → Stratified K-Fold is better.

1. Use:

   python
   from sklearn.model_selection import StratifiedKFold
   

2. Evaluate two models using **5-fold stratified CV**:

   * Logistic Regression
   * Random Forest Classifier

3. Compute:

   * Accuracy mean
   * Accuracy standard deviation
   * Compare the two models

Small analysis:

* Which model is more stable (lower std)?
* Which is more accurate?
* Does the improvement justify model complexity?

---

# **5. Cross Validation with Multiple Metrics**

Use:

python
scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']


Perform 5-fold Stratified CV on:

* Logistic Regression
* Random Forest

For each metric:

* Print mean
* Print std

Questions to answer:

* Which model performs best across all metrics?
* Does one metric show a bigger difference between models?

---

# **6. Hyperparameter Tuning with Cross Validation**

Use **GridSearchCV** with StratifiedKFold for one model:

### Example: Random Forest

Parameters to search:

python
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 4, 6]
}


Steps:

1. Use:

   python
   from sklearn.model_selection import GridSearchCV
   
2. Use scoring = 'accuracy'
3. Perform `cv=5`
4. Print:

   * Best accuracy
   * Best hyperparameters
   * Best estimator

Then:

* Train the best estimator on the full training data
* Evaluate on test set
* Compare with baseline accuracy