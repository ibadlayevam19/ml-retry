# **Polynomial Features: Modeling Disease Progression (Sklearn Diabetes Dataset)**

**Objective:**
In this homework, you will use **Polynomial Features** to model the progression of diabetes using the built-in **Diabetes dataset** from scikit-learn.
You will compare a simple **Linear Regression** model versus **Polynomial Regression** and analyze the impact of adding non-linear features.

---

## **Dataset (Sklearn Built-in)**

Import using:

python
from sklearn.datasets import load_diabetes


The dataset includes:

* **10 numerical features**, already normalized:

  1. age
  2. sex
  3. bmi
  4. bp (blood pressure)
  5. s1
  6. s2
  7. s3
  8. s4
  9. s5
  10. s6
* **Target**: a quantitative measure of diabetes progression one year after baseline.

All values are numerical — no categorical encoding needed.

---

# **Homework Steps**

## **1. Data Loading and Exploration**

1. Load the dataset:

   python
   data = load_diabetes()
   
2. Convert to pandas DataFrame.
3. Display:

   * First 5 rows
   * Shape (rows, columns)
   * Feature names
4. Print basic statistics (`df.describe()`).

> Note: the dataset contains no missing values.

---

## **2. Polynomial Feature Engineering**

1. Use:

   python
   from sklearn.preprocessing import PolynomialFeatures
   

2. Create **degree-2 polynomial features** for the following features only:

   * bmi
   * bp
   * s5
     (These are known to have non-linear relationships with the target.)

3. Keep all other features unchanged.

4. Print:

   * Number of new polynomial features
   * Full expanded feature name list using `get_feature_names_out()`

---

## **3. Model Building**

1. Split the dataset:

   * **80% training**
   * **20% testing**

2. Train two models:

   * **Baseline model:** Linear Regression using original features
   * **Polynomial model:** Linear Regression using expanded polynomial features

3. Extract and print the **top 10 largest coefficients** (absolute value) in the polynomial model
   and explain at least **three** of them in the report.

---

## **4. Model Evaluation**

Evaluate both models using:

* **MAE**
* **RMSE**
* **R² score**

Then answer the following:

* Did polynomial features improve the model?
* Why or why not? (Hint: Diabetes dataset is small — polynomial features may overfit.)

---

## **5. Visualization**

Create the following plots:

### **1. Predictions vs Actual (Scatter Plot)**

* For both baseline and polynomial model
* Overlay a diagonal line representing perfect predictions

### **2. Error Comparison Bar Chart**

* Baseline vs Polynomial:

  * MAE
  * RMSE

### **3. Polynomial Curve Visualization**

Pick **one** feature (e.g., bmi → target):

* Plot scatter: `(bmi, target)`
* Fit a **single-feature polynomial regression** to visualize curvature
* Plot the polynomial curve
  (Sort x values so curve is smooth)

> This is separate from the multi-feature model.

---

## **6. Predictions**

Predict the disease progression for the following input (a sample from the dataset):

* **age**: 0.05
* **sex**: 0.02
* **bmi**: 0.04
* **bp**: 0.03
* **s1**: -0.02
* **s2**: -0.01
* **s3**: 0.00
* **s4**: 0.02
* **s5**: 0.03
* **s6**: 0.01

Steps:

1. Convert this to a DataFrame row.
2. Apply the same polynomial transformation.
3. Predict using both:

   * Linear Model
   * Polynomial Model
4. Compare the outputs.