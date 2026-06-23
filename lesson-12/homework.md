# **Dataset (Sklearn Built-in)**

Use the **Breast Cancer Wisconsin dataset**:

python
from sklearn.datasets import load_breast_cancer


The dataset contains:

* **569 samples**
* **30 numerical features**, such as:

  * mean_radius
  * mean_texture
  * mean_smoothness
  * mean_compactness
  * mean_concavity
* **Binary target**:

  * `0` → malignant
  * `1` → benign

This is a **binary classification** problem.

---

# **Homework**

# **Baseline Model (Without Pipeline)**

Before introducing pipelines:

1. Perform an **80/20 train-test split**
2. Apply `StandardScaler` to the data **manually**
3. Train a `LogisticRegression` model
4. Compute test accuracy

This model will act as a **baseline reference**.

---

# **First Pipeline (Preprocessing + Model)**

Now build your first pipeline.

1. Import:

   python
   from sklearn.pipeline import Pipeline
   
2. Create a pipeline with:

   * `StandardScaler`
   * `LogisticRegression`
3. Fit the pipeline on training data
4. Evaluate on test data

Comparison questions:

* Is the accuracy different from the baseline?
* Why should the pipeline version be preferred, even if accuracy is similar?