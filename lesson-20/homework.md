# **Ensemble Learning & Voting Classifier**

### **Dataset**

Use the **Wine dataset** from sklearn:

python
from sklearn.datasets import load_wine


This is a **multiclass classification** problem.

---

### **Step 1 — Baseline Models**

1. Perform an **80/20 train-test split**
2. Train the following models **individually**:

   * Logistic Regression
   * K-Nearest Neighbors
   * Decision Tree
3. Evaluate and record **test accuracy** for each model


### **Step 2 — Hard Voting Classifier**

1. Create a **VotingClassifier** using the three models above
2. Use **hard voting**
3. Train the ensemble
4. Evaluate test accuracy


### **Step 3 — Soft Voting Classifier**

1. Modify the VotingClassifier to use **soft voting**
2. Ensure all base models support probability prediction
3. Train and evaluate the model