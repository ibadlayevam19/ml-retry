## **Boosting and Stacking**

### Tasks

1. Load a **binary classification dataset**:

   * Breast Cancer dataset from `sklearn`, **OR**
   * A CSV provided by the instructor
2. Split into:

   * Train set
   * Test set
3. Apply scaling where appropriate


## AdaBoost (Boosting)

### Tasks

1. Train:

   python
   AdaBoostClassifier(
       n_estimators=50,
       learning_rate=0.1
   )
   
2. Evaluate on train and test sets
3. Record:

   * Accuracy
   * F1-score
---

## Gradient Boosting

### Tasks

1. Train:

   python
   GradientBoostingClassifier
   
2. Run **at least 3 experiments** with different:

   * `n_estimators`
   * `learning_rate`
3. Create a small table of results

---

## Stacking Using sklearn

### Tasks

1. Build a:

   python
   StackingClassifier
   
2. Base models:

   * Logistic Regression
   * Decision Tree
   * KNN
3. Meta-learner:

   * Logistic Regression
4. Evaluate:

   * Accuracy
   * ROC-AUC