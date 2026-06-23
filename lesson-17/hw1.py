# 1. A Decision Tree is a supervised learning model that makes predictions by
#    recursively splitting data based on feature values until a leaf node
#    gives a final class or value.

# 2. A node is pure if all samples belong to the same class.
#    A node is impure if it contains samples from multiple classes.

# 3. Entropy and Gini impurity measure how impure a node is.
#    Trees use them to decide the best feature to split on.

# 4. Information Gain measures the reduction in impurity after a split.
#    It is used to choose the split that best separates the data.

# 5. Decision Trees are greedy because they choose the best split at each step
#    without considering future splits or global optimality.

# 6. Deep trees overfit because they learn noise and very specific patterns
#    from the training data instead of general trends.

# 7. Shallow trees have high bias and low variance (underfitting).
#    Deep trees have low bias and high variance (overfitting).

# 8. Pruning removes unnecessary branches from a tree to reduce overfitting
#    and improve generalization.

# 9. Decision Trees do not require feature scaling because they use comparisons
#    (>, <) rather than distance-based calculations.

# 10. Advantage: Easy to interpret and visualize.
#     Limitation: Prone to overfitting if not properly constrained.
