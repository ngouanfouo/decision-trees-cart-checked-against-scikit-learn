"""
Decision Trees: CART, Checked Against Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - gini
import numpy as np

def gini(y):
    """
    Return the Gini impurity of a label array.
    Empty arrays have impurity 0.0.
    """
    y = np.asarray(y)

    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()

    return float(1.0 - np.sum(p ** 2))

# Step 2 - best_split
import numpy as np

def best_split(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    n = y.size
    if n == 0 or X.size == 0:
        return None, None, 0.0

    # Allow 1D X for convenience
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    parent = gini(y)
    best_feature = None
    best_threshold = None
    best_gain = 0.0

    n_features = X.shape[1]

    for j in range(n_features):
        vals = np.unique(X[:, j])

        # Need at least two distinct values to split
        if vals.size < 2:
            continue

        thresholds = (vals[:-1] + vals[1:]) / 2.0

        for threshold in thresholds:
            left_mask = X[:, j] <= threshold
            n_left = np.sum(left_mask)
            n_right = n - n_left

            if n_left == 0 or n_right == 0:
                continue

            left_gini = gini(y[left_mask])
            right_gini = gini(y[~left_mask])

            weighted_child = (n_left * left_gini + n_right * right_gini) / n
            gain = parent - weighted_child

            # Strict > keeps the first best split on ties
            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = float(threshold)

    if best_feature is None:
        return None, None, 0.0

    return best_feature, best_threshold, float(best_gain)

# Step 3 - grow_tree (not yet solved)
# TODO: implement

# Step 4 - predict_tree (not yet solved)
# TODO: implement

# Step 5 - fit_sklearn_tree (not yet solved)
# TODO: implement

# Step 6 - sklearn_splits (not yet solved)
# TODO: implement

# Step 7 - compare_trees (not yet solved)
# TODO: implement

# Step 8 - moons_data (not yet solved)
# TODO: implement

# Step 9 - overfit_vs_regularized (not yet solved)
# TODO: implement

# Step 10 - rotation_sensitivity (not yet solved)
# TODO: implement

# Step 11 - regression_tree (not yet solved)
# TODO: implement

# Step 12 - tree_rules (not yet solved)
# TODO: implement

# Step 13 - save_and_reload_tree (not yet solved)
# TODO: implement

# Step 14 - predict_species (not yet solved)
# TODO: implement

