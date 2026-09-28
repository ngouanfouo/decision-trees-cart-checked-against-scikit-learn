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

# Step 3 - grow_tree
import numpy as np


def _majority(y):
    """Majority class as int; ties go to the smallest class."""
    vals, counts = np.unique(y, return_counts=True)
    top = counts.max()
    return int(vals[counts == top].min())


def _best_split_constrained(X, y, min_samples_leaf):
    """Like best_split, but rejects splits with a child smaller than min_samples_leaf."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = y.size
    if n == 0 or X.size == 0:
        return None, None, 0.0
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    parent = gini(y)
    best_feature = None
    best_threshold = None
    best_gain = 0.0

    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        if vals.size < 2:
            continue
        thresholds = (vals[:-1] + vals[1:]) / 2.0
        for t in thresholds:
            left_mask = X[:, j] <= t
            n_left = int(np.sum(left_mask))
            n_right = n - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            left_gini = gini(y[left_mask])
            right_gini = gini(y[~left_mask])
            weighted = (n_left * left_gini + n_right * right_gini) / n
            gain = parent - weighted
            # strict > keeps the first best on ties
            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = float(t)

    if best_feature is None:
        return None, None, 0.0
    return best_feature, best_threshold, float(best_gain)


def grow_tree(X, y, max_depth=2, min_samples_leaf=1, depth=0):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = int(y.size)

    leaf = {
        "leaf": True,
        "value": _majority(y) if n > 0 else 0,
        "n": n,
    }

    # Stopping rules
    if n == 0:
        return leaf
    if np.unique(y).size == 1:            # pure node
        return leaf
    if depth >= max_depth:                 # depth cap
        return leaf

    feature, threshold, gain = _best_split_constrained(X, y, min_samples_leaf)
    if feature is None or gain <= 0:       # no useful split
        return leaf

    left_mask = X[:, feature] <= threshold
    left = grow_tree(X[left_mask], y[left_mask],
                     max_depth, min_samples_leaf, depth + 1)
    right = grow_tree(X[~left_mask], y[~left_mask],
                      max_depth, min_samples_leaf, depth + 1)

    return {
        "leaf": False,
        "feature": feature,
        "threshold": threshold,
        "n": n,
        "left": left,
        "right": right,
    }

# Step 4 - predict_tree
import numpy as np


def predict_tree(tree, X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    preds = np.empty(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        node = tree
        while not node["leaf"]:
            if X[i, node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        preds[i] = node["value"]

    return preds

# Step 5 - fit_sklearn_tree
from sklearn.tree import DecisionTreeClassifier


def fit_sklearn_tree(X, y, max_depth=2, min_samples_leaf=1, random_state=42):
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf

# Step 6 - sklearn_splits
def sklearn_splits(clf):
    tree = clf.tree_
    splits = []
    for i in range(tree.node_count):
        if tree.children_left[i] == -1:  # leaf
            continue
        splits.append((int(tree.feature[i]), round(float(tree.threshold[i]), 3)))
    return splits

# Step 7 - compare_trees
def _my_splits(tree):
    """Depth-first, left-first (feature, rounded threshold) pairs for internal nodes."""
    if tree is None or tree["leaf"]:
        return []
    here = (int(tree["feature"]), round(float(tree["threshold"]), 3))
    return [here] + _my_splits(tree["left"]) + _my_splits(tree["right"])


def compare_trees(X, y, max_depth=2):
    X = np.asarray(X)
    y = np.asarray(y)

    mine = grow_tree(X, y, max_depth=max_depth, min_samples_leaf=1)
    clf = fit_sklearn_tree(X, y, max_depth=max_depth)

    my_splits = _my_splits(mine)
    their_splits = sklearn_splits(clf)

    same_splits = my_splits == their_splits

    my_preds = predict_tree(mine, X)
    their_preds = clf.predict(X)
    agreement = float((my_preds == their_preds).mean()) if len(y) else 1.0

    return {
        "same_splits": same_splits,
        "agreement": agreement,
        "my_splits": my_splits,
    }

# Step 8 - moons_data
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def moons_data(n_samples=300, noise=0.25, random_state=42, test_size=0.3):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

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

