# Refactoring of this tutorial:
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

# CART on the Bank Note dataset
import csv
import pprint
import random
import typing


def main():
    """Test CART on Bank Note dataset."""
    random.seed(1)
    # load and prepare data
    filename = "data/banknote_authentication.csv"
    dataset = load_csv(filename)
    # convert string attributes to integers
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    # evaluate algorithm
    n_folds = 5
    max_depth = 5
    min_size = 10
    scores = evaluate_algorithm(
        dataset, decision_tree, n_folds, max_depth, min_size
    )
    print("Scores: %s" % scores)
    mean_accuracy = sum(scores) / float(len(scores))
    print("Mean Accuracy: %.3f%%" % mean_accuracy)

    EXPECTED_SCORES = [
        96.35036496350365,
        97.08029197080292,
        97.44525547445255,
        98.17518248175182,
        97.44525547445255,
    ]
    assert scores == EXPECTED_SCORES
    EXPECTED_MEAN_ACCURACY = 97.299  # rounded to thousandths
    assert round(mean_accuracy, 3) == EXPECTED_MEAN_ACCURACY


# Load a CSV file
def load_csv(filename: str) -> typing.List[str]:
    file_ = open(filename, "rt")
    lines = csv.reader(file_)
    dataset = list(lines)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def decision_tree(train, test, max_depth, min_size):
    """Create a decision tree and run predictions on it.

    Args:
        train: Training set
        test: Test set.
        max_depth: Maximum allowable depth for the decision tree.
        min_size: Minimum number of patterns (?) for a node.

    Returns:
        List of the predictions for each item in the test set.
    """
    tree = build_tree(train, max_depth, min_size)
    pprint.pprint(tree)
    predictions = [predict(tree, record) for record in test]
    return predictions


def build_tree(train, max_depth, min_size):
    """Builds a decision tree recursively.

    Args:
        train: Training set.
        max_depth: Maximum allowable depth for the decision tree.
        min_size: Minimum number of patterns (?) for a node.

    Returns:
        Root of the decision tree.
    """
    root = build_internal_node(train)
    split(root, max_depth, min_size, 1)
    return root


def build_internal_node(dataset):
    """Build an internal node by evaluating the Gini score for each
    feature through a greedy algorithm.

    The split point is defined as an independent variable and value for
    it.

    Args:
        dataset: Training set.

    Returns:
        Dict representing the node.
    """
    class_values = list(set(observation[-1] for observation in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    # For each feature:
    for feature in range(len(dataset[0]) - 1):
        # For each observation/sample:
        for observation in dataset:
            # Test the Gini score of splitting the dataset on the value
            # of the current feature at the current observation.
            groups = test_split(feature, observation[feature], dataset)
            gini = gini_index(groups, class_values)
            # If the Gini score for this split is the best one yet, use that.
            if gini < b_score:
                b_index, b_value, b_score, b_groups = (
                    feature,
                    observation[feature],
                    gini,
                    groups,
                )
    return {"index": b_index, "value": b_value, "groups": b_groups}


# TODO potential refactoring: change dataset to DataFrame and index to
# a column.
def test_split(index, partition_value, dataset):
    """Split a dataset based on a feature and a specified cutoff value
    for that feature.

    Args:
        index: Column of the attribute in the input matrix.
        partition_value: Threshold/cutoff value for the feature.
        dataset: Training set.

    Returns:
        Pair of lists partitioning the training set by attribute value.
        First list is observations where the feature is less than the
        partition value, second has all observations where the features
        are greater than or equal to the partition value.
    """
    lt, ge = list(), list()
    for observation in dataset:
        if observation[index] < partition_value:
            lt.append(observation)
        else:
            ge.append(observation)
    return lt, ge


def gini_index(groups, classes):
    """Calculate the Gini index for a specific split.

    Args:
        groups (List[List]): Branches of the decision tree.
        classes (List[int]): List of output labels.

    Returns:
        float: The Gini score.
    """
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node["groups"]
    del node["groups"]
    # check for a no split
    if not left or not right:
        node["left"] = node["right"] = build_leaf(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node["left"], node["right"] = build_leaf(left), build_leaf(right)
        return
    # process left child
    if len(left) <= min_size:
        node["left"] = build_leaf(left)
    else:
        node["left"] = build_internal_node(left)
        split(node["left"], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node["right"] = build_leaf(right)
    else:
        node["right"] = build_internal_node(right)
        split(node["right"], max_depth, min_size, depth + 1)


# Create a terminal node value
def build_leaf(group: typing.Iterable) -> float:
    """Build a leaf node for the decision tree.

    Args:
        group:

    Returns:
        The predicted value.
    """
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def predict(node: dict, x: list) -> float:
    """Predict the class of x. Recursive function.

    Args:
        node: current node of the tree.
        x: sample to predict.

    Returns:
        prediction
    """
    # TODO refactor this to handle multiple records at once.
    if x[node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], x)
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], x)
        else:
            return node["right"]


if __name__ == "__main__":
    main()
