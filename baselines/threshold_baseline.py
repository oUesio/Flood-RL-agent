from sklearn.tree import _tree
import numpy as np

SEASON_TO_IDX = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}

FEATURE_NAMES = [
    "precipitation", "flood_depth", "holiday", "soil_moisture",
    "water_density", "water_distance", "elevation", "impervious_surface",
    "historical_flood_flag", "deprivation_index", "residential", "commercial",
    "industrial", "agriculture", "transport", "population_density",
    "season_sin", "season_cos"
]

def extract_threshold_policy(clf):
    """
    Extracts and prints the threshold policy from a fitted decision tree,
    and returns a callable policy function.
    """
    tree = clf.tree_
    feature_name = [
        FEATURE_NAMES[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree.feature
    ]

    def predict(obs: dict) -> int:
        node = 0
        while tree.children_left[node] != _tree.TREE_LEAF:
            fname = feature_name[node]
            threshold = tree.threshold[node]
            if obs[fname] <= threshold:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]
        return int(np.argmax(tree.value[node]))

    return predict


class ThresholdAgent:
    def __init__(self, clf):
        self.policy = extract_threshold_policy(clf)

    def predict(self, obs: dict) -> int:
        return self.policy(obs)
