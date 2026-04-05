import pickle
from sklearn.tree import _tree
import numpy as np

# Load the saved decision tree model
with open("dt_model.pkl", "rb") as f:
    clf = pickle.load(f)

def extract_threshold_policy(clf, feature_names):
    """
    Extracts and prints the threshold policy from a fitted decision tree,
    and returns a callable policy function.
    """
    tree = clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree.feature
    ]

    def predict(obs: dict) -> int:
        node = 0
        while tree.feature[node] != _tree.TREE_LEAF:
            fname = feature_name[node]
            threshold = tree.threshold[node]
            if obs[fname] <= threshold:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]
        return int(np.argmax(tree.value[node]))

    return predict


class Threshold:
    def __init__(self, clf, feature_names):
        self.policy = extract_threshold_policy(clf, feature_names)

    def predict(self, obs: dict) -> int:
        return self.policy(obs)


'''
agent = ThresholdAgent(clf, feature_names)
obs = environment.get_observable_features()
warning = agent.predict(obs)
print(f"Issued warning level: {warning}")
'''