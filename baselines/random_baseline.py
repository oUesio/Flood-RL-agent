import numpy as np

class RandomAgent:
    def __init__(self):
        # Values from historical distribution of warnings
        total = 6935
        self.probs = [
            4993 / total,  # none
            1474 / total,  # yellow
            448 / total,   # amber
            20 / total,    # red
        ]

    def predict(self):
        return np.random.choice([0, 1, 2, 3], p=self.probs)
