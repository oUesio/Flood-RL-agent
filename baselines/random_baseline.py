import numpy as np

class RandomAgent:
    def __init__(self):
        # Based on class distribution: 355571, 102618, 38134, 3677
        total = 6935
        self.probs = [
            4993 / total,  # none
            1474 / total,  # yellow
            448 / total,   # amber
            20 / total,    # red
        ]

    def predict(self, obs: dict) -> int:
        return np.random.choice([0, 1, 2, 3], p=self.probs)

'''
agent = RandomAgent(y)
obs = environment.get_observable_features()
warning = agent.predict(obs)
print(f"Issued warning level: {warning}")
'''