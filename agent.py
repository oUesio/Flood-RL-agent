from environment import Environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

N_FEATURES = 18
SEASON_TO_IDX = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
MAX_STEPS = 200

IMPACT_THRESHOLDS = {
    "none":   0.0447,
    "yellow": 0.0673,
    "amber":  0.1657,
}

MAX_RESAMPLE_ATTEMPTS = 100  # avoid infinite loop if severe events are very rare

class FloodWarningEnv(gym.Env):
    def __init__(self, false_weight=1.5, missed_weight=2, jump_weight=0.5, severe_prob=0.0):
        super().__init__()
        self.env = Environment()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(N_FEATURES,), dtype=np.float32)
        self.prev_action = 0
        self.current_step = 0

        # Reward/Penalty weights
        self.false_weight = false_weight
        self.missed_weight = missed_weight
        self.jump_weight = jump_weight

        # Curriculum learning
        self.severe_prob = severe_prob  # probability of forcing a severe starting condition

    def set_severe_prob(self, prob):
        # Called externally by curriculum callback to decay severe_prob over training
        self.severe_prob = np.clip(prob, 0.0, 1.0)

    def _sample_features(self):
        if np.random.random() < self.severe_prob:
            # Force a severe starting condition (amber or red)
            for _ in range(MAX_RESAMPLE_ATTEMPTS):
                self.env.sample_features()
                self.env.update_derived()
                if self.env.impact > IMPACT_THRESHOLDS["amber"]:
                    return
            # Fallback to normal sample if no severe found within attempts
            self.env.sample_features()
            self.env.update_derived()
        else:
            self.env.sample_features()
            self.env.update_derived()

    def step(self, action):
        self.env.update_warning(action)
        reward = self._get_reward(action)
        self.prev_action = action
        self.current_step += 1
        self.env.update_features()
        terminated = False
        truncated = self.current_step >= MAX_STEPS
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_features()  # uses curriculum logic
        self.prev_action = 0
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        f = self.env.get_observable_features()
        idx = SEASON_TO_IDX[f["season"]]
        season_sin = math.sin(2 * math.pi * idx / 4)
        season_cos = math.cos(2 * math.pi * idx / 4)
        return np.array([
            f["precipitation"],
            f["flood_depth"],
            season_sin,
            season_cos,
            f["holiday"],
            f["soil_moisture"],
            f["water_density"],
            f["water_distance"],
            f["elevation"],
            f["impervious_surface"],
            float(f["historical_flood_flag"]),
            f["deprivation_index"],
            f["residential"],
            f["commercial"],
            f["industrial"],
            f["agriculture"],
            f["transport"],
            f["population_density"],
        ], dtype=np.float32)

    def _get_impact_level(self, impact):
        if impact < IMPACT_THRESHOLDS["none"]:
            return 0
        elif impact < IMPACT_THRESHOLDS["yellow"]:
            return 1
        elif impact < IMPACT_THRESHOLDS["amber"]:
            return 2
        else:
            return 3

    def _get_reward(self, action):
        impact_level = self._get_impact_level(self.env.impact)

        alignment = 1.0 if action == impact_level else 0.0
        false_alarm = self.false_weight * action * max(0, action - impact_level)
        missed = self.missed_weight * impact_level * max(0, impact_level - action)
        jump = self.jump_weight * max(0, (action - self.prev_action) - 1)
        return alignment - false_alarm - missed - jump