from environment import Environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

'''
PPO

Observable features in real life:
Precipitation (mm)
Flood depth (m)
Season 
Soil moisture (percent or fraction)

Observable features estimable from public data:
Water density (fraction)
Water distance (m)
Elevation (m)
Impervious surface (fraction)
Historic flood flag (binary)
Deprivation index (1-10 normalised to 0-1)
Residential (fraction)
Commercial (fraction)
Industrial (fraction)
Agriculture (fraction)
Transport (fraction)
Population density (of an area)

'''

N_FEATURES = 18
SEASON_TO_IDX = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
MAX_STEPS = 200  # number of timesteps per episode

IMPACT_THRESHOLDS = {
    "none":   0.0447,
    "yellow": 0.0673,
    "amber":  0.1657,
}

class FloodWarningEnv(gym.Env):
    def __init__(self, false_weight, missed_weight, jump_weight):
        super().__init__()
        self.env = Environment()
        self.action_space = spaces.Discrete(4)  # 0=none, 1=yellow, 2=amber, 3=red
        self.observation_space = spaces.Box(low=0, high=1, shape=(N_FEATURES,), dtype=np.float32)
        self.prev_action = 0
        self.current_step = 0

        # Reward/Penalty weights
        self.false_weight = false_weight
        self.missed_weight = missed_weight
        self.jump_weight = jump_weight

    def step(self, action):
        # Apply warning to environment so preparedness is affected
        self.env.update_warning(action)

        reward = self._get_reward(action)
        self.prev_action = action
        self.current_step += 1

        # Evolve dynamic features and recompute impact
        self.env.update_features()

        terminated = False
        truncated = self.current_step >= MAX_STEPS
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.sample_features()
        self.env.update_derived()
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
            return 0  # no warning
        elif impact < IMPACT_THRESHOLDS["yellow"]:
            return 1  # yellow
        elif impact < IMPACT_THRESHOLDS["amber"]:
            return 2  # amber
        else:
            return 3  # red

    def _get_reward(self, action):
        impact_level = self._get_impact_level(self.env.impact)

        alignment = 1.0 if action == impact_level else 0.0
        false_alarm = self.false_weight * action * max(0, action - impact_level)
        missed = self.missed_weight * impact_level * max(0, impact_level - action)
        jump = self.jump_weight * max(0, (action - self.prev_action) - 1)  # now meaningful across timesteps

        return alignment - false_alarm - missed - jump

