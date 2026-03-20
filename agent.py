from environment import Environment
from collections import defaultdict
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
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

IMPACT_THRESHOLDS = {
    "none":   0.0447,
    "yellow": 0.0673,
    "amber":  0.1657,
}

class FloodWarningEnv(gym.Env):
    def __init__(self):
        self.env = Environment()
        self.action_space = spaces.Discrete(4)  # 0=none, 1=yellow, 2=amber, 3=red
        self.observation_space = spaces.Box(low=0, high=1, shape=(N_FEATURES,), dtype=np.float32)
        #self.prev_action = 0

    def step(self, action):
        obs = self._get_obs()
        reward = self._get_reward(action)
        #self.prev_action = action
        terminated = True  # single timestep episode
        truncated = False
        return obs, reward, terminated, truncated, {}
    
    def reset(self):
        self.env.sample_features()
        self.env.update_derived()
        #self.prev_action = 0
        return self._get_obs(), {}

    # No normalisation
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
            f["historical_flood_flag"],
            f["deprivation_index"],
            f["residential"],
            f["commercial"],
            f["industrial"],
            f["agriculture"],
            f["transport"],
            f["population_density"],
        ], dtype=np.float32)

    def _get_impact_level(self, impact):
        thresholds = {
            "none":   0.0456,
            "yellow": 0.0725,
            "amber":  0.2346,
        }
        if impact < thresholds["none"]:
            return 0  # no warning
        elif impact < thresholds["yellow"]:
            return 1  # yellow
        elif impact < thresholds["amber"]:
            return 2  # amber
        else:
            return 3  # red

    def _get_reward(self, action):
        impact_level = self._get_impact_level(self.env.impact)

        f = 0.3  # false alarm weight
        m = 2.0  # missed warning weight
        #j = 0.5  # jump penalty weight

        alignment = 1.0 if action == impact_level else 0.0
        false_alarm = f * action * max(0, action - impact_level)
        missed = m * impact_level * max(0, impact_level - action)
        #jump = j * max(0, (action - self.prev_action) - 1)

        return alignment - false_alarm - missed #- jump


