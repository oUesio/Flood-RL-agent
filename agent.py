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

'''with open("data/min_max.json", "r") as file:
    MIN_MAX = json.load(file)'''

class FloodWarningEnv(gym.Env):
    def __init__(self):
        self.env = Environment()
        self.action_space = spaces.Discrete(4)  # 0=none, 1=yellow, 2=amber, 3=red
        self.observation_space = spaces.Box(low=0, high=1, shape=(N_FEATURES,), dtype=np.float32)
        self.prev_action = 0

    def step(self, action):
        obs = self._get_obs()
        reward = self._get_reward(action)
        self.prev_action = action
        terminated = True  # single timestep episode
        truncated = False
        return obs, reward, terminated, truncated, {}
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.env.sample_features()
        self.env.update_derived()
        self.prev_action = 0
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
    
    # Normalisation using data min and max
    '''
    def _normalise(self, value, feature): # NEW
        min_val = self.MIN_MAX[feature]["min"]
        max_val = self.MIN_MAX[feature]["max"]
        return float(np.clip((value - min_val) / (max_val - min_val + 1e-8), 0, 1))

    def _get_obs(self):
        f = self.env.get_observable_features()

        idx = SEASON_TO_IDX[f["season"]]
        season_sin = math.sin(2 * math.pi * idx / 4)
        season_cos = math.cos(2 * math.pi * idx / 4)
        return np.array([
            self._normalise(f["precipitation"], "precipitation"),
            self._normalise(f["flood_depth"], "flood_depth"),
            season_sin,
            season_cos,
            f["holiday"],
            f["soil_moisture"],
            f["water_density"],
            self._normalise(f["water_distance"], "water_distance"),
            self._normalise(f["elevation"], "elevation"),
            f["impervious_surface"],
            f["historical_flood_flag"],
            f["deprivation_index"],
            f["residential"],
            f["commercial"],
            f["industrial"],
            f["agriculture"],
            f["transport"],
            self._normalise(f["population_density"], "population_density"),
        ], dtype=np.float32)
    '''

    def _get_impact_level(self, impact):
        # THRESHOLDS AND RETURN VALUES TEMPORARY, TEMPORARY, WILL CHANGE TO REFLECT ACTUAL WARNING LEVELS TO THE IMPACT SCORE, RETURN CONTINUOUS VALUE SCALED TO IMPACT?
        if impact < 0.25:
            return 0  # no warning
        elif impact < 0.50:
            return 1  # yellow
        elif impact < 0.75:
            return 2  # amber
        else:
            return 3  # red

    def _get_reward(self, action):
        impact_level = self._get_impact_level(self.env.impact)
        warning_level = action / 3  

        f = 0.3  # false alarm weight
        m = 2.0  # missed warning weight
        j = 0.5  # jump penalty weight

        # ASSUMES WARNING_LEVEL PROPORTIONAL TO IMPACT SCORE, TEMPORARY, WILL CHANGE
        alignment = 1 - abs(warning_level - impact) # reward for matching warning level to impact
        false_alarm = f * warning_level * max(0, warning_level - impact) # when warning level > impact, penalty for warning too high relative to impact
        missed = m * impact * max(0, impact - warning_level) # when impact > warning level, penalty for warning too low relative to impact
        jump = j * max(0, (action - self.prev_action) - 1) # when action increases by more than 1 step from previous, penalty for skipping warning levels in a single step

        return alignment - false_alarm - missed - jump



