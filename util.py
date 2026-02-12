import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import lognorm, skewnorm, truncnorm
import random
from datetime import datetime

FILE_PATHS = {
    "urban": "data/urban_rural.csv",
    "population_density": "data/population_density.csv",
    "property_value": "data/property_value.csv",
    "property_age": "data/property_age.csv",
    "response_times": "data/response_times.csv",
    "ambulance_handover": "data/ambulance_handover.csv",
    "hospital_beds": "data/hospital_beds.csv",
    "english_proficiency": "data/english_proficiency.csv",
    "vehicle": "data/vehicle.csv",
    "second_address": "data/second_address.csv",
    "precipitation": "data/precipitation.csv",
    "depth_damage": "data/depth_damage.csv",
    "disabled": "data/disabled.csv",
    "general_health": "data/general_health.csv",
    "age": "data/age.csv"
}

GRID_FILE = "data/grid/grid.shp"

FILE_PATHS_DIR = {
    "river_flow_dir": "data/river_flow",
    "river_level_dir": "data/river_level",
    "soil_moisture_dir": "data/soil_moisture",
}

SEASON_MONTHS = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
    "Winter": [12, 1, 2],
}

SEASON_MONTHS_FULL = {
    "Spring": ["March", "April", "May"],
    "Summer": ["June", "July", "August"],
    "Autumn": ["September", "October", "November"],
    "Winter": ["December", "January", "February"],
}

def hhmmss_to_hours(time_str: str) -> float:
    h, m, s = map(int, time_str.split(":"))
    return h + m / 60 + s / 3600


def trunc_normal(low, high, mean, sd):
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd)

def map_proficiency(category):
    if category in [
        "Main language is English (English or Welsh in Wales)",
        "Main language is not English (English or Welsh in Wales): Can speak English very well or well"
    ]:
        return "Good English Proficiency"
    elif category == "Main language is not English (English or Welsh in Wales): Cannot speak English or cannot speak English well":
        return "Bad English Proficiency"
    else:
        return None
    
def sample_time_from_band(band):
    if band == "Under 15 min":
        return trunc_normal(0, 15, mean=8, sd=3)
    if band == "15–30 min":
        return trunc_normal(15, 30, mean=22, sd=4)
    if band == "30–60 min":
        return trunc_normal(30, 60, mean=45, sd=7)
    if band == "Over 60 min":
        return 60 + np.random.exponential(scale=20)
    
def sample_beta(df, category_col, a, b, value_col):
    total = df.groupby(category_col)[value_col].sum()
    alpha = total[a] + 1
    beta = total[b] + 1
    return np.random.beta(alpha, beta, size=1)[0]

def sample_beta_observed(df, category_col, a, b, observed_a, observed_b, value_col):
    total = df.groupby(category_col)[value_col].sum()
    alpha = total[a] + 1 + observed_a
    beta = total[b] + 1 + observed_b
    return np.random.beta(alpha, beta, size=1)[0]

def exp_damage(d, k):
    return 1 - np.exp(-k * d)