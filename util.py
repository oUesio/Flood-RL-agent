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

BUCKET_RANGES = {
    "BP_PRE_1900": (1800, 1899),
    "BP_1900_1918": (1900, 1918),
    "BP_1919_1929": (1919, 1929),
    "BP_1930_1939": (1930, 1939),
    "BP_1945_1954": (1945, 1954),
    "BP_1955_1964": (1955, 1964),
    "BP_1965_1972": (1965, 1972),
    "BP_1973_1982": (1973, 1982),
    "BP_1983_1992": (1983, 1992),
    "BP_1993_1999": (1993, 1999),
    "BP_2000_2009": (2000, 2009),
    "BP_2010_2015": (2010, 2015),
}

def hhmmss_to_hours(time_str: str) -> float:
    h, m, s = map(int, time_str.split(":"))
    return h + m / 60 + s / 3600

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