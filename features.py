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
}

GRID = "data/grid/grid.shp"

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


def sample_time_from_band(band):
    if band == "Under 15 min":
        return trunc_normal(0, 15, mean=8, sd=3)
    if band == "15–30 min":
        return trunc_normal(15, 30, mean=22, sd=4)
    if band == "30–60 min":
        return trunc_normal(30, 60, mean=45, sd=7)
    if band == "Over 60 min":
        return 60 + np.random.exponential(scale=20)
    raise ValueError(f"Unknown band: {band}")


def sample_handover_time(probs, n=1):
    bands = probs.index.to_numpy()
    p = probs.to_numpy()
    sampled = np.random.choice(bands, size=n, p=p)
    times = np.array([sample_time_from_band(b) for b in sampled])
    return times if n > 1 else times[0]


def sample_beta(df, category_col, a, b, value_col):
    total = df.groupby(category_col)[value_col].sum()
    alpha = total[a] + 1
    beta = total[b] + 1
    return np.random.beta(alpha, beta, size=1)[0]


def sample_cell_with_noise(gdf, noise_scale=0.05):
    features = [
        "water_dens", # more water = more risk (more water)
        "water_dist", # shorter distance = more risk (closer to river increases exposure)
        "elevation", # lower elevation = more risk (water flows downhill)
        "impervious", # more impervious = more risk (less infiltration, more runoff)
        "historic", # flag
        "road_dens", # less roads = more risk (better access, but less roads means less impervious)
        "road_dist", # longer distance = more risk (harder access to evac)
        "hospital", # longer distance = more risk (slower emergency response)
    ]
    row = gdf.sample(1).iloc[0]
    sample = {}
    eps=1e-6 # prevent zero division
    for f in features:
        val = row[f]
        if f == "historic":
            sample["historic"] = val
        else:
            noise = np.random.normal(0, noise_scale * abs(val + 1e-6))
            s = max(val + noise, 0)
            mean = gdf[f].mean()
            if f in ["impervious", "water_dens", "road_dist", "hospital"]:
                sample[f+'_ratio'] = s / (mean + eps) 
            else:
                sample[f+'_ratio'] = mean / (s + eps)
    sample["geometry"] = row.geometry
    return sample, row

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

def sample_flood_depth(row):
    prob_cols = [
        "p_0",
        "p_0p0_0p2",
        "p_0p2_0p3",
        "p_0p3_0p6",
        "p_0p6_0p9",
        "p_0p9_1p2",
        "p_gt_1p2"
    ]
    bins = [
        (0.0, 0.0),
        (0.0, 0.2),
        (0.2, 0.3),
        (0.3, 0.6),
        (0.6, 0.9),
        (0.9, 1.2),
        (1.2, np.inf),
    ]

    # Build the probs dict
    probs = {
        bin_range: float(row[col])
        for bin_range, col in zip(bins, prob_cols)
    }

    range_probs = np.array(list(probs.values()))
    range_index = np.random.choice(len(bins), p=range_probs)
    low, high = bins[range_index]

    if low == 0.0 and high == 0.0:
        return 0.0
    elif np.isinf(high):
        return low + np.random.exponential(scale=0.3)
    else:
        return np.random.uniform(low, high)

def generate_feature_samples():
    samples = {}
    dfs = {key: pd.read_csv(path) for key, path in FILE_PATHS.items()}
    gdf_grid = gpd.read_file(GRID)
    
    # Urban/Rural
    urban = dfs["urban"]
    probs = urban["Urban_rural_flag"].value_counts(normalize=True)
    samples["urban"] = np.random.binomial(1, probs["Urban"])

    # Population density
    popden = dfs["population_density"]
    merged = popden.merge(urban, on="OA21CD", how="left")

    flag = "urban" if samples["urban"] == 1 else "rural"
    popden_total = merged.loc[
        merged["Urban_rural_flag"].str.lower() == flag, "Total"
    ]

    log_popden = np.log(popden_total)
    mu, sigma = log_popden.mean(), log_popden.std()
    popden_mean = popden_total.mean()
    popden_sample = np.random.lognormal(mu, sigma)
    samples["population_density_ratio"] = popden_sample / popden_mean

    # Mean property value
    property_df = dfs["property_value"]
    property_df = property_df.dropna(subset=["price", "property_type", "duration"])
    property_df = property_df[property_df["price"] > 0]

    log_price = np.log(property_df["price"])
    shape, loc, scale = skewnorm.fit(log_price)

    samples["property_value_ratio"] = property_df["price"].mean() / np.exp(skewnorm.rvs(shape, loc=loc, scale=scale, size=1)[0]) # lower property value = more susceptible

    # Building age
    building_age = dfs["property_age"]
    age_columns = [
        "BP_PRE_1900","BP_1900_1918","BP_1919_1929","BP_1930_1939",
        "BP_1945_1954","BP_1955_1964","BP_1965_1972","BP_1973_1982",
        "BP_1983_1992","BP_1993_1999","BP_2000_2009","BP_2010_2015"
    ]
    age_totals = building_age[age_columns].sum()
    year_counts = {}

    for col, count in age_totals.items():
        if col == "BP_PRE_1900":
            years = range(1850, 1900)
        else:
            a, b = map(int, col.split("_")[1:])
            years = range(a, b + 1)

        for y in years:
            year_counts[y] = year_counts.get(y, 0) + count / len(years)

    years = np.array(list(year_counts.keys()))
    counts = np.array(list(year_counts.values()))
    counts /= counts.sum()

    current_year = datetime.now().year
    mean_year = np.sum(years * counts)
    mean_building_age = current_year - mean_year
    std_year = np.sqrt(np.sum(counts * (years - mean_year)**2))
    building_age = current_year - int(np.random.normal(loc=mean_year, scale=std_year))
    samples["building_age_ratio"] = building_age / mean_building_age

    # Season
    samples["season"] = random.choice(list(SEASON_MONTHS))

    # Holiday
    samples["holiday"] = int(random.random() < (28 / 365))

    # Emergency Response Times
    response = dfs["response_times"]
    months = SEASON_MONTHS_FULL[samples["season"]]
    response_season = response[response["month"].isin(months)]

    c2 = response_season["C2_mean"].apply(hhmmss_to_hours)
    c3 = response_season["C3_mean"].apply(hhmmss_to_hours)

    c2_count = response_season["C2_count"].str.replace(",", "").astype(int).sum()
    c3_count = response_season["C3_count"].str.replace(",", "").astype(int).sum()

    category = np.random.choice(
        ["C2", "C3"],
        p=[c2_count / (c2_count + c3_count), c3_count / (c2_count + c3_count)],
    )

    mean_response_time = c2.mean() if category == 'C2' else c3.mean()

    data = c2 if category == "C2" else c3
    shape, loc, scale = lognorm.fit(data, floc=0)
    response_sample = lognorm.rvs(shape, loc=loc, scale=scale)

    factor = 1 + 0.5 * ((popden_mean - popden_sample) / popden_mean)
    samples["response_time_ratio"] = response_sample * factor / mean_response_time # longer delays = more risk

    # Ambulance handover delays !!!
    handover = dfs["ambulance_handover"]

    for col in ["Handover time known", "Over 15 minutes", "Over 30 minutes", "Over 60 minutes", "Handover time unknown", "All handovers"]:
        handover[col] = handover[col].str.replace(",", "").astype(int)

    handover["Under 15 min"] = handover["Handover time known"] - handover["Over 15 minutes"]
    handover["15–30 min"] = handover["Over 15 minutes"] - handover["Over 30 minutes"]
    handover["30–60 min"] = handover["Over 30 minutes"] - handover["Over 60 minutes"]
    handover["Over 60 min"] = handover["Over 60 minutes"]

    handover["Date_parsed"] = pd.to_datetime(handover["Date"], format="%b'%y")
    handover_season = handover[handover["Date_parsed"].dt.month.isin(SEASON_MONTHS[samples["season"]])]

    counts = handover_season[["Under 15 min", "15–30 min", "30–60 min", "Over 60 min"]].sum()
    midpoints = np.array([7.5, 22.5, 45, 75])
    mean_handover = np.sum(counts * midpoints) / counts.sum()
    probs = counts / counts.sum()
    samples["handover_time_ratio"] = sample_handover_time(probs) / mean_handover

    # Hospital bed availability 
    beds = dfs["hospital_beds"]

    beds["Period"] = pd.to_datetime(beds["Period"], format="%d/%m/%Y")
    # Aggregate by month (sum across all hospitals)
    beds = beds.groupby('Period')[['Available', 'Occupied', 'Free']].sum().reset_index()

    # Compute occupancy percentage
    beds['OccupancyPct'] = beds['Occupied'] / beds['Available']

    beds_season = beds[beds["Period"].dt.month.isin(SEASON_MONTHS[samples["season"]])]

    alpha = beds_season['Free'].sum() + 1
    beta = beds_season['Occupied'].sum() + 1

    samples["bed_occupancy"] = np.random.beta(alpha, beta, size=1)[0]

    # English proficiency rate
    english = dfs["english_proficiency"] 
    english['Proficiency_Group'] = english['Proficiency in English language (4 categories)'].apply(map_proficiency)
    samples["english_proficiency"] = sample_beta(english, 'Proficiency_Group', 'Good English Proficiency', 'Bad English Proficiency', 'Observation')

    # Vehicle rate 
    vehicle = dfs["vehicle"]
    samples["vehicle_rate"] = sample_beta(vehicle, "Car or van availability (3 categories)", "1 or more cars or vans in household", "No cars or vans in household", "Observation")

    '''# Second address rate
    second_add = dfs["second_address"]
    second_add["second_address_combined"] = second_add["Second address indicator (3 categories)"].apply(
        lambda x: "No second address" if x == "No second address" else "Has second address"
    )
    samples["second_address"] = sample_beta(second_add, "second_address_combined", "Has second address", "No second address", "Observation")'''

    # Grid
    samples["grid"], row = sample_cell_with_noise(gdf_grid)

    # Flood depth
    samples["depth"] = sample_flood_depth(row)

    return samples

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Store all samples in a list
all_values = defaultdict(list)

# Collect values across all samples
for _ in range(100):
    samples = generate_feature_samples()
    for key, values in samples['grid'].items():
        if key != 'geometry':
            if np.isfinite(values):
                all_values[key].append(values)  # combine all iterations
            else:
                print(key, values)

# Plot histograms for each feature
num_features = len(all_values)
cols = 2
rows = int(np.ceil(num_features / cols))

fig, axes = plt.subplots(rows, cols, figsize=(10, rows*3))
axes = axes.flatten()

for i, (feature, values) in enumerate(all_values.items()):
    axes[i].hist(values, bins=20, alpha=0.7)
    axes[i].set_title(feature)

# Remove empty subplots if any
for j in range(num_features, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('temp.png')
