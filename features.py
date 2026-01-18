import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import lognorm, skewnorm, truncnorm
import random

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


def sample_cell_with_noise(gdf, features, noise_scale=0.05):
    row = gdf.sample(1).iloc[0]
    sample = {}
    for f in features:
        val = row[f]
        noise = np.random.normal(0, noise_scale * abs(val + 1e-6))
        sample[f] = max(val + noise, 0)
    sample["geometry"] = row.geometry
    return sample

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

    samples["population_density"] = np.random.lognormal(mu, sigma)

    # Mean property value
    property_df = dfs["property_value"]
    property_df = property_df.dropna(subset=["price", "property_type", "duration"])
    property_df = property_df[property_df["price"] > 0]

    log_price = np.log(property_df["price"])
    shape, loc, scale = skewnorm.fit(log_price)

    samples["property_value"] = np.exp(skewnorm.rvs(shape, loc=loc, scale=scale, size=1)[0])

    # Building age
    building_age = dfs["property_age"]
    age_columns = [
        "BP_PRE_1900","BP_1900_1918","BP_1919_1929","BP_1930_1939",
        "BP_1945_1954","BP_1955_1964","BP_1965_1972","BP_1973_1982",
        "BP_1983_1992","BP_1993_1999","BP_2000_2009","BP_2010_2015"
    ]
    age_totals = building_age[age_columns].sum()
    age_probs = age_totals / age_totals.sum()
    age_categories = age_totals.index.to_numpy()

    building_range = np.random.choice(age_categories, p=age_probs)

    if building_range == "BP_PRE_1900":
        building_year = 1900
    else:
        start, end = map(int, building_range.split("_")[1:])
        building_year = np.random.randint(start, end + 1)
    samples["building_year"] = building_year

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

    data = c2 if category == "C2" else c3
    shape, loc, scale = lognorm.fit(data, floc=0)
    response_sample = lognorm.rvs(shape, loc=loc, scale=scale)

    mean_popden = popden_total.mean()
    factor = 1 + 0.5 * ((samples["population_density"] - mean_popden) / mean_popden)
    samples["response_time"] = response_sample * factor

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
    probs = counts / counts.sum()

    samples["handover_time"] = sample_handover_time(probs)

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

    # Second address rate
    second_add = dfs["second_address"]
    second_add["second_address_combined"] = second_add["Second address indicator (3 categories)"].apply(
        lambda x: "No second address" if x == "No second address" else "Has second address"
    )
    second_add["second_address_combined"] = sample_beta(second_add, "second_address_combined", "Has second address", "No second address", "Observation")

    # Grid
    grid_features = [
        "water_dens",
        "water_dist",
        "risk_score",
        "elevation",
        "impervious",
        "historic",
        "road_dens",
        "road_dist",
        "hospital",
    ]
    samples["grid"] = sample_cell_with_noise(gdf_grid, grid_features)

    return samples

samples = generate_feature_samples()
print(samples)