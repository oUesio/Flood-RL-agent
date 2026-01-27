import pandas as pd
import numpy as np
from scipy.stats import lognorm, skewnorm, gamma
import os
import random
from datetime import datetime
from util import *
from household import generate_household_samples
from scipy.optimize import curve_fit
import geopandas as gpd

class Sampler:
    def __init__(self):
        self.dist_params = {}
        self.features = {}
        self.dfs = {key: pd.read_csv(path) for key, path in FILE_PATHS.items()}
        self.generate_feature_samples()

    def sample_handover_time(self, probs, max_time=120):
        bands = probs.index.to_numpy()
        p = probs.to_numpy()
        sampled = np.random.choice(bands, size=1, p=p)
        time = sample_time_from_band(sampled)

        norm_time = np.clip(time / max_time, 0, 1)
        return norm_time

    def sample_cell(self, noise_scale=0.05):
        grid = gpd.read_file(GRID_FILE)
        self.dfs['grid'] = grid
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
        row = grid.sample(1).iloc[0]
        sample = {}
        eps=1e-6 # prevent zero division
        for f in features:
            val = row[f]
            if f == "historic":
                sample["historic"] = val
            else:
                noise = np.random.normal(0, noise_scale * abs(val + 1e-6))
                s = max(val + noise, 0) # ensure positive
                mean = self.dfs['grid'][f].mean()
                if f in ["impervious", "water_dens", "road_dist"]:
                    if f == "impervious":
                        s = max(0, min(s, 1))
                        sample["impervious"] = s # fraction
                    sample[f+'_ratio'] = s / mean
                elif f == 'hospital':
                    min_hosp = self.dfs['grid'][f].min()
                    max_hosp = self.dfs['grid'][f].max()
                    s = min(s, max_hosp) # ensure does not exceed max
                    sample[f+'_norm'] = (s - min_hosp) / (max_hosp - min_hosp)
                else:
                    sample[f+'_ratio'] = mean / (s + eps)
        sample["geometry"] = row.geometry
        self.features["grid"] = sample
        return row

    def sample_flood_depth(self, row):
        prob_cols = ["p_0", "p_0p0_0p2", "p_0p2_0p3", "p_0p3_0p6", "p_0p6_0p9", "p_0p9_1p2", "p_gt_1p2"]
        bins = [(0.0, 0.0), (0., 0.2), (0.2, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, np.inf)]

        # Build the probs dict
        probs = {
            bin_range: float(row[col])
            for bin_range, col in zip(bins, prob_cols)
        }

        range_probs = np.array(list(probs.values()))
        range_index = np.random.choice(len(bins), p=range_probs)
        low, high = bins[range_index]

        if low == 0.0 and high == 0.0:
            depth = 0.0
        elif np.isinf(high):
            depth = low + np.random.exponential(scale=0.3)
        else:
            depth = np.random.uniform(low, high)
        self.features["depth"] = depth
        
    def sample_population_density(self):
        popden = self.dfs["population_density"]
        merged = popden.merge(self.dfs["urban"], on="OA21CD", how="left")

        flag = "urban" if  self.features["urban"] == 1 else "rural"
        popden_total = merged.loc[
            merged["Urban_rural_flag"].str.lower() == flag, "Total"
        ]

        log_popden = np.log(popden_total)
        mu, sigma = log_popden.mean(), log_popden.std()
        popden_mean = popden_total.mean()
        popden_sample = np.random.lognormal(mu, sigma)
        self.features["population_density_ratio"] = popden_sample / popden_mean
        return popden_mean, popden_sample

    def sample_mean_property(self):
        property_df = self.dfs["property_value"]
        property_df = property_df.dropna(subset=["price", "property_type", "duration"])
        property_df = property_df[property_df["price"] > 0]

        log_price = np.log(property_df["price"])
        shape, loc, scale = skewnorm.fit(log_price)

        sample_value = np.exp(skewnorm.rvs(shape, loc=loc, scale=scale, size=1)[0])
        cdf_value = skewnorm.cdf(np.log(sample_value), shape, loc=loc, scale=scale)
        self.features["property_value_norm"] = cdf_value

    def sample_building_age(self):
        building_age = self.dfs["property_age"]
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
        sample_year = int(np.random.normal(loc=mean_year, scale=std_year))
        sample_year = min(sample_year, current_year) # ensure not a future year
        building_age = current_year - sample_year
        self.features["building_age_ratio"] = building_age / mean_building_age

    def sample_emergency_response(self, popden_mean, popden_sample):
        response = self.dfs["response_times"]
        months = SEASON_MONTHS_FULL[self.features["season"]]
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

        data_min = data.min()
        data_max = data.max()

        popden_factor = 1 + 0.5 * ((popden_mean - popden_sample) / popden_mean)
        prec_factor = np.exp(self.features['precipitation'] / 50) 
        self.features["response_time_norm"] = (
            (response_sample * popden_factor * prec_factor - data_min) / (data_max - data_min)
        ).clip(0, 1) # longer delays = more risk

    def sample_ambulance_handover(self):
        handover = self.dfs["ambulance_handover"]

        for col in ["Handover time known", "Over 15 minutes", "Over 30 minutes", "Over 60 minutes", "Handover time unknown", "All handovers"]:
            handover[col] = handover[col].str.replace(",", "").astype(int)

        handover["Under 15 min"] = handover["Handover time known"] - handover["Over 15 minutes"]
        handover["15–30 min"] = handover["Over 15 minutes"] - handover["Over 30 minutes"]
        handover["30–60 min"] = handover["Over 30 minutes"] - handover["Over 60 minutes"]
        handover["Over 60 min"] = handover["Over 60 minutes"]

        handover["Date_parsed"] = pd.to_datetime(handover["Date"], format="%b'%y")
        handover_season = handover[handover["Date_parsed"].dt.month.isin(SEASON_MONTHS[self.features["season"]])]

        counts = handover_season[["Under 15 min", "15–30 min", "30–60 min", "Over 60 min"]].sum()
        handover_probs = counts / counts.sum()
        self.features["handover_time_norm"] = self.sample_handover_time(handover_probs)

    def sample_hospital_bed(self):
        beds = self.dfs["hospital_beds"]

        beds["Period"] = pd.to_datetime(beds["Period"], format="%d/%m/%Y")
        # Aggregate by month (sum across all hospitals)
        beds = beds.groupby('Period')[['Available', 'Occupied', 'Free']].sum().reset_index()

        # Compute occupancy percentage
        beds['OccupancyPct'] = beds['Occupied'] / beds['Available']

        beds_season = beds[beds["Period"].dt.month.isin(SEASON_MONTHS[self.features["season"]])]

        alpha_bed = beds_season['Free'].sum() + 1
        beta_bed = beds_season['Occupied'].sum() + 1

        self.features["bed_occupancy"] = np.random.beta(alpha_bed, beta_bed, size=1)[0]

    def sample_age(self, observed):
        # Age
        age = self.dfs["age"]
        age["children"] = age["Age (6 categories)"] == "Aged 15 years and under"
        age["elderly"] = age["Age (6 categories)"] == "Aged 65 years and over"
        age = (age.groupby("Lower tier local authorities")
            .apply(
                lambda x: pd.Series({
                    "total_children": x.loc[x["children"], "Observation"].sum(),
                    "total_elderly": x.loc[x["elderly"], "Observation"].sum(),
                    "total_not_elderly": x.loc[~x["elderly"], "Observation"].sum(),
                    "total_not_children": x.loc[~x["children"], "Observation"].sum()
                }),
                include_groups=False
            )
            .reset_index())

        age = age.melt(
            id_vars=["Lower tier local authorities"],
            var_name="category",
            value_name="Observation"
        )

        # Elderly rate
        self.features['elderly'] = sample_beta_observed(age, "category", "total_elderly", "total_not_elderly", observed['elderly_a'], observed['elderly_b'], "Observation")

        # Children rate
        self.features['children'] = sample_beta_observed(age, "category", "total_children", "total_not_children",observed['child_a'], observed['child_b'], "Observation")

    def sample_depth_damage(self):
        depths = np.array([float(c) for c in self.dfs['depth_damage'].columns[1:]])
        # Compute overall damage fraction (mean across all types)
        overall_damage = self.dfs['depth_damage'].iloc[:, 1:].astype(float).mean(axis=0).values
        # Fit the exponential model
        params, _ = curve_fit(exp_damage, depths, overall_damage, bounds=(0, np.inf))
        k = params[0]
        # Add noise and ensures stays within bounds
        self.features['damage_fraction'] = np.clip(exp_damage(self.features['depth'], k) + np.random.normal(0, 0.05), 0, 1)

    def sample_precipitation(self):
        # 24-Hour Precipitation
        # mm
        prec_dist = {}
        prec = self.dfs["precipitation"]
        prec["time"] = pd.to_datetime(prec["time"])
        rainfall = prec.loc[
            prec["time"].dt.month.isin(SEASON_MONTHS[self.features['season']]),
            "rainfall"
        ]
        prec_dist['pct_zero'] = (rainfall == 0).sum() / len(rainfall)
        zero_sample = np.random.binomial(n=1, p=prec_dist['pct_zero'], size=1)[0]

        rainfall_nonzero = rainfall[rainfall > 0]
        prec_dist['shape'], prec_dist['loc'], prec_dist['scale'] = gamma.fit(rainfall_nonzero)
        self.features['precipitation'] = 0 if zero_sample == 1 else gamma.rvs(a=prec_dist['shape'], loc=prec_dist['loc'], scale=prec_dist['scale'], size=1)[0]
 
        self.dist_params['precipitation'] = prec_dist

    def sample_river(self):
        # River level
        # m
        river_level_files = [f for f in os.listdir(FILE_PATHS_DIR['river_level_dir']) if f.lower().endswith(".csv")]
        rl_file = random.choice(river_level_files)
        path = os.path.join(FILE_PATHS_DIR['river_level_dir'], rl_file)
        df = pd.read_csv(path)
        values = pd.to_numeric(df["value"], errors="coerce").dropna()
        shape, loc, scale = gamma.fit(values, floc=0)
        self.features['river_level'] = gamma.rvs(a=shape, loc=loc, scale=scale, size=1)[0]

    def generate_feature_samples(self):     
        self.features["season"] = random.choice(list(SEASON_MONTHS))

        self.features["household"], observed, self.features['home_insure_rate'], self.features['income_norm'] = generate_household_samples(10)

        self.features['disabled'] = sample_beta_observed(self.dfs["disabled"], 'Disability (3 categories)', 'Disabled under the Equality Act', 'Not disabled under the Equality Act', observed['disable_a'], observed['disable_b'], 'Observation')
        
        self.features['general_health'] = sample_beta_observed(self.dfs["general_health"], 'General health (3 categories)', 'Good health', 'Not good health', observed['health_a'], observed['health_b'], 'Observation')

        self.sample_age(observed)

        self.sample_precipitation()

        self.sample_river()

        probs = self.dfs["urban"]["Urban_rural_flag"].value_counts(normalize=True)
        self.features["urban"] = np.random.binomial(1, probs["Urban"])
        popden_mean, popden_sample = self.sample_population_density()

        self.sample_mean_property()

        self.sample_building_age()

        self.features["holiday"] = int(random.random() < (28 / 365))

        self.sample_emergency_response(popden_mean, popden_sample)

        self.sample_ambulance_handover()

        self.sample_hospital_bed()

        english = self.dfs["english_proficiency"] 
        english['Proficiency_Group'] = english['Proficiency in English language (4 categories)'].apply(map_proficiency)
        self.features["english_proficiency"] = sample_beta(english, 'Proficiency_Group', 'Good English Proficiency', 'Bad English Proficiency', 'Observation')

        vehicle = self.dfs["vehicle"]
        self.features["vehicle_rate"] = sample_beta(vehicle, "Car or van availability (3 categories)", "1 or more cars or vans in household", "No cars or vans in household", "Observation")

        row = self.sample_cell()

        self.sample_flood_depth(row)

        self.sample_depth_damage()


'''import matplotlib.pyplot as plt
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
'''