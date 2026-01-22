
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import gamma, lognorm, norm, skewnorm, truncnorm
import random
import os
import rasterio
import glob
import datetime as dt
from scipy.optimize import curve_fit

import time

from household import generate_household_samples # MODIFY TO AVERAGED NOT ARRAY, ALSO OUTPUT TOTAL TO UPDATE THE BETA DISTS
from features import generate_feature_samples, SEASON_MONTHS


FILE_PATHS = {
    "precipitation": "data/precipitation.csv",
    "depth_damage": "data/depth_damage.csv",
    "disabled": "data/disabled.csv",
    "general_health": "data/general_health.csv",
    "age": "data/age.csv",
}

FILE_PATHS_DIR = {
    "groundwater_dir": "data/groundwater_level",
    "river_flow_dir": "data/river_flow",
    "river_level_dir": "data/river_level",
    "soil_moisture_dir": "data/soil_moisture",
    "flood_risk_dir": "data/flood_risk",
}

DEPTH_THRESHOLDS = [0.2, 0.3, 0.6, 0.9, 1.2]

RISK_BAND_WEIGHTS = {
    "Very Low": 0.0005, # (0 + 0.1)/2 
    "Low": 0.00055, # (0.1 + 1)/2
    "Medium": 0.0215, # (1 + 3.3)/2
    "High": 0.033 # (3.3 + 3.3)/2
}

FILE_DEPTH = {
    "merged_rofsw_4bandPolygon.shp": 0.0,
    "merged_rofsw_4band_0_2m_depthPolygon.shp": 0.2,
    "merged_rofsw_4band_0_3m_depthPolygon.shp": 0.3,
    "merged_rofsw_4band_0_6m_depthPolygon.shp": 0.6,
    "merged_rofsw_4band_0_9m_depthPolygon.shp": 0.9,
    "merged_rofsw_4band_1_2m_depthPolygon.shp": 1.2
}

class Environment():
    def __init__(self):
        self.features = generate_feature_samples()
        self.household = None
        self.derived = {}
        self.warning_issued = False
        self.impact = None
    
    def exp_damage(self, d, k):
        return 1 - np.exp(-k * d)
    
    def sample_beta(self, df, category_col, a, b, observed_a, observed_b, value_col):
        total = df.groupby(category_col)[value_col].sum()
        alpha = total[a] + 1 + observed_a
        beta = total[b] + 1 + observed_b
        return np.random.beta(alpha, beta, size=1)[0]

    def update_precipitation(self):
        # update precipitation based on previous precipitation
        pass

    def update(self):
        # groundwater level, river flow, river level, soil moisture saturation, emergency response
        # self.features['precipitation']
        # modify those variables based on self.precipitation
        pass

    def update_depth(self):
        # flood depth, depth-damage
        # modify based on the hydrological features
        pass

    def update_beta(self, observed, disabled, gen_health, age):
        # Disability rate
        self.features['disabled'] = self.sample_beta(disabled, 'Disability (3 categories)', 'Disabled under the Equality Act', 'Not disabled under the Equality Act', observed['disable_a'], observed['disable_b'], 'Observation')

        # General health rate
        self.features['general_health'] = self.sample_beta(gen_health, 'General health (3 categories)', 'Good health', 'Not good health', observed['health_a'], observed['health_b'], 'Observation')

        # Age
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
        self.features['elderly'] = self.sample_beta(age, "category", "total_elderly", "total_not_elderly", observed['elderly_a'], observed['elderly_b'], "Observation")

        # Children rate
        self.features['children'] = self.sample_beta(age, "category", "total_children", "total_not_children",observed['child_a'], observed['child_b'], "Observation")

    def physical_vul(self):
        weights = {
            "elevation_ratio": 0.30,
            "impervious_ratio": 0.20,
            "water_dist_ratio": 0.05,
            "water_dens_ratio": 0.10,
            "road_dens_ratio": 0.10,
            "road_dist_ratio": 0.05,
            "building_age_ratio": 0.10,
        }

        features = {
            "elevation_ratio": self.features["grid"]["elevation_ratio"],
            "impervious_ratio": self.features["grid"]["impervious_ratio"],
            "water_dist_ratio": self.features["grid"]["water_dist_ratio"],
            "water_dens_ratio": self.features["grid"]["water_dens_ratio"],
            "road_dens_ratio": self.features["grid"]["road_dens_ratio"],
            "road_dist_ratio": self.features["grid"]["road_dist_ratio"],
            "building_age_ratio": self.features["building_age_ratio"],
        }
        print(features)

        total_weight = sum(weights.values())

        self.derived["physical"] = (
            sum(weights[k] * features[k] for k in weights) / total_weight
        )

    
    def socioeconomic_vul(self):
        weights = {
            "elderly": 0.15,
            "disabled": 0.15,
            "children": 0.15,
            "general_health": 0.15,
            "english_proficiency": 0.05,
            "household": 0.30, ###
            "property_value_ratio": 0.10,
        }

        features = {
            "elderly": self.features["elderly"],
            "disabled": self.features["disabled"],
            "children": self.features["children"],
            "general_health": self.features["general_health"],
            "english_proficiency": self.features["english_proficiency"],
            "household": self.household,
            "property_value_ratio": self.features["property_value_ratio"],
        }
        print(features)

        total_weight = sum(weights.values())

        self.derived["socioeconomic"] = (
            sum(weights[k] * features[k] for k in weights) / total_weight
        )


    def preparedness(self):
        weights = {
            "response_time_ratio": 0.20,
            "handover_time_ratio": 0.10,
            "bed_occupancy": 0.10,
            "vehicle": 0.15,
            "hospital_ratio": 0.10,
        }

        features = {
            "response_time_ratio": self.features["response_time_ratio"],
            "handover_time_ratio": self.features["handover_time_ratio"],
            "bed_occupancy": self.features["bed_occupancy"],
            "vehicle": self.features["vehicle_rate"],
            "hospital_ratio": self.features["grid"]["hospital_ratio"],
        }
        print(features)

        total_weight = sum(weights.values())
        warning_factor = 1.2 if self.warning_issued else 1

        self.derived["preparedness"] = (
            (sum(weights[k] * features[k] for k in weights) / total_weight) * warning_factor
        )

    def recovery(self):
        weights = {
            "income_rate": 0.40,
            "home_insure_rate": 0.20,
        }

        features = {
            "income_rate": self.features["income_rate"],
            "home_insure_rate": self.features["home_insure_rate"],
        }
        print(features)

        total_weight = sum(weights.values())

        self.derived["recovery"] = (
            sum(weights[k] * features[k] for k in weights) / total_weight
        )


    def exposure(self):
        weights = {
            "population_density_ratio": 0.30,
        }

        features = {
            "population_density_ratio": self.features["population_density_ratio"],
        }
        print(features)

        total_weight = sum(weights.values())
        historic_factor = 1.2 if self.features["grid"]["historic"] else 1
        holiday_factor = 1.1 if self.features["holiday"] else 1

        self.derived["exposure"] = (
            (sum(weights[k] * features[k] for k in weights) / total_weight) * historic_factor * holiday_factor
        )

    def impact_score(self):
        weights = {
            "exposure": 0.25,
            "physical": 0.20,
            "socioeconomic": 0.15,
            "depth": 0.15,
            "damage_fraction": 0.10,
            "preparedness": 0.10,
            "recovery": 0.05,
        }

        features = {
            "exposure": self.derived["exposure"],
            "physical": self.derived["physical"],
            "socioeconomic": self.derived["socioeconomic"],
            "depth": self.features["depth"],
            "damage_fraction": self.features["damage_fraction"],
            # invert: higher preparedness/recovery = lower impact
            "preparedness": -1 * self.derived["preparedness"],
            "recovery": -1 * self.derived["recovery"],
        }

        total_weight = sum(weights.values())

        self.impact = (
            sum(weights[k] * features[k] for k in weights) / total_weight
        )


    def init_prec(self, precipitation):
        # 24-Hour Precipitation
        rainfall = precipitation["rainfall"]
        pct_zero = (rainfall == 0).sum() / len(rainfall)
        zero_sample = np.random.binomial(n=1, p=pct_zero, size=1)[0]

        rainfall_nonzero = precipitation.loc[precipitation["rainfall"] > 0, "rainfall"]
        shape, loc, scale = gamma.fit(rainfall_nonzero)
        self.features['precipitation'] = 0 if zero_sample == 1 else gamma.rvs(a=shape, loc=loc, scale=scale, size=1)[0]

    def init_groundwater(self):
        # Groundwater level
        # mAOD
        # add prec dependency, modify ground water level based on previous prec_sample
        groundwater_files = [f for f in os.listdir(FILE_PATHS_DIR['groundwater_dir']) if f.lower().endswith(".csv")]
        gw_file = random.choice(groundwater_files)
        path = os.path.join(FILE_PATHS_DIR['groundwater_dir'], gw_file)
        df = pd.read_csv(path)
        values = pd.to_numeric(df["value"], errors="coerce").dropna()
        a, loc, scale = skewnorm.fit(values)
        self.features['groundwater'] = skewnorm.rvs(a, loc, scale, size=1)[0]

    def init_river(self):
        # River flow (WORK IN PREC)
        # m3/s
        # add prec dependency, modify ground water level based on previous prec_sample
        river_flow_files = [f for f in os.listdir(FILE_PATHS_DIR['river_flow_dir']) if f.lower().endswith(".csv")]
        rf_file = random.choice(river_flow_files)
        path = os.path.join(FILE_PATHS_DIR['river_flow_dir'], rf_file)
        df = pd.read_csv(path)
        values = pd.to_numeric(df["value"], errors="coerce").dropna()
        shape, loc, scale = gamma.fit(values, floc=0)
        self.features['river_flow'] = gamma.rvs(a=shape, loc=loc, scale=scale, size=1)[0]

        # River level (WORK IN PREC)
        # m
        # add prec dependency, modify ground water level based on previous prec_sample
        file_prefix = rf_file.split('-')[0]
        river_level_files = [f for f in os.listdir(FILE_PATHS_DIR['river_level_dir']) if f.lower().endswith(".csv")]
        for f in river_level_files:
            if file_prefix in f:
                rl_file = f
                break
        path = os.path.join(FILE_PATHS_DIR['river_level_dir'], rl_file)
        df = pd.read_csv(path)
        values = pd.to_numeric(df["value"], errors="coerce").dropna()
        shape, loc, scale = gamma.fit(values, floc=0)
        self.features['river_level'] = gamma.rvs(a=shape, loc=loc, scale=scale, size=1)[0]
    
    def init_soil_moisture(self):
        # Soil moisture saturation (WORK IN PREC)
        # Choose pixel that contains the overall grid cell chosen
        # sample from normal distribution of time-series values for the pixel
        # filters by season

        cell_geom = self.features['grid']["geometry"]  # polygon of the sampled grid cell

        # Load soil moisture rasters
        tiff_files = sorted(glob.glob(os.path.join(FILE_PATHS_DIR['soil_moisture_dir'], "*.tif")))
        season_tiffs = []

        for f in tiff_files:
            # Extract date from filename
            date_str = f.split("_")[-1].replace(".tif", "")
            file_date = dt.datetime.strptime(date_str, "%Y-%m-%d")
            if file_date.month in SEASON_MONTHS[self.features['season']]:
                season_tiffs.append(f)

        # Load soil moisture rasters
        stack = []
        with rasterio.open(season_tiffs[0]) as src:
            transform = src.transform
            nodata = src.nodata
            for f in season_tiffs:
                with rasterio.open(f) as s:
                    data = s.read(1).astype(np.float32)
                    if nodata is not None:
                        data[data == nodata] = np.nan
                    stack.append(data)

        stack = np.stack(stack, axis=0)  # shape: (time, rows, cols)

        # Find the single pixel containing the centroid of the sampled grid cell
        centroid_x, centroid_y = cell_geom.centroid.x, cell_geom.centroid.y
        col, row = ~transform * (centroid_x, centroid_y)
        row, col = int(row), int(col)

        # Ensure row/col are within raster bounds
        row = np.clip(row, 0, stack.shape[1]-1)
        col = np.clip(col, 0, stack.shape[2]-1)

        # Extract time series for that pixel
        values = stack[:, row, col]
        values = values[~np.isnan(values)]

        # Fit normal distribution safely
        if len(values) == 0:
            self.features['soil_moisture'] = np.nan
        else:
            mu, sigma = norm.fit(values)
            sigma = max(sigma, 1e-6)
            self.features['soil_moisture'] = norm.rvs(mu, sigma)

    def init_depth(self):
        # Depth-damage
        df = pd.read_csv(FILE_PATHS['depth_damage'])
        depths = np.array([float(c) for c in df.columns[1:]])
        # Compute overall damage fraction (mean across all types)
        overall_damage = df.iloc[:, 1:].astype(float).mean(axis=0).values
        # Fit the exponential model
        params, _ = curve_fit(self.exp_damage, depths, overall_damage, bounds=(0, np.inf))
        k = params[0]
        # Add noise and ensures stays within bounds
        self.features['damage_fraction'] = np.clip(self.exp_damage(self.features['depth'], k) + np.random.normal(0, 0.05), 0, 1)

    def init_variable_samples(self):
        dfs = {key: pd.read_csv(path) for key, path in FILE_PATHS.items()}

        self.household, observed, self.features['home_insure_rate'], self.features['income_rate'] = generate_household_samples(10) # PLACEHOLDER VALUE
        self.update_beta(observed, dfs['disabled'], dfs['general_health'], dfs['age'])
        
        self.init_prec(dfs['precipitation'])
        self.init_groundwater()
        self.init_river()
        self.init_soil_moisture()
        self.init_depth()

        # Emergency response times
        prec_factor = np.exp(self.features['precipitation'] / 50) 
        self.features['response_time_ratio'] = self.features['response_time_ratio'] * prec_factor

        # Derived
        self.physical_vul()
        self.socioeconomic_vul()
        self.preparedness()
        self.recovery()
        self.exposure()

        # Impact score
        self.impact_score()
        
'''np.random.seed(42)
start = time.time()
temp = Environment()
temp.init_variable_samples()
print('Runtime full: ', time.time() - start)
print(temp.features)
print(temp.household)
print(temp.derived)
print(temp.impact)
# self.derived["exposure"], self.derived["physical"], self.derived["socioeconomic"], self.features["depth"], self.features["damage_fraction"], 1 - self.derived["preparedness"], "recovery": 1 - self.derived["recovery"], self.impact'''

import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(42)

N = 10  # number of simulations

results = {
    "exposure": [],
    "physical": [],
    "socioeconomic": [],
    "depth": [],
    "damage_fraction": [],
    "preparedness_inv": [],
    "recovery_inv": [],
    "impact": [],
}

for i in range(N):
    print(i)
    temp = Environment()
    temp.init_variable_samples()
    print(temp.derived)
    print(temp.features['depth'])
    print(temp.impact)

    results["exposure"].append(temp.derived["exposure"])
    results["physical"].append(temp.derived["physical"])
    results["socioeconomic"].append(temp.derived["socioeconomic"])
    results["depth"].append(temp.features["depth"])
    results["damage_fraction"].append(temp.features["damage_fraction"])
    results["preparedness_inv"].append(temp.derived["preparedness"])
    results["recovery_inv"].append(temp.derived["recovery"])
    results["impact"].append(temp.impact)

'''fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for ax, (key, values) in zip(axes, results.items()):
    ax.hist(values, bins=30)
    ax.set_title(key)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.save_fig('temp.png')
'''