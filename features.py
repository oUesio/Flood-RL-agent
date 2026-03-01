import pandas as pd
import numpy as np
from scipy.stats import lognorm, skewnorm, gamma, norm
import os
import random
from datetime import datetime
from util import *
from household import generate_household_samples
from scipy.optimize import curve_fit
import geopandas as gpd
import rasterio
import glob
import datetime as dt

class Sampler:
    def __init__(self):
        # Resampling
        self.popden_params = {} # flag{mu, sigma, mean}
        self.property_params = {} # shape, loc, scale
        self.handover_params = {} # seasons{probs}
        self.response_params = {} # road_factor, prec_factor, popden_factor, seasons{data_min, data_max, shape, loc, scale}
        self.prec_params = {} # season{pct_zero, non_zero_max, shape, loc, scale}
        self.river_params = {} # river{shape, loc, scale}, river_list, river
        self.soil_params = {} # transform, stack, seasons{transform, stack}
        self.bed_params = {} # season{alpha, beta}
        self.urban_prob = None
        self.age = None
        self.depth_damage_k = None
        self.building_age_params = {} # probs, curr year, mean
        self.grid_params = {} # 

        self.features = {}
        self.dfs = {key: pd.read_csv(path) for key, path in FILE_PATHS.items()}

        self.init_ambulance_handover()
        self.init_soil_moisture()
        self.init_emergency_response()
        self.init_mean_property()
        self.init_cell()
        self.init_age()
        self.init_hospital_bed()
        self.init_precipitation()
        self.init_river()
        self.init_urban()
        self.init_english_proficiency()
        self.init_building_age()
        self.init_depth_damage()
        self.init_population_density()

    def init_cell(self):
        grid = gpd.read_file(GRID_FILE)
        self.dfs['grid'] = grid
        for f in GRID_FEATURES:
            if f == "hospital":
                self.grid_params['hospital_min'] = self.dfs['grid'][f].min()
                self.grid_params['hospital_max'] = self.dfs['grid'][f].max()   
            elif f != "historic": # exclude historic and hospital
                self.grid_params[f] = self.dfs['grid'][f].mean()

    def sample_cell(self, noise_scale=0.05):
        row = self.dfs['grid'].sample(1).iloc[0]
        for f in GRID_FEATURES:
            val = row[f]
            if f == "historic":
                self.features["historic"] = val
            else:
                noise = np.random.normal(0, noise_scale * abs(val + 1e-6))
                s = max(val + noise, 0) # ensure positive

                #### 
                self.features[f] = s
                ####

                if f in ["impervious", "water_dens", "road_dist", "road_dens"]:
                    if f == "impervious":
                        s = max(0, min(s, 1))
                        self.features["impervious"] = s # fraction
                    self.features[f+'_ratio'] = s / self.grid_params[f]
                elif f == 'hospital':
                    s = min(s, self.grid_params['hospital_max']) # ensure does not exceed max
                    self.features[f+'_norm'] = max(0, min((s - self.grid_params['hospital_min']) / (self.grid_params['hospital_max'] - self.grid_params['hospital_min']), 1)) # 0-1 range
                else:
                    self.features[f+'_ratio'] = self.grid_params[f] / s

        # Road factor for respone time
        road = self.features['road_dist_ratio']
        road_factor = 0.95 + 0.15 * (np.log1p(road) - np.log1p(self.grid_params['road_dist']))
        self.response_params['road_factor'] = np.clip(road_factor, 0.75, 1.1)

        self.features["geometry"] = row.geometry
        return row

    def sample_flood_depth(self, row):
        prob_cols = ["p_0", "p_0p0_0p2", "p_0p2_0p3", "p_0p3_0p6", "p_0p6_0p9", "p_0p9_1p2", "p_gt_1p2"]
        bins = [(0.0, 0.0), (0., 0.2), (0.2, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, np.inf)]

        probs = np.array([row[c] for c in prob_cols], dtype=float)
        probs = probs / probs.sum()

        low, high = bins[np.random.choice(len(bins), p=probs)]

        if low == 0.0 and high == 0.0:
            depth = 0.0
        elif np.isinf(high):
            depth = min(low + np.random.exponential(scale=0.3), 3) # max 3m depth
        else:
            depth = np.random.uniform(low, high)
        self.features["depth"] = depth

    def init_population_density(self):  
        popden = self.dfs["population_density"]
        merged = popden.merge(self.dfs["urban"], on="OA21CD", how="left")

        for flag in ['urban', 'rural']:
            params = {}

            popden_total = merged.loc[
                merged["Urban_rural_flag"].str.lower() == flag, "Total"
            ]

            log_popden = np.log(popden_total)
            mu, sigma = log_popden.mean(), log_popden.std()
            popden_mean = popden_total.mean()

            params['mu'] = mu
            params['sigma'] = sigma
            params['mean'] = popden_mean

            self.popden_params[flag] = params

    def sample_population_density(self):
        params = self.popden_params['urban' if self.features['urban'] else 'rural']

        sample = np.random.lognormal(params['mu'], params['sigma'])
        self.features["population_density_ratio"] = sample / params['mean']

        # Population density scale factor for response time
        log_pop = np.log1p(sample)
        log_min = np.log1p(4) # min population density from data
        log_max = np.log1p(2561) # max population density from data

        scaled = (log_pop - log_min) / (log_max - log_min) # 0–1
        self.response_params['popden_factor'] = 0.95 + 0.15 * (1 - scaled) # 0.95–1.1 range

    def init_mean_property(self):
        property_df = self.dfs["property_value"]
        property_df = property_df.dropna(subset=["price", "property_type", "duration"])
        property_df = property_df[property_df["price"] > 0]

        log_price = np.log(property_df["price"])
        self.property_params['shape'], self.property_params['loc'], self.property_params['scale'] = skewnorm.fit(log_price)

    def sample_mean_property(self):
        sample_value = np.exp(skewnorm.rvs(self.property_params['shape'], loc=self.property_params['loc'], scale=self.property_params['scale'], size=1)[0])

        #### 
        self.features["property_value"] = sample_value
        ####

        cdf_value = skewnorm.cdf(np.log(sample_value), self.property_params['shape'], loc=self.property_params['loc'], scale=self.property_params['scale'])
        self.features["property_value_norm"] = cdf_value

    def init_building_age(self):
        self.building_age_params['curr_year'] = datetime.now().year

        counts = self.dfs["property_age"][list(BUCKET_RANGES)].sum().to_numpy()

        mids = np.array([(start + end) / 2 for start, end in BUCKET_RANGES.values()])
        mean_construction_year = np.dot(counts, mids) / counts.sum()
        self.building_age_params['mean'] = self.building_age_params['curr_year'] - mean_construction_year
        self.building_age_params['probs'] = counts / counts.sum()

    def sample_building_age(self):
        bucket = np.random.choice(list(BUCKET_RANGES), p=self.building_age_params['probs'])

        start, end = BUCKET_RANGES[bucket]
        sampled_year = np.random.randint(start, end + 1)
        sampled_age = self.building_age_params['curr_year'] - sampled_year

        self.features["building_age"] = sampled_age

        self.features["building_age_ratio"] = sampled_age / self.building_age_params['mean']

    def init_emergency_response(self):
        response = self.dfs["response_times"]
        for season in list(SEASON_MONTHS):
            months = SEASON_MONTHS_FULL[season]
            response_season = response[response["month"].isin(months)]

            c2 = response_season["C2_mean"].apply(hhmmss_to_hours)
            c3 = response_season["C3_mean"].apply(hhmmss_to_hours)

            c2_count = response_season["C2_count"].str.replace(",", "").astype(int).sum()
            c3_count = response_season["C3_count"].str.replace(",", "").astype(int).sum()

            self.response_params[season] = [c2_count / (c2_count + c3_count), c3_count / (c2_count + c3_count)]
            for category in ["C2", "C3"]:
                data = c2 if category == "C2" else c3
                params = {}
                params['shape'], params['loc'], params['scale'] = lognorm.fit(data, floc=0)

                params['data_min'] = data.min()
                params['data_max'] = data.max()
                self.response_params[season+'_'+category] = params

    def sample_emergency_response(self):
        category = np.random.choice( 
            ["C2", "C3"],
            p=self.response_params[self.features['season']],
        )
        params = self.response_params[self.features["season"]+'_'+category]

        response_sample = lognorm.rvs(params['shape'], loc=params['loc'], scale=params['scale'])

        #### 
        self.features["response_time"] = response_sample * self.response_params['popden_factor'] * self.response_params['prec_factor'] * self.response_params['road_factor']
        ####

        self.features["response_time_norm"] = (
            (response_sample * self.response_params['popden_factor'] * self.response_params['prec_factor'] * self.response_params['road_factor'] -params['data_min']) / (params['data_max'] - params['data_min'])
        ).clip(0, 1) # longer delays = more risk
    
    def init_ambulance_handover(self):
        handover = self.dfs["ambulance_handover"]

        for col in ["Handover time known", "Over 15 minutes", "Over 30 minutes", "Over 60 minutes", "Handover time unknown", "All handovers"]:
            handover[col] = handover[col].str.replace(",", "").astype(int)

        handover["Under 15 min"] = handover["Handover time known"] - handover["Over 15 minutes"]
        handover["15–30 min"] = handover["Over 15 minutes"] - handover["Over 30 minutes"]
        handover["30–60 min"] = handover["Over 30 minutes"] - handover["Over 60 minutes"]
        handover["Over 60 min"] = handover["Over 60 minutes"]

        handover["Date_parsed"] = pd.to_datetime(handover["Date"], format="%b'%y")

        for season in list(SEASON_MONTHS):
            handover_season = handover[handover["Date_parsed"].dt.month.isin(SEASON_MONTHS[season])]

            counts = handover_season[["Under 15 min", "15–30 min", "30–60 min", "Over 60 min"]].sum()
            self.handover_params[season] = counts / counts.sum()

    def sample_ambulance_handover(self):
        probs = self.handover_params[self.features["season"]]
        band = np.random.choice(probs.index, p=probs.values)

        max_time=120
        if band == "Under 15 min":
            time = np.random.uniform(0, 15)

        elif band == "15–30 min":
            time = np.random.uniform(15, 30)

        elif band == "30–60 min":
            time = np.random.uniform(30, 60)
        else: # Over 60 min
            while True: # Resample until valid
                sample = 60 + np.random.exponential(scale=20)
                if sample <= max_time:
                    time = sample
                    break
        
        self.features["handover_time"] = time
        self.features["handover_time_norm"] = np.clip(time / max_time, 0, 1)

    def init_hospital_bed(self):
        beds = self.dfs["hospital_beds"]

        beds["Period"] = pd.to_datetime(beds["Period"], format="%d/%m/%Y")
        # Aggregate by month (sum across all hospitals)
        beds = beds.groupby('Period')[['Available', 'Occupied', 'Free']].sum().reset_index()

        for season in list(SEASON_MONTHS):
            params = {}
            beds_season = beds[beds["Period"].dt.month.isin(SEASON_MONTHS[season])]

            params['alpha'] = beds_season['Free'].sum() + 1
            params['beta'] = beds_season['Occupied'].sum() + 1
            self.bed_params[season] = params

    def sample_hospital_bed(self):
        params = self.bed_params[self.features['season']]
        self.features["bed_occupancy"] = np.random.beta(params['alpha'], params['beta'], size=1)[0]

    def init_age(self):
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

        self.age = age.melt(
            id_vars=["Lower tier local authorities"],
            var_name="category",
            value_name="Observation"
        )
        
    def sample_age(self, observed):
        # Elderly rate
        self.features['elderly'] = sample_beta_observed(self.age, "category", "total_elderly", "total_not_elderly", observed['elderly_a'], observed['elderly_b'], "Observation")

        # Children rate
        self.features['children'] = sample_beta_observed(self.age, "category", "total_children", "total_not_children",observed['child_a'], observed['child_b'], "Observation")

    def init_depth_damage(self):
        depths = np.array([float(c) for c in self.dfs['depth_damage'].columns[1:]])
        # Compute overall damage fraction (mean across all types)
        overall_damage = self.dfs['depth_damage'].iloc[:, 1:].astype(float).mean(axis=0).values
        # Fit the exponential model
        params, _ = curve_fit(exp_damage, depths, overall_damage, bounds=(0, np.inf))
        self.depth_damage_k = params[0]

    def sample_depth_damage(self):
        # Add noise and ensures stays within bounds
        self.features['damage_fraction'] = np.clip(exp_damage(self.features['depth'], self.depth_damage_k) + np.random.normal(0, 0.05), 0, 1)

    def init_precipitation(self):
        # 24-Hour Precipitation
        # mm
        prec = self.dfs["precipitation"]
        prec["time"] = pd.to_datetime(prec["time"])
        for season in list(SEASON_MONTHS):
            params = {}
            rainfall = prec.loc[
                prec["time"].dt.month.isin(SEASON_MONTHS[season]),
                "rainfall"
            ]
            params['pct_zero'] = (rainfall == 0).sum() / len(rainfall)
            non_zero = rainfall[rainfall > 0]
            params['non_zero_max'] = non_zero.max()
            params['shape'], params['loc'], params['scale'] = gamma.fit(non_zero)
            self.prec_params[season] = params

    def sample_precipitation(self):
        params = self.prec_params[self.features["season"]]
        zero_sample = np.random.binomial(n=1, p=params['pct_zero'], size=1)[0]
        self.features['precipitation'] = 0 if zero_sample == 1 else gamma.rvs(a=params['shape'], loc=params['loc'], scale=params['scale'], size=1)[0]

        # Precipitation scale factor for response time
        log_prec = np.log1p(self.features['precipitation'])
        log_max = np.log1p(params['non_zero_max'])

        scaled = log_prec / log_max # 0–1
        self.response_params['prec_factor'] = 0.95 + 0.25 * scaled # 0.95–1.2 range

    def init_river(self):
        # River level
        # m
        self.river_params['river_list'] = glob.glob(os.path.join(FILE_PATHS_DIR['river_level_dir'], "*.csv"))
        for river in self.river_params['river_list']:
            params = {}
            df = pd.read_csv(river)
            values = pd.to_numeric(df["value"], errors="coerce").dropna()
            params['shape'], params['loc'], params['scale'] = gamma.fit(values, floc=0)
            self.river_params[river] = params

    def sample_river(self):
        river = random.choice(self.river_params['river_list'])
        params = self.river_params[river]
        self.river_params['river'] = river # use for future updating logic? otherwise delete

        self.features['river_level'] = gamma.rvs(a=params['shape'], loc=params['loc'], scale=params['scale'], size=1)[0]

    def init_soil_moisture(self):
        # Load soil moisture rasters
        tiff_files = sorted(glob.glob(os.path.join(FILE_PATHS_DIR['soil_moisture_dir'], "*.tif")))

        for season in list(SEASON_MONTHS):
            season_tiffs = []
            for f in tiff_files:
                # Extract date from filename
                date_str = f.split("_")[-1].replace(".tif", "")
                file_date = dt.datetime.strptime(date_str, "%Y-%m-%d")
                if file_date.month in SEASON_MONTHS[season]:
                    season_tiffs.append(f)

            # Load soil moisture rasters
            stack = []
            params = {}
            with rasterio.open(season_tiffs[0]) as src:
                params['transform'] = src.transform
                nodata = src.nodata
                for f in season_tiffs:
                    with rasterio.open(f) as s:
                        data = s.read(1).astype(np.float32)
                        if nodata is not None:
                            data[data == nodata] = np.nan
                        stack.append(data)

            params['stack'] = np.stack(stack, axis=0)  # shape: (time, rows, cols)
            self.soil_params[season] = params

    def sample_soil_moisture(self):
        params = self.soil_params[self.features["season"]]
        cell_geom = self.features["geometry"]  # polygon of the sampled grid cell

        # Find the single pixel containing the centroid of the sampled grid cell
        centroid_x, centroid_y = cell_geom.centroid.x, cell_geom.centroid.y
        col, row = ~params['transform'] * (centroid_x, centroid_y)
        row, col = int(row), int(col)

        # Ensure row/col are within raster bounds
        row = np.clip(row, 0, params['stack'].shape[1]-1)
        col = np.clip(col, 0, params['stack'].shape[2]-1)

        # Extract time series for that pixel
        values = params['stack'][:, row, col]
        values = values[~np.isnan(values)]

        mu, sigma = norm.fit(values)
        sigma = max(sigma, 1e-6)
        self.features['soil_moisture'] = norm.rvs(mu, sigma)

    def init_urban(self):
        self.urban_prob = self.dfs["urban"]["Urban_rural_flag"].value_counts(normalize=True)

    def sample_urban(self):
        self.features["urban"] = np.random.binomial(1, self.urban_prob["Urban"])

    def init_english_proficiency(self):
        self.dfs["english_proficiency"]['Proficiency_Group'] = self.dfs["english_proficiency"]['Proficiency in English language (4 categories)'].apply(map_proficiency)

    def sample_english_proficiency(self):
        self.features["english_proficiency"] = sample_beta(self.dfs["english_proficiency"], 'Proficiency_Group', 'Good English Proficiency', 'Bad English Proficiency', 'Observation')

    def sample_vehicle(self):
        self.features["vehicle_rate"] = sample_beta(self.dfs["vehicle"], "Car or van availability (3 categories)", "1 or more cars or vans in household", "No cars or vans in household", "Observation")
        
    def sample_holiday(self):
        self.features["holiday"] = int(random.random() < (28 / 365))

    def sample_season(self):
        self.features["season"] = random.choice(list(SEASON_MONTHS))
    
    def sample_household(self, n=100):
        household_features, observed = generate_household_samples(n)
        self.features |= household_features
        return observed
    
    def sample_health(self, observed):
        self.features['disabled'] = sample_beta_observed(self.dfs["disabled"], 'Disability (3 categories)', 'Disabled under the Equality Act', 'Not disabled under the Equality Act', observed['disable_a'], observed['disable_b'], 'Observation')
        self.features['general_health'] = sample_beta_observed(self.dfs["general_health"], 'General health (3 categories)', 'Good health', 'Not good health', observed['health_a'], observed['health_b'], 'Observation')

    def sample_features(self):  
        self.sample_season()

        self.sample_holiday()

        self.sample_building_age()

        self.sample_precipitation()

        self.sample_river()

        self.sample_urban()

        self.sample_mean_property()

        self.sample_ambulance_handover()

        self.sample_hospital_bed()

        self.sample_english_proficiency()

        self.sample_vehicle()

        row = self.sample_cell()

        observed = self.sample_household()

        # Depends on urban
        self.sample_population_density()

        # Depends on cell
        self.sample_soil_moisture()
        self.sample_flood_depth(row)

        # Depends on household
        self.sample_health(observed)
        self.sample_age(observed)
        
        # Depends on precipitation, population density, cell
        self.sample_emergency_response()

        # Depends on flood depth # ADD DEPENDENCY FOR OTHERS
        self.sample_depth_damage()

