import pandas as pd
import numpy as np
from scipy.stats import lognorm, skewnorm, gamma, norm
import os
import random
from datetime import datetime
from util import *
from household import Household
from scipy.optimize import curve_fit
import geopandas as gpd
import rasterio
import glob
import datetime as dt

class Sampler:
    def __init__(self, use_historic):
        # Resampling
        self.property_params = {} # shape, loc, scale
        self.handover_params = {} # seasons{probs}
        self.response_params = {} # road_factor, prec_factor, popden_factor, seasons{data_min, data_max, shape, loc, scale}
        self.prec_params = {} # season{pct_zero, non_zero_max, shape, loc, scale}
        self.soil_params = {} # transform, stack, seasons{transform, stack}
        self.bed_params = {} # season{alpha, beta}
        self.urban_prob = None
        self.age = None
        self.depth_damage_k = {}
        self.building_age_params = {} # probs, curr year, mean
        self.grid_params = {} # 

        self.features = {}
        self.dfs = {key: pd.read_csv(path) for key, path in FILE_PATHS.items()}
        self.household = Household()

        self.use_historic = use_historic
        if self.use_historic == "yellow":
            self.historic_data = pd.read_csv("warning/yellow_warnings.csv")
        elif self.use_historic == "amber":
            self.historic_data = pd.read_csv("warning/amber_warnings.csv")
        elif self.use_historic == "red":
            self.historic_data = pd.read_csv("warning/red_warnings.csv")
        self.historic_row = None

        self.init_ambulance_handover()
        self.init_soil_moisture()
        self.init_emergency_response()
        self.init_mean_property()
        self.init_cell()
        self.init_age()
        self.init_hospital_bed()
        self.init_precipitation()
        self.init_english_proficiency()
        self.init_building_age()
        self.init_damage_cost()

    def init_cell(self):
        grid = gpd.read_file(GRID_FILE)
        self.dfs['grid'] = grid
        for f in GRID_FEATURES:
            if f in ["hospital", "emergency", "infra", "transport", "deprived"]: # norm
                self.grid_params[f+'_min'] = self.dfs['grid'][f].min()
                self.grid_params[f+'_max'] = self.dfs['grid'][f].max()   
            if f not in ["hospital", "emergency", "infra", "historic"]: # exclude historic, the rest are for ratio
                self.grid_params[f] = self.dfs['grid'][f].mean()
            if f in ["water_dens", "water_dist"]:
                self.grid_params[f+'_log_mean'] = np.log1p(self.dfs['grid'][f]).mean()

    def sample_cell(self):
        row = self.dfs['grid'].sample(1).iloc[0]
        for f in GRID_FEATURES:
            val = row[f]
            scaled_noise = np.random.normal(0, 0.05 * abs(val + 1e-6))
            noise = np.random.normal(0, 0.02)

            if f in ["historic"]: # no noise
                self.features[f] = val

            if f in ["impervious", "water_dens", "road_dist", "transport", "buildings", "popden"]: # ratio
                noisy_val = max(val + scaled_noise, 0)
                if f == "water_dens":
                    self.features[f+'_ratio'] = np.log1p(noisy_val) / (self.grid_params[f+'_log_mean'] + 1e-6)
                else:
                    self.features[f+'_ratio'] = noisy_val / self.grid_params[f]

            if f in ["water_dist", "elevation"]: # inverse ratio
                noisy_val = max(val + scaled_noise, 0)
                if f == "water_dist":
                    self.features[f+'_ratio'] = (self.grid_params[f+'_log_mean'] + 1e-6) / np.log1p(noisy_val)
                else:
                    self.features[f+'_ratio'] = self.grid_params[f] / noisy_val
            
            if f in ["hospital", "emergency", "infra", "transport", "deprived"]: # normalise
                val = max(0, min((val - self.grid_params[f+'_min']) / (self.grid_params[f+'_max'] - self.grid_params[f+'_min']), 1))
                if f == "infra":
                    val = val * 0.3 # scaled since electrical lines cover less area (line polygons)
                elif f == "transport":
                    val = val * 0.4 # scaled since roads and rail tracks cover less area (line polygons)

            if f in ["impervious", "deprived", "transport", "hospital", "emergency", "education", "resident", "commercial", "indust", "agri", "infra"]: # noise, fraction
                noisy_val = max(0, min(val+noise, 1))
                self.features[f] = noisy_val 

            if f in ["elevation", "water_dens", "water_dist", "popden"]: # scaled noise, non-fraction
                noisy_val = max(val + scaled_noise, 0)
                self.features[f] = noisy_val 


        # Population density factor for response time
        self.response_params['popden_factor'] = np.clip(0.8 + 0.4 * (self.features['popden_ratio'] / 2), 0.8, 1.2)

        # Road factor for response time
        road = self.features['road_dist_ratio']
        road_factor = 0.95 + 0.15 * (np.log1p(road) - np.log1p(self.grid_params['road_dist']))
        self.response_params['road_factor'] = np.clip(road_factor, 0.75, 1.1)
        # For rescaling road factor with flood depth
        self.response_params['base_road_factor'] = self.response_params['road_factor']


        self.features["geometry"] = row.geometry
        return row

    def sample_flood_depth(self, row, max_depth=3):
        prob_cols = ["p_0", "p_0p0_0p2", "p_0p2_0p3", "p_0p3_0p6", "p_0p6_0p9", "p_0p9_1p2", "p_gt_1p2"]
        bins = [(0.0, 0.0), (0., 0.2), (0.2, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, np.inf)]

        probs = np.array([row[c] for c in prob_cols], dtype=float)
        probs = probs / probs.sum()

        low, high = bins[np.random.choice(len(bins), p=probs)]

        if low == 0.0 and high == 0.0:
            depth = 0.0
        elif np.isinf(high):
            depth = min(low + np.random.exponential(scale=0.3), max_depth) # max 3m depth
        else:
            depth = np.random.uniform(low, high)
        self.features["depth"] = depth

    def update_flood_depth(self):
        # Upward push
        prec_influence = self.features['precipitation_norm']
        absorption_resistance = self.features['impervious'] + (1 - self.features['impervious']) * self.features['soil_moisture']
        push_factor = 0.05
        push = prec_influence * absorption_resistance * push_factor

        # Downward recession
        # Proportional to current depth, deeper floods drain faster=
        base_recession = 0.05
        recession = self.features['depth'] * base_recession * self.features['water_dist_ratio']

        # Positive delta = flooding worsening, negative = water receding
        delta = push - recession
        new_depth = np.clip(self.features['depth'] + delta, 0, 3.0)
        self.features['depth'] = new_depth

    def init_mean_property(self):
        property_df = self.dfs["property_value"]
        property_df = property_df.dropna(subset=["price", "property_type", "duration"])
        property_df = property_df[property_df["price"] > 0]

        log_price = np.log(property_df["price"])
        self.property_params['shape'], self.property_params['loc'], self.property_params['scale'] = skewnorm.fit(log_price)

    def sample_mean_property(self):
        sample_value = np.exp(skewnorm.rvs(self.property_params['shape'], loc=self.property_params['loc'], scale=self.property_params['scale'], size=1)[0])

        #### For histograms only
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

        if self.use_historic is not None:
            while self.historic_row['YEAR'].item() < sampled_year:
                bucket = np.random.choice(list(BUCKET_RANGES), p=self.building_age_params['probs'])
                start, end = BUCKET_RANGES[bucket]
                sampled_year = np.random.randint(start, end + 1)
            sampled_age = self.historic_row['YEAR'].item() - sampled_year
        else:
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
        response_sample = 4.5 # Capped at 4.5 hours, Approximately max mean for C3 in the data
        while response_sample >= 4.5:
            if self.use_historic is not None and pd.notna(self.historic_row["C2_count"].item()):
                C2_count, C3_count, C2_mean, C3_mean = self.historic_row["C2_count"].item(), self.historic_row["C3_count"].item(), self.historic_row["C2_mean"].item(), self.historic_row["C3_mean"].item()
                total = C2_count + C3_count
                if random.random() < (C2_count / total):
                    category = 'C2'
                    response_sample = max(0, C2_mean * (1 + np.random.normal(0, 0.1)))  # scaled by noise
                else:
                    category = 'C3'
                    response_sample = max(0, C3_mean * (1 + np.random.normal(0, 0.1)))
                    self.response_params['season_category'] = self.historic_row["SEASON"].item()+'_'+category
                params = self.response_params[self.response_params['season_category']]
            else:
                category = np.random.choice( 
                    ["C2", "C3"],
                    p=self.response_params[self.features['season']],
                )
                self.response_params['season_category'] = self.features["season"]+'_'+category
                params = self.response_params[self.response_params['season_category']]

                response_sample = lognorm.rvs(params['shape'], loc=params['loc'], scale=params['scale'])

        # Use for rescaling
        self.features['response_time_raw'] = response_sample

        scaled_sample = response_sample * self.response_params['popden_factor'] * self.response_params['prec_factor'] * self.response_params['road_factor']

        #### For histograms only
        self.features["response_time"] = scaled_sample
        ####

        self.features["response_time_norm"] = ((scaled_sample - params['data_min']) / (params['data_max'] - params['data_min'])).clip(0, 1) # longer delays = more risk
    
    def update_emergency_response(self):
        # Update road factor
        # Flood depth worsens road conditions on top of the base road distance factor
        depth_norm = self.features['depth'] / 3.0
        depth_road_weight = 0.2
        updated_road_factor = self.response_params['base_road_factor'] + depth_road_weight * depth_norm
        self.response_params['road_factor'] = np.clip(updated_road_factor, 0.75, 1.5)

        params = self.response_params[self.response_params['season_category']]
        scaled_sample = self.features['response_time_raw'] * self.response_params['prec_factor'] * self.response_params['road_factor'] * self.response_params['popden_factor']

        self.features['response_time'] = scaled_sample
        self.features["response_time_norm"] = ((scaled_sample - params['data_min']) / (params['data_max'] - params['data_min'])).clip(0, 1) # longer delays = more risk

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

    def init_damage_cost(self):
        depths = np.array([float(c) for c in self.dfs['depth_damage'].columns[1:]])
        land_use = ["transport", "residential", "commercial", "industrial", "agriculture", "infrastructure"]
        for lu in land_use:
            rows = self.dfs['depth_damage'][self.dfs['depth_damage']['type'] == lu]

            damage = rows.iloc[:, 1:].astype(float).mean(axis=0).values
            
            # Fit exponential model
            params, _ = curve_fit(exp_damage, depths, damage, bounds=(0, np.inf))
            self.depth_damage_k[lu] = params[0]
        
    def sample_damage_cost(self):
        transport = self.features["transport"]
        resident = self.features["resident"]
        commercial = self.features["commercial"]
        industrial = self.features["indust"]
        agriculture = self.features["agri"]
        infrastructure = self.features["infra"]

        # Weighted sum of damage fractions by land use proportion
        weight_sum_damage = (
            transport * exp_damage(self.features['depth'], self.depth_damage_k["transport"])  +
            resident * exp_damage(self.features['depth'], self.depth_damage_k["residential"]) +
            commercial * exp_damage(self.features['depth'], self.depth_damage_k["commercial"]) +
            industrial * exp_damage(self.features['depth'], self.depth_damage_k["industrial"]) +
            agriculture * exp_damage(self.features['depth'], self.depth_damage_k["agriculture"]) +
            infrastructure * exp_damage(self.features['depth'], self.depth_damage_k["infrastructure"])
        )

        total_weight = transport + resident + commercial + industrial + agriculture + infrastructure
        if total_weight > 1: # Norm if damage fraction exceeding 1 (assumes grid cell not always covered by landuse types)
            weight_sum_damage /= total_weight

        self.features['cost_fraction'] = np.clip(weight_sum_damage + np.random.normal(0, 0.02), 0, 1)

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

        if self.use_historic is not None:
            self.features['precipitation'] = max(0, self.historic_row['RAINFALL'].item() + np.random.normal(0, 0.02))
        else: 
            zero_sample = np.random.binomial(n=1, p=params['pct_zero'], size=1)[0]
            self.features['precipitation'] = 0 if zero_sample == 1 else gamma.rvs(a=params['shape'], loc=params['loc'], scale=params['scale'], size=1)[0]
        # Precipitation scale factor for response time
        log_prec = np.log1p(self.features['precipitation'])
        log_max = np.log1p(params['non_zero_max'])

        scaled = log_prec / log_max # 0–1
        self.response_params['prec_factor'] = 0.95 + 0.25 * scaled # 0.95–1.2 range

        self.features['precipitation_norm'] = np.clip(log_prec / log_max, 0, 1)

    def update_precipitation(self):
        params = self.prec_params[self.features["season"]]
        
        # Sample delta from normal distribution of scale 2mm
        delta = np.random.normal(loc=0, scale=2.0)
        new_prec = self.features['precipitation'] + delta
        new_prec = np.clip(new_prec, 0, params['non_zero_max'])
        
        self.features['precipitation'] = new_prec

        log_prec = np.log1p(self.features['precipitation'])
        log_max = np.log1p(params['non_zero_max'])

        scaled = log_prec / log_max
        self.response_params['prec_factor'] = 0.95 + 0.25 * scaled
        self.features['precipitation_norm'] = np.clip(scaled, 0, 1)

    def init_soil_moisture(self):
        # Load soil moisture rasters
        tiff_files = sorted(glob.glob(os.path.join(FILE_PATHS_DIR['soil_moisture_dir'], "*.tif")))

        #all_values = []
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
            #all_values.append(np.nanmean(params['stack']))
            self.soil_params[season] = params

        #self.soil_params['global_mean'] = np.mean(all_values)

    def sample_soil_moisture(self):
        if self.use_historic is not None and pd.notna(self.historic_row["SOIL_MOISTURE_PATH"].item()):
            with rasterio.open(self.historic_row["SOIL_MOISTURE_PATH"].item()) as src:
                data = src.read(1)
                nodata = src.nodata
                while True:
                    row = np.random.randint(0, data.shape[0])
                    col = np.random.randint(0, data.shape[1])
                    value = data[row, col]
                    if value != nodata:
                        break
            soil_sample = np.clip(data[row, col] + np.random.normal(0, 0.02), 0, 1)
        else:
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
            soil_sample = norm.rvs(mu, sigma)
            
        self.features['soil_moisture'] = soil_sample
        #self.features['soil_moisture_ratio'] = soil_sample / self.soil_params['global_mean']

    def update_soil_moisture(self):
        prec_influence = self.features['precipitation_norm']
        # Normalise flood depth using max possible depth (3.0m)
        depth_influence = self.features['depth'] / 3.0

        # Combined push
        prec_weight = 0.02
        depth_weight = 0.05 
        push = prec_weight * prec_influence + depth_weight * depth_influence

        # Constant natural drainage between time steps
        drain = 0.01

        delta = push - drain
        new_soil = np.clip(self.features['soil_moisture'] + delta, 0, 1)

        self.features['soil_moisture'] = new_soil
        #self.features['soil_moisture_ratio'] = new_soil / self.soil_params['global_mean']

    def init_english_proficiency(self):
        self.dfs["english_proficiency"]['Proficiency_Group'] = self.dfs["english_proficiency"]['Proficiency in English language (4 categories)'].apply(map_proficiency)

    def sample_english_proficiency(self):
        self.features["english_proficiency"] = sample_beta(self.dfs["english_proficiency"], 'Proficiency_Group', 'Good English Proficiency', 'Bad English Proficiency', 'Observation')

    def sample_vehicle(self):
        self.features["vehicle_rate"] = sample_beta(self.dfs["vehicle"], "Car or van availability (3 categories)", "1 or more cars or vans in household", "No cars or vans in household", "Observation")
        
    def sample_holiday(self):
        self.features["holiday"] = int(random.random() < (28 / 365))

    def sample_season(self):
        if self.use_historic is not None:
            self.features["season"] = self.historic_row['SEASON'].item()
        else:
            self.features["season"] = random.choice(list(SEASON_MONTHS))
    
    def sample_household(self, n=100): 
        household_features, observed = self.household.sample_household_features(n)
        self.features |= household_features
        return observed
    
    def sample_health(self, observed):
        self.features['disabled'] = sample_beta_observed(self.dfs["disabled"], 'Disability (3 categories)', 'Disabled under the Equality Act', 'Not disabled under the Equality Act', observed['disable_a'], observed['disable_b'], 'Observation')
        self.features['general_health'] = sample_beta_observed(self.dfs["general_health"], 'General health (3 categories)', 'Good health', 'Not good health', observed['health_a'], observed['health_b'], 'Observation')

    def update(self):
        self.update_precipitation()
        self.update_soil_moisture()
        self.update_flood_depth()
        self.update_emergency_response()

    def sample_features(self):  
        if self.use_historic != None:
            self.historic_row = self.historic_data.sample(1).copy()

        self.sample_season()

        self.sample_holiday()

        self.sample_building_age()

        self.sample_precipitation()

        self.sample_mean_property()

        self.sample_ambulance_handover()

        self.sample_hospital_bed()

        self.sample_english_proficiency()

        self.sample_vehicle()

        row = self.sample_cell()

        # Depends on cell
        observed = self.sample_household() 
        self.sample_soil_moisture()
        self.sample_flood_depth(row)

        # Depends on household
        self.sample_health(observed)
        self.sample_age(observed)
        
        # Depends on precipitation, population density, cell
        self.sample_emergency_response()

        # Depends on flood depth # ADD DEPENDENCY FOR OTHERS
        self.sample_damage_cost()

