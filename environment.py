
import numpy as np
from features import Sampler

class Environment():
    def __init__(self, use_historic=None):
        self.derived = {}
        self.impact = None
        self.samples = Sampler(use_historic) # None, "yellow", "amber", "red"
        self.warning_issued = 0 # 0 (No), 1 (yellow), 2 (amber), 3 (red)

    # Calculate derived features
    def physical_vul(self): # high physical = high risk
        weights = {
            "elevation_ratio": 0.24,
            "impervious_ratio": 0.08,
            "water_dist_ratio": 0.06,
            "water_dens_ratio": 0.12,
            "transport_ratio": 0.06, # less than impervious since easier access
            "building_age_ratio": 0.10,
            "buildings_ratio": 0.10,
            #"soil_moisture_ratio": 0.07, # less than impervious since can still absorb water
        }

        features = {
            "elevation_ratio": np.log10(self.samples.features["elevation_ratio"]),
            "impervious_ratio": self.samples.features["impervious_ratio"],
            "water_dist_ratio": self.samples.features["water_dist_ratio"],
            "water_dens_ratio": self.samples.features["water_dens_ratio"],
            "transport_ratio": self.samples.features["transport_ratio"],
            "building_age_ratio": self.samples.features["building_age_ratio"],
            "buildings_ratio": self.samples.features["buildings_ratio"],
            #"soil_moisture_ratio": self.samples.features["soil_moisture_ratio"],
        }

        total_weight = sum(weights.values())

        self.derived["physical"] = (
            sum(weights[k] * features[k] for k in weights) / total_weight
        )

    def socioeconomic_vul(self): # high socioeconomic = high risk
        weights = {
            "elderly": 0.15,
            "disabled": 0.15,
            "children": 0.15,
            "general_health": 0.15,
            "english_proficiency": 0.10,
            "household": 0.20,
            "property_value_norm": 0.10,
            "deprived": 0.25,
            "education": 0.08,
        }

        features = {
            "elderly": self.samples.features["elderly"], # high = high risk
            "disabled": self.samples.features["disabled"], # high = high risk
            "children": self.samples.features["children"], # high = high risk
            "general_health": 1 - self.samples.features["general_health"], # high = low risk (good health) (invert)
            "english_proficiency": 1 - self.samples.features["english_proficiency"], # high = low risk (proficient at english) (invert)
            "household": self.samples.features["household"], # high = high risk
            "property_value_norm": 1 - self.samples.features["property_value_norm"], # low = high risk (CDF) (invert)
            "deprived": 1 - self.samples.features["deprived"], # low deprivation index rank = high risk
            "education": 1 - self.samples.features["education"], # far from education = slight possibility less educated or more deprived = high risk
        }

        total_weight = sum(weights.values())

        self.derived["socioeconomic"] = (
            sum(weights[k] * features[k] for k in weights) / total_weight
        )

    def preparedness(self): # high preparedness = lower risk
        weights = {
            "response_time_norm": 0.20,
            "handover_time_norm": 0.10,
            "bed_occupancy": 0.10,
            "vehicle": 0.15, 
            "hospital": 0.05,
            "emergency": 0.05,
        }

        features = {
            "response_time_norm": 1 - self.samples.features["response_time_norm"], # long response time = high risk
            "handover_time_norm": 1 - self.samples.features["handover_time_norm"], # long handover time = high risk
            "bed_occupancy": 1 - self.samples.features["bed_occupancy"], # high bed occupancy = high risk (free beds)
            "vehicle": self.samples.features["vehicle_rate"], # low vehicle rate = high risk
            "hospital": 1 - self.samples.features["hospital"],  # long distance = high risk
            "emergency": 1 - self.samples.features["emergency"],  # long distance = high risk
        }

        total_weight = sum(weights.values())
        warning_factor = 1 + 0.1 * self.warning_issued # higher warning issued, more prepared

        self.derived["preparedness"] = np.clip((
            (sum(weights[k] * features[k] for k in weights) / total_weight) * warning_factor
        ), 0, 1)

    def recovery(self): # high recovery = lower risk
        weights = {
            "income_norm": 0.30,
            "home_insure_rate": 0.20,
        }

        features = {
            "income_norm": self.samples.features["income_norm"], # low income = high risk
            "home_insure_rate": self.samples.features["home_insure_rate"], # less home insurance = high risk
        }

        total_weight = sum(weights.values())

        self.derived["recovery"] = (
            sum(weights[k] * features[k] for k in weights) / total_weight
        )

    def exposure(self):
        weights = {
            "popden_ratio": 0.30,
        }

        features = {
            "popden_ratio": self.samples.features["popden_ratio"],
        }

        total_weight = sum(weights.values())
        historic_factor = 1.2 if self.samples.features["historic"] else 1
        holiday_factor = 1.1 if self.samples.features["holiday"] else 1

        self.derived["exposure"] = np.clip((
            (sum(weights[k] * features[k] for k in weights) / total_weight) * historic_factor * holiday_factor
        ), 0, 1)

    def impact_score(self):
        physical_impact = ( # no exposure or no damage = no physical impact
            self.derived["exposure"] # more exposure = high impact
            * self.derived["physical"] # high physical = high impact
            * self.samples.features["cost_fraction"] # high cost = high impact
            * (1 + self.samples.features["precipitation_norm"]) # high precipitation = high impact
        )

        socio_impact = ( # reduce socio vulnerability if preparedness and recovery are high
            self.derived["socioeconomic"] # high socio = high impact
            * (1 - self.derived["preparedness"]) # high preparedness = low impact
            * (1 - self.derived["recovery"]) # high recovery = low impact
        )

        self.impact = self.impact = physical_impact + socio_impact

    def sample_features(self):
        self.samples.sample_features()

    def update_warning(self, warning):
        # Used when rl agent makes choice to issue
        self.warning_issued = warning

    def update_features(self):
        self.samples.update()
        self.update_derived()

    def update_derived(self):
        self.physical_vul()
        self.socioeconomic_vul()
        self.preparedness()
        self.recovery()
        self.exposure()
        self.impact_score()

    def get_observable_features(self): 
        return {
            # Measurements
            "precipitation": self.samples.features['precipitation'], # millimetres
            "flood_depth": self.samples.features['depth'], # metres
            "season": self.samples.features['season'], # Spring, Summer, Autumn, Winter
            "holiday": self.samples.features["holiday"], # Holiday flag
            "soil_moisture": self.samples.features["soil_moisture"], # fraction
            # From data
            "water_density": self.samples.features["water_dens"], # fraction
            "water_distance": self.samples.features["water_dist"], # metres (crs 27700)
            "elevation": self.samples.features["elevation"], # metres
            "impervious_surface": self.samples.features["impervious"], # fraction
            "historical_flood_flag": self.samples.features["historic"], # boolean
            "deprivation_index": self.samples.features["deprived"], # normalised (max=10, min=1)
            "residential": self.samples.features["resident"], # fraction
            "commercial": self.samples.features["commercial"], # fraction
            "industrial": self.samples.features["indust"], # fraction
            "agriculture": self.samples.features["agri"], # fraction
            "transport": self.samples.features["transport"], # fraction
            # infrastructure excluded since it only represents electricity transmission lines
            "population_density": self.samples.features["popden"], # of a 1km x 1km grid cell
        }
