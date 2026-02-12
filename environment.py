
import numpy as np
from scipy.stats import gamma, lognorm
from features import Sampler

class Environment():
    def __init__(self):
        self.derived = {}
        self.warning_issued = False
        self.impact = None
        self.samples = Sampler()
        self.update_deprived()

    def update_prec(self):
        prec_dist = self.samples.features.dist_params['precipitation']
        zero_sample = np.random.binomial(n=1, p=prec_dist['pct_zero'], size=1)[0]
        sample = 0 if zero_sample == 1 else gamma.rvs(a=prec_dist['shape'], loc=prec_dist['loc'], scale=prec_dist['scale'], size=1)[0]
        alpha = 0.7
        self.samples.features['precipitation'] = alpha * self.samples.features['precipitation'] + (1 - alpha) * sample
    
    def update_hydro(self):
        # Soil moisture
        pervious = 1 - self.samples.features["impervious"]
        old_sm = self.samples.features["soil_moisture"] 
        sm_max = 80
        sm_mm = old_sm * sm_max
        rain_mm = self.samples.features["precipitation"] * pervious
        flood_mm = min(self.samples.features["depth"] * 1000, 10) * pervious 
        infiltration_factor = 1 - old_sm
        drying_factor = 0.01
        sm_new = sm_mm + (rain_mm + flood_mm) * infiltration_factor - (drying_factor * sm_mm)
        self.samples.features["soil_moisture"] = np.clip((sm_new / sm_max), 0, 1)

        # River level
        '''river_dist = self.samples.features.dist_params['river']
        river_sample = gamma.rvs(a=river_dist['shape'], loc=river_dist['loc'], scale=river_dist['scale'], size=1)[0]
        depth_scale = 1.2
        depth_factor = 1 + (self.features["depth"] / depth_scale)
        print('RIVER LEVEL')
        print(depth_factor)
        prec_factor = np.exp(self.features["precipitation"] / 100)
        print(prec_factor)
        river_sample = river_sample * depth_factor * prec_factor

        alpha = 0.6 # higher = more stable river levels
        print(self.features['river_level'])
        self.features['river_level'] = (alpha * self.features['river_level'] + (1 - alpha) * river_sample)
        print(self.features['river_level'])'''
        old_river = self.samples.features["river_level"]  # m

        # Inputs
        prec_mm = self.samples.features["precipitation"] # mm
        soil_moisture = self.samples.features["soil_moisture"] # 0–1
        impervious = self.samples.features["impervious"]
        flood_depth = self.samples.features["depth"] # m

        pervious = 1 - impervious

        # Parameters
        runoff_coeff = 0.6
        soil_runoff_sensitivity = 1.5
        impervious_runoff_boost = 1.0
        flood_gain = 0.3
        recession_rate = 0.05
        dt_scale = 0.001

        # 1. Runoff from rainfall
        pervious_runoff = prec_mm * runoff_coeff * pervious * (soil_moisture ** soil_runoff_sensitivity)
        impervious_runoff = prec_mm * impervious * impervious_runoff_boost

        runoff_mm = pervious_runoff + impervious_runoff
        runoff_rise = runoff_mm * dt_scale

        # 2. Flood backwater
        flood_rise = flood_gain * flood_depth

        # 3. Recession
        recession = recession_rate * old_river

        # 4. Update
        river_new = old_river + runoff_rise + flood_rise - recession
        self.samples.features["river_level"] = max(river_new, 0)

        '''
        Uses soil moisture as a mediator,
        Separates inputs (rain, flood) from state (river),
        Includes memory and decay,
        Is stable and interpretable at sub-daily resolution.

        Impervious areas
            Generate near-instant runoff.
            Do not depend on soil moisture.
            Dominate response in urban catchments.
        Pervious areas
            Runoff increases as soil wets up.
            Already captured by your soil moisture state.
        '''


    def update_emergency(self):
        # response time, bed occupancy, handover time
        # self.samples.features["response_time_norm"]
        response_dist = self.samples.features.dist_params['response_time']
        response_sample = lognorm.rvs(response_dist['shape'], loc=response_dist['loc'], scale=response_dist['scale'])
        prec_factor = np.exp(self.features['precipitation'] / 50) 
        response_sample = (
            (response_sample * response_dist['popden_factor'] * prec_factor - response_dist['data_min']) / (response_dist['data_max'] - response_dist['data_min'])
        ).clip(0, 1) # longer delays = more risk
        alpha = 0.7
        self.features["response_time_norm"] = alpha * self.samples.features['response_time_norm'] + (1 - alpha) * response_sample

        # Bed occupancy
        bed_sample = np.random.beta(self.hospital_bed_stats[0], self.hospital_bed_stats[1], size=1)[0]
        alpha = 0.8
        self.samples.features['bed_occupancy'] = alpha * self.samples.features['bed_occupancy'] + (1 - alpha) * bed_sample

        # Handover time
        handover_sample = self.samples.features.sample_handover_time(self.handover_prob)
        alpha = 0.7
        self.samples.features['handover_time_norm'] = alpha * self.samples.features['handover_time_norm'] + (1 - alpha) * handover_sample

    def update_depth(self):
        # flood depth, depth-damage
        # self.samples.features['depth'] 
        '''
        precpitation, river level and scaled? using watercourse distance
        '''
        # self.samples.features['damage_fraction']
        self.depth_damage()
        pass

    def update_warning(self):
        # used when rl agent makes choice to issue
        self.warning_issued = 1

    # Calculate derived features
    def physical_vul(self): # high physicial = high risk
        weights = {
            "elevation_ratio": 0.25,
            "impervious_ratio": 0.10,
            "water_dens_ratio": 0.15,
            "road_dens_ratio": 0.10,
            "building_age_ratio": 0.10,
        }

        features = {
            "elevation_ratio": np.log10(self.samples.features["elevation_ratio"]),
            "impervious_ratio": self.samples.features["impervious_ratio"],
            "water_dens_ratio": self.samples.features["water_dens_ratio"],
            "road_dens_ratio": self.samples.features["road_dens_ratio"],
            "building_age_ratio": self.samples.features["building_age_ratio"],
        }
        #print(features)

        total_weight = sum(weights.values())

        self.derived["physical"] = (
            sum(weights[k] * features[k] for k in weights) / total_weight
        )
        #print(self.derived["physical"])

    def socioeconomic_vul(self): # high socioeconomic = high risk
        weights = {
            "elderly": 0.15,
            "disabled": 0.15,
            "children": 0.15,
            "general_health": 0.15,
            "english_proficiency": 0.10,
            "household": 0.30,
            "property_value_norm": 0.10,
        }

        features = {
            "elderly": self.samples.features["elderly"], # high = high risk
            "disabled": self.samples.features["disabled"], # high = high risk
            "children": self.samples.features["children"], # high = high risk
            "general_health": 1 - self.samples.features["general_health"], # high = low risk (good health) (invert)
            "english_proficiency": 1 - self.samples.features["english_proficiency"], # high = low risk (proficient at english) (invert)
            "household": self.samples.features["household"], # high = high risk
            "property_value_norm": 1 - self.samples.features["property_value_norm"], # low = high risk (CDF) (invert)
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
            "hospital_norm": 0.10,
        }

        features = {
            "response_time_norm": 1 - self.samples.features["response_time_norm"], # long response time = high risk
            "handover_time_norm": 1 - self.samples.features["handover_time_norm"], # long handover time = high risk
            "bed_occupancy": self.samples.features["bed_occupancy"], # low bed occupancy = high risk (free beds)
            "vehicle": self.samples.features["vehicle_rate"], # low vehicle rate = high risk
            "hospital_norm": 1 - self.samples.features["hospital_norm"],  # long distance = high risk
        }

        total_weight = sum(weights.values())
        warning_factor = 1.4 if self.warning_issued else 1

        self.derived["preparedness"] = np.clip((
            (sum(weights[k] * features[k] for k in weights) / total_weight) * warning_factor
        ), 0, 1)

    def recovery(self): # high recovery = lower risk
        weights = {
            "income_norm": 0.40,
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
            "population_density_ratio": 0.30,
        }

        features = {
            "population_density_ratio": self.samples.features["population_density_ratio"],
        }

        total_weight = sum(weights.values())
        historic_factor = 1.2 if self.samples.features["historic"] else 1
        holiday_factor = 1.1 if self.samples.features["holiday"] else 1

        self.derived["exposure"] = np.clip((
            (sum(weights[k] * features[k] for k in weights) / total_weight) * historic_factor * holiday_factor
        ), 0, 1)


    def impact_score(self):
        physical_impact = ( # no exposure or no damage = no physical impact
            self.derived["exposure"] # more exposure = high risk
            * self.derived["physical"] # high physical = high risk
            * self.samples.features["damage_fraction"] # high damage = high risk
        )

        socio_impact = ( # reduce socio vulnerability if preparedness and recovery are high
            self.derived["socioeconomic"] # high socio = high risk
            * (1 - self.derived["preparedness"]) # high preparedness = low risk
            * (1 - self.derived["recovery"]) # high recovery = low risk
        )

        self.impact = physical_impact + socio_impact
    
    def update_deprived(self):
        self.physical_vul()
        self.socioeconomic_vul()
        self.preparedness()
        self.recovery()
        self.exposure()
        self.impact_score()

    def update(self):
        # New precipitation
        self.update_prec()   

        self.update_hydro()
        self.update_depth()

        self.update_emergency()

        self.update_deprived()
        
'''np.random.seed(42)
start = time.time()
temp = Environment()
temp.init_variable_samples()
print('Runtime full: ', time.time() - start)
print(temp.features)
print(temp.household)
print(temp.derived)
print(temp.impact)
# self.derived["exposure"], self.derived["physical"], self.derived["socioeconomic"], self.samples.features["depth"], self.samples.features["damage_fraction"], 1 - self.derived["preparedness"], "recovery": 1 - self.derived["recovery"], self.impact'''

'''import numpy as np
import matplotlib.pyplot as plt'''

#np.random.seed(42)

'''N = 4  # number of simulations

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
    print(temp.samples.features)
    print(temp.derived)
    print(temp.impact)
    #temp.update_hydro()'''
