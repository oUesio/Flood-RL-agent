
import numpy as np
from scipy.stats import gamma
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
    
    # Update features for next time step
    def update_hydro(self):
        '''# Soil moisture, updates based on precipitation, flood depth and impervious surface area
        loss_frac = 0.002 # fractional soil moisture loss per timestep
        S_max = 80 # assume max water soil can absorb in mm
        depth = self.samples.features['depth']*1000 # depth to mm
        SM_mm = self.samples.features['soil_moisture'] * S_max
        water = (self.samples.features['precipitation'] + depth) * (1 - self.samples.features["impervious_frac"])
        loss = loss_frac * SM_mm # Water loss from evaporation
        self.samples.features['soil_moisture'] = min(max(SM_mm + water - loss, 0), S_max) / S_max'''

        '''# self.samples.features['river_flow']'''

        # River level
        '''
        - update river level using precipitation and flood depth
        '''
         
        pass

    def update_emergency(self):
        # response time, bed occupancy, handover time
        # self.samples.features["response_time_norm"]

        # Bed occupancy
        bed_sample = np.random.beta(self.hospital_bed_stats[0], self.hospital_bed_stats[1], size=1)[0]
        alpha = 0.7
        self.samples.features['bed_occupancy'] = alpha * self.samples.features['bed_occupancy'] + (1 - alpha) * bed_sample

        # Handover time
        handover_sample = self.samples.features.sample_handover_time(self.handover_prob)
        alpha = 0.7
        self.samples.features['handover_time_norm'] = alpha * self.samples.features['handover_time_norm'] + (1 - alpha) * handover_sample

    def update_depth(self):
        # flood depth, depth-damage
        # self.samples.features['depth'] 
        '''
        based on river level and scaled for water density and watercourse distance
        '''
        # self.samples.features['damage_fraction']
        self.depth_damage()
        pass

    def update_warning(self):
        # used when rl agent makes choice to issue
        self.warning_issued = 1

    def update(self):
        # New precipitation
        self.update_prec()   

        self.update_hydro()
        self.update_depth()

        self.update_emergency()

        self.update_deprived()

    # Calculate derived features
    def physical_vul(self): # high physicial = high risk
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
            "elevation_ratio": self.samples.features["grid"]["elevation_ratio"],
            "impervious_ratio": self.samples.features["grid"]["impervious_ratio"],
            "water_dist_ratio": self.samples.features["grid"]["water_dist_ratio"],
            "water_dens_ratio": self.samples.features["grid"]["water_dens_ratio"],
            "road_dens_ratio": self.samples.features["grid"]["road_dens_ratio"],
            "road_dist_ratio": self.samples.features["grid"]["road_dist_ratio"],
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
            "hospital_norm": 1 - self.samples.features["grid"]["hospital_norm"],  # long distance = high risk
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
        historic_factor = 1.2 if self.samples.features["grid"]["historic"] else 1
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

import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(42)

N = 1  # number of simulations

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
    '''print(temp.features['precipitation'])

    results["exposure"].append(temp.derived["exposure"])
    results["physical"].append(temp.derived["physical"])
    results["socioeconomic"].append(temp.derived["socioeconomic"])
    results["depth"].append(temp.features["depth"])
    results["damage_fraction"].append(temp.features["damage_fraction"])
    results["preparedness_inv"].append(temp.derived["preparedness"])
    results["recovery_inv"].append(temp.derived["recovery"])
    results["impact"].append(temp.impact)'''

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