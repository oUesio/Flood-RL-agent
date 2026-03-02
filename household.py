import itertools
import pandas as pd
import numpy as np
from scipy.stats import skewnorm

FILE_PATHS = {
    "disability": "data/disabled.csv",
    "general_health": "data/general_health.csv",
    "age": "data/age.csv",
    "qualification": "data/qualification-age.csv",
    "accommodation": "data/accommodation_type.csv",
    "household_size": "data/household_size.csv",
    "economic_activity": "data/nssec_economic_age.csv",
    "nssec": "data/nssec_economic_age.csv",
    "mean_income": "data/mean_income.csv",
    "household_employed": "data/household_employed_size.csv",
    "household_disabled": "data/household_disabled_size.csv",
    "household_longterm": "data/household_long-term_size.csv",
    "deprived_education": "data/deprived_education+deps.csv",
    "deprived_employment": "data/deprived_employment+deps.csv",
    "deprived_health": "data/deprived_health+deps.csv",
    "people_per_room": "data/people_per_room_hsize.csv",
    "occupancy": "data/occupancy_rating_nopeopleper.csv",
    "deprived_housing": "data/deprived_housing+deps.csv",
    "tenure": "data/tenure.csv"
}

TENURE_RISK = {
    "Owned: Owns outright": 0.0,
    "Owned: Owns with a mortgage or loan or shared ownership": 0.1,
    "Private rented: Private landlord or letting agency": 0.6,
    "Private rented: Other private rented or lives rent free": 0.6,
    "Social rented: Rents from council or Local Authority": 0.8,
    "Social rented: Other social rented": 0.8,
}

ACCO_RISK = {
    "Whole house or bungalow: Detached": 0.1,
    "Whole house or bungalow: Semi-detached": 0.1,
    "Whole house or bungalow: Terraced": 0.3,
    "Flat, maisonette or apartment": 0.5,
    "A caravan or other mobile or temporary structure": 0.8,
}

SIZE_RISK = {
    "1 person in household": 0.5,
    "2 people in household": 0.2,
    "3 people in household": 0.3,
    "4 or more people in household": 0.6,
}

INTERNET_RISK = {1: 0.0, 0: 0.6}

DEPRIVATION_RISK = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}

LOW_INCOME_RISK = {0: 0.0, 1: 0.5}

HOME_INSURE_RISK = {0: 0.4, 1: 0.0}

HEALTH_INSURE_RISK = {0: 0.2, 1: 0.0}

WEIGHTS = {
    "tenure_risk": 0.12,
    "acco_risk": 0.08,
    "size_risk": 0.08,
    "internet_risk": 0.08,
    "deprivation_risk": 0.36,
    "low_income_risk": 0.16,
    "home_insure_risk": 0.08,
    "health_insure_risk": 0.04,
}

NSSEC_MULTIPLIERS = {
    "L1, L2 and L3: Higher managerial, administrative and professional occupations": 1.90,
    "L4, L5 and L6: Lower managerial, administrative and professional occupations": 1.35,
    "L7: Intermediate occupations": 1.00,
    "L8 and L9: Small employers and own account workers": 1.10,
    "L10 and L11: Lower supervisory and technical occupations": 0.90,
    "L12: Semi-routine occupations": 0.75,
    "L13: Routine occupations": 0.65,
    "L14.1 and L14.2: Never worked and long-term unemployed": 0.40,
    "L15: Full-time students": 0.35,
    "Does not apply": 0.00
}


class Household:
    def __init__(self):
        # Resampling
        self.categorical_census_probs = {}
        self.income_params = {}
        self.weight_sum = sum(WEIGHTS.values())

        self.dfs = {key: pd.read_csv(path) for key, path in FILE_PATHS.items()}

        self.init_categories()
        self.init_income()


    def calculate_household_risk(self, tenure_sample, acco_sample, size_sample, internet_sample, deprivation_sample, low_income_sample, home_insure_sample, health_insure_sample):
        risk_score = (
            WEIGHTS["tenure_risk"] * TENURE_RISK[tenure_sample] +
            WEIGHTS["acco_risk"] * ACCO_RISK[acco_sample] +
            WEIGHTS["size_risk"] * SIZE_RISK[size_sample] +
            WEIGHTS["internet_risk"] * INTERNET_RISK[internet_sample] +
            WEIGHTS["deprivation_risk"] * DEPRIVATION_RISK[deprivation_sample] +
            WEIGHTS["low_income_risk"] * LOW_INCOME_RISK[low_income_sample] +
            WEIGHTS["home_insure_risk"] * HOME_INSURE_RISK[home_insure_sample] +
            WEIGHTS["health_insure_risk"] * HEALTH_INSURE_RISK[health_insure_sample]
        )
        risk_score = risk_score / self.weight_sum
        return np.clip(risk_score + np.random.normal(0, 0.01), 0, 1)
    
    def init_categorical_census(self, category, category_col, value_col="Observation", filters=None, ignore=None):
        filters = filters or []
        ignore = ignore or []
        df = self.dfs[category]
        if not filters: 
            totals = df.groupby(category_col)[value_col].sum()
            totals = totals[~totals.index.isin(ignore)] # remove ignore only for the category_col
            self.categorical_census_probs[category] = totals / totals.sum()
        else:
            probs = {}
            filter_values = [[v for v in df[col].unique()] for col in filters]
            for combo in itertools.product(*filter_values):
                key = tuple(combo)
                filtered = df
                for col, val in zip(filters, combo):
                    filtered = filtered[filtered[col] == val]
                totals = filtered.groupby(category_col)[value_col].sum()
                totals = totals[~totals.index.isin(ignore)]
                probs[key] = totals / totals.sum()
            self.categorical_census_probs[category] = probs

    def init_categories(self):
        self.init_categorical_census("age", 'Age (6 categories)')
        self.init_categorical_census("qualification", 'Highest level of qualification (7 categories)', filters=['Age (6 categories)'])
        self.init_categorical_census("accommodation", 'Accommodation type (5 categories)')
        self.init_categorical_census("household_size", 'Household size (5 categories)', ignore=['0 people in household'])
        self.init_categorical_census("economic_activity", 'Economic activity status (4 categories)', filters=['Age (6 categories)'])
        self.init_categorical_census("nssec", 'National Statistics Socio-economic Classification (NS-SeC) (10 categories)', filters=['Age (6 categories)', 'Economic activity status (4 categories)'])
        self.init_categorical_census("household_employed", 'Number of adults in employment in household (5 categories)', filters=['Household size (5 categories)'], ignore=['Does not apply'])
        self.init_categorical_census("household_disabled", 'Number of disabled people in household (4 categories)', filters=['Household size (5 categories)'], ignore=['Does not apply'])
        self.init_categorical_census("household_longterm", 'Number of people in household with a long-term heath condition but are not disabled (4 categories)', filters=['Household size (5 categories)'], ignore=['Does not apply'])
        self.init_categorical_census("deprived_education", 'Household deprived in the education dimension (3 categories)', filters=['Highest level of qualification (7 categories)'], ignore=['Does not apply'])
        self.init_categorical_census("deprived_employment", 'Household deprived in the employment dimension (3 categories)', filters=['Number of adults in employment in household (5 categories)', 'National Statistics Socio-economic Classification (NS-SeC) (10 categories)'])
        self.init_categorical_census("deprived_health", 'Household deprived in the health and disability dimension (3 categories)', filters=['Number of people in household with a long-term heath condition but are not disabled (4 categories)', 'Number of disabled people in household (4 categories)'])
        self.init_categorical_census("people_per_room", 'Number of people per room in household (5 categories)', filters=['Household size (5 categories)'], ignore=['Does not apply'])
        self.init_categorical_census("occupancy", 'Occupancy rating for rooms (5 categories)', filters=['Number of people per room in household (5 categories)'], ignore=['Does not apply'])
        self.init_categorical_census("deprived_housing", 'Household deprived in the housing dimension (3 categories)', filters=['Number of people per room in household (5 categories)','Occupancy rating for rooms (5 categories)'], ignore=['Does not apply'])
        self.init_categorical_census("tenure", 'Tenure of household (7 categories)', ignore=['Does not apply'])

    def sample_categorical_census(self, category, filters=None):
        filters = filters or []
        if filters:
            lookup_key =  tuple(filters)
            probs = self.categorical_census_probs[category][lookup_key]
        else:
            probs = self.categorical_census_probs[category]

        return np.random.choice(probs.index.to_numpy(), size=1, p=probs.to_numpy())[0]
    
    def init_income(self):
        params = {}
        income_df = self.dfs["mean_income"]
        income_df['Total annual income (£)'] = (
            income_df['Total annual income (£)']
            .str.strip()
            .str.replace(',', '')
            .astype(float)
        )
        log_income = np.log(income_df['Total annual income (£)'])
        params['shape'], params['loc'], params['scale'] = skewnorm.fit(log_income)
        params['log_min'] = log_income.min()
        params['log_max'] = log_income.max()
        params['median'] = income_df['Total annual income (£)'].median()
        self.income_params = params

    def sample_income(self, nssec_sample):
        income_sample = np.exp(skewnorm.rvs(self.income_params['shape'], loc=self.income_params['loc'], scale=self.income_params['scale'], size=1))[0]
        log_sample = np.log(income_sample)
        income_norm = (log_sample - self.income_params['log_min']) / (self.income_params['log_max'] - self.income_params['log_min'])

        income_sample *= NSSEC_MULTIPLIERS[nssec_sample]
        low_income_sample = int(income_sample < 0.6 * self.income_params['median'])

        return income_norm, low_income_sample

    def sample_household_dep(self, dep_edu_sample, dep_employ_sample, dep_health_sample, dep_housing_sample):
        household_dep_sample = sum([
            dep_edu_sample == 'Household is deprived in the education dimension',
            dep_employ_sample == 'Household is deprived in the employment dimension',
            dep_health_sample == 'Household is deprived in the health and disability dimension',
            dep_housing_sample == 'Household is deprived in the housing dimension'
        ])
        return household_dep_sample

    def sample_internet(self, age_sample, house_size_sample):
        internet_prob = 0.85 if age_sample == 'Aged 65 years and over' and house_size_sample == '1 person in household' else 0.98
        return np.random.binomial(1, internet_prob)

    def sample_insurances(self):
        home_insure_rate = np.random.beta(30, 10)  # mean ~ 0.75
        home_insure_sample = np.random.binomial(1, home_insure_rate)
        health_insure_sample = np.random.binomial(1, 0.14)
        return home_insure_rate, home_insure_sample, health_insure_sample

    def sample_household_features(self, num_households):
        total_risk = 0
        total_home_insure = 0
        total_income_norm = 0
        total_deprived = 0
        total_low = 0
        observed = {
            "disable_a": 0, "disable_b": 0,
            "elderly_a": 0, "elderly_b": 0,
            "child_a": 0, "child_b": 0,
            "health_a": 0, "health_b": 0,
        }

        for _ in range(num_households):
            age_sample = self.sample_categorical_census("age")
            observed['elderly_a' if age_sample == 'Aged 65 years and over' else 'elderly_b'] += 1
            observed['child_a' if age_sample == 'Aged 15 years and under' else 'child_b'] += 1

            qual_sample = self.sample_categorical_census("qualification", [age_sample])
 
            acco_type_sample = self.sample_categorical_census("accommodation")

            house_size_sample = self.sample_categorical_census("household_size")

            eas_sample = self.sample_categorical_census("economic_activity", [age_sample])

            nssec_sample = self.sample_categorical_census("nssec", [age_sample, eas_sample])

            income_norm, low_income_sample = self.sample_income(nssec_sample)
            total_income_norm += income_norm
            total_low += low_income_sample

            num_adults_sample = self.sample_categorical_census("household_employed", [house_size_sample])

            num_disable_sample = self.sample_categorical_census("household_disabled", [house_size_sample])
            if '1' in num_disable_sample:
                observed['disable_a'] += 1
            elif '2' in num_disable_sample:
                observed['disable_a'] += 2
            elif 'No people' in num_disable_sample:
                observed['disable_b'] += 1

            num_long_sample = self.sample_categorical_census("household_longterm", [house_size_sample])

            dep_edu_sample = self.sample_categorical_census("deprived_education", [qual_sample])

            dep_employ_sample = self.sample_categorical_census("deprived_employment", [num_adults_sample, nssec_sample])

            dep_health_sample = self.sample_categorical_census("deprived_health", [num_long_sample, num_disable_sample])
            if 'is not deprived' in dep_health_sample:
                observed['health_a'] += 1
            elif 'is deprived' in dep_health_sample:
                observed['health_b'] += 1

            num_people_sample = self.sample_categorical_census("people_per_room", [house_size_sample])

            num_occupancy_sample = self.sample_categorical_census("occupancy", [num_people_sample])

            dep_housing_sample = self.sample_categorical_census("deprived_housing", [num_people_sample, num_occupancy_sample])

            household_dep_sample = self.sample_household_dep(dep_edu_sample, dep_employ_sample, dep_health_sample, dep_housing_sample)
            total_deprived += household_dep_sample

            tenure_sample = self.sample_categorical_census("tenure")

            internet_sample = self.sample_internet(age_sample, house_size_sample)

            home_insure_rate, home_insure_sample, health_insure_sample = self.sample_insurances()
            total_home_insure += home_insure_rate

            total_risk += self.calculate_household_risk(tenure_sample, acco_type_sample, house_size_sample, internet_sample, household_dep_sample, low_income_sample, home_insure_sample, health_insure_sample)

        features = {}
        features["household"] = total_risk / num_households
        features['home_insure_rate'] = total_home_insure / num_households
        features['income_norm'] = total_income_norm / num_households
        features['deprived'] = total_deprived / num_households
        features['low_income'] = total_low / num_households
        return features, observed
