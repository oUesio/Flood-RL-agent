import pandas as pd
import numpy as np
from scipy.stats import skewnorm

# ======================
# File paths
# ======================
FILE_PATHS = {
    "disability": "data/disabled.csv",
    "general_health": "data/general_health.csv",
    "age": "data/age.csv",
    "qualification": "data/qualification-age.csv",
    "accommodation": "data/accomodation_type.csv",
    "household_size": "data/household_size.csv",
    "economic_activity": "data/nssec_economic_age.csv",
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

# ======================
# Risk maps & weights
# ======================
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

HEALTH_INSURE_RISK = {0: 0.4, 1: 0.0}

WEIGHTS = {
    "tenure_risk": 0.12,
    "acco_risk": 0.08,
    "size_risk": 0.08,
    "internet_risk": 0.08,
    "deprivation_risk": 0.30,
    "low_income_risk": 0.18,
    "home_insure_risk": 0.08,
    "health_insure_risk": 0.08,
}

# ======================
# Helper functions
# ======================
def sample_categorical_census(df: pd.DataFrame, category_col: str, value_col: str, ignore_categories: list):
    """Sample from a categorical distribution."""
    if ignore_categories:
        df = df[~df[category_col].isin(ignore_categories)]

    totals = df.groupby(category_col)[value_col].sum()
    probs = totals / totals.sum()
    samples = np.random.choice(
        probs.index.to_numpy(),
        size=1,
        p=probs.to_numpy()
    )

    return samples[0]

def sample_from_csv(
    filepath: str,
    category_col: str,
    value_col: str = "Observation",
    filters: dict = None,
    ignore_categories: list = None
):
    """Filter CSV by optional filters and sample a category."""
    df = pd.read_csv(filepath)
    if filters:
        for col, val in filters.items():
            df = df[df[col] == val]
    return sample_categorical_census(df, category_col, value_col, ignore_categories or [])

def map_proficiency(category: str) -> str | None:
    if category in [
        "Main language is English (English or Welsh in Wales)",
        "Main language is not English (English or Welsh in Wales): Can speak English very well or well"
    ]:
        return "Good English Proficiency"
    elif category == "Main language is not English (English or Welsh in Wales): Cannot speak English or cannot speak English well":
        return "Bad English Proficiency"
    return None

def calculate_household_risk(
    tenure_sample, acco_sample, size_sample, internet_sample, deprivation_sample, 
    low_income_sample, home_insure_sample, health_insure_sample
) -> float:
    tenure_risk = TENURE_RISK[tenure_sample]
    acco_risk = ACCO_RISK[acco_sample]
    size_risk = SIZE_RISK[size_sample]
    internet_risk = INTERNET_RISK[internet_sample]
    deprivation_risk = DEPRIVATION_RISK[deprivation_sample]
    low_income_risk = LOW_INCOME_RISK[low_income_sample]
    home_insure_risk = HOME_INSURE_RISK[home_insure_sample]
    health_insure_risk = HEALTH_INSURE_RISK[health_insure_sample]

    risk_score = (
        WEIGHTS["tenure_risk"] * tenure_risk +
        WEIGHTS["acco_risk"] * acco_risk +
        WEIGHTS["size_risk"] * size_risk +
        WEIGHTS["internet_risk"] * internet_risk +
        WEIGHTS["deprivation_risk"] * deprivation_risk +
        WEIGHTS["low_income_risk"] * low_income_risk +
        WEIGHTS["home_insure_risk"] * home_insure_risk +
        WEIGHTS["health_insure_risk"] * health_insure_risk
    )

    noise = np.random.normal(0, 0.03)
    return np.clip(risk_score + noise, 0, 1)

# ======================
# Load all CSVs once
# ======================
dfs = {key: pd.read_csv(path) for key, path in FILE_PATHS.items()}

def generate_household_samples(num_households: int) -> list[float]:
    """
    Generate household samples and compute a household risk score for each.
    Returns a list of risk scores.
    """
    risk_scores = []

    for _ in range(num_households):
        # -------------------------
        # General household categorical variables
        # -------------------------
        age_sample = sample_from_csv(FILE_PATHS["age"], 'Age (6 categories)')

        qual_sample = sample_from_csv(FILE_PATHS["qualification"],
                                      'Highest level of qualification (7 categories)',
                                      filters={'Age (6 categories)': age_sample})

        acco_type_sample = sample_from_csv(FILE_PATHS["accommodation"], 'Accommodation type (5 categories)')

        house_size_sample = sample_from_csv(FILE_PATHS["household_size"],
                                            'Household size (5 categories)',
                                            ignore_categories=['0 people in household'])

        eas_sample = sample_from_csv(FILE_PATHS["economic_activity"],
                                     'Economic activity status (4 categories)',
                                     filters={'Age (6 categories)': age_sample})

        nssec_sample = sample_from_csv(FILE_PATHS["economic_activity"],
                                       'National Statistics Socio-economic Classification (NS-SeC) (10 categories)',
                                       filters={'Age (6 categories)': age_sample,
                                                'Economic activity status (4 categories)': eas_sample})

        # -------------------------
        # Income
        # -------------------------
        income_df = dfs["mean_income"].copy()
        income_df['Total annual income (£)'] = (
            income_df['Total annual income (£)']
            .str.strip()
            .str.replace(',', '')
            .astype(float)
        )
        log_income = np.log(income_df['Total annual income (£)'])
        shape, loc, scale = skewnorm.fit(log_income)
        income_sample = np.exp(skewnorm.rvs(shape, loc=loc, scale=scale, size=1))[0]
        NSSEC = {
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
        income_sample *= NSSEC[nssec_sample]

        median_income = income_df['Total annual income (£)'].median()
        low_income_sample = int(income_sample < 0.6 * median_income)

        # -------------------------
        # Household composition
        # -------------------------
        num_adults_sample = sample_from_csv(FILE_PATHS["household_employed"],
                                            'Number of adults in employment in household (5 categories)',
                                            filters={'Household size (5 categories)': house_size_sample},
                                            ignore_categories=['Does not apply'])

        num_disable_sample = sample_from_csv(FILE_PATHS["household_disabled"],
                                             'Number of disabled people in household (4 categories)',
                                             filters={'Household size (5 categories)': house_size_sample},
                                             ignore_categories=['Does not apply'])

        num_long_sample = sample_from_csv(FILE_PATHS["household_longterm"],
                                          'Number of people in household with a long-term heath condition but are not disabled (4 categories)',
                                          filters={'Household size (5 categories)': house_size_sample},
                                          ignore_categories=['Does not apply'])

        dep_edu_sample = sample_from_csv(FILE_PATHS["deprived_education"],
                                         'Household deprived in the education dimension (3 categories)',
                                         filters={'Highest level of qualification (7 categories)': qual_sample},
                                         ignore_categories=['Does not apply'])

        dep_employ_sample = sample_from_csv(FILE_PATHS["deprived_employment"],
                                            'Household deprived in the employment dimension (3 categories)',
                                            filters={'Number of adults in employment in household (5 categories)': num_adults_sample,
                                                     'National Statistics Socio-economic Classification (NS-SeC) (10 categories)': nssec_sample})

        dep_health_sample = sample_from_csv(FILE_PATHS["deprived_health"],
                                            'Household deprived in the health and disability dimension (3 categories)',
                                            filters={'Number of people in household with a long-term heath condition but are not disabled (4 categories)': num_long_sample,
                                                     'Number of disabled people in household (4 categories)': num_disable_sample})

        # -------------------------
        # Housing and occupancy
        # -------------------------
        num_people_sample = sample_from_csv(FILE_PATHS["people_per_room"],
                                           'Number of people per room in household (5 categories)',
                                           filters={'Household size (5 categories)': house_size_sample},
                                           ignore_categories=['Does not apply'])

        num_occupancy_sample = sample_from_csv(FILE_PATHS["occupancy"],
                                              'Occupancy rating for rooms (5 categories)',
                                              filters={'Number of people per room in household (5 categories)': num_people_sample},
                                              ignore_categories=['Does not apply'])

        dep_housing_sample = sample_from_csv(FILE_PATHS["deprived_housing"],
                                            'Household deprived in the housing dimension (3 categories)',
                                            filters={'Number of people per room in household (5 categories)': num_people_sample,
                                                     'Occupancy rating for rooms (5 categories)': num_occupancy_sample},
                                            ignore_categories=['Does not apply'])

        # -------------------------
        # Household deprivation
        # -------------------------
        household_dep_sample = sum([
            dep_edu_sample == 'Household is deprived in the education dimension',
            dep_employ_sample == 'Household is deprived in the employment dimension',
            dep_health_sample == 'Household is deprived in the health and disability dimension',
            dep_housing_sample == 'Household is deprived in the housing dimension'
        ])

        # -------------------------
        # Tenure & insurance
        # -------------------------
        tenure_sample = sample_from_csv(FILE_PATHS["tenure"],
                                       'Tenure of household (7 categories)',
                                       ignore_categories=['Does not apply'])

        internet_prob = 0.85 if age_sample == 'Aged 65 years and over' and house_size_sample == '1 person in household' else 0.98
        internet_sample = np.random.binomial(1, internet_prob)

        home_insure_sample = np.random.binomial(1, 0.75)
        health_insure_sample = np.random.binomial(1, 0.14)

        # -------------------------
        # Household risk score
        # -------------------------
        household_risk = calculate_household_risk(
            tenure_sample, acco_type_sample, house_size_sample, internet_sample,
            household_dep_sample, low_income_sample, home_insure_sample, health_insure_sample
        )

        risk_scores.append(household_risk)

    return risk_scores

np.random.seed(42)
risk_scores = generate_household_samples(100)
print(risk_scores)