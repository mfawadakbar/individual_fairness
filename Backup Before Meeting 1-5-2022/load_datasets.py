"""
This file loads different datasets
BANK DATASET
COMPAS DATASET
ADULT INCOME DATASET
GERMAN CREDIT DATASET
"""
import pandas as pd
import numpy as np
import fnmatch
import random
import preprocess

random.seed(preprocess.seed)
np.random.seed(preprocess.seed)


def find_patterns(df, patterns, found_patterns):
    for pattern in patterns:
        matches = fnmatch.filter(df.columns, pattern + "*")
        found_patterns.extend(matches)
    return found_patterns


def style_specific_cell(x):
    color_thresh = "background-color: lightpink"
    df_color = pd.DataFrame("", index=x.index, columns=x.columns)
    rows_number = len(x.index)
    column_number = len(x.columns)
    for r in range(0, rows_number):
        for c in range(0, column_number):
            try:
                val = float(x.iloc[r, c])
                if x.iloc[r, 0] == "Percentage":
                    if val < 10:
                        df_color.iloc[r, c] = color_thresh
            except:
                pass

    return df_color


def style_stats_specific_cell(x):
    color_thresh = "background-color: lightpink"
    df_color = pd.DataFrame("", index=x.index, columns=x.columns)
    rows_number = len(x.index)
    for r in range(0, rows_number):
        try:
            val = x.iloc[r, 1]
            if val > 0.05:
                df_color.iloc[r, 1] = color_thresh
        except:
            pass
    return df_color


def load_dataset():
    print(
        "Load Dataset: Type name from the following 1) bank 2) adult 3) compas 4) german"
    )
    dataset_name = str(input("Enter Name: "))

    if dataset_name == "bank":
        print("BANK DATASET LOADED")
        # Define variables
        input_file = "https://query.data.world/s/jzrt4xirncrw4p7z5zrxgkshexekot"
        df = pd.read_csv(input_file, delimiter=";", header=0)
        df = df.drop(columns=["cons.conf.idx"])
        LAST_COLUMN = df.columns[-1]
        df = df.rename(columns={LAST_COLUMN: "y"})
        all_columns = df.columns
        categorical_cols = df.columns.difference(df._get_numeric_data().columns)
        categorical_cols = categorical_cols.drop(["y", "marital"])
        # print("Categorical variables:", categorical_cols)
        df = df[(df["marital"] != "unknown")]
        drop_columns = ["marital"]
        df["marital"] = df["marital"].replace("divorced", "single")
        marital_attr = {"single": 1, "married": 0}
        df["marital"] = df["marital"].map(marital_attr)
        y_attr = {"no": 0, "yes": 1}
        df["y"] = df["y"].map(y_attr)
        print(f"Value Counts for Sensitive Attribute: \n {df.marital.value_counts()}")
        print("Value Counts for Output Attribute: ", df.y.value_counts())

    elif dataset_name == "adult":
        print("ADULT DATASET LOADED")
        # Define variables
        input_file = "https://query.data.world/s/irpia6espshpvhps2az6vivh4tbpnr"
        df = pd.read_csv(input_file, header=0)
        LAST_COLUMN = df.columns[-1]
        df = df.rename(columns={LAST_COLUMN: "y"})
        df = df.rename(
            columns={
                "marital-status": "marital_status",
                "native--country": "native_country",
                "education-num": "education_num",
                "capital-gain": "capital_gain",
                "capital-loss": "capital_loss",
                "hours-per-week": "hours_per_week",
            }
        )
        all_columns = df.columns
        categorical_cols = df.columns.difference(df._get_numeric_data().columns)
        categorical_cols = categorical_cols.drop(["y"])
        # print("Categorical variables:", categorical_cols)
        sex_attr = {df["sex"].unique()[0]: 0, df["sex"].unique()[1]: 1}
        df["sex"] = df["sex"].map(sex_attr)
        y_attr = {df["y"].unique()[0]: 0, df["y"].unique()[1]: 1}
        df["y"] = df["y"].map(y_attr)
        print("Value Counts for Output Attribute: ", df.y.value_counts())
        print(f"Value Counts for Sensitive Attribute:", df["sex"].value_counts())
        drop_columns = ["sex"]

    elif dataset_name == "compas":
        print("COMPAS DATASET LOADED")
        import warnings

        warnings.filterwarnings("ignore")
        # filter dplyr warnings
        import urllib.request

        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        urllib.request.urlretrieve(url, "compas-scores-two-years.csv")
        raw_data = pd.read_csv("./compas-scores-two-years.csv")
        # print("Num rows: %d" % len(raw_data))
        time_to_hours = lambda x: x.total_seconds() / (60 * 60)
        df = raw_data[
            (
                (raw_data["days_b_screening_arrest"] <= 30)
                & (raw_data["days_b_screening_arrest"] >= -30)
                & (raw_data["is_recid"] != -1)
                & (raw_data["c_charge_degree"] != "O")
                & (raw_data["score_text"] != "N/A")
            )
        ]

        # print("Num rows filtered: %d" % len(df))
        df.c_jail_in = pd.to_datetime(df.c_jail_in)
        df.c_jail_out = pd.to_datetime(df.c_jail_out)
        df["time_inside"] = list(map(time_to_hours, (df.c_jail_out - df.c_jail_in)))
        patterns = [
            "id",
            "name",
            "first",
            "last",
            "dob",
            "age_cat",
            "compas_screening_date",
            "r_jail_in",
            "r_jail_out",
            "c_jail_in",
            "c_jail_out",
            "c_case_number",
            "c_offense_date",
            "r_case_number",
            "r_offense_date",
            "vr_case_number",
            "vr_offense_date",
            "vr_charge_degree",
            "vr_charge_desc",
            "c_arrest_date",
            "is_recid",
            "violent_recid",
            "decile_score",
            "decile_score.1",
            "screening_date",
            "v_screening_date",
            "in_custody",
            "out_custody",
        ]
        drop_columns_not_needed = []
        drop_columns_not_needed = find_patterns(df, patterns, drop_columns_not_needed)
        df["is_med_or_high_risk"] = (df["decile_score"] >= 5).astype(int)
        # Define variables
        df = df.drop(columns=drop_columns_not_needed)
        LAST_COLUMN = "is_med_or_high_risk"
        df = df.rename(columns={LAST_COLUMN: "y"})
        all_columns = df.columns
        categorical_cols = df.columns.difference(df._get_numeric_data().columns)
        df.r_days_from_arrest = df.r_days_from_arrest.fillna(0)
        # print("Categorical variables:", categorical_cols)
        df = df[(df["race"] == "African-American") | (df["race"] == "Caucasian")]
        drop_columns = ["race"]
        race = {"African-American": 1, "Caucasian": 0}
        df["race"] = df["race"].map(race)
        y_attr = {df["y"].unique()[0]: 0, df["y"].unique()[1]: 1}
        df["y"] = df["y"].map(y_attr)
        print("Value Counts for Output Attribute: ", df.y.value_counts())
        print("Value Counts for Sensitive Attribute: \n", df["race"].value_counts())

    elif dataset_name == "german":
        print("GERMAN DATASET LOADED")
        # Reading Dataset from  http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
        df = pd.read_csv(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
            sep=" ",
            header=None,
        )
        headers = [
            "Status of existing checking account",
            "Duration in month",
            "Credit history",
            "Purpose",
            "Credit amount",
            "Savings account/bonds",
            "Present employment since",
            "Installment rate in percentage of disposable income",
            "Personal status and sex",
            "Other debtors / guarantors",
            "Present residence since",
            "Property",
            "Age in years",
            "Other installment plans",
            "Housing",
            "Number of existing credits at this bank",
            "Job",
            "Number of people being liable to provide maintenance for",
            "Telephone",
            "foreign worker",
            "Cost Matrix(Risk)",
        ]
        df.columns = headers
        df.to_csv("german_data_credit_cat.csv", index=False)  # save as csv file
        # for structuring only
        Status_of_existing_checking_account = {
            "A14": "no checking account",
            "A11": "<0 DM",
            "A12": "0 <= <200 DM",
            "A13": ">= 200 DM ",
        }
        df["Status of existing checking account"] = df[
            "Status of existing checking account"
        ].map(Status_of_existing_checking_account)

        Credit_history = {
            "A34": "critical account",
            "A33": "delay in paying off",
            "A32": "existing credits paid back duly till now",
            "A31": "all credits at this bank paid back duly",
            "A30": "no credits taken",
        }
        df["Credit history"] = df["Credit history"].map(Credit_history)

        Purpose = {
            "A40": "car (new)",
            "A41": "car (used)",
            "A42": "furniture/equipment",
            "A43": "radio/television",
            "A44": "domestic appliances",
            "A45": "repairs",
            "A46": "education",
            "A47": "vacation",
            "A48": "retraining",
            "A49": "business",
            "A410": "others",
        }
        df["Purpose"] = df["Purpose"].map(Purpose)

        Saving_account = {
            "A65": "no savings account",
            "A61": "<100 DM",
            "A62": "100 <= <500 DM",
            "A63": "500 <= < 1000 DM",
            "A64": ">= 1000 DM",
        }
        df["Savings account/bonds"] = df["Savings account/bonds"].map(Saving_account)

        Present_employment = {
            "A75": ">=7 years",
            "A74": "4<= <7 years",
            "A73": "1<= < 4 years",
            "A72": "<1 years",
            "A71": "unemployed",
        }
        df["Present employment since"] = df["Present employment since"].map(
            Present_employment
        )

        Personal_status_and_sex = {
            "A95": "female:single",
            "A94": "male:married/widowed",
            "A93": "male:single",
            "A92": "female:divorced/separated/married",
            "A91": "male:divorced/separated",
        }
        df["Personal status and sex"] = df["Personal status and sex"].map(
            Personal_status_and_sex
        )

        Other_debtors_guarantors = {
            "A101": "none",
            "A102": "co-applicant",
            "A103": "guarantor",
        }
        df["Other debtors / guarantors"] = df["Other debtors / guarantors"].map(
            Other_debtors_guarantors
        )

        Property = {
            "A121": "real estate",
            "A122": "savings agreement/life insurance",
            "A123": "car or other",
            "A124": "unknown / no property",
        }
        df["Property"] = df["Property"].map(Property)

        Other_installment_plans = {"A143": "none", "A142": "store", "A141": "bank"}
        df["Other installment plans"] = df["Other installment plans"].map(
            Other_installment_plans
        )

        Housing = {"A153": "for free", "A152": "own", "A151": "rent"}
        df["Housing"] = df["Housing"].map(Housing)

        Job = {
            "A174": "management/ highly qualified employee",
            "A173": "skilled employee / official",
            "A172": "unskilled - resident",
            "A171": "unemployed/ unskilled  - non-resident",
        }
        df["Job"] = df["Job"].map(Job)

        Telephone = {"A192": "yes", "A191": "none"}
        df["Telephone"] = df["Telephone"].map(Telephone)

        foreign_worker = {"A201": "yes", "A202": "no"}
        df["foreign worker"] = df["foreign worker"].map(foreign_worker)

        risk = {1: "Good Risk", 2: "Bad Risk"}
        df["Cost Matrix(Risk)"] = df["Cost Matrix(Risk)"].map(risk)

        df = pd.read_csv("german_data_credit_cat.csv")
        number_of_credit = {1: 1, 2: 2, 3: 2, 4: 2}
        df["Number of existing credits at this bank"] = df[
            "Number of existing credits at this bank"
        ].map(number_of_credit)

        Status_of_existing_checking_account = {
            "A14": "no checking account",
            "A11": "<0 DM",
            "A12": ">0 DM",
            "A13": ">0 DM",
        }
        df["Status of existing checking account"] = df[
            "Status of existing checking account"
        ].map(Status_of_existing_checking_account)

        Credit_history = {
            "A34": "critical account/delay in paying off",
            "A33": "critical account/delay in paying off",
            "A32": "all credit / existing credits paid back duly till now",
            "A31": "all credit / existing credits paid back duly till now",
            "A30": "no credits taken",
        }
        df["Credit history"] = df["Credit history"].map(Credit_history)

        Purpose = {
            "A40": "car (new)",
            "A41": "car (used)",
            "A42": "Home Related",
            "A43": "Home Related",
            "A44": "Home Related",
            "A45": "Home Related",
            "A46": "others",
            "A47": "others",
            "A48": "others",
            "A49": "others",
            "A410": "others",
        }
        df["Purpose"] = df["Purpose"].map(Purpose)

        Saving_account = {
            "A65": "no savings account",
            "A61": "<100 DM",
            "A62": "<500 DM",
            "A63": ">500 DM",
            "A64": ">500 DM",
        }
        df["Savings account/bonds"] = df["Savings account/bonds"].map(Saving_account)

        Present_employment = {
            "A75": ">=7 years",
            "A74": "4<= <7 years",
            "A73": "1<= < 4 years",
            "A72": "<1 years",
            "A71": "<1 years",
        }
        df["Present employment since"] = df["Present employment since"].map(
            Present_employment
        )

        Personal_status_and_sex = {
            "A95": "female",
            "A94": "male",
            "A93": "male",
            "A92": "female",
            "A91": "male",
        }
        df["Personal status and sex"] = df["Personal status and sex"].map(
            Personal_status_and_sex
        )

        Other_debtors_guarantors = {
            "A101": "none",
            "A102": "co-applicant/guarantor",
            "A103": "co-applicant/guarantor",
        }
        df["Other debtors / guarantors"] = df["Other debtors / guarantors"].map(
            Other_debtors_guarantors
        )

        Property = {
            "A121": "real estate",
            "A122": "savings agreement/life insurance",
            "A123": "car or other",
            "A124": "unknown / no property",
        }
        df["Property"] = df["Property"].map(Property)

        Other_installment_plans = {
            "A143": "none",
            "A142": "bank/store",
            "A141": "bank/store",
        }
        df["Other installment plans"] = df["Other installment plans"].map(
            Other_installment_plans
        )

        Housing = {"A153": "for free", "A152": "own", "A151": "rent"}
        df["Housing"] = df["Housing"].map(Housing)

        Job = {
            "A174": "employed",
            "A173": "employed",
            "A172": "unemployed",
            "A171": "unemployed",
        }
        df["Job"] = df["Job"].map(Job)

        Telephone = {"A192": "yes", "A191": "none"}
        df["Telephone"] = df["Telephone"].map(Telephone)

        foreign_worker = {"A201": "yes", "A202": "no"}
        df["foreign worker"] = df["foreign worker"].map(foreign_worker)

        risk = {1: "Good Risk", 2: "Bad Risk"}
        df["Cost Matrix(Risk)"] = df["Cost Matrix(Risk)"].map(risk)

        column_names = df.columns.tolist()
        column_names.remove("Credit amount")  # numerical variable
        column_names.remove("Age in years")  # numerical variable
        column_names.remove("Duration in month")  # numerical variable

        # ----------------------------------------------------------------------------------------
        column_names_cat = {}
        for name in column_names:
            column_names_cat[name] = len(df[name].unique().tolist())

            marginal_report_cluster = {}
        for itr in range(0, np.asarray(list(column_names_cat.values())).max() + 1):
            if [k for k, v in column_names_cat.items() if v == itr]:
                marginal_report_cluster[itr] = [
                    k for k, v in column_names_cat.items() if v == itr
                ]

        # ----------------------------------------------------------------------------------------
        for key in marginal_report_cluster.keys():
            marginal_percentage_report = []
            for name in sorted(marginal_report_cluster[key]):
                data = (
                    pd.crosstab(df[name], columns=["Percentage"])
                    .apply(lambda r: (round((r / r.sum()) * 100, 2)), axis=0)
                    .reset_index()
                )
                data.columns = [name, "Percentage"]
                data = data.transpose().reset_index()
                [marginal_percentage_report.append(x) for x in data.values.tolist()]
                options = []
            marginal_percentage_report = pd.DataFrame(marginal_percentage_report)
            [
                options.append("Category Option " + str(itr))
                for itr in range(1, len(marginal_percentage_report.columns))
            ]
            marginal_percentage_report.columns = ["Attribute"] + options
            # display(marginal_percentage_report.style.apply(style_specific_cell, axis=None))

        attr_significant = [
            "Status of existing checking account",
            "Credit history",
            "Purpose",
            "Savings account/bonds",
            "Present employment since",
            "Personal status and sex",
            "Property",
            "Other installment plans",
            "Housing",
            "foreign worker",
            "Credit amount",
            "Age in years",
            "Duration in month",
        ]
        sex = {"female": 1, "male": 0}
        df["Personal status and sex"] = df["Personal status and sex"].map(sex)

        all_columns = df.columns
        # df=df[attr_significant+target_variable]
        df = df.rename(columns={"Cost Matrix(Risk)": "y"})

        categorical_cols = df.columns.difference(df._get_numeric_data().columns)
        categorical_cols = categorical_cols.drop(["y"])
        print("Categorical variables:", categorical_cols)

        LAST_COLUMN = all_columns[-1]
        drop_columns = ["Personal status and sex"]
        y_attr = {df["y"].unique()[0]: 0, df["y"].unique()[1]: 1}
        df["y"] = df["y"].map(y_attr)
        print("Value Counts for Output Attribute: ", df.y.value_counts())
        print(
            "Value Counts for Sensitive Attribute: ",
            df["Personal status and sex"].value_counts(),
        )
    return df, dataset_name, drop_columns, categorical_cols


if __name__ == "__main__":
    load_dataset()
