# data_preprocessing.py

# Importing the Basic packages for the data preparation and cleaning
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt, timedelta
import warnings

warnings.filterwarnings('ignore')

# Define the base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the raw data folder
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Importing the all datasets
def load_data():
    """
    Function to load the raw datasets.
    """
    mt = pd.read_csv(os.path.join(RAW_DATA_DIR, "Step1_Mount_Terminals.csv"))  # mt --> Mount Terminals
    mtr = pd.read_csv(os.path.join(RAW_DATA_DIR, "Step1_Mount_Terminal_Resin.csv"))  # mtr --> Mount Terminal Resin
    ww = pd.read_csv(os.path.join(RAW_DATA_DIR, "Step2_Wind_Wire.csv"))  # ww --> Wind Wire
    pw = pd.read_csv(os.path.join(RAW_DATA_DIR, "Step3_Peel_Wire.csv"))  # pw --> Peel Wire
    ca = pd.read_csv(os.path.join(RAW_DATA_DIR, "Step4_Check_Alignment.csv"))  # ca --> Check Alignment
    dr = pd.read_csv(os.path.join(RAW_DATA_DIR, "Defect Rates.csv"))  # dr --> Defect Rate
    return mt, mtr, ww, pw, ca, dr


# Columns to rename
columns_to_rename = ["DateTime", "Time", "MeasurementCount", "OverallJudgment", "OutputBufferMargin"]

# User-defined function to add a prefix to selected columns in the DataFrame
def add_prefix_to_columns(df, prefix, columns_to_rename):
    """
    Function to add a prefix to the selected columns in the DataFrame.
    """
    df.sort_values('MeasurementCount', ascending=True, inplace=True)
    new_column_names = {col: f"{prefix}_{col}" for col in columns_to_rename}
    df.rename(columns=new_column_names, inplace=True)


# Apply the UDF to add prefix to the selected columns
def preprocess_data():
    mt, mtr, ww, pw, ca, dr = load_data()
    # List of tuples containing DataFrame and corresponding prefixes
    sample_prefix = [(mt, "mt"), (mtr, "mtr"), (ww, "ww"), (pw, "pw"), (ca, "ca")]
    for df, prefix in sample_prefix:
        add_prefix_to_columns(df, prefix, columns_to_rename)

    # Production process Data concatenation
    production_df = pd.concat([mt, mtr, ww, pw, ca], axis=1).reset_index().drop(columns='index')

    # Ensure the defect rate data is in the correct order to align with other datasets
    dr = dr.loc[::-1].reset_index(drop=True)  # Reversing the defect rate data and resetting index

    return production_df, dr


# Calling the preprocessing function
production_df, dr = preprocess_data()

# Optionally, save the processed dataframes to CSV or use them for further analysis
# production_df.to_csv('processed_data.csv', index=False)
# dr.to_csv('processed_defect_rate.csv', index=False)
