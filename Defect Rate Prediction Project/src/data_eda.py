# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Directories exist for saving results and visualizations
os.makedirs("../results/eda/", exist_ok=True)  # Directory for saving EDA results
os.makedirs("../results/eda/visualizations/", exist_ok=True)  # Directory for visualizations

# Function for EDA processing
def perform_eda(production_df, dr):
    # Drop redundant date and time columns
    # These columns are redundant as we are combining DateTime later.
    production_df.drop(columns=['mt_DateTime', 'mt_Time', 'mtr_DateTime', 'mtr_Time', 
                                'ww_DateTime', 'ww_Time', 'pw_DateTime', 'pw_Time'], inplace=True)

    # Adding hour information to Check Alignment data
    # Time is in minutes we add hours to accurately represent the timestamp.
    production_df.loc[0:2119, 'ca_Time'] = '00:' + production_df.ca_Time
    production_df.loc[2120:6058, 'ca_Time'] = '01:' + production_df.ca_Time
    production_df.loc[6059:, 'ca_Time'] = '02:' + production_df.ca_Time

    # Combining date and time columns into a single datetime column
    # Easier to work with datetime objects for time-based analysis.
    production_df.ca_DateTime = pd.to_datetime(production_df.ca_DateTime + ' ' + production_df.ca_Time)
    production_df.drop(columns='ca_Time', inplace=True)

    # Defect Rate data timestamps
    # add hours for accuracy in defect rates' timestamps.
    dr.loc[:2398, 'Time'] = '00:' + dr.Time
    dr.loc[2399:6179, 'Time'] = '01:' + dr.Time
    dr.loc[6180:9946, 'Time'] = '02:' + dr.Time
    dr.loc[9947:, 'Time'] = '03:' + dr.Time

    # Combining Date and Time into a single DateTime column
    dr['DateTime'] = pd.to_datetime(dr.Date + ' ' + dr.Time)
    dr.drop(columns=['Date', 'Time'], inplace=True)

    # Converting Defect Rate to numerical format
    # Percentages are stored as strings; convert to numerical values for analysis.
    dr['Defect Rate'] = dr['Defect Rate'].str.replace('%', '').astype('float32') / 100

    # Droping columns with no variance
    # Columns with no variance won't contribute to meaningful insights.
    no_variance_cols = [
        'mt_MeasurementCount', 'mt_OverallJudgment', 'Trg1OverallJudgment', 'Trg1NGItem',
        'Trg2OverallJudgment', 'Trg2NGItem', 'mt_OutputBufferMargin', 
        'mtr_MeasurementCount', 'mtr_OverallJudgment', 'Cam1TerminalNo', 
        'Cam1Judgment', 'Cam1NGItem', 'Cam2TerminalNo', 'Cam2Judgment', 
        'Cam2NGItem', 'mtr_OutputBufferMargin', 'ww_MeasurementCount', 
        'ww_OverallJudgment', 'ww_OutputBufferMargin', 'pw_MeasurementCount', 
        'pw_OverallJudgment', '#d1', '#d2', 'Judgment1', 'Judgment2', 
        'Judgment3', 'Judgment4', 'pw_OutputBufferMargin', 'ca_MeasurementCount', 
        'ca_OverallJudgment', 'Alarm_No', 'ca_OutputBufferMargin', 
        'InspectionExecutionID'
    ]
    production_df.drop(columns=no_variance_cols, inplace=True)

    # Generating data summary and saving as CSV
    # Generating key statistics for a high-level view of the data.
    prod_summary = production_df.describe(percentiles=[0.01, 0.03, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99]).T
    prod_summary.to_csv("../results/eda/production_summary.csv")
    

    # Visualizing the box plots for all features and save
    # outliers detection and the spread of data.
    fig, axes = plt.subplots(20, 5, figsize=(25, 60))
    axes = axes.flatten()
    for i, col in enumerate(production_df.columns):
        sns.boxplot(data=production_df, y=col, ax=axes[i])
        axes[i].set_title(col)
    for j in range(len(production_df.columns), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig("../results/eda/visualizations/boxplots.png")
    plt.close()

    # Visualizing distribution of features and save
    # Understanding the distribution and presence of skewness in data.
    fig, axes = plt.subplots(20, 5, figsize=(25, 60))
    axes = axes.flatten()
    for i, col in enumerate(production_df.columns):
        sns.histplot(production_df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    for j in range(len(production_df.columns), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig("../results/eda/visualizations/distributions.png")
    plt.close()

    # Clipping outliers for float-type columns
    # Reducing the impact of extreme values on analysis and models.
    def clip_outliers_for_floats(df, z_threshold=3):
        float_cols = df.select_dtypes(include=['float']).columns
        df_out = df.copy()
        for col in float_cols:
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            lower_bound = mean - z_threshold * std
            upper_bound = mean + z_threshold * std
            df_out[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df_out

    df_updated = clip_outliers_for_floats(production_df, z_threshold=3)

    # Resample the data to the minute level
    # Aligning timestamps and prepare for time-series analysis.
    df_updated.rename(columns={'ca_DateTime': 'DateTime'}, inplace=True)
    df_updated.set_index('DateTime', inplace=True)
    main_df = df_updated.resample('1T').mean()
    main_dr = dr.set_index('DateTime').resample('1T').mean()
    main_dr.fillna(method='ffill', inplace=True)
    main_dr2 = main_dr.iloc[:main_df.shape[0]]
    main_dr2.index = main_df.index

    # Adding Hour and Minute columns for temporal features
    main_df.reset_index(inplace=True)
    main_df['Hour'] = main_df.DateTime.dt.hour
    main_df['Minute'] = main_df.DateTime.dt.minute
    main_df.set_index('DateTime', inplace=True)

    # Saving processed datasets
    # Saving for further steps in the pipeline.
    main_df.to_csv("../results/eda/processed_production_data.csv")
    main_dr2.to_csv("../results/eda/processed_defect_rates.csv")

    print(main_df.shape, main_dr2.shape)
    return main_df, main_dr2, main_dr2[['Defect Rate']]
    
