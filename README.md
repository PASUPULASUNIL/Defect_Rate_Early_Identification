# Defect_Rate_Early_Identification


## Table of Contents
#### 1. Project Structure
#### 2. Setup Instructions
#### 3. Dataset Description
#### 4. Workflow
  Data Preparation
  
  Exploratory Data Analysis
  
  Model Building
  
  Time-Series Forecasting
  
#### 5. Results
#### 6. Resources

## 1. Project Structure
    Defect Rate Prediction Project/
    │
    ├── src/
    │   ├── data_preprocessing.py             # Handles data cleaning and preparation
    │   ├── data_eda.py                       # Performs exploratory data analysis
    │   ├── regression_model_trainer.py       # Trains regression models for defect prediction
    │   ├── classification_model_trainer.py   # Trains classification models for defect category prediction
    │   └── forecasting_model_trainer.py      # Implements ARIMA/SARIMAX models for forecasting defect rates
    │
    ├── data/
    │   ├── raw/                              # Contains raw input data
    │   │   ├── Step1_Mount_Terminals.csv
    │   │   ├── Step1_Mount_Terminal_Resin.csv
    │   │   ├── Step2_Wind_Wire.csv
    │   │   ├── Step3_Peel_Wire.csv
    │   │   ├── Step4_Check_Alignment.csv
    │   │   └── Defect Rates.csv
    │   └── processed/                        # Contains preprocessed datasets
    │
    ├── results/
    │   ├── forecasting_results/              # Time-series forecasting outputs
    │   ├── regression_results/               # Regression model outputs
    │   ├── classification_results/           # Classification model outputs
    │   └── eda/                              # EDA visualizations
    │
    ├── models/
    │   ├── forecast_models/                  # Stored forecasting models
    │   ├── regression_models/                # Stored regression models
    │   └── classification_models/            # Stored classification models
    │
    ├── main.py                               # Main script to execute the project workflow
    ├── requirements.txt                      # Dependencies for the project
    └── Resource/
        ├── Defect Rate Prediction Project_Dictionary.txt
        ├── Defect Rate Early Identification Documentation.pdf


## Dataset Description

#### Raw Data

The project includes datasets for various production steps:

Step1_Mount_Terminals.csv: Parameters for the first production stage.

Step1_Mount_Terminal_Resin.csv: Data for applying resin to terminals.

Step2_Wind_Wire.csv: Data for winding wires.

Step3_Peel_Wire.csv: Data for peeling wires.

Step4_Check_Alignment.csv: Final alignment check parameters.

Defect Rates.csv: Recorded defect rates.

#### Processed Data
The processed datasets align all production steps chronologically and clean redundant or irrelevant columns.

## Workflow

#### 1. Data Preparation
Sorting and alignment: Align datasets chronologically based on time and measurement count.
Prefix addition: Add unique prefixes to columns to differentiate features across steps.
Feature scaling: Standardize features using StandardScaler.
Handling missing data: Logical assumptions were made to fill missing time information.
Outlier treatment: Outliers were clipped at the 1st and 99th percentiles.

#### 2. Exploratory Data Analysis
Visualizations: Created histograms, boxplots, and line charts to understand distributions and trends.
Feature importance: Identified critical parameters impacting defect rates.
#### 3. Model Building
Regression
Random Forest: Achieved the lowest Mean Absolute Percentage Error (MAPE) of 0.0339.
XGBoost: Alternative model with MAPE of 0.0340.
Classification
Defect Categories: Binary classification for low and high defect rates.
Random Forest Classifier: F1-score of 0.4318.
XGBoost Classifier: F1-score of 0.4283.
#### 4. Time-Series Forecasting
ARIMA: Best model order (4, 1, 1) achieved the lowest RMSE and MAPE.
SARIMAX: Seasonal order (2, 1, 2, 6) captured periodic fluctuations better, making it the preferred model for forecasting.

## Results
**Regression**
Random Forest performed the best with the lowest MAPE.
Identified significant features such as alignment measurements and terminal dimensions.

**Classification**
Key features from different production stages impact defect rates.
Random Forest outperformed XGBoost in classification accuracy.

**Forecasting**
The SARIMAX model successfully predicted defect rates for the next 7 days.
Seasonal patterns with a 6-minute periodicity were effectively captured.
## Resources
Defect Rate Prediction Project Dictionary: Key definitions and data descriptions.

Defect Rate Early Identification Documentation: Detailed methodology and assumptions.
