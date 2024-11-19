# importing necessary packages

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import itertools
import joblib
from datetime import datetime as dt
import os

# Setting up directories for saving models, results, and visualizations
results_folder = '../results/forecasting_results'
models_folder = '../models/forecast_models'

# Creating directories if they don't exist
os.makedirs(results_folder, exist_ok=True)
os.makedirs(f"{results_folder}/visualizations", exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# Storing current date for results filenames
current_date = dt.now().strftime('%Y-%m-%d_%H-%M-%S')

# Functions for RMSE and RMSPE calculation
def RMSE(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def RMSPE(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    percentage_error = (actual - predicted) / actual
    mean_square_percentage_error = np.mean(percentage_error**2)
    rmspe = np.sqrt(mean_square_percentage_error)
    return rmspe

# Decompose the defect rate (y) time series to observing its trend, seasonality, and residuals
def decompose_time_series(y):
    decompose = seasonal_decompose(y, period=6, model='additive')

    # Saving the trend Plot
    plt.figure(figsize=[20, 2])
    plt.title('Trend')
    plt.plot(decompose.trend)
    plt.ylabel('Defect Rate')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(f"{results_folder}/visualizations/trend_{current_date}.png")
    plt.close()

    # Saving the Seasonality Plot
    plt.figure(figsize=[20, 2])
    plt.title('Seasonality')
    plt.plot(decompose.seasonal)
    plt.ylabel('Defect Rate')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(f"{results_folder}/visualizations/seasonality_{current_date}.png")
    plt.close()

    # Saving Residual plot
    plt.figure(figsize=[20, 2])
    plt.title('Residual')
    plt.plot(decompose.resid)
    plt.ylabel('Defect Rate')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(f"{results_folder}/visualizations/residual_{current_date}.png")
    plt.close()

# Performing the Augmented Dickey-Fuller (ADF) test to check for stationarity
def adf_test(y):
    adf_result = adfuller(y)
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")

    if adf_result[1] > 0.05:
        print("Target variable is non-stationary. Differencing is needed.")
        return y.diff().dropna()
    else:
        print("Target variable is stationary.")
        return y

# Forecast using ARIMA model and evaluate using RMSE, RMSPE, and MAPE
def arima_forecasting(train, test):
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    order_list = []
    rmse_list, rmspe_list, mape_list, aic_list = [], [], [], []

    p_range = range(1, 5)
    d_range = range(1, 5)
    q_range = range(1, 5)

    # Iterating the model for the best p,q,d values and scores
    for i in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(train, order=i)
            model_fit = model.fit()

            forecast = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)

            aic_list.append(model_fit.aic)
            order_list.append(i)
            rmse_list.append(np.sqrt(mean_squared_error(test, forecast)))
            rmspe_list.append(RMSPE(test, forecast))
            mape_list.append(mean_absolute_percentage_error(test, forecast))
        except Exception as e:
            print(f"Skipping order {i} due to error: {e}")
            continue

    # creating and saving metrics list of dataframe 
    metric_list = pd.DataFrame({
        'Order_list': order_list,
        'AIC_score': aic_list,
        'RMSE': rmse_list,
        'RMSPE': rmspe_list,
        'MAPE': mape_list
    })
    metric_list.to_csv(f"{results_folder}/arima_results_{current_date}.csv", index=False)

    # Taking Best order
    best_model_idx = np.argmin(rmse_list)
    best_order = order_list[best_model_idx]
    print(f"Best ARIMA model order: {best_order}")
    return best_order

# Forecast using SARIMAX model and evaluate using RMSE, RMSPE, and MAPE
def sarimax_forecasting(train, test):
    order_list = []
    seasonal_order_list = []
    rmse_list, rmspe_list, mape_list, aic_list = [], [], [], []

    p_range = range(1, 3)
    d_range = range(1, 3)
    q_range = range(1, 3)
    P_range = range(1, 4)
    D_range = range(1, 3)
    Q_range = range(1, 5)

    # Iterating the process for the best seasonal orders
    for i in itertools.product(p_range, d_range, q_range):
        for j in itertools.product(P_range, D_range, Q_range):
            try:
                model = SARIMAX(train, order=i, seasonal_order=j + (6,))
                model_fit = model.fit()

                forecast = model_fit.forecast(len(test))

                aic_list.append(model_fit.aic)
                order_list.append(i)
                seasonal_order_list.append(j)
                rmse_list.append(np.sqrt(mean_squared_error(test, forecast)))
                rmspe_list.append(RMSPE(test, forecast))
                mape_list.append(mean_absolute_percentage_error(test, forecast))
            except Exception as e:
                print(f"Skipping order {i}, seasonal order {j} due to error: {e}")
                continue

    metric_list = pd.DataFrame({
        'Order_list': order_list,
        'Seasonal_Order': seasonal_order_list,
        'AIC_score': aic_list,
        'RMSE': rmse_list,
        'RMSPE': rmspe_list,
        'MAPE': mape_list
    })
    metric_list.to_csv(f"{results_folder}/sarimax_results_{current_date}.csv", index=False)

    best_model_idx = np.argmin(rmse_list)
    best_order = order_list[best_model_idx]
    best_seasonal_order = seasonal_order_list[best_model_idx]
    print(f"Best SARIMAX model order: {best_order}, seasonal order: {best_seasonal_order}")
    return best_order, best_seasonal_order

# Plotting and saving the best model forecasts
def plot_and_save_best_model_forecasts(train, test, best_order, best_seasonal_order):
    best_sarimax_model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order + (6,))
    best_sarimax_model_fit = best_sarimax_model.fit()

    backward_predict = best_sarimax_model_fit.predict(start=0, end=len(train) - 1)
    forward_predict = best_sarimax_model_fit.forecast(len(test))

    # ploting  and saving actual and predicted Defect rates
    plt.figure(figsize=[40, 5])
    plt.title('Train Data\nActual vs Predicted')
    plt.plot(train, label='Actual')
    plt.plot(backward_predict, label='Train_Predict')
    plt.legend()
    plt.savefig(f"{results_folder}/visualizations/train_actual_vs_predicted_{current_date}.png")
    plt.close()

    # ploting and saving actual and predicted Defect rates
    plt.figure(figsize=[40, 5])
    plt.title('Test Data\nActual vs Predicted')
    plt.plot(test, label='Actual')
    plt.plot(forward_predict, label='Test_Predict')
    plt.legend()
    plt.savefig(f"{results_folder}/visualizations/test_actual_vs_predicted_{current_date}.png")
    plt.close()


    # Predicting Defect rates for the next 7 Days
    predict_7_days = best_sarimax_model_fit.forecast(7)
    forecast_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=7, freq='D')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': predict_7_days})

    # Ploting and saving forecast
    plt.figure(figsize=[40, 5])
    plt.title('7 Day Forecast')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast for next 7 days')
    plt.legend()
    plt.savefig(f"{results_folder}/visualizations/7_day_forecast_{current_date}.png")
    plt.close()

    forecast_df.to_csv(f"{results_folder}/forecast_{current_date}.csv", index=False)
    joblib.dump(best_sarimax_model_fit, f"{models_folder}/best_sarimax_model_{current_date}.pkl")

# Main function
def train_forecasting_models(y):
    decompose_time_series(y)
    y_stationary = adf_test(y)

    train_length = int(len(y_stationary) * 0.8)
    train = y_stationary.iloc[:train_length]
    test = y_stationary.iloc[train_length:]

    best_arima_order = arima_forecasting(train, test)
    best_sarimax_order, best_sarimax_seasonal_order = sarimax_forecasting(train, test)

    plot_and_save_best_model_forecasts(train, test, best_sarimax_order, best_sarimax_seasonal_order)


