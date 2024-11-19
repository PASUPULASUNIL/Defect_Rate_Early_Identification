# Import necessary libraries
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt

# Defining base paths for saving results and models
results_folder = '../results/regression_results'
models_folder = '../models/regression_models'

# Creating directories for results and models if they don't exist
os.makedirs(results_folder, exist_ok=True)
os.makedirs(f"{results_folder}/visualizations", exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# Storing current date for results filenames
current_date = dt.now().strftime('%Y-%m-%d_%H-%M-%S')

# Traing Multiple model
class RegressionModelTrainer:
    # initializing the variables
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.best_model = None
        self.best_score = float('inf')
        self.best_model_name = None

    # Function for traing the RF Model
    def train_random_forest(self):
        print("Training Random Forest Regressor...")
        rf = RandomForestRegressor(random_state=143)

        # Hyperparameter grid for Random Forest
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [2, 4, 6]
        }

        # GridSearchCV for Random Forest
        grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
        grid_rf.fit(self.train_x, self.train_y)
        
	# Finding Best Parametsrs
        best_rf = grid_rf.best_estimator_
        print("Best parameters for Random Forest:", grid_rf.best_params_)

        # Evaluating on test data
        predictions = best_rf.predict(self.test_x)
        mape_score = mean_absolute_percentage_error(self.test_y, predictions)
        print(f"Random Forest MAPE: {mape_score:.4f}")

        # Saving model and results 
        if mape_score < self.best_score:
            self.best_score = mape_score
            self.best_model = best_rf
            self.best_model_name = "Random Forest"
            self._save_results(predictions, "Random Forest")
    
    #Training Xgboost Model
    def train_xgboost(self):
        print("Training XGBoost Regressor...")
        xgb = XGBRegressor(random_state=143, verbosity=0)

        # Hyperparameter grid for XGBoost
        param_grid_xgb = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.001, 0.01, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        # GridSearchCV for XGBoost
        grid_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
        grid_xgb.fit(self.train_x, self.train_y)
        
        # Taking the best parameters
        best_xgb = grid_xgb.best_estimator_
        print("Best parameters for XGBoost:", grid_xgb.best_params_)

        # Evaluating on test data
        predictions = best_xgb.predict(self.test_x)
        mape_score = mean_absolute_percentage_error(self.test_y, predictions)
        print(f"XGBoost MAPE: {mape_score:.4f}")

        # Saving model and results 
        if mape_score < self.best_score:
            self.best_score = mape_score
            self.best_model = best_xgb
            self.best_model_name = "XGBoost"
            self._save_results(predictions, "XGBoost")

    # Function for saving the models
    def _save_results(self, predictions, model_name):
        # Calculate evaluation metrics
        mape_score = mean_absolute_percentage_error(self.test_y, predictions)
        mse_score = mean_squared_error(self.test_y, predictions)
        r2 = r2_score(self.test_y, predictions)

        # Save metrics as a CSV
        metrics = {
            "Model": [model_name],
            "MAPE": [mape_score],
            "MSE": [mse_score],
            "R2_Score": [r2]
        }
        pd.DataFrame(metrics).to_csv(f"{results_folder}/metrics_{model_name}_{current_date}.csv", index=False)

        # Saving predictions as a CSV
        prediction_df = pd.DataFrame({
            "Actual": self.test_y.values.flatten() if isinstance(self.test_y, pd.DataFrame) else self.test_y.values, 
            "Predicted": predictions
        })
        prediction_df.to_csv(f"{results_folder}/predictions_{model_name}_{current_date}.csv", index=False)

        # Plotting and saving the visualizations
        self._plot_predictions(predictions, model_name)
        self._plot_actual_vs_predicted(predictions, model_name)  # New line chart after best model evaluation

        # Saving model
        model_filename = os.path.join(models_folder, f'{model_name}_best_model_{current_date}.pkl')
        joblib.dump(self.best_model, model_filename)
        print(f"Saved {model_name} model to {model_filename}")

    # Function for ploting the Actual vs predited 
    def _plot_predictions(self, predictions, model_name):
        # Check the shape and flatten test_y if necessary
        test_y_flat = self.test_y.values.flatten() if isinstance(self.test_y, pd.DataFrame) else self.test_y
        plt.figure(figsize=(40, 6))
        plt.scatter(test_y_flat, predictions, alpha=0.6, label="Predicted vs Actual")
        plt.plot([test_y_flat.min(), test_y_flat.max()], [test_y_flat.min(), test_y_flat.max()], color='red', label="Perfect Fit")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_name} - Predicted vs Actual")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{results_folder}/visualizations/{model_name}_predicted_vs_actual_{current_date}.png")
        plt.close()

    # Function for ploting the Actual vs predited 
    def _plot_actual_vs_predicted(self, predictions, model_name):

        # Extracting actual and predicted defect rates for both training and testing sets
        actual_train = self.train_y.values.flatten() if isinstance(self.train_y, pd.DataFrame) else self.train_y
        predicted_train = self.best_model.predict(self.train_x)

        actual_test = self.test_y.values.flatten() if isinstance(self.test_y, pd.DataFrame) else self.test_y
        predicted_test = predictions

        # Creating a figure and plot the lines
        plt.figure(figsize=(30, 6))

        # Plot training and testing actual vs predicted
        plt.plot(actual_train, label="Actual Training Defect Rate", color='blue', linestyle='-', linewidth=2)
        plt.plot(predicted_train, label="Predicted Training Defect Rate", color='red', linestyle='--', linewidth=2)
        plt.plot(actual_test, label="Actual Testing Defect Rate", color='green', linestyle='-', linewidth=2)
        plt.plot(predicted_test, label="Predicted Testing Defect Rate", color='green', linestyle='--', linewidth=2)

        # Labels and title
        plt.ylabel("Defect Rate")
        plt.title(f"{model_name} - Actual vs Predicted Defect Rate (Training and Testing)")
        plt.legend()
        plt.grid(True)

        # Saving the plot to the appropriate folder
        plt.tight_layout()
        plt.savefig(f"{results_folder}/visualizations/{model_name}_actual_vs_predicted_lines_{current_date}.png")
        plt.close()

    # Function for traing and identify the best model
    def evaluate_models(self):

        # Train both models and identify the best one
        self.train_random_forest()
        self.train_xgboost()

        print(f"\nBest model: {self.best_model_name} with MAPE: {self.best_score:.4f}")

        # Generate line chart for actual vs predicted defect rates
        self._plot_actual_vs_predicted(self.best_model.predict(self.test_x), self.best_model_name)

        return self.best_model

    # function for Extracting feature importance for the best model
    def get_feature_importance(self):

        # Extracting feature importance for the best model
        if self.best_model_name == 'Random Forest':
            feature_importance = self.best_model.feature_importances_
            feature_names = self.train_x.columns
        elif self.best_model_name == 'XGBoost':
            feature_importance = self.best_model.feature_importances_
            feature_names = self.train_x.columns
        else:
            return pd.DataFrame()

        # Saving feature importance
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        feature_importance_df.to_csv(f"{results_folder}/feature_importance_{self.best_model_name}_{current_date}.csv", index=False)

        # Plotting and saving feature importance
        plt.figure(figsize=(6, 30))
        sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f"{self.best_model_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{results_folder}/visualizations/{self.best_model_name}_feature_importance_{current_date}.png")
        plt.close()

        return feature_importance_df
