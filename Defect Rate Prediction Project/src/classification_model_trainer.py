# importing the necessary packageds
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from datetime import datetime as dt

# Defining base paths
results_folder = '../results/classification_results'
models_folder = '../models/classification_models'

# Creating directories if they don't exist
os.makedirs(results_folder, exist_ok=True)
os.makedirs(f"{results_folder}/visualizations", exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# Storing current date for results filenames
current_date = dt.now().strftime('%Y-%m-%d_%H-%M-%S')

# Traning Multiple models 
class ClassificationModelTrainer:
    def __init__(self, new_train_x, new_train_y, new_test_x, new_test_y):
        self.new_train_x = new_train_x
        self.new_train_y = new_train_y
        self.new_test_x = new_test_x
        self.new_test_y = new_test_y
        self.best_model = None
        self.best_score = float('-inf')  # Start with lowest score since F1 ranges from 0 to 1
        self.best_model_name = None

    # Traing Random Forest Model
    def train_random_forest(self):
        print("Training Random Forest Classifier...")
        rf = RandomForestClassifier(random_state=23)

        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 5, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # GridSearch CV hyper parameter tuning
        grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, scoring='f1_weighted', n_jobs=-1)
        grid_rf.fit(self.new_train_x, self.new_train_y)

        # Taking Best Estimator and Parameters
        best_rf = grid_rf.best_estimator_
        print("Best parameters for Random Forest:", grid_rf.best_params_)

        predictions = best_rf.predict(self.new_test_x)
        f1 = f1_score(self.new_test_y, predictions, average='weighted')
        print(f"Random Forest F1-Score: {f1:.4f}")

        # checking for the best score
        if f1 > self.best_score:
            self.best_score = f1
            self.best_model = best_rf
            self.best_model_name = "Random Forest"
            self._save_results(predictions, "Random Forest")

    # Traing Xgboost Model
    def train_xgboost(self):
        print("Training XGBoost Classifier...")
        xgb = XGBClassifier(random_state=143)

        param_grid_xgb = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.001, 0.1],
            'subsample': [0.8, 1.0]
        }

        # GridSearch CV & Hyperparameter Tuning
        grid_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3, scoring='f1_weighted', n_jobs=-1)
        grid_xgb.fit(self.new_train_x, self.new_train_y)

        best_xgb = grid_xgb.best_estimator_
        print("Best parameters for XGBoost:", grid_xgb.best_params_)

        predictions = best_xgb.predict(self.new_test_x)
        f1 = f1_score(self.new_test_y, predictions, average='weighted')
        print(f"XGBoost F1-Score: {f1:.4f}")

	# checking for the best score
        if f1 > self.best_score:
            self.best_score = f1
            self.best_model = best_xgb
            self.best_model_name = "XGBoost"
            self._save_results(predictions, "XGBoost")

    # Saving the models
    def _save_results(self, predictions, model_name):

        # Saving metrics and classification report
        f1 = f1_score(self.new_test_y, predictions, average='weighted')
        cm = confusion_matrix(self.new_test_y, predictions)
        report = classification_report(self.new_test_y, predictions)

        # Saving metrics as text
        metrics_file = os.path.join(results_folder, f'{model_name}_classification_report_{current_date}.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write(report)
        print(f"Saved classification report to {metrics_file}")

        # Saving predictions
        prediction_df = pd.DataFrame({
            'Actual': self.new_test_y,
            'Predicted': predictions
        })
        prediction_df.to_csv(f"{results_folder}/predictions_{model_name}_{current_date}.csv", index=False)

        # Plotting and saving confusion matrix
        self._plot_confusion_matrix(cm, model_name)

        # Saving model
        model_filename = os.path.join(models_folder, f'{model_name}_best_model_{current_date}.pkl')
        joblib.dump(self.best_model, model_filename)
        print(f"Saved {model_name} model to {model_filename}")


    # Plotting confusion matrix
    def _plot_confusion_matrix(self, cm, model_name):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{results_folder}/visualizations/{model_name}_confusion_matrix_{current_date}.png")
        plt.close()

    # Evaluating the best model
    def evaluate_models(self):
        self.train_random_forest()
        self.train_xgboost()

        print(f"\nBest model: {self.best_model_name} with F1-Score: {self.best_score:.4f}")

        return self.best_model

    # Getting Future Importance
    def get_feature_importance(self):
        if self.best_model_name == 'Random Forest':
            feature_importance = self.best_model.feature_importances_
            feature_names = self.new_train_x.columns
        elif self.best_model_name == 'XGBoost':
            feature_importance = self.best_model.feature_importances_
            feature_names = self.new_train_x.columns
        else:
            return pd.DataFrame()

        # Saving feature importance
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        feature_importance_df.to_csv(f"{results_folder}/feature_importance_{self.best_model_name}_{current_date}.csv", index=False)

        # Plotting and saving feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f"{self.best_model_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{results_folder}/visualizations/{self.best_model_name}_feature_importance_{current_date}.png")
        plt.close()

        return feature_importance_df

