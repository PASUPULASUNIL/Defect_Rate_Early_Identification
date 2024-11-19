

# Import necessary modules
from src.data_preprocessing import preprocess_data
from src.data_eda import perform_eda
from src.regression_model_trainer import RegressionModelTrainer
from src.classification_model_trainer import ClassificationModelTrainer
from src.forecasting_model_trainer import train_forecasting_models

# Data Preprocessing
# Load and preprocess the data
production_df, dr = preprocess_data()

# Exploratory Data Analysis (EDA)
main_df, main_dr2, y = perform_eda(production_df, dr)

# Regression Model Training
# Prepare features and target for regression
train_x = main_df  # Features for training
train_y = y  # Target for training

# Splitting the data into train and test sets for regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Creating an instance of the RegressionModelTrainer
regression_trainer = RegressionModelTrainer(X_train, y_train, X_test, y_test)

# Train models and get the best model
best_model = regression_trainer.evaluate_models()

# Getting feature importance for the best model (if required)
feature_importance_df = regression_trainer.get_feature_importance()

# Classification Model Training
# Assuming we need to train classification models 

# Defining target variable for classification 

classification_target = [1 if i >= main_dr2['Defect Rate'].mean() else 0 for i in main_dr2['Defect Rate']]

# features for classification 
classification_features = main_df 

# Splitting the data into train and test sets for classification
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(classification_features, classification_target, test_size=0.2, random_state=42)

# Creating an instance of the ClassificationModelTrainer
classification_trainer = ClassificationModelTrainer(X_class_train, y_class_train, X_class_test, y_class_test)

# Train models and get the best classification model
best_classification_model = classification_trainer.evaluate_models()

# Getting feature importance for the best classification model
classification_feature_importance_df = classification_trainer.get_feature_importance()

# Forecasting Model Training
# Train forecasting models using defect rate data (main_dr2)
train_forecasting_models(main_dr2)

# Final Comment
print("Process completed successfully!")
