import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import mlflow
import mlflow.sklearn

# --- Step 1: Data Preprocessing and EDA ---

# Load the dataset
df = pd.read_csv('insurance.csv')

# Step 1.1 & 1.2: Load & Explore the Dataset
print("Dataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

# Step 2.1: Handle Missing Values (Assuming none, but keeping the placeholder)

# Visualize distribution of the new target feature
plt.figure(figsize=(10, 6))
# Corrected column name from 'charges' to 'Premium Amount'
sns.histplot(df['Premium Amount'], kde=True)
plt.title('Distribution of Premium Amount')
plt.show()

# Step 2.2: Convert Categorical Variables to Numerical Form
categorical_cols = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',
                    'Policy Type', 'Customer Feedback', 'Smoking Status', 'Property Type',
                    'Exercise Frequency']

# Convert all identified categorical columns to numerical using one-hot encoding
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop the non-numericl date column ('Policy Start Date') before correlation
df_processed = df_processed.drop(columns=['Policy Start Date'])

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
# Now all columns are numerical, so .corr() will work
sns.heatmap(df_processed.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()

print("\nProcessed Data Info:")
print(df_processed.info())

# Save the preprocessed data for model training
df_processed.to_csv('insurance_processed.csv', index=False)

# --- Step 3: Model Training and Evaluation ---

# Load the preprocessed data
df_processed = pd.read_csv('insurance_processed.csv')

# Define features (X) and target (y)
# Corrected the target column name from 'charges' to 'Premium Amount'
X = df_processed.drop('Premium Amount', axis=1)
y = df_processed['Premium Amount']

# Step 3.1: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3.2: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary of models to train
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_rmse = float('inf')

# Step 4: Track experiments with MLflow
with mlflow.start_run():
    mlflow.set_tag("project", "SmartPremium")
    for name, model in models.items():
        # Using nested runs for each model to prevent parameter key collisions
        with mlflow.start_run(nested=True):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Calculate evaluation metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"\n--- {name} Results ---")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R2 Score: {r2:.2f}")

            # Log metrics to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            # Save the best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                joblib.dump(scaler, 'scaler.pkl')
                joblib.dump(best_model, 'best_model.pkl')
                print(f"\n--- Saved new best model: {name} ---")

    # Log the best model as an artifact in the parent run
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_model_artifact",
        input_example=X_train_scaled[0:2]
        )
    print("Models and metrics logged to MLflow.")