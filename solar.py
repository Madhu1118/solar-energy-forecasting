# ============================================================
# SOLAR ENERGY FORECASTING PROJECT
# Week 1-3: Complete Source Code
# ============================================================

# ------------------------------------------------------------
# WEEK 1: DATA LOADING AND CLEANING
# ------------------------------------------------------------

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Data
print("Loading Plant 1 Data...")
gen1 = pd.read_csv('data/plant1/generation.csv')
weather1 = pd.read_csv('data/plant1/weather.csv')

print("Loading Plant 2 Data...")
gen2 = pd.read_csv('data/plant2/generation.csv')
weather2 = pd.read_csv('data/plant2/weather.csv')

print("Data loaded successfully!")

# Step 3: Preview Data
print("\n--- Plant 1 Generation Data Preview ---")
print(gen1.head())
print("\nData Shape:", gen1.shape)
print("\nColumn Names:", gen1.columns.tolist())

print("\n--- Plant 1 Weather Data Preview ---")
print(weather1.head())
print("\nData Shape:", weather1.shape)
print("\nColumn Names:", weather1.columns.tolist())

# Step 4: Check for Missing Values
print("\n--- Checking Missing Values in Plant 1 Generation Data ---")
print(gen1.isnull().sum())

print("\n--- Checking Missing Values in Plant 1 Weather Data ---")
print(weather1.isnull().sum())

# Step 5: Clean Data (Remove Missing Values)
print("\n--- Cleaning Data ---")
gen1_clean = gen1.dropna()
weather1_clean = weather1.dropna()

print(f"Original Generation Data Size: {len(gen1)}")
print(f"Cleaned Generation Data Size: {len(gen1_clean)}")
print(f"Rows Removed: {len(gen1) - len(gen1_clean)}")

# Step 6: Save Cleaned Data
gen1_clean.to_csv('data/plant1/generation_clean.csv', index=False)
weather1_clean.to_csv('data/plant1/weather_clean.csv', index=False)
print("\nCleaned data saved successfully!")

# ------------------------------------------------------------
# WEEK 2: MODEL BUILDING AND TRAINING
# ------------------------------------------------------------

# Step 7: Feature Selection
# Display available columns to choose features
print("\n--- Available Columns for Modeling ---")
print(gen1_clean.columns.tolist())

# Select features and target
# NOTE: Replace these column names with your actual column names
# Common columns might be: 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD'

# Example feature selection (adjust based on your actual columns):
feature_columns = ['DC_POWER', 'AC_POWER', 'DAILY_YIELD']  # Replace with your columns
target_column = 'TOTAL_YIELD'  # Replace with your target column

# If you're unsure of column names, uncomment this to see them:
# print(gen1_clean.columns)

X = gen1_clean[feature_columns]
y = gen1_clean[target_column]

print(f"\nFeatures selected: {feature_columns}")
print(f"Target variable: {target_column}")
print(f"Dataset size: {len(X)} samples")

# Step 8: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Step 9: Train the Linear Regression Model
print("\n--- Training Linear Regression Model ---")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")

# Step 10: Make Predictions
print("\n--- Making Predictions ---")
y_pred = model.predict(X_test)
print("Predictions completed!")

# Step 11: Compare Predictions with Actual Values
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Difference': y_test.values - y_pred
})
print("\n--- Sample Predictions vs Actual Values ---")
print(comparison.head(10))

# Step 12: Calculate Model Accuracy Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance Metrics ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Interpretation
if r2 > 0.7:
    print("✓ Model Performance: Good")
elif r2 > 0.5:
    print("✓ Model Performance: Moderate")
else:
    print("✓ Model Performance: Needs Improvement")

# Step 13: Visualize Results - Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Actual vs Predicted Values - Linear Regression Model', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nGraph saved as 'actual_vs_predicted.png'")

# Step 14: Visualize Results - Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.title('Residual Plot', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("Residual plot saved as 'residual_plot.png'")

# ------------------------------------------------------------
# WEEK 3: MODEL EVALUATION AND SUMMARY
# ------------------------------------------------------------

# Step 15: Feature Importance (Coefficients)
print("\n--- Feature Importance (Model Coefficients) ---")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)
print(feature_importance)

# Step 16: Model Summary
print("\n" + "="*60)
print("SOLAR ENERGY FORECASTING - MODEL SUMMARY")
print("="*60)
print(f"Model Type: Linear Regression")
print(f"Number of Features: {len(feature_columns)}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"\nPerformance Metrics:")
print(f"  - R² Score: {r2:.4f}")
print(f"  - RMSE: {rmse:.2f}")
print(f"  - MSE: {mse:.2f}")
print("="*60)

# Step 17: Save Model (Optional - using joblib)
try:
    import joblib
    joblib.dump(model, 'solar_energy_model.pkl')
    print("\n✓ Model saved as 'solar_energy_model.pkl'")
except ImportError:
    print("\n⚠ joblib not installed. Model not saved.")
    print("  Install with: pip install joblib")

# Step 18: Create Results Summary DataFrame
results_summary = pd.DataFrame({
    'Metric': ['R² Score', 'RMSE', 'MSE', 'Training Samples', 'Testing Samples'],
    'Value': [r2, rmse, mse, len(X_train), len(X_test)]
})
results_summary.to_csv('model_results_summary.csv', index=False)
print("✓ Results summary saved as 'model_results_summary.csv'")

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
