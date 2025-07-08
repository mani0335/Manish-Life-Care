import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the trained model
model_path = "models/svc.pkl"  # Adjust path if needed
svc = pickle.load(open('models/.ipynb_checkpoints/svc.pkl', "rb"))

# Load test dataset
test_data_path = "datasets/test_data.csv"  # Adjust path if needed
df_test = pd.read_csv(test_data_path)

# Drop unnecessary columns (like 'Unnamed: 0' if present)
df_test = df_test.drop(columns=["Unnamed: 0"], errors="ignore")

# Ensure test features match training features
train_features = svc.feature_names_in_  # Get feature names used during training
df_test = df_test.reindex(columns=train_features, fill_value=0)  # Align columns

# Separate features and labels
X_test = df_test.drop(columns=["Disease"])  # Adjust based on dataset
y_test = df_test["Disease"]

# Make predictions
y_pred = svc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
