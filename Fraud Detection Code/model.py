import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Step 1: Load the dataset
df = pd.read_csv("onlinefraud.csv")  # Make sure this CSV is in the same folder

# Step 2: Encode 'type' column (handle unknown types if needed)
df['type'] = df['type'].map({'CASH_OUT': 0, 'PAYMENT': 1, 'CASH_IN': 2, 'TRANSFER': 3, 'DEBIT': 4})

# Drop rows with missing values after encoding
df = df.dropna()

# Step 3: Select features and label
X = df[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = df['isFraud']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Save the model
with open('payments.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as payments.pkl")
