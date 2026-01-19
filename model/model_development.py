import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# 1. Load Dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

# 2. Feature Selection (Selecting exactly 6 features as required)
selected_features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'flavanoids']
X = df[selected_features]
y = data.target

# 3. Scaling (Mandatory)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Implement Random Forest (One of the permitted algorithms)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 6. Save Model and Scaler into the /model/ folder
if not os.path.exists('model'): os.makedirs('model')

with open('model/wine_cultivar_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print("Project files saved in /model/ directory.")