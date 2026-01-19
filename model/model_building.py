import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load Dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

# 2. Select 6 features & Target
selected_features = ['alcohol', 'magnesium', 'flavanoids', 'color_intensity', 'hue', 'proline']
X = df[selected_features]
y = data.target

# 3. Scale and Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 4. Save files to the Colab environment
joblib.dump(model, 'wine_cultivar_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler have been trained and saved!")