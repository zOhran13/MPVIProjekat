import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt

# 🔹 1. UČITAVANJE PODATAKA (ISTI DATASET KAO PRI TRENIRANJU)
data_path = r"C:\Users\zlata\Documents\Projekti\MPVI\MPVIprojekat\instafake-dataset-master\data\fake-v1.0"

fake_df = pd.read_json(os.path.join(data_path, "fakeAccountData.json"))
real_df = pd.read_json(os.path.join(data_path, "realAccountData.json"))

fake_df["isFake"], real_df["isFake"] = 1, 0
df = pd.concat([fake_df, real_df], ignore_index=True).drop_duplicates().fillna(0)

# 🔹 2. NORMALIZACIJA PODATAKA
features = ["userFollowerCount", "userFollowingCount", "userBiographyLength",
            "userMediaCount", "userHasProfilPic", "userIsPrivate",
            "usernameDigitCount", "usernameLength"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

X, y = df[features].values, df["isFake"].values

# 🔹 3. PODJELA NA TRAIN/TEST (ISTE POSTAVKE KAO U `train_model.py`)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 4. PRIPREMA TEST PODATAKA ZA LSTM
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# 🔹 5. UČITAVANJE SPREMLJENOG MODELA
model_path = r"C:\Users\zlata\Documents\Projekti\MPVI\model_4"
model = load_model(model_path)
print(f"✅ Učitani model iz {model_path}")

# 🔹 6. TESTIRANJE MODELA
results = model.evaluate(X_test, y_test)
print("\nTest Loss and Accuracy:", results)

# 🔹 7. PREDIKCIJE I METRIKE
y_pred = (model.predict(X_test) > 0.5).astype(int)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_pred):.4f}')
print(f'F1 score: {f1_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')

# 🔹 8. VIZUALIZACIJA
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy", "Recall", "Precision", "F1 Score"],
        [accuracy_score(y_test, y_pred), recall_score(y_test, y_pred),
         precision_score(y_test, y_pred), f1_score(y_test, y_pred)],
        color=['blue', 'green', 'orange', 'red'])
plt.ylabel("Score")
plt.title("Evaluacija modela na test podacima")
plt.ylim(0, 1)
plt.show()
