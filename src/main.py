import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt

#  1. UČITAVANJE PODATAKA
data_path = r"C:\Users\zlata\Documents\Projekti\MPVI\MPVIprojekat\instafake-dataset-master\data\fake-v1.0"

fake_df = pd.read_json(os.path.join(data_path, "fakeAccountData.json"))
real_df = pd.read_json(os.path.join(data_path, "realAccountData.json"))

fake_df["isFake"], real_df["isFake"] = 1, 0
df = pd.concat([fake_df, real_df], ignore_index=True).drop_duplicates().fillna(0)

#  2. NORMALIZACIJA PODATAKA
features = ["userFollowerCount", "userFollowingCount", "userBiographyLength",
            "userMediaCount", "userHasProfilPic", "userIsPrivate",
            "usernameDigitCount", "usernameLength"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

X, y = df[features].values, df["isFake"].values

#  3. BALANSIRANJE KLASE
y_unique, y_counts = np.unique(y, return_counts=True)
class_weights = dict(enumerate(compute_class_weight('balanced', classes=y_unique, y=y)))

#  4. PODJELA NA TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#  5. KREIRANJE MODELA
model = Sequential([
    Bidirectional(LSTM(256, return_sequences=True, activation='relu'), input_shape=(1, X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
    Dropout(0.3),
    LSTM(64, activation='relu', return_sequences=False),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

#  6. KOMPILACIJA I CALLBACKS
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

#  7. TRENING MODELA
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=100,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights,
                    callbacks=[early_stopping, reduce_lr])

model.save(r"C:\Users\zlata\Documents\Projekti\MPVI\model_6")


# 8. EVALUACIJA
results = model.evaluate(X_test, y_test)
print("\nTest Loss and Accuracy:", results)

# 9. PREDIKCIJE I METRIKE
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_pred):.4f}')
print(f'F1 score: {f1_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')

#  10. VIZUALIZACIJA TRENIRANJA
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')
plt.show()
