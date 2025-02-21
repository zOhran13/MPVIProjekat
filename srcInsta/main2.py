import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. UČITAVANJE PODATAKA
data_path = r"..\twitterDataset.csv"
df = pd.read_csv(data_path)

# 2️. ČIŠĆENJE PODATAKA
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# 3️. PODJELA NA FEATURES I TARGET
y = df["class_bot"]
X = df.drop("class_bot", axis=1)

# 4️. NORMALIZACIJA PODATAKA
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 5️. PREOBLIKOVANJE ZA LSTM
X = X.reshape(X.shape[0], 1, X.shape[1])

# 6️. PODJELA NA TRAIN/VAL/TEST (70-20-10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)  # 20% val, 10% test

print(f'Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}')

# 7️. KREIRANJE BIDIRECTIONAL LSTM MODELA
model = Sequential([
    Bidirectional(LSTM(256, return_sequences=True, activation='relu'), input_shape=(1, X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
    Dropout(0.3),

    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.3),

    LSTM(32, activation='relu', return_sequences=False),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

# 8️. KOMPILACIJA I CALLBACKS
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

# 9️. TRENING MODELA
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=100,
                    validation_data=(X_val, y_val),  # Validacija na 20% podataka
                    callbacks=[early_stopping, reduce_lr])

# 10. ČUVANJE MODELA
model.save(r"..\models\model2_twitter.h5")

# 1️1. EVALUACIJA MODELA NA TESTNOM SETU
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 1️2️. PREDIKCIJE I METRIKE
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nKlasifikacijski izvještaj:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# 1️3️. VIZUALIZACIJA REZULTATA
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')
plt.show()

