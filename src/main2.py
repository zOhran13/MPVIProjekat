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
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt

#  1. UČITAVANJE PODATAKA
data_path = r"C:\Users\zlata\Documents\Projekti\MPVI\twitterDataset.csv"
df = pd.read_csv(data_path)

#  2. ČIŠĆENJE PODATAKA
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

#  3. PODJELA NA FEATURES I TARGET
y = df["class_bot"]
X = df.drop("class_bot", axis=1)

#  4. NORMALIZACIJA PODATAKA
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#  5. PREOBLIKOVANJE ZA LSTM
X = X.reshape(X.shape[0], 1, X.shape[1])

#  6. PODJELA NA TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  7. KREIRANJE BIDIRECTIONAL LSTM MODELA
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

#  8. KOMPILACIJA I CALLBACKS
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

#  9. TRENING MODELA
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=100,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

#  10. ČUVANJE MODELA
model.save(r"C:\Users\zlata\Documents\Projekti\MPVI\model1_twitter.h5")

# 11. EVALUACIJA MODELA
results = model.evaluate(X_test, y_test)
print("\nTest Loss and Accuracy:", results)

#  12. PREDIKCIJE I METRIKE
y_pred = (model.predict(X_test) > 0.4).astype(int)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_pred):.4f}')
print(f'F1 score: {f1_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')

#  13. VIZUALIZACIJA REZULTATA
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')
plt.show()
