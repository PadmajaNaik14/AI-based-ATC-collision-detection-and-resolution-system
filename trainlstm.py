import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("flights.csv")
df = df.sort_values(["flight_id", "timestep"]).reset_index(drop=True)

FEATURES = ["latitude", "longitude", "altitude", "speed", "heading"]
SEQ_LEN = 10
FUTURE_STEPS = 20

# ------------------------------
# Compute deltas
# ------------------------------
df_list = []
for fid, g in df.groupby("flight_id"):
    g = g.copy()
    for f in FEATURES:
        g["d_" + f] = g[f].diff().fillna(0)
    df_list.append(g)

df = pd.concat(df_list, ignore_index=True)

# ------------------------------
# Build sequences
# ------------------------------
X_list, Y_list = [], []

for fid, g in df.groupby("flight_id"):
    arrX = g[FEATURES].values
    arrY = g[["d_" + f for f in FEATURES]].values

    if len(arrX) < SEQ_LEN + FUTURE_STEPS:
        continue

    for i in range(len(arrX) - SEQ_LEN - FUTURE_STEPS):
        X_list.append(arrX[i:i+SEQ_LEN])
        Y_list.append(arrY[i+SEQ_LEN:i+SEQ_LEN+FUTURE_STEPS].flatten())

X = np.array(X_list)
Y = np.array(Y_list)

print("Samples:", X.shape, Y.shape)

# ------------------------------
# Scale data
# ------------------------------
flat_X = X.reshape(-1, len(FEATURES))
flat_Y = Y.reshape(-1, len(FEATURES))

scaler_X = MinMaxScaler().fit(flat_X)
scaler_Y = MinMaxScaler().fit(flat_Y)

X_scaled = scaler_X.transform(flat_X).reshape(X.shape)
Y_scaled = scaler_Y.transform(flat_Y).reshape(Y.shape)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler_X, "models/scaler_X.save")
joblib.dump(scaler_Y, "models/scaler_Y.save")

# ------------------------------
# Train/Val split
# ------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, Y_scaled, test_size=0.15, random_state=42
)

# ------------------------------
# Build Model
# ------------------------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(FEATURES))),
    Dropout(0.2),
    LSTM(128),
    Dense(256, activation="relu"),
    Dense(FUTURE_STEPS * len(FEATURES))
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ------------------------------
# Train
# ------------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=128,
    verbose=1
)

model.save("models/delta_multi_lstm.h5")
print("✅ Saved model to models/delta_multi_lstm.h5")
