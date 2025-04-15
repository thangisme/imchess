import tensorflow
from tensorflow import keras
import numpy as np
import os
import time

PROCESSED_DATA_FILE = "lichess_data_processed.npz"
SAVED_MODEL_FILE = "chess_eval.keras"

HIDDEN_LAYER_1_NEURONS = 256
HIDDEN_LAYER_2_NEURONS = 128

LEARNING_RATE = 0.001

BATCH_SIZE = 1024
EPOCHS = 20
VALIDATION_SPLIT = 0.15

if not os.path.exists(PROCESSED_DATA_FILE):
    print(f"ERROR: Processed data file not found")
    exit()

data = np.load(PROCESSED_DATA_FILE)
x_data = data['X']
y_data = data['y']

print("Data loaded")
print(f"Shape of x (board vectors): {x_data.shape}")
print(f"Shape of y (board vectors): {y_data.shape}")

if x_data.shape[0] != y_data.shape[0]:
    print("ERROR: mismatch between number of x and y")
    exit()
if x_data.shape[0] == 0:
    print("ERROR: NO data found")
    exit()

input_dimension = x_data.shape[1]
print(f"features per board: {input_dimension}")

layers = keras.layers
model = keras.Sequential(
    [
        layers.Input(shape=(input_dimension,), name="Board_Input"),
        layers.Dense(HIDDEN_LAYER_1_NEURONS, activation="relu", name="Hidden_layer_1"),
        layers.Dense(HIDDEN_LAYER_2_NEURONS, activation="relu", name="Hidden_layer_2"),
        layers.Dense(1, activation="tanh", name="Evaluation_Output")
    ],
    name="Chess_Evaluator"
)

model.summary()

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss = keras.losses.MeanSquaredError(),
)
print("Model compiled")

print("Starting model training")
print(f"Dataset size: {x_data.shape[0]} samples")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Validation split: {VALIDATION_SPLIT*100:.1f}%")

training_start_time = time.time()

history = model.fit(
    x_data,
    y_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    shuffle=True,
    verbose=1
)
training_duration = time.time() - training_start_time

print(f"Model training taken: {training_duration:.2f} s")

model.save(SAVED_MODEL_FILE)
print("Model saved")
