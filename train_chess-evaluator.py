import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

BOARD_MEMMAP_FILENAME = "board_states.dat"
SCORE_MEMMAP_FILENAME = "eval_scores.dat"
VALID_POSITIONS_COUNT_FILE = "valid_positions_count.txt"

BATCH_SIZE = 1024
EPOCHS = 20
VALIDATION_SPLIT_FRACTION = 0.15
N_PLANES = 13

if not os.path.exists(VALID_POSITIONS_COUNT_FILE):
    print("ERROR: count file not found")
    exit(1)
total_number_of_positions = int(open(VALID_POSITIONS_COUNT_FILE).read())

board_states = np.memmap(
    BOARD_MEMMAP_FILENAME,
    mode="r",
    dtype=np.uint8,
    shape=(total_number_of_positions, 8, 8, N_PLANES),
)
evaluation_scores = np.memmap(
    SCORE_MEMMAP_FILENAME,
    mode="r",
    dtype=np.float16,
    shape=(total_number_of_positions,),
)

dataset_all_positions = tf.data.Dataset.from_tensor_slices(
    (board_states, evaluation_scores)
)

number_of_validation_positions = int(
    total_number_of_positions * VALIDATION_SPLIT_FRACTION
)
number_of_training_positions = (
    total_number_of_positions - number_of_validation_positions
)

dataset_shuffled_once = dataset_all_positions.shuffle(
    buffer_size=total_number_of_positions
).repeat()

validation_raw_dataset = dataset_shuffled_once.take(number_of_validation_positions)
training_raw_dataset = dataset_shuffled_once.skip(number_of_validation_positions)


def cast_to_float32(board_uint8, score_float16):
    board_float32 = tf.cast(board_uint8, tf.float32)
    score_float32 = tf.cast(score_float16, tf.float32)
    return board_float32, score_float32


training_dataset = (
    training_raw_dataset.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = (
    validation_raw_dataset.map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


def build_chess_evaluator_model(input_shape=(8, 8, 13)):
    inputs = keras.Input(shape=input_shape, name="board_input")
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="tanh", name="eval_output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ChessEvaluator")
    return model


model = build_chess_evaluator_model()
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
)
model.summary()

model.fit(
    training_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=1
)

model.save("chess_eval.keras")
print("Model saved to chess_eval.keras")
