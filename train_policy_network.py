import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
import time
import sys

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import tf2onnx
    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False


DATA_DIR = "policy_training_data_mp"
MODEL_OUTPUT_DIR = "policy_models"

BATCH_SIZE = 4096
EPOCHS = 100
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01

N_PLANES = 13
INPUT_SHAPE = (8, 8, N_PLANES)
MAX_MOVE_ID = 63 * (64 * 5) + 63 * 5 + 4
NUM_POLICY_OUTPUTS = MAX_MOVE_ID + 1

NUM_RES_BLOCKS = 12
NUM_FILTERS_RES = 192
DROPOUT_RATE = 0.3

EARLY_STOPPING_PATIENCE = 10
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.2
MIN_LR = 1e-7

BEST_MODEL_FILENAME = os.path.join(MODEL_OUTPUT_DIR, "best_policy_value_model.keras")
FINAL_MODEL_FILENAME = os.path.join(MODEL_OUTPUT_DIR, "final_policy_value_model.keras")
ONNX_MODEL_FILENAME = os.path.join(MODEL_OUTPUT_DIR, "policy_value_model.onnx")
HISTORY_PLOT_FILENAME = os.path.join(MODEL_OUTPUT_DIR, "training_history.png")
TENSORBOARD_LOG_DIR = os.path.join(MODEL_OUTPUT_DIR, "logs")

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


def residual_block(x, filters):
    x_skip = x
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, x_skip])
    x = layers.Activation('relu')(x)
    return x


def build_policy_value_network(
    input_shape=INPUT_SHAPE,
    num_res_blocks=NUM_RES_BLOCKS,
    num_filters_res=NUM_FILTERS_RES,
    dropout_rate=DROPOUT_RATE,
    num_policy_outputs=NUM_POLICY_OUTPUTS,
):
    inputs = keras.Input(shape=input_shape, name="board_input")

    x = layers.Conv2D(num_filters_res, 3, padding='same', use_bias=False, name="conv_stem")(inputs)
    x = layers.BatchNormalization(name="bn_stem")(x)
    x = layers.Activation('relu', name="relu_stem")(x)

    for i in range(num_res_blocks):
        x = residual_block(x, num_filters_res)

    policy_conv = layers.Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False, name="policy_conv")(x)
    policy_bn = layers.BatchNormalization(name="policy_bn")(policy_conv)
    policy_relu = layers.Activation('relu', name="policy_relu")(policy_bn)
    policy_flat = layers.Flatten(name="policy_flatten")(policy_relu)
    policy_dropout = layers.Dropout(dropout_rate/2, name="policy_dropout")(policy_flat)
    policy_head = layers.Dense(num_policy_outputs, activation='softmax', name='policy_head')(policy_dropout)

    value_conv = layers.Conv2D(filters=32, kernel_size=1, padding='same', use_bias=False, name="value_conv")(x)
    value_bn = layers.BatchNormalization(name="value_bn")(value_conv)
    value_relu = layers.Activation('relu', name="value_relu")(value_bn)
    value_flat = layers.Flatten(name="value_flatten")(value_relu)
    value_hidden = layers.Dense(256, activation='relu', name='value_dense_hidden')(value_flat)
    value_dropout = layers.Dropout(dropout_rate/2, name="value_dropout")(value_hidden)
    value_head = layers.Dense(1, activation='tanh', name='value_head')(value_dropout)

    model = keras.Model(
        inputs=inputs, outputs=[policy_head, value_head], name="ChessPolicyValueResNet"
    )
    return model


def data_generator(x_path, y_policy_path, y_value_path, indices, batch_size):
    x_mmap = np.load(x_path, mmap_mode="r")
    y_policy_mmap = np.load(y_policy_path, mmap_mode="r")
    y_value_mmap = np.load(y_value_path, mmap_mode="r")

    num_samples = len(indices)
    num_batches = num_samples // batch_size

    while True:
        np.random.shuffle(indices)
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            batch_indices = indices[batch_start:batch_end]
            try:
                batch_x = np.array(x_mmap[batch_indices], dtype=np.float32)
                batch_y_policy = np.array(y_policy_mmap[batch_indices], dtype=np.int32)
                batch_y_value = np.array(y_value_mmap[batch_indices], dtype=np.float32)
                yield (
                    batch_x,
                    {"policy_head": batch_y_policy, "value_head": batch_y_value},
                )
            except IndexError as e:
                print(
                    f"\nWarning: IndexError during batch generation (i={i}, indices={batch_indices}). "
                    f"Check data file integrity and index range. Error: {e}",
                    file=sys.stderr,
                )
                continue
            except Exception as e:
                print(f"\nError during batch generation: {e}", file=sys.stderr)
                raise


def train_model():
    print("Setting up data generators...")
    try:
        X_train_path = os.path.join(DATA_DIR, "X_train.npy")
        y_policy_train_path = os.path.join(DATA_DIR, "y_policy_train.npy")
        y_value_train_path = os.path.join(DATA_DIR, "y_value_train.npy")
        X_val_path = os.path.join(DATA_DIR, "X_val.npy")
        y_policy_val_path = os.path.join(DATA_DIR, "y_policy_val.npy")
        y_value_val_path = os.path.join(DATA_DIR, "y_value_val.npy")

        required_files = [
            X_train_path,
            y_policy_train_path,
            y_value_train_path,
            X_val_path,
            y_policy_val_path,
            y_value_val_path,
        ]
        for p in required_files:
            if not os.path.exists(p):
                print(f"ERROR: Data file not found: {p}", file=sys.stderr)
                print(
                    "Please ensure 'prepare_policy_data.py' was run successfully.",
                    file=sys.stderr,
                )
                sys.exit(1)

        def get_npy_shape(filepath):
            with open(filepath, "rb") as f:
                version = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format._read_array_header(f, version)
                return shape

        shape_train = get_npy_shape(X_train_path)
        shape_val = get_npy_shape(X_val_path)
        train_size = shape_train[0]
        val_size = shape_val[0]

        if shape_train[1:] != INPUT_SHAPE or shape_val[1:] != INPUT_SHAPE:
            print(
                f"ERROR: Data shape mismatch. Expected {INPUT_SHAPE}, "
                f"found Train: {shape_train[1:]}, Val: {shape_val[1:]}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"Found {train_size:,} training samples, {val_size:,} validation samples."
        )
        if train_size == 0 or val_size == 0:
            print(
                "ERROR: Found 0 samples in data files. Check data preparation script.",
                file=sys.stderr,
            )
            sys.exit(1)

        train_indices = np.arange(train_size)
        val_indices = np.arange(val_size)

        output_signature = (
            tf.TensorSpec(
                shape=(None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]),
                dtype=tf.float32,
            ),
            {
                "policy_head": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "value_head": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            },
        )

        print("Creating tf.data.Dataset from generators...")
        train_dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(
                X_train_path,
                y_policy_train_path,
                y_value_train_path,
                train_indices,
                BATCH_SIZE,
            ),
            output_signature=output_signature,
        )

        val_dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(
                X_val_path, y_policy_val_path, y_value_val_path, val_indices, BATCH_SIZE
            ),
            output_signature=output_signature,
        )

        steps_per_epoch = train_size // BATCH_SIZE
        validation_steps = val_size // BATCH_SIZE
        if steps_per_epoch == 0 or validation_steps == 0:
            print(
                f"ERROR: Batch size {BATCH_SIZE} is too large for dataset sizes "
                f"(Train: {train_size}, Val: {val_size}). Please reduce batch size.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}"
        )

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        print("Dataset pipelines created.")

    except Exception as e:
        print(f"ERROR setting up data generators or datasets: {e}", file=sys.stderr)
        sys.exit(1)

    print("Building the ResNet policy-value network model...")
    model = build_policy_value_network()
    model.summary()

    print("Compiling the model...")
    optimizer = keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    model.compile(
        optimizer=optimizer,
        loss={
            "policy_head": keras.losses.SparseCategoricalCrossentropy(),
            "value_head": keras.losses.MeanSquaredError(),
        },
        metrics={
            "policy_head": [
                keras.metrics.SparseCategoricalAccuracy(name="policy_accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
            ],
            "value_head": [keras.metrics.MeanAbsoluteError(name="value_mae")],
        },
        loss_weights={"policy_head": 1.0, "value_head": 1.0},
    )
    print("Model compiled.")

    print("Setting up callbacks...")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_FILENAME,
        save_weights_only=False,
        monitor="val_policy_head_policy_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_policy_head_policy_accuracy",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_policy_head_policy_accuracy",
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_lr=MIN_LR,
        mode="max",
        verbose=1,
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1
    )

    callbacks_list = [
        checkpoint_callback,
        early_stopping,
        reduce_lr,
        tensorboard_callback,
    ]
    print("Callbacks defined.")

    print(f"\n--- Starting Training for up to {EPOCHS} epochs ---")
    start_time = time.time()
    history = None
    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1,
        )
    except Exception as e:
        print(f"\nERROR during model training: {e}", file=sys.stderr)
        sys.exit(1)

    training_time = time.time() - start_time
    print(f"\n--- Training Finished ---")
    print(
        f"Total training time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)"
    )

    print(f"Saving final model to {FINAL_MODEL_FILENAME}...")
    try:
        model.save(FINAL_MODEL_FILENAME)
        print("Final model saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save final model: {e}", file=sys.stderr)

    if TF2ONNX_AVAILABLE:
        print("\nConverting best model to ONNX format...")
        if os.path.exists(BEST_MODEL_FILENAME):
            try:
                print(f"Loading Keras model from {BEST_MODEL_FILENAME} for ONNX conversion...")
                keras_model_to_convert = keras.models.load_model(BEST_MODEL_FILENAME)

                print(f"Converting Keras model to ONNX: {BEST_MODEL_FILENAME} -> {ONNX_MODEL_FILENAME}")
                tf2onnx.convert.from_keras(
                    keras_model_to_convert,
                    output_path=ONNX_MODEL_FILENAME,
                    opset=13
                )
                
                if os.path.exists(ONNX_MODEL_FILENAME):
                    print(f"ONNX model successfully saved to {ONNX_MODEL_FILENAME}")
                else:
                    print("ERROR: ONNX conversion function ran but output file not found.")
            except Exception as e:
                print(f"ERROR during ONNX conversion process: {e}", file=sys.stderr)
        else:
            print(
                f"Skipping ONNX conversion: Best model file '{BEST_MODEL_FILENAME}' not found."
            )
    else:
        print("\nSkipping ONNX conversion: 'tf2onnx' or 'onnxruntime' library not found.")
        print("Install them with: pip install tf2onnx onnxruntime")

    if MATPLOTLIB_AVAILABLE and history is not None:
        print("\nPlotting training history...")
        try:
            plt.figure(figsize=(15, 6))

            plt.subplot(1, 3, 1)
            plt.plot(
                history.history["policy_head_policy_accuracy"], label="Policy Accuracy"
            )
            plt.plot(
                history.history["val_policy_head_policy_accuracy"],
                label="Val Policy Accuracy",
            )
            plt.title("Policy Head Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(history.history["policy_head_loss"], label="Policy Loss")
            plt.plot(history.history["val_policy_head_loss"], label="Val Policy Loss")
            plt.title("Policy Head Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(history.history["value_head_value_mae"], label="Value MAE")
            plt.plot(history.history["val_value_head_value_mae"], label="Val Value MAE")
            plt.title("Value Head Error")
            plt.xlabel("Epoch")
            plt.ylabel("MAE")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(HISTORY_PLOT_FILENAME)
            print(f"Training history plot saved to {HISTORY_PLOT_FILENAME}")

        except Exception as e:
            print(f"\nAn error occurred during plotting: {e}")
    elif not MATPLOTLIB_AVAILABLE:
        print("\nSkipping plotting training history: 'matplotlib' not found.")
        print("Install it with: pip install matplotlib")

    print("\nScript finished.")


if __name__ == "__main__":
    train_model()

