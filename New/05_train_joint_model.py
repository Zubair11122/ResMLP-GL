#!/usr/bin/env python3
# 06_train_joint_model_server.py — stable GPU training (GBM+COAD) using preprocessed matrices
# Mirrors 05_train_nn_safe.py style (no Optuna, no ADASYN; class weights instead)

import os
# --- Stability switches (set before TF import) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# Non-interactive plotting for servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- PATHS (match your server) -----
BASE = "/mnt/820f42a7-6768-4c07-a318-b6345e4826df/zubei/rep_error_project/out"
X_TRAIN_PATH = os.path.join(BASE, "X_train_proc.tsv")
X_TEST_PATH  = os.path.join(BASE, "X_test_proc.tsv")
Y_TRAIN_PATH = os.path.join(BASE, "y_train.tsv")
Y_TEST_PATH  = os.path.join(BASE, "y_test.tsv")
MODEL_OUT    = os.path.join(BASE, "driver_prediction_model.keras")
PLOT_OUT     = os.path.join(BASE, "model_performance_curves.png")
PRED_OUT     = os.path.join(BASE, "test_predictions.tsv")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set memory growth:", e)
print("GPUs visible to TF:", tf.config.list_physical_devices('GPU'))

# ----- LOAD DATA (float32 for TF) -----
X_train = pd.read_csv(X_TRAIN_PATH, sep="\t").astype("float32").values
X_test  = pd.read_csv(X_TEST_PATH,  sep="\t").astype("float32").values
y_train = pd.read_csv(Y_TRAIN_PATH, sep="\t").iloc[:,0].values.astype("float32")
y_test  = pd.read_csv(Y_TEST_PATH,  sep="\t").iloc[:,0].values.astype("float32")

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train pos rate: {y_train.mean():.3f}, y_test pos rate: {y_test.mean():.3f}")

# ----- CLASS WEIGHTS (replace ADASYN) -----
neg = float((y_train == 0).sum()); pos = float((y_train == 1).sum())
class_weight = {0: 1.0, 1: (neg / max(pos, 1.0))}
print(f"Class weights -> {class_weight}")

# ----- MODEL: Cross-connection style but simplified & stable -----
def build_cross_conn_model(input_dim, neurons=256, dropout_rate=0.3, l2_reg=1e-3, lr=1e-4):
    inputs = layers.Input(shape=(input_dim,), name="input")
    skip_in = inputs

    # Block 1
    skip1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2_reg), name="skip1")(skip_in)
    x = layers.Dense(neurons, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Add(name="add1")([x, skip1])
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # Block 2
    skip2 = layers.Dense(neurons // 2, kernel_regularizer=regularizers.l2(l2_reg), name="skip2")(x)
    x = layers.Dense(neurons // 2, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Add(name="add2")([x, skip2])
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    # Block 3 + gating via input projection
    x = layers.Dropout(dropout_rate, name="drop3")(x)
    x = layers.Dense(neurons // 2, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense3")(x)
    gate = layers.Dense(neurons // 2, activation="sigmoid", kernel_regularizer=regularizers.l2(l2_reg), name="gate_proj")(skip_in)
    x = layers.Multiply(name="gated")([x, gate])

    outputs = layers.Dense(1, activation="sigmoid", name="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="cross_conn_model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 "accuracy"]
    )
    return model

model = build_cross_conn_model(input_dim=X_train.shape[1], neurons=256, dropout_rate=0.3, l2_reg=1e-3, lr=1e-4)
es = callbacks.EarlyStopping(monitor="val_auc", patience=12, mode="max", restore_best_weights=True)

# Train with validation split (keep test untouched)
hist = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=120,
    batch_size=32,            # safe default; try 64/128 if GPU RAM allows
    callbacks=[es],
    verbose=1,
    class_weight=class_weight
)

# ----- EVALUATION -----
y_prob = model.predict(X_test, batch_size=256).ravel().astype("float32")
auc = roc_auc_score(y_test, y_prob)
ap  = average_precision_score(y_test, y_prob)
print(f"\n✅ Final Test AUROC: {auc:.4f} | AUPRC: {ap:.4f}")

# Save predictions & model
pd.DataFrame({"y_true": y_test, "y_prob": y_prob}).to_csv(PRED_OUT, sep="\t", index=False)
model.save(MODEL_OUT)
print(f"🧠 saved {MODEL_OUT}")

# ----- PLOTS -----
fpr, tpr, _ = roc_curve(y_test, y_prob)
prec, rec, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={auc:.3f})")
plt.subplot(1, 2, 2)
plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={ap:.3f})")
plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=900, bbox_inches="tight")
print(f"📈 saved {PLOT_OUT}")
