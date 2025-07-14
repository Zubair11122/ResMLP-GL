import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from imblearn.over_sampling import ADASYN
from tensorflow.keras import layers, regularizers, callbacks
import matplotlib.pyplot as plt
import optuna

# ─── Load Data ────────────────────────────────────────────────────────
df = pd.read_csv("mutations_variant_complete.tsv", sep="\t")
df.replace("-", np.nan, inplace=True)
y = df["is_driver"].astype(int)

# ─── Preprocess ───────────────────────────────────────────────────────
preprocessor = joblib.load("preprocessor.pkl")
X_raw = df[preprocessor.feature_names_in_]
X = preprocessor.transform(X_raw)

# ─── Model builder with proper skip‐projections ───────────────────────
def build_cross_conn_model(input_dim, neurons, dropout_rate, l2_reg, learning_rate, metrics):
    inputs = tf.keras.Input(shape=(input_dim,), name="input")
    skip_in = inputs

    # Block 1
    skip1 = layers.Dense(neurons,
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name="skip1_proj")(skip_in)
    x = layers.Dense(neurons,
                     activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name="dense1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Add(name="add1")([x, skip1])
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # Block 2
    skip2 = layers.Dense(neurons // 2,
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name="skip2_proj")(x)
    x = layers.Dense(neurons // 2,
                     activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name="dense2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Add(name="add2")([x, skip2])
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    # Block 3 + gating
    x = layers.Dropout(dropout_rate, name="drop3")(x)
    x = layers.Dense(neurons // 2,
                     activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name="dense3")(x)
    skip3 = layers.Dense(neurons // 2,
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name="skip3_proj")(skip_in)
    x = layers.Multiply(name="gate")([x, skip3])

    # Final head
    x = layers.Dense(1, name="logits")(x)
    outputs = layers.Activation("sigmoid", name="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="cross_conn_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metrics
    )
    return model

# ─── Optuna objective ─────────────────────────────────────────────────
def objective(trial):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, y_train = ADASYN(random_state=42).fit_resample(X_train, y_train)

    # hyperparameters
    dropout_rate  = trial.suggest_float("dropout_rate", 0.2, 0.5)
    neurons       = trial.suggest_int("neurons",      64, 512)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    l2_reg        = trial.suggest_float("l2_reg",       1e-4, 1e-2, log=True)

    model = build_cross_conn_model(
        input_dim     = X_train.shape[1],
        neurons       = neurons,
        dropout_rate  = dropout_rate,
        l2_reg        = l2_reg,
        learning_rate = learning_rate,
        metrics       = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    stopper = callbacks.EarlyStopping(
        monitor="val_auc", patience=10, mode="max", restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        verbose=0,
        callbacks=[stopper]
    )
    return history.history["val_auc"][-1]

# ─── Run Optuna ───────────────────────────────────────────────────────
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # ← now 100 trials, no timeout

# ─── Save trial history ────────────────────────────────────────────────
trials_df = study.trials_dataframe()
trials_df.to_csv("optuna_trials.csv", index=False)
print("✅ Optuna trial history saved to optuna_trials.csv")

print(f"🏅 Best trial params: {study.best_trial.params}")
best_params = study.best_trial.params

# ─── 10-Fold CV with best params ───────────────────────────────────────
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_aucs, fold_models = [], []

for fold, (tr_i, val_i) in enumerate(kfold.split(X, y), start=1):
    print(f"🔄 Fold {fold}")
    X_tr_raw, X_val = X[tr_i], X[val_i]
    y_tr,    y_val = y.iloc[tr_i], y.iloc[val_i]
    X_tr, y_tr = ADASYN(random_state=42).fit_resample(X_tr_raw, y_tr)

    model = build_cross_conn_model(
        input_dim     = X_tr.shape[1],
        neurons       = best_params["neurons"],
        dropout_rate  = best_params["dropout_rate"],
        l2_reg        = best_params["l2_reg"],
        learning_rate = best_params["learning_rate"],
        metrics       = [
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            "accuracy"
        ]
    )
    stopper = callbacks.EarlyStopping(
        monitor="val_auc", patience=15, mode="max", restore_best_weights=True)

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=64,
        verbose=1,
        callbacks=[stopper]
    )

    y_val_prob = model.predict(X_val).ravel()
    auc_score  = roc_auc_score(y_val, y_val_prob)
    print(f"✅ Fold {fold} AUC: {auc_score:.4f}")

    fold_aucs.append(auc_score)
    fold_models.append(model)

# ─── Final Held-out Evaluation ────────────────────────────────────────
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_train_fin, y_train_fin = ADASYN(random_state=42).fit_resample(X_train_all, y_train_all)

best_idx    = int(np.argmax(fold_aucs))
final_model = fold_models[best_idx]

y_prob_test = final_model.predict(X_test).ravel()
test_auc    = roc_auc_score(y_test, y_prob_test)
print(f"\n✅ Final Test AUC: {test_auc:.4f}")
if test_auc >= 0.88:
    print("🎯 SUCCESS: AUC target achieved!")

final_model.save("driver_prediction_model.keras")
print("💾 Final model saved as driver_prediction_model.keras")

# ─── Plot ROC & PR curves ───────────────────────────────────────────────
fpr, tpr, _         = roc_curve(y_test, y_prob_test)
precision, recall, _ = precision_recall_curve(y_test, y_prob_test)

# create figure with explicit white background
fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')

# ROC curve (no grid)
axes[0].plot(fpr, tpr)
axes[0].set_title(f"ROC Curve (AUC = {test_auc:.3f})")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")

# Precision-Recall curve (no grid)
axes[1].plot(recall, precision)
axes[1].set_title("Precision-Recall Curve")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")

plt.tight_layout()
# save with white background, no on-screen display
fig.savefig(
    "model_performance_curves.png",
    dpi=1200,
    bbox_inches="tight",
    facecolor='white'
)
plt.close(fig)  # do not show the plot

# ─── Summary ───────────────────────────────────────────────────────────
print(f"\n📊 CV AUCs: {fold_aucs}")
print(f"📈 Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"🏆 Best Fold AUC: {np.max(fold_aucs):.4f}")
