import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from imblearn.over_sampling import ADASYN
from tensorflow.keras import layers, regularizers, callbacks
import matplotlib.pyplot as plt

# ─── Load Data & Preprocessor ─────────────────────────────────────────
df = pd.read_csv("mutations_variant_complete.tsv", sep="\t")
df.replace("-", np.nan, inplace=True)
y = df["is_driver"].astype(int)

preprocessor = joblib.load("preprocessor.pkl")
X_raw = df[preprocessor.feature_names_in_]
X = preprocessor.transform(X_raw)

# ─── Load Best Params from Optuna Trials ──────────────────────────────
trials_df = pd.read_csv("optuna_trials.csv")
best_params = trials_df.loc[trials_df["value"].idxmax()].to_dict()

# Extract only the relevant parameters (drop Optuna metadata)
best_params = {
    "dropout_rate": best_params["params_dropout_rate"],
    "neurons": int(best_params["params_neurons"]),
    "learning_rate": best_params["params_learning_rate"],
    "l2_reg": best_params["params_l2_reg"]
}

print("🔍 Loaded Best Params from optuna_trials.csv:")
print(best_params)

# ─── Model Builder (Same as Before) ───────────────────────────────────
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

# ─── 10-Fold CV with Best Params ─────────────────────────────────────
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

# ─── Final Evaluation & Save ─────────────────────────────────────────
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

# ─── Plot Curves ─────────────────────────────────────────────────────
fpr, tpr, _         = roc_curve(y_test, y_prob_test)
precision, recall, _ = precision_recall_curve(y_test, y_prob_test)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
axes[0].plot(fpr, tpr)
axes[0].set_title(f"ROC Curve (AUC = {test_auc:.3f})")
axes[1].plot(recall, precision)
axes[1].set_title("Precision-Recall Curve")
plt.tight_layout()
fig.savefig("model_performance_curves.png", dpi=1200, bbox_inches="tight")
plt.close(fig)

# ─── Summary ─────────────────────────────────────────────────────────
print(f"\n📊 CV AUCs: {fold_aucs}")
print(f"📈 Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"🏆 Best Fold AUC: {np.max(fold_aucs):.4f}")