import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, f1_score, accuracy_score
)

# 1. Load the models
dl_driver_model = load_model('dl_driver_auc90_model.keras')
tabtransformer_model = load_model('tabtransformer_final.keras')

# 2. Load the preprocessor
with open('preprocessor1.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# 3. Load your raw test data (unseen data)
# Example: You would load your raw test data here, for example:
# raw_test_data = pd.read_csv('path_to_your_raw_test_data.csv')

# 4. Preprocess the raw test data
X_test = preprocessor.transform(raw_test_data)  # Preprocess using the saved preprocessor

# 5. Generate model predictions
p_test_tabtransformer = tabtransformer_model.predict(X_test)[:, 1]  # Get probabilities for the positive class
p_test_dl_driver = dl_driver_model.predict(X_test)[:, 1]  # Get probabilities for the positive class

# 6. Combine predictions (average of both models)
p_test = (p_test_tabtransformer + p_test_dl_driver) / 2

# 7. Set the threshold for classification (e.g., 0.5, but you can adjust based on your needs)
best_thresh = 0.5
y_pred_test = (p_test > best_thresh).astype(int)

# 8. If you have the true labels for your test data, you can calculate performance metrics
# Example:
# y_test = <your_actual_test_labels_here>  # Replace with the actual labels for your test data

# If you have the true labels, evaluate the performance metrics
if 'y_test' in globals():  # Check if y_test is available
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, p_test)

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, p_test)

    # Plot ROC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, p_test):.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve_testset.png", dpi=300)
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 5))
    plt.plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_test, p_test):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_curve_testset.png", dpi=300)
    plt.show()

    # Final Performance Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_test),
        "F1": f1_score(y_test, y_pred_test),
        "ROC-AUC": roc_auc_score(y_test, p_test),
        "PR-AUC": average_precision_score(y_test, p_test)
    }

    plt.figure(figsize=(7, 4))
    plt.bar(metrics.keys(), metrics.values(), color="skyblue")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Model Performance Metrics")
    plt.tight_layout()
    plt.savefig("model_performance_bar.png", dpi=300)
    plt.show()

else:
    print("y_test (true labels) is not available. Please ensure you have the true labels for evaluation.")
