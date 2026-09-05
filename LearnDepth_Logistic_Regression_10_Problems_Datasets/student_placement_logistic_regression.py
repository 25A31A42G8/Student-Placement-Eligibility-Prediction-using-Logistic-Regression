import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve
)

# ============================================================
# STUDENT PLACEMENT ELIGIBILITY - LOGISTIC REGRESSION
# ============================================================

# -----------------------------
# Configuration
# -----------------------------

DATASET_PATH = "datasets/dataset_07_student_placement_eligibility.csv"
OUTPUT_DIR = "student_placement_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------

print("=" * 70)
print("STUDENT PLACEMENT ELIGIBILITY - LOGISTIC REGRESSION")
print("=" * 70)

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset not found.")
    print("Make sure this script is placed in the main extracted folder.")
    print("Expected path:")
    print(DATASET_PATH)
    input("\nPress Enter to exit...")
    raise SystemExit

df = pd.read_csv(DATASET_PATH)

print("\n1. DATASET INFORMATION")
print("-" * 70)

print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

print("\nColumns:")
for col in df.columns:
    print(f"  - {col}")

print("\nFirst 5 rows:")
print(df.head())

# -----------------------------
# Data Quality Checks
# -----------------------------

print("\n\n2. DATA QUALITY CHECK")
print("-" * 70)

missing_values = df.isnull().sum()
duplicate_count = df.duplicated().sum()

print("\nMissing values:")
print(missing_values)

print(f"\nDuplicate rows: {duplicate_count}")

print("\nData types:")
print(df.dtypes)

# -----------------------------
# Target Distribution
# -----------------------------

target_column = "target"

print("\n\n3. TARGET / CLASS BALANCE")
print("-" * 70)

class_counts = df[target_column].value_counts().sort_index()

print(class_counts)

print("\nClass percentages:")
print((df[target_column].value_counts(normalize=True) * 100).round(2))

# Save class distribution
class_distribution = pd.DataFrame({
    "Class": class_counts.index,
    "Count": class_counts.values,
    "Percentage": [
        round((class_counts[c] / len(df)) * 100, 2)
        for c in class_counts.index
    ]
})

class_distribution.to_csv(
    os.path.join(OUTPUT_DIR, "class_distribution.csv"),
    index=False
)

# -----------------------------
# Feature / Target Separation
# -----------------------------

X = df.drop(columns=[target_column])
y = df[target_column]

print("\n\n4. FEATURES")
print("-" * 70)

print("Input features:")
for feature in X.columns:
    print(f"  - {feature}")

print(f"\nTarget variable: {target_column}")

# -----------------------------
# Train / Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n\n5. TRAIN / TEST SPLIT")
print("-" * 70)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")
print("Test size       : 20%")
print("Random state    : 42")
print("Stratification  : Yes")

# -----------------------------
# Logistic Regression Pipeline
# -----------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("\n\n6. MODEL TRAINING")
print("-" * 70)

print("Model: Logistic Regression")
print("Preprocessing: StandardScaler")
print("Maximum iterations: 1000")

model.fit(X_train, y_train)

print("Training completed successfully.")

# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Evaluation Metrics
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n\n7. MODEL PERFORMANCE")
print("-" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_metrics.csv"),
    index=False
)

# -----------------------------
# Classification Report
# -----------------------------

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w"
) as f:
    f.write(report)

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n\n8. CONFUSION MATRIX")
print("-" * 70)

print(cm)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

cm_df.to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
)

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Eligible", "Eligible"]
).plot(ax=ax)

plt.title("Confusion Matrix - Student Placement Eligibility")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.close()

# -----------------------------
# ROC Curve
# -----------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Student Placement Eligibility")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve.png"),
    dpi=300
)

plt.close()

# -----------------------------
# Feature Coefficients
# -----------------------------

logistic_model = model.named_steps["logistic_regression"]

coefficients = logistic_model.coef_[0]

coefficient_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefficients,
    "Absolute_Coefficient": np.abs(coefficients)
})

coefficient_df = coefficient_df.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

print("\n\n9. FEATURE COEFFICIENTS")
print("-" * 70)

print(coefficient_df.to_string(index=False))

coefficient_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_coefficients.csv"),
    index=False
)

# -----------------------------
# Coefficient Plot
# -----------------------------

plt.figure(figsize=(10, 6))

sorted_coefficients = coefficient_df.sort_values(
    "Coefficient"
)

plt.barh(
    sorted_coefficients["Feature"],
    sorted_coefficients["Coefficient"]
)

plt.axvline(
    x=0,
    linestyle="--"
)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")
plt.title("Feature Coefficients - Student Placement Eligibility")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "feature_coefficients.png"),
    dpi=300
)

plt.close()

# -----------------------------
# Predictions File
# -----------------------------

predictions_df = X_test.copy()

predictions_df["Actual_Target"] = y_test.values
predictions_df["Predicted_Target"] = y_pred
predictions_df["Placement_Probability"] = y_probability

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False
)

# -----------------------------
# Generate Final Report
# -----------------------------

report_path = os.path.join(
    OUTPUT_DIR,
    "final_report.txt"
)

with open(report_path, "w") as f:

    f.write("=" * 70 + "\n")
    f.write("STUDENT PLACEMENT ELIGIBILITY USING LOGISTIC REGRESSION\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. PROBLEM STATEMENT\n")
    f.write("-" * 70 + "\n")
    f.write(
        "Predict whether a student is likely to meet a placement "
        "eligibility criterion using Logistic Regression.\n\n"
    )

    f.write("2. DATASET DESCRIPTION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Number of records: {len(df)}\n")
    f.write(f"Number of features: {len(X.columns)}\n")
    f.write("Target variable: target\n\n")

    f.write("Features:\n")
    for feature in X.columns:
        f.write(f"- {feature}\n")

    f.write("\n3. DATA QUALITY\n")
    f.write("-" * 70 + "\n")
    f.write(f"Missing values: {int(missing_values.sum())}\n")
    f.write(f"Duplicate rows: {duplicate_count}\n\n")

    f.write("4. PREPROCESSING\n")
    f.write("-" * 70 + "\n")
    f.write(
        "All input features are numerical. StandardScaler was used "
        "to standardize the feature values before Logistic Regression.\n"
    )
    f.write(
        "The preprocessing and model were combined using a Scikit-learn "
        "Pipeline to prevent data leakage.\n\n"
    )

    f.write("5. TRAIN / TEST SPLIT\n")
    f.write("-" * 70 + "\n")
    f.write("80% training data and 20% testing data.\n")
    f.write("Random state: 42\n")
    f.write("Stratified split: Yes\n\n")

    f.write("6. MODEL\n")
    f.write("-" * 70 + "\n")
    f.write("Algorithm: Logistic Regression\n")
    f.write("Maximum iterations: 1000\n\n")

    f.write("7. PERFORMANCE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {roc_auc:.4f}\n\n")

    f.write("8. CONFUSION MATRIX\n")
    f.write("-" * 70 + "\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("9. FEATURE INTERPRETATION\n")
    f.write("-" * 70 + "\n")

    for _, row in coefficient_df.iterrows():
        direction = "positive" if row["Coefficient"] > 0 else "negative"

        f.write(
            f"{row['Feature']}: coefficient = "
            f"{row['Coefficient']:.4f} ({direction} association)\n"
        )

    f.write("\n10. LIMITATIONS\n")
    f.write("-" * 70 + "\n")
    f.write(
        "The dataset is synthetic and intended for educational practice. "
        "Therefore, model performance may not represent performance on "
        "real-world student placement data. Real-world deployment would "
        "require validation using representative data, additional feature "
        "analysis, fairness checks, and monitoring.\n\n"
    )

    f.write("11. CONCLUSION\n")
    f.write("-" * 70 + "\n")
    f.write(
        "A Logistic Regression classifier was developed to predict student "
        "placement eligibility. The model was evaluated using accuracy, "
        "precision, recall, F1 score, ROC-AUC and a confusion matrix. "
        "The results provide an educational demonstration of binary "
        "classification using Logistic Regression.\n"
    )

# -----------------------------
# Final Output
# -----------------------------

print("\n\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nAll results have been saved to:")
print(os.path.abspath(OUTPUT_DIR))

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  ✓ {filename}")

print("\nYou can now open the 'student_placement_results' folder.")

input("\nPress Enter to exit...")
