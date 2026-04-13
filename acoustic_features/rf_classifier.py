# =============================================================================
# Random Forest Classifier - Acoustic Features
# SONIVA PAPER - Clinical Language Cognition Lab
#
# Binary classification: Healthy vs Patient
# Participant-level aggregation from segment-level acoustic features
# Class imbalance handled via SMOTE (applied within each CV fold only)
# =============================================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ============================================================
# Reproducibility
# ============================================================
np.random.seed(42)

# ============================================================
# Load pre-split acoustic feature files
# Update paths as needed
# ============================================================
train_path = "train_acoustic_features.csv"
test_path  = "test_acoustic_features.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

print("Original train shape (segments):", train_df.shape)
print("Original test shape  (segments):", test_df.shape)

# ============================================================
# Basic cleaning
# ============================================================
for df in [train_df, test_df]:
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str).str.strip()

train_df = train_df[train_df["Label"].isin(["Healthy", "Patient"])].copy()
test_df  = test_df[test_df["Label"].isin(["Healthy", "Patient"])].copy()

print("\nSegment-level label counts")
print("Train:", train_df["Label"].value_counts(dropna=False).to_dict())
print("Test :", test_df["Label"].value_counts(dropna=False).to_dict())

# ============================================================
# Grouping column
# ============================================================
group_col = "ID"

if group_col not in train_df.columns:
    raise ValueError(f"Column '{group_col}' not found in train dataframe.")
if group_col not in test_df.columns:
    raise ValueError(f"Column '{group_col}' not found in test dataframe.")

# ============================================================
# Columns to exclude from features
# ============================================================
exclude_cols = {
    "Label", "ID", "ID_numeric", "filename", "basefile",
    "basefile_normalized", "matched_wav", "wav_stem",
    "start", "end", "Age", "Sex", "years", "Source"
}

feature_cols = [c for c in train_df.columns if c not in exclude_cols]
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]

print("\nNumber of candidate numeric feature columns:", len(feature_cols))
print("Example features:", feature_cols[:10])

if len(feature_cols) == 0:
    raise ValueError("No numeric feature columns found after exclusions.")

# ============================================================
# Keep only relevant columns before aggregation
# ============================================================
train_keep = train_df[[group_col, "Label"] + feature_cols].copy()
test_keep  = test_df[[group_col, "Label"] + feature_cols].copy()

# ============================================================
# Handle missing / inf values BEFORE aggregation
# ============================================================
def clean_numeric_df(df):
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

train_keep[feature_cols] = clean_numeric_df(train_keep[feature_cols])
test_keep[feature_cols]  = clean_numeric_df(test_keep[feature_cols])

# Impute using train medians only
train_medians = train_keep[feature_cols].median(axis=0)
train_keep[feature_cols] = train_keep[feature_cols].fillna(train_medians)
test_keep[feature_cols]  = test_keep[feature_cols].fillna(train_medians)

train_keep[feature_cols] = train_keep[feature_cols].astype(float)
test_keep[feature_cols]  = test_keep[feature_cols].astype(float)

# ============================================================
# Aggregate segments to participant level (mean)
# ============================================================
def aggregate_by_speaker(df, group_col, feature_cols):
    label_df = (
        df.groupby(group_col)["Label"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )
    feat_df = (
        df.groupby(group_col)[feature_cols]
        .mean()
        .reset_index()
    )
    return label_df.merge(feat_df, on=group_col, how="inner")

train_agg = aggregate_by_speaker(train_keep, group_col, feature_cols)
test_agg  = aggregate_by_speaker(test_keep, group_col, feature_cols)

print("\nAggregated train shape (participants):", train_agg.shape)
print("Aggregated test shape  (participants):", test_agg.shape)
print("\nParticipant-level label counts")
print("Train:", train_agg["Label"].value_counts(dropna=False).to_dict())
print("Test :", test_agg["Label"].value_counts(dropna=False).to_dict())

# ============================================================
# Labels and groups
# ============================================================
y_trainval = (train_agg["Label"] == "Patient").astype(int)
y_test     = (test_agg["Label"] == "Patient").astype(int)

groups_train = train_agg[group_col].copy()
groups_test  = test_agg[group_col].copy()

# ============================================================
# Feature matrices
# ============================================================
X_trainval = train_agg[feature_cols].copy()
X_test     = test_agg[feature_cols].copy()

common_cols = [c for c in X_trainval.columns if c in X_test.columns]
X_trainval  = X_trainval[common_cols].copy()
X_test      = X_test[common_cols].copy()
feature_names = common_cols

print("\nFinal aligned participant-level feature columns:", len(feature_names))

# ============================================================
# Leakage sanity check
# ============================================================
train_ids = set(groups_train.astype(str))
test_ids  = set(groups_test.astype(str))
overlap   = train_ids.intersection(test_ids)

print("\nUnique train participants:", len(train_ids))
print("Unique test participants :", len(test_ids))
print("Train/test ID overlap    :", len(overlap))

if len(overlap) > 0:
    raise ValueError("Data leakage detected: some participant IDs appear in both train and test.")

# ============================================================
# Model and CV setup
# ============================================================
rf = RandomForestClassifier(
    n_estimators=70,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

smote = SMOTE(random_state=42)
kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

feature_importances = []
cv_reports          = []
cv_accuracies       = []

# ============================================================
# 5-fold cross-validation on TRAIN participants only
# ============================================================
print("\nStarting 5-fold cross-validation on TRAIN participants...")
start_time = time.time()

for fold, (tr_idx, val_idx) in enumerate(
    kfold.split(X_trainval, y_trainval, groups=groups_train)
):
    print(f"\nFold {fold + 1}")

    X_train_fold = X_trainval.iloc[tr_idx].copy()
    X_val_fold   = X_trainval.iloc[val_idx].copy()
    y_train_fold = y_trainval.iloc[tr_idx]
    y_val_fold   = y_trainval.iloc[val_idx]

    print("  Train participants:", groups_train.iloc[tr_idx].nunique())
    print("  Val participants  :", groups_train.iloc[val_idx].nunique())

    # Scale within fold
    scaler = StandardScaler()
    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
    X_val_fold_scaled   = scaler.transform(X_val_fold)

    # SMOTE applied only to training fold
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold_scaled, y_train_fold)

    print("  After SMOTE training shape:", X_train_resampled.shape)
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print("  Class distribution after SMOTE:", dict(zip(unique, counts)))

    rf.fit(X_train_resampled, y_train_resampled)
    feature_importances.append(rf.feature_importances_)

    y_val_pred  = rf.predict(X_val_fold_scaled)
    acc_fold    = accuracy_score(y_val_fold, y_val_pred)
    report_fold = classification_report(
        y_val_fold, y_val_pred,
        target_names=["Healthy", "Patient"],
        output_dict=True, zero_division=0
    )

    cv_reports.append(report_fold)
    cv_accuracies.append(acc_fold)
    print(f"  Fold {fold + 1} Accuracy: {acc_fold:.4f}")

end_time = time.time()
print(f"\nTotal CV time: {(end_time - start_time)/60:.2f} minutes")

# ============================================================
# Average CV metrics
# ============================================================
def extract_average_metrics(cv_reports):
    avg_metrics = {}
    for key in cv_reports[0].keys():
        if isinstance(cv_reports[0][key], dict):
            avg_metrics[key] = {
                metric: np.mean([r[key][metric] for r in cv_reports])
                for metric in cv_reports[0][key].keys()
            }
        else:
            avg_metrics[key] = np.mean([r[key] for r in cv_reports])
    return avg_metrics

avg_importance = np.mean(feature_importances, axis=0)
avg_report     = extract_average_metrics(cv_reports)
avg_accuracy   = np.mean(cv_accuracies)
std_accuracy   = np.std(cv_accuracies)

print(f"\nAverage Accuracy Across Folds: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
print("\nAverage Classification Report Across Folds:")
print(pd.DataFrame(avg_report).round(3))

# ============================================================
# Final model trained on ALL TRAIN, evaluated on TEST
# ============================================================
scaler_final = StandardScaler()
X_trainval_scaled = scaler_final.fit_transform(X_trainval)
X_test_scaled     = scaler_final.transform(X_test)

X_trainval_resampled, y_trainval_resampled = smote.fit_resample(X_trainval_scaled, y_trainval)

print("\nFinal train shape after SMOTE:", X_trainval_resampled.shape)
unique, counts = np.unique(y_trainval_resampled, return_counts=True)
print("Final class distribution after SMOTE:", dict(zip(unique, counts)))

rf.fit(X_trainval_resampled, y_trainval_resampled)
y_test_pred = rf.predict(X_test_scaled)

print("\nClassification Report on Test Set (participant level):")
print(classification_report(y_test, y_test_pred, target_names=["Healthy", "Patient"], zero_division=0))
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

# ============================================================
# Misclassified participants
# ============================================================
misclassified_indices = np.where(y_test_pred != y_test)[0]
misclassified_df = test_agg.iloc[misclassified_indices].copy()
misclassified_df["True_Label_Binary"]      = y_test.iloc[misclassified_indices].values
misclassified_df["Predicted_Label_Binary"] = y_test_pred[misclassified_indices]
misclassified_df["Predicted_Label"]        = np.where(
    misclassified_df["Predicted_Label_Binary"] == 1, "Patient", "Healthy"
)

print("\nMisclassified participants:", len(misclassified_df))
if "ID" in misclassified_df.columns:
    print("Misclassified IDs:", sorted(misclassified_df["ID"].astype(str).unique().tolist()))

misclassified_df.to_csv("misclassified_patients_rf_patient_level.csv", index=False)

# ============================================================
# Feature importance
# ============================================================
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": avg_importance
}).sort_values("importance", ascending=False)

print("\nTop 20 RF features by average importance across folds:")
print(importance_df.head(20).to_string(index=False))
importance_df.to_csv("rf_feature_importance_patient_level.csv", index=False)

top_imp = importance_df.head(20).iloc[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top_imp["feature"], top_imp["importance"])
plt.xlabel("Average feature importance")
plt.title("Top 20 acoustic features (RF, patient-level)")
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Save test predictions
# ============================================================
test_output = test_agg.copy()
test_output["y_true"] = y_test.values
test_output["y_pred"] = y_test_pred

if hasattr(rf, "predict_proba"):
    test_output["y_proba_patient"] = rf.predict_proba(X_test_scaled)[:, 1]

test_output.to_csv("test_predictions_rf_patient_level.csv", index=False)

print("\nDone.")
