# =============================================================================
# Neural Network Classifier - Acoustic Features
# SONIVA PAPER - Clinical Language Cognition Lab
#
# Binary classification: Healthy vs Patient
# Participant-level aggregation from segment-level acoustic features
# Class imbalance handled via weighted BCEWithLogitsLoss (no SMOTE)
# Threshold tuning: maximise specificity subject to sensitivity >= 0.90
# =============================================================================

import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================================
# Reproducibility
# ============================================================
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Threshold tuning
# ============================================================
def find_best_threshold(y_true, y_prob, metric="sensitivity_target", min_sens=None):
    """
    Tune classification threshold on validation probabilities.

    metric:
        - "bacc"               : maximise balanced accuracy
        - "f1"                 : maximise F1
        - "youden"             : maximise Youden's J = sensitivity + specificity - 1
        - "sensitivity_target" : among thresholds with sensitivity >= min_sens,
                                 choose the one with highest specificity
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        bacc   = balanced_accuracy_score(y_true, y_pred)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        sens   = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rows.append({
            "threshold": thr, "bacc": bacc, "f1": f1,
            "sensitivity": sens, "specificity": spec,
            "youden": sens + spec - 1.0
        })

    results_df = pd.DataFrame(rows)

    if metric == "bacc":
        best_idx   = results_df["bacc"].idxmax()
        best_score = float(results_df.loc[best_idx, "bacc"])
    elif metric == "f1":
        best_idx   = results_df["f1"].idxmax()
        best_score = float(results_df.loc[best_idx, "f1"])
    elif metric == "youden":
        best_idx   = results_df["youden"].idxmax()
        best_score = float(results_df.loc[best_idx, "youden"])
    elif metric == "sensitivity_target":
        if min_sens is None:
            raise ValueError("min_sens must be provided for metric='sensitivity_target'")
        eligible = results_df[results_df["sensitivity"] >= min_sens].copy()
        best_idx = eligible["specificity"].idxmax() if len(eligible) > 0 else results_df["sensitivity"].idxmax()
        best_score = float(results_df.loc[best_idx, "specificity"])
    else:
        raise ValueError("metric must be one of: 'bacc', 'f1', 'youden', 'sensitivity_target'")

    return float(results_df.loc[best_idx, "threshold"]), best_score, results_df


# ============================================================
# Weighted BCE loss (handles class imbalance without SMOTE)
# ============================================================
def make_bce_loss(y_binary, device):
    y_binary  = np.asarray(y_binary).astype(int)
    n_pos     = max((y_binary == 1).sum(), 1)
    n_neg     = max((y_binary == 0).sum(), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


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
    return df.copy().replace([np.inf, -np.inf], np.nan)

train_keep[feature_cols] = clean_numeric_df(train_keep[feature_cols])
test_keep[feature_cols]  = clean_numeric_df(test_keep[feature_cols])

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
    feat_df = df.groupby(group_col)[feature_cols].mean().reset_index()
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

groups_train = train_agg["ID"].copy()
groups_test  = test_agg["ID"].copy()

# ============================================================
# Feature matrices
# ============================================================
X_trainval = train_agg[feature_cols].copy()
X_test     = test_agg[feature_cols].copy()

common_cols = [c for c in X_trainval.columns if c in X_test.columns]
X_trainval  = X_trainval[common_cols].copy()
X_test      = X_test[common_cols].copy()
feature_names = common_cols

print("Final aligned participant-level feature columns:", len(feature_names))

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
# Neural network architecture
# ============================================================
class AcousticNet(nn.Module):
    """
    Fully connected network: input -> 64 -> 32 -> 16 -> 1 (logits)
    Each hidden layer uses ReLU activation and dropout.
    """
    def __init__(self, input_size, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

    def forward(self, x):
        return self.net(x)

# ============================================================
# Data loaders
# ============================================================
def create_data_loaders(X_train, y_train, X_val, y_val, batch_size):
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader

# ============================================================
# Train / evaluate helpers
# ============================================================
def train_model(model, train_loader, criterion, optimizer, device, threshold=0.5):
    model.train()
    running_loss, predictions, true_labels = 0.0, [], []
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        logits = model(inputs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        probs = torch.sigmoid(logits)
        predictions.extend((probs >= threshold).detach().cpu().numpy().astype(int).flatten())
        true_labels.extend(labels.cpu().numpy().astype(int).flatten())
    return running_loss / max(len(train_loader), 1), accuracy_score(true_labels, predictions)


def evaluate_model(model, val_loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss, predictions, true_labels, probabilities = 0.0, [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            logits = model(inputs)
            loss   = criterion(logits, labels)
            running_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            probabilities.extend(probs.tolist())
            predictions.extend((probs >= threshold).astype(int).tolist())
            true_labels.extend(labels.cpu().numpy().astype(int).flatten().tolist())
    return (
        running_loss / max(len(val_loader), 1),
        accuracy_score(true_labels, predictions),
        np.array(true_labels),
        np.array(predictions),
        np.array(probabilities)
    )

# ============================================================
# Hyperparameters
# ============================================================
batch_size         = 64
num_epochs         = 120
learning_rate      = 0.001
weight_decay       = 1e-4
patience           = 15
threshold_metric   = "sensitivity_target"
target_sensitivity = 0.90

# ============================================================
# 5-fold cross-validation on TRAIN participants only
# ============================================================
kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)

cv_reports    = []
cv_accuracies = []
all_losses    = []
cv_thresholds = []

print("\nStarting 5-fold cross-validation on TRAIN participants...")
start_time = time.time()

for fold, (tr_idx, val_idx) in enumerate(
    kfold.split(X_trainval, y_trainval, groups=groups_train)
):
    print(f"\nFold {fold + 1}")

    X_train_fold = X_trainval.iloc[tr_idx].copy()
    X_val_fold   = X_trainval.iloc[val_idx].copy()
    y_train_fold = y_trainval.iloc[tr_idx].copy()
    y_val_fold   = y_trainval.iloc[val_idx].copy()

    print("  Train participants:", groups_train.iloc[tr_idx].nunique())
    print("  Val participants  :", groups_train.iloc[val_idx].nunique())

    scaler_fold            = StandardScaler()
    X_train_fold_scaled    = scaler_fold.fit_transform(X_train_fold)
    X_val_fold_scaled      = scaler_fold.transform(X_val_fold)

    # No SMOTE — class imbalance handled by weighted loss
    X_train_resampled = X_train_fold_scaled
    y_train_resampled = y_train_fold.values

    train_loader, val_loader = create_data_loaders(
        X_train_resampled, y_train_resampled,
        X_val_fold_scaled, y_val_fold.values,
        batch_size=batch_size
    )

    model     = AcousticNet(input_size=X_train_resampled.shape[1], dropout=0.15).to(device)
    criterion = make_bce_loss(y_train_resampled, device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5, verbose=True)

    fold_losses       = []
    best_val_loss     = float("inf")
    early_counter     = 0
    best_model_state  = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_true, val_pred, val_prob = evaluate_model(
            model, val_loader, criterion, device
        )
        fold_losses.append([train_loss, val_loss])
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            early_counter    = 0
        else:
            early_counter += 1

        if early_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    _, _, val_true, _, val_prob = evaluate_model(model, val_loader, criterion, device)

    best_thr, best_score, _ = find_best_threshold(
        val_true, val_prob, metric=threshold_metric, min_sens=target_sensitivity
    )
    cv_thresholds.append(best_thr)

    val_pred_tuned = (val_prob >= best_thr).astype(int)
    val_acc_tuned  = accuracy_score(val_true, val_pred_tuned)

    all_losses.append(fold_losses)
    cv_accuracies.append(val_acc_tuned)
    cv_reports.append(classification_report(
        val_true, val_pred_tuned,
        target_names=["Healthy", "Patient"],
        output_dict=True, zero_division=0
    ))

    print(f"  Fold {fold + 1} best threshold: {best_thr:.3f} | Val Acc: {val_acc_tuned:.4f}")

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

avg_report     = extract_average_metrics(cv_reports)
avg_accuracy   = np.mean(cv_accuracies)
std_accuracy   = np.std(cv_accuracies)
final_threshold = float(np.median(cv_thresholds))

print("\nAverage Classification Report Across Folds:")
print(pd.DataFrame(avg_report).round(3))
print(f"\nAverage Accuracy Across Folds: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Fold thresholds: {[round(t, 3) for t in cv_thresholds]}")
print(f"Final threshold applied to test (median of CV): {final_threshold:.3f}")

# ============================================================
# Final training on TRAIN with internal validation split
# ============================================================
final_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)
tr_idx_final, val_idx_final = next(
    final_splitter.split(X_trainval, y_trainval, groups=groups_train)
)

X_train_final = X_trainval.iloc[tr_idx_final].copy()
X_val_final   = X_trainval.iloc[val_idx_final].copy()
y_train_final = y_trainval.iloc[tr_idx_final].copy()
y_val_final   = y_trainval.iloc[val_idx_final].copy()

scaler_final          = StandardScaler()
X_train_final_scaled  = scaler_final.fit_transform(X_train_final)
X_val_final_scaled    = scaler_final.transform(X_val_final)
X_test_scaled         = scaler_final.transform(X_test)

train_loader_final, val_loader_final = create_data_loaders(
    X_train_final_scaled, y_train_final.values,
    X_val_final_scaled, y_val_final.values,
    batch_size=batch_size
)

test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test.values)),
    batch_size=batch_size, shuffle=False
)

final_model     = AcousticNet(input_size=X_train_final_scaled.shape[1], dropout=0.15).to(device)
criterion_final = make_bce_loss(y_train_final.values, device)
optimizer_final = optim.AdamW(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler_final = ReduceLROnPlateau(optimizer_final, mode="min", factor=0.5, patience=5, min_lr=1e-5, verbose=True)

best_final_loss  = float("inf")
best_final_state = copy.deepcopy(final_model.state_dict())
early_counter    = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_model(final_model, train_loader_final, criterion_final, optimizer_final, device)
    val_loss, val_acc, _, _, _ = evaluate_model(final_model, val_loader_final, criterion_final, device)
    scheduler_final.step(val_loss)

    if epoch % 10 == 0:
        print(f"Final Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    if val_loss < best_final_loss:
        best_final_loss  = val_loss
        best_final_state = copy.deepcopy(final_model.state_dict())
        early_counter    = 0
    else:
        early_counter += 1

    if early_counter >= patience:
        print(f"Final training early stopping at epoch {epoch}")
        break

final_model.load_state_dict(best_final_state)

# ============================================================
# Test evaluation
# ============================================================
final_model.eval()
test_predictions, test_probabilities, test_true_labels = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        probs  = torch.sigmoid(final_model(inputs)).detach().cpu().numpy().flatten()
        preds  = (probs >= final_threshold).astype(int)
        test_probabilities.extend(probs.tolist())
        test_predictions.extend(preds.tolist())
        test_true_labels.extend(labels.numpy().tolist())

test_prob_arr        = np.array(test_probabilities)
test_true_arr        = np.array(test_true_labels)
test_predictions_arr = np.array(test_predictions)

print("\n✅ Test Set Performance (participant level):")
print(f"Test Accuracy @ tuned threshold ({final_threshold:.3f}): {accuracy_score(test_true_arr, test_predictions_arr):.4f}")
print(f"Test Accuracy @ default 0.5   : {accuracy_score(test_true_arr, (test_prob_arr >= 0.5).astype(int)):.4f}")
print("\nTest Classification Report:")
print(classification_report(test_true_arr, test_predictions_arr, target_names=["Healthy", "Patient"], zero_division=0))

# Derived metrics
cm = confusion_matrix(test_true_arr, test_predictions_arr, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Sensitivity : {tp / (tp + fn) if (tp + fn) > 0 else 0.0:.4f}")
print(f"Specificity : {tn / (tn + fp) if (tn + fp) > 0 else 0.0:.4f}")
print(f"Precision   : {tp / (tp + fp) if (tp + fp) > 0 else 0.0:.4f}")
print(f"NPV         : {tn / (tn + fn) if (tn + fn) > 0 else 0.0:.4f}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Patient"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
plt.title("Confusion Matrix - Test Set (participant level)")
plt.tight_layout()
plt.savefig("confusion_matrix_nn_patient_level.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Misclassified participants
# ============================================================
misclassified_indices = np.where(test_predictions_arr != test_true_arr)[0]
misclassified_df = test_agg.iloc[misclassified_indices].copy()
misclassified_df["True_Label_Binary"]          = test_true_arr[misclassified_indices]
misclassified_df["Predicted_Label_Binary"]     = test_predictions_arr[misclassified_indices]
misclassified_df["Predicted_Label"]            = np.where(
    misclassified_df["Predicted_Label_Binary"] == 1, "Patient", "Healthy"
)
misclassified_df["Predicted_Probability_Patient"] = test_prob_arr[misclassified_indices]

print("\nMisclassified participants:", len(misclassified_df))
if "ID" in misclassified_df.columns:
    print("Misclassified IDs:", sorted(misclassified_df["ID"].astype(str).unique().tolist()))

misclassified_df.to_csv("misclassified_patients_nn_patient_level.csv", index=False)

# ============================================================
# Save test predictions
# ============================================================
test_output = test_agg.copy()
test_output["y_true"]           = test_true_arr
test_output["y_pred"]           = test_predictions_arr
test_output["y_proba_patient"]  = test_prob_arr
test_output["threshold_used"]   = final_threshold
test_output.to_csv("test_predictions_nn_patient_level.csv", index=False)

# ============================================================
# CV learning curves
# ============================================================
plt.figure(figsize=(12, 6))
for fold_idx in range(len(all_losses)):
    losses = np.array(all_losses[fold_idx])
    if len(losses) > 0:
        plt.plot(losses[:, 0], alpha=0.3, label=f"Train Fold {fold_idx+1}")
        plt.plot(losses[:, 1], alpha=0.3, label=f"Val Fold {fold_idx+1}")
plt.title("Training and Validation Loss Curves (CV Folds)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("nn_learning_curves.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nDone.")
