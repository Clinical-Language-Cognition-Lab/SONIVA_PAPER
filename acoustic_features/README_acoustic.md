# Acoustic Features – SONIVA

This folder contains the participant-level train/test acoustic feature files used for binary classification of **Healthy vs Patient** speech in the SONIVA dataset, together with example classification scripts used in the paper.

Features were extracted from speech using acoustic analysis pipelines and used in downstream machine learning experiments.

---

## Data Availability

The metadata and transcripts associated with this dataset are publicly available via the Helix repository:

**DOI:** [https://doi.org/10.82186/ra9dv-4az59](https://doi.org/10.82186/ra9dv-4az59)

This folder contains:

- `train_acoustic_features.csv`
- `test_acoustic_features.csv`
- `svm_classifier.py`
- `rf_classifier.py`
- `nn_classifier.py`

---

## Files in this Folder

### Acoustic feature files

- `train_acoustic_features.csv`: training split
- `test_acoustic_features.csv`: held-out test split

These files contain segment-level acoustic features together with participant identifiers and labels.  
During model training, segments are aggregated to the **participant level** by averaging features across segments belonging to the same `ID`.

### Classification scripts

- `svm_classifier.py`: Support Vector Machine classifier with RBF kernel
- `rf_classifier.py`: Random Forest classifier
- `nn_classifier.py`: Feed-forward neural network implemented in PyTorch

All three scripts perform:

- filtering to `Healthy` and `Patient` labels
- participant-level aggregation using the `ID` column
- train/test separation checks to prevent participant leakage
- evaluation at the **participant level**

Additional model-specific details:

- **SVM** and **Random Forest** use **SMOTE only within training folds**
- **Neural Network** uses **weighted BCEWithLogitsLoss** instead of SMOTE
- **Neural Network** also performs validation-based threshold tuning, selecting the final threshold from cross-validation

---

## Data Format

Each CSV file contains rows corresponding to speech segments and includes:

- `ID`: unique participant identifier
- `Label`: class label (`Healthy` or `Patient`)
- multiple numeric acoustic feature columns
- additional metadata columns, some of which are excluded automatically by the scripts

Examples of excluded metadata columns include:

- `ID`
- `Label`
- `filename`
- `start`
- `end`
- `Age`
- `Sex`
- `Source`

Only numeric acoustic features are retained for modelling.

---

## Train/Test Split Summary

The dataset is split at the **participant level**.

### Train set
- Patients: 514
- Controls: 92
- Total: 606 participants

### Test set
- Patients: 57
- Controls: 11
- Total: 68 participants

Counts refer to unique individuals.

---

## Requirements

These scripts were written in Python and require the following main packages:

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn torch
```

### Optional: Create a Virtual Environment

Depending on your environment, you may wish to create a dedicated virtual environment first.

```bash
python -m venv soniva_env
source soniva_env/bin/activate
pip install numpy pandas matplotlib scikit-learn imbalanced-learn torch
```

For Windows users:

```bash
python -m venv soniva_env
soniva_env\\Scripts\\activate
pip install numpy pandas matplotlib scikit-learn imbalanced-learn torch
```

---

## How to Run

Place yourself inside the `acoustic_features` folder:

```bash
cd acoustic_features
```

Then run any of the classifiers:

### SVM
```bash
python svm_classifier.py
```

### Random Forest
```bash
python rf_classifier.py
```

### Neural Network
```bash
python nn_classifier.py
```

The scripts assume that the following files are present in the same folder:

- `train_acoustic_features.csv`
- `test_acoustic_features.csv`

If you move the scripts elsewhere, update the file paths inside the code accordingly.

---

## What Each Script Does

### `svm_classifier.py`
- Standardises features
- Uses `StratifiedGroupKFold` cross-validation on training participants only
- Applies SMOTE only to training folds
- Trains an RBF-kernel SVM
- Computes participant-level test predictions
- Exports misclassified participants and permutation feature importance

### `rf_classifier.py`
- Standardises features
- Uses `StratifiedGroupKFold` cross-validation on training participants only
- Applies SMOTE only to training folds
- Trains a Random Forest classifier
- Computes participant-level test predictions
- Exports misclassified participants and feature importance rankings

### `nn_classifier.py`
- Standardises features within each fold
- Uses `StratifiedGroupKFold` cross-validation on training participants only
- Trains a feed-forward neural network with weighted binary cross-entropy loss
- Tunes the classification threshold on validation data with a sensitivity constraint
- Applies the median cross-validated threshold to the held-out test set
- Exports misclassified participants, prediction probabilities, confusion matrix, and learning curves

---

## Output Files

Running the scripts will generate result files such as:

- test-set predictions
- misclassified participant lists
- feature importance tables
- confusion matrix plots
- learning-curve plots

Exact filenames depend on the script.

---

## Methodological Notes

- Evaluation is performed with participant-level separation to avoid leakage across segments from the same individual.
- Segment-level acoustic features are aggregated to participant level using the mean.
- Cross-validation is performed only on the training split.
- The held-out test split remains untouched until final evaluation.

These scripts are provided to support reproducibility of the classification experiments reported in the associated work.

---

## Contact

- First Author: gs2022@ic.ac.uk
- Corresponding Author: fatemeh.geranmayeh00@imperial.ac.uk

For technical questions, please use GitHub Issues where possible.

