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

---

## 🧪 Methodological Notes

* Features were extracted using openSMILE (e.g., eGeMAPS configuration)
* Classification experiments were conducted using standard machine learning approaches (e.g., SVM, Random Forest, Neural Networks)
* Evaluation was performed using subject-level separation (e.g., group-based cross-validation)

This release focuses on **data transparency** and allows users to reproduce or extend the classification experiments using their own models and pipelines.

---

## ⚠️ Notes

* This repository intentionally does **not** include trained models or training pipelines
* Users are expected to implement their own classification models if needed
* The provided features are sufficient to reproduce the experimental setup described in the associated work

---

## 📞 Contact

* First Author: [gs2022@ic.ac.uk](mailto:gs2022@ic.ac.uk)
* Corresponding Author: [fatemeh.geranmayeh00@imperial.ac.uk](mailto:fatemeh.geranmayeh00@imperial.ac.uk)
* Issues: Use GitHub Issues for technical questions

