# Acoustic Classification – SONIVA Project

This folder contains the acoustic feature set used for the classification of aphasic vs. control speech segments in the SONIVA dataset.

The features were extracted using openSMILE and used in downstream machine learning experiments described in the paper.

---

## 📊 Data Availability

The metadata and transcripts associated with this dataset are publicly available via the Helix repository:

DOI: [https://doi.org/10.82186/ra9dv-4az59](https://doi.org/10.82186/ra9dv-4az59)

This folder contains **only the extracted acoustic features** used for classification experiments.

### This repository includes:

* Acoustic feature matrix

### This repository does NOT include:

* Raw audio data
* Transcripts
* Full metadata
* Trained model weights
* Training or inference code

These are available via the Helix release (see above) or can be implemented by the user.

---

## 📂 Data Format

The acoustic feature file is provided in Excel (`.xlsx`) format with the following structure:

| ID   | Feature1 | Feature2 | ... | FeatureN | Label   |
| ---- | -------- | -------- | --- | -------- | ------- |
| S001 | 0.123    | 0.456    | ... | 0.789    | Control |
| S002 | 0.234    | 0.567    | ... | 0.890    | Patient |

* **ID**: Unique subject identifier (used for group-based evaluation)
* **Label**: `Control` or `Patient`
* **Features**: Numeric acoustic features extracted from speech

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

