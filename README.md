# SONIVA: Speech recOgNItion Validation in Aphasia 

**Authors:**
Giulia Sanguedolce, Cathy J. Price, Sophie Brook, Dragos C. Gruia, Niamh V. Parkinson, Patrick A. Naylor, Fatemeh Geranmayeh

---

This repository provides resources developed for the **SONIVA database** available via DOI: [https://doi.org/10.82186/ra9dv-4az59](https://doi.org/10.82186/ra9dv-4az59) (see below for description). Specifically, this page includes:

* Acoustic features for downstream analyses and the associated classification algorithms 
* A fine-tuned Whisper ASR model
* Data Usage Agreement file for data sharing purposes

---

### 1.🎙️ Acoustic Features and Classification Models

The `acoustic_features/` directory contains participant-level acoustic feature datasets and reproducible classification scripts used in the SONIVA study.

### Included Files

- `train_acoustic_features.csv`
- `test_acoustic_features.csv`
- `svm_classifier.py`
- `rf_classifier.py`
- `nn_classifier.py`

These scripts reproduce acoustic-only experiments for binary classification of **Healthy vs Patient** speech.

### Methodological Characteristics

- Binary classification: **Healthy vs Patient**
- Participant-level modelling via aggregation of segment-level features
- Group-aware cross-validation using `StratifiedGroupKFold`
- Strict separation between training and test participants to prevent data leakage

### Model-Specific Details

- **Support Vector Machine (SVM)**
  - Radial Basis Function (RBF) kernel
  - SMOTE applied within training folds only

- **Random Forest (RF)**
  - Ensemble-based classification
  - SMOTE applied within training folds only

- **Neural Network (NN)**
  - Fully connected feed-forward architecture implemented in PyTorch
  - Weighted `BCEWithLogitsLoss` to address class imbalance
  - Validation-based threshold tuning with sensitivity constraints

For detailed instructions on running these models, see:

📄 `acoustic_features/README_acoustic.md`

---

#### Data Structure

* **Format:** Excel (.xlsx)
* **Groups:** Control and Patient participants

#### Feature Categories

The dataset includes various acoustic measures extracted using openSMILE:

* Fundamental frequency: Mean, standard deviation, range
* Formant frequencies: F1, F2, F3 characteristics
* Spectral features: Centroid, bandwidth, rolloff
* Voice quality: Jitter, shimmer, harmonics-to-noise ratio
* Prosodic measures: Intensity, duration, pause characteristics
* Cepstral features: MFCC coefficients

#### Data Quality

* All features are numeric (float values)
* Standardized subject ID format
* Validated label consistency

---

### 2. ASR Fine-Tuned Model

The `ASR_finetuned/` folder contains:

* A fine-tuned Whisper Medium model
* Stored as split archive parts due to GitHub file size limits
* A lightweight inference script (`test_asr.py`)

This enables evaluation of **automatic speech recognition (ASR)** on impaired speech.

---

## 3.📊 Data Access

The SONIVA database is distributed across this repository and an external open-access archive.

### Open Access (Helix Repository)

Metadata and transcripts are publicly available via:

DOI: [https://doi.org/10.82186/ra9dv-4az59](https://doi.org/10.82186/ra9dv-4az59)

This includes:

* Participant metadata
* Orthographic transcripts (.cha format)

---

### 🔐 Audio Data Access (Controlled)

Additionally, SONIVA contains matched audio recordings from which the transcriptions are derived. Nevertheless, due to the sensitivity of the speech audio data, these files are not suitable for public sharing and are available upon request only. Imperial will review individual data access requests and will execute appropriate institutional data sharing agreements depending on the legal status and location of the data requestor.

The following conditions must be met:

i) The academic requester should have appropriate ethical clearance according to their local regulations

ii) The data must be stored in a secure academic research environment

iii) There should be no attempt to re-identify participants

iv) In the unlikely event that a participant is identified through the audio recordings, the data controller must be informed

To request access, please contact:

**[fatemeh.geranmayeh00@imperial.ac.uk](mailto:fatemeh.geranmayeh00@imperial.ac.uk)**

A Data Usage Agreement (DUA) is provided in this repository and must be completed as part of the request process.

---

## 🧠 Scientific Contributions

SONIVA supports research at the intersection of artificial intelligence, speech processing, and clinical neuroscience. The dataset enables studies in:

- Speech-based neurological assessment
- Post-stroke aphasia analysis
- Clinical automatic speech recognition
- Computational paralinguistics
- Machine learning for healthcare
- Foundation models for clinical speech processing
---

## 📚 Citation

If you use this repository or dataset, please cite:

```bibtex
@article{sanguedolce2025soniva,
  ...
}
```

---

## 📄 License

### Code
All code in this repository is released under the MIT License.

### Data (Non-Audio)
All non-audio data, including:
- acoustic features  
- transcripts  
- metadata  
- fine-tuned ASR model  

are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

### Audio Data
Given the sensitivity of the speech audio data, the authors will review individual data access requests and will execute appropriate institutional data sharing agreement depending on the legal status and location of the data requestor.  The following conditions must be met: i) The academic requester should have appropriate ethical clearance according to their local regulations, ii) The data needs to be stored in a secure academic research environment, iii) There should be no attempt to re-identify participants, iv) in the unlikely event that a participant is identified through the audio recordings, the authors needs to be informed. Access will be granted via a secure institutional cloud link. This process ensures compliance with UK GDPR and the ethical standards governing sensitive biometric data. 

---

## 📞 Contact

* First Author: [gs2022@ic.ac.uk](mailto:gs2022@ic.ac.uk)
* Corresponding Author: [fatemeh.geranmayeh00@imperial.ac.uk](mailto:fatemeh.geranmayeh00@imperial.ac.uk)

For technical issues, please use GitHub Issues.

---

**Last Updated:** 13/04/2026

