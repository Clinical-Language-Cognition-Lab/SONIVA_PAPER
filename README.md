# SONIVA: Speech recOgNItion Validation in Aphasia

**Authors:**
Giulia Sanguedolce, Cathy J. Price, Sophie Brook, Dragos C. Gruia, Niamh V. Parkinson, Patrick A. Naylor, Fatemeh Geranmayeh

---

This repository provides resources developed for the **SONIVA database**, including:

* Acoustic features for downstream analyses
* A fine-tuned Whisper ASR model

The repository is designed as a **data and model release**, with a focus on transparency and reproducibility.

---

## 📦 Repository Components

### 1. Acoustic Features

The `acoustic_features/` folder contains:

* Precomputed acoustic feature matrices (train/test splits used in the paper)
* Features extracted using openSMILE from segmented speech

These features can be used for **classification tasks of the user's choice** or to support broader research pipelines, including studies aligned with the transcripts available via the Helix repository.

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

## 📊 Data Access

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

## 🧪 Methodological Overview

### Acoustic Features

* Extracted using openSMILE (e.g., eGeMAPS configuration)
* Derived from **utterance-level segmentation** aligned with CHAT (.cha) transcripts
* Interviewer speech removed to retain only participant speech

### ASR Model

* Based on Whisper Medium (OpenAI)
* Fully fine-tuned (encoder + decoder)
* Trained on SONIVA audios and transcripts

---

## 📂 Repository Structure

```text
SONIVA_PAPER/
│
├── acoustic_features/       # Acoustic feature dataset
├── ASR_finetuned/           # Fine-tuned Whisper model
├── .gitattributes           # Git LFS configuration
├── .gitignore
└── README.md
```

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
The raw speech recordings are **not publicly distributed** due to ethical and data protection constraints, as they contain biometric and potentially identifiable information.

Access to the audio data is provided **upon request only**, subject to:
- approval by the data controllers  
- completion of a Data Usage Agreement (DUA)  
- compliance with institutional and legal requirements  

See the Data Access section for details on how to request access.
---

## 📞 Contact

* First Author: [gs2022@ic.ac.uk](mailto:gs2022@ic.ac.uk)
* Corresponding Author: [fatemeh.geranmayeh00@imperial.ac.uk](mailto:fatemeh.geranmayeh00@imperial.ac.uk)

For technical issues, please use GitHub Issues.

---

**Last Updated:** 31/03/2026

