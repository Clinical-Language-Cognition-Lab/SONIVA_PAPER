# SONIVA: Speech recOgNItion Validation in Aphasia

This repository contains the code, models, and data developed for the **SONIVA project**.  
It includes **two main components**:

### 1. Acoustic Classification
We provide openSMILE-extracted acoustic features, along with machine learning models — **Support Vector Machine (SVM)**, **Random Forest (RF)**, and a **Neural Network (NN)** using focal loss and dropout regularization. These models are trained to classify speech segments as either **post-stroke (Patient)** or **healthy (Control)**, demonstrating the potential of acoustic biomarkers in clinical classification.

### 2. ASR Whisper Fine-Tuning
We release a **fine-tuned Whisper ASR model** trained on sentence-wise `.cha` transcripts derived from SONIVA recordings. These transcripts (but not raw audio) are included for benchmarking and adaptation, enabling the evaluation of **Word Error Rate (WER)** on post-stroke speech and supporting further research on clinical ASR performance.

---

## 📥 Data Access

All datasets are hosted on **OneDrive** for secure sharing:

- **Acoustic Classification Dataset:**  
  [🔗 Download Acoustic Features + Metadata](PUT_ONEDRIVE_LINK_ACOUSTIC_HERE)

- **ASR Whisper Fine-Tuning & Transcripts:**  
  [🔗 Download Fine-Tuned Whisper Model + CHA Files](PUT_ONEDRIVE_LINK_ASR_HERE)

Each OneDrive package includes:
- `acoustic_features_with_id.xlsx` – Extracted acoustic features with IDs and labels.
- `metadata.xlsx` – Feature descriptions, sample details, and quality control information.
- `.cha` transcript files – Sentence-wise speech annotations (for ASR).
- Fine-tuned ASR model – Ready-to-use Whisper model.

---

## 📂 Repository Structure

```bash
SONIVA_PAPER/
├── acoustic_classification/ # Acoustic feature classification (Experiment 1)
│ ├── main.py
│ ├── requirements.txt
│ ├── README_acoustic.md
│ └── data/
│ └── README.md # Points to OneDrive dataset
│
├── asr_whisper_finetuning/ # Whisper ASR fine-tuning + transcripts (Experiment 2)
│ ├── README_asr.md
│ ├── download_model.sh
│ └── transcripts/
│ └── README.md # Points to CHA files
│
├── DATA_ACCESS.md
├── README.md # This file
├── setup.sh
├── .gitignore
└── .github/
└── workflows/
└── test.yml

```


---

## 🚀 Quick Start

### Acoustic Classification
1. Download the acoustic features from OneDrive and place them in `acoustic_classification/data/`.
2. Install dependencies and run:
   ```bash
   cd acoustic_classification
   pip install -r requirements.txt
   python main.py --data_path data/acoustic_features_with_id.xlsx --model all
   ```

## ASR Whisper Fine-Tuning

1. **Download the fine-tuned Whisper model and `.cha` transcripts** from OneDrive:  
   [🔗 Download ASR Model + Transcripts](PUT_ONEDRIVE_LINK_ASR_HERE)

2. **Follow the detailed instructions** provided in [README_asr.md](asr_whisper_finetuning/README_asr.md).

## 📚 Citation

If you use this repository or dataset, please cite:

```bibtex
@article{sanguedolce2025soniva,
  title={SONIVA: Speech recOgNItion Validation in Aphasia},
  author={Sanguedolce, Giulia and Price, Cathy J. and Brook, Sophie and Gruia, Dragos C. and Parkinson, Niamh V. and Naylor, Patrick A. and Geranmayeh, Fatemeh},
  journal={[Journal Name]},
  year={2025},
  doi={To be assigned}
}



