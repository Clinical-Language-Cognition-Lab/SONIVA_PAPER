# SONIVA: Speech recOgNItion Validation in Aphasia

This repository contains the code, models, and data developed for the **SONIVA project**.  
It includes **two main components**:

### 1. Acoustic Classification
We provide openSMILE-extracted acoustic features, along with machine learning models — **Support Vector Machine (SVM)**, **Random Forest (RF)**, and a **Neural Network (NN)** using focal loss and dropout regularization. These models are trained to classify speech segments as either **post-stroke (Patient)** or **healthy (Control)**, demonstrating the potential of acoustic biomarkers in clinical classification.

### 2. ASR Whisper Fine-Tuning
We release a **fine-tuned Whisper ASR model** trained on sentence-wise `.cha` transcripts derived from SONIVA recordings. These transcripts (but not raw audio) are included for benchmarking and adaptation, enabling the evaluation of **Word Error Rate (WER)** on post-stroke speech and supporting further research on clinical ASR performance.

---


## 📥 Data Access

All SONIVA datasets (acoustic features, metadata, ASR model, and transcripts) are hosted together on **OneDrive** for secure sharing:

**[🔗 Download SONIVA Data and Models](https://imperiallondon-my.sharepoint.com/shared?id=%2Fpersonal%2Ffg00%5Fic%5Fac%5Fuk%2FDocuments%2FSpeech%5FRecognition%5Fshared%2FSONIVA%2F%280%29%20SONIVA%5FFor%5FPublication%20%28GIULIA%29%2FSONIVA%5FPAPER&sortField=LinkFilename&isAscending=true)**

The OneDrive package includes:
- `acoustic_features_with_id.xlsx` – Extracted acoustic features with subject IDs and labels.  
- `metadata.xlsx` – Clinical and demographic metadata (IC3 and PLORAS columns).  
- `.cha` transcript files – Orthographic and phonetic transcripts for ASR benchmarking.  
- Fine-tuned Whisper Medium model – Fully trained ASR model ready for testing.

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
If you use this code or dataset in your research, please cite our paper:
```bibtex
@article{sanguedolce2025soniva,
  ...
}
```
---

## 🙏 Acknowledgments
- [Funding sources]
- [Collaborators]

---

## 📞 Contact
For questions about this code or paper:
- **First Author:** gs2022@ic.ac.uk
- **Corresponding Author:** fatemeh.geranmayeh00@imperial.ac.uk
- **Issues:** Please use GitHub Issues for technical problems

---

**Last Updated:** 29/07/2025  
**Version:** 1.0.0


