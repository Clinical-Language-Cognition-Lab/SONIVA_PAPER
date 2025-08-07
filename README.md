# SONIVA: Speech recOgNItion Validation in Aphasia

Authors:
Giulia Sanguedolce, Cathy J. Price, Sophie Brook, Dragos C. Gruia, Niamh V. Parkinson4, Patrick A. Naylor and Fatemeh Geranmayeh

This repository contains the code, models, and data developed for the **SONIVA project**.  
It includes **two main components**:

### 1. Acoustic Classification
We provide openSMILE-extracted acoustic features, along with machine learning models — **Support Vector Machine (SVM)**, **Random Forest (RF)**, and a **Neural Network (NN)** using focal loss and dropout regularization. These models are trained to classify speech segments as either **post-stroke (Patient)** or **healthy (Control)**, demonstrating the potential of acoustic biomarkers in clinical classification.

### 2. ASR Whisper Fine-Tuning
We release a **fine-tuned Whisper ASR model** trained on sentence-wise `.cha` transcripts derived from SONIVA recordings. These transcripts (but not raw audio) are included for benchmarking and adaptation, enabling the evaluation of **Word Error Rate (WER)** on post-stroke speech and supporting further research on clinical ASR performance.

---


## 📥 Data Access

All SONIVA datasets (acoustic features, metadata, ASR model, and transcripts) are hosted together:

**[🔗 Download SONIVA Data and Models](https://drive.google.com/drive/folders/1lqyKebne8jIBaTeD9MjTh6M2Kf5hsbVW?usp=sharing)**

The folder includes:
- `acoustic_features_with_id.xlsx` – Extracted acoustic features with subject IDs and labels.  
- `metadata.xlsx` – Clinical and demographic metadata (IC3 and PLORAS columns).  
- `.cha` transcript files – Orthographic and phonetic transcripts for ASR benchmarking.  
- Fine-tuned Whisper Medium model – Fully trained ASR model ready for testing.

---

## 🔐 Audio Data Access Policy (Controlled)

While acoustic features and transcripts are **openly available**, access to the **raw audio recordings** is restricted due to GDPR and ethical regulations. The audio files contain **biometric data (voice)** and are considered potentially re-identifiable.

To request access to the audio:

1. Download and review the [**SONIVA Data Usage Agreement (DUA)**](./DATA_USAGE_AGREEMENT_AUDIO.docx)
2. Sign the agreement and email it to **fatemeh.geranmayeh00@imperial.ac.uk** with:
   - Your institutional affiliation
   - A brief description of your intended research use
   - A valid **Good Clinical Practice (GCP)** training certificate 

Upon approval, secure access to the audio folder will be granted via the Drive.

Please note that access is limited to researchers affiliated with academic or healthcare institutions, and all use must comply with the conditions outlined in the DUA.

---


## 📂 Repository Structure

```bash
SONIVA_PAPER/
├── acoustic_classification/ # Acoustic feature classification (Experiment 1)
│ ├── main.py
│ ├── requirements.txt
│ ├── README_acoustic.md
│ └── data/
│ └── README.md # Points to the dataset
│
├── asr_whisper_finetuning/ # Whisper ASR fine-tuning + transcripts (Experiment 2)
│ ├── README_asr.md
│ ├── download_model.sh
│ └── transcripts/
│ └── README.md # Points to CHA files
│
├── DATA_ACCESS.md
├── DATA_USAGE_AGREEMENT_AUDIO.docx
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
1. Download the acoustic features from the Drive and place them in `acoustic_classification/data/`.
2. Install dependencies and run:
   ```bash
   cd acoustic_classification
   pip install -r requirements.txt
   python main.py --data_path data/acoustic_features_with_id.xlsx --model all
   ```

## ASR Whisper Fine-Tuning

1. **Download the fine-tuned Whisper model and `.cha` transcripts** from the Drive:  
   [🔗 Download ASR Model + Transcripts](https://drive.google.com/drive/folders/1lqyKebne8jIBaTeD9MjTh6M2Kf5hsbVW?usp=sharing)

2. **Follow the detailed instructions** provided in [README_asr.md](asr_whisper_finetuning/README_asr.md).


## 📚 Citation
If you use this code or dataset in your research, please cite our paper:
```bibtex
@article{sanguedolce2025soniva,
  ...
}
```
---


## 📄 License

This repository is made available under the following licensing terms:

### Code
All code in this repository is released under the [MIT License](https://opensource.org/licenses/MIT)
> © 2025 Imperial College London & Contributors. 

### Data (Transcripts, Metadata, Acoustic Features, Fine-Tuned Models)
All non-audio data files are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. 


### Audio Recordings
The raw speech recordings are **not licensed for public use** due to GDPR and ethical restrictions, as they contain **biometric and potentially re-identifiable information**. Access is governed via a **Data Usage Agreement (DUA)**:

- Download and review the DUA [here](./DATA_USAGE_AGREEMENT_AUDIO.docx)
- Submit the signed DUA and GCP certificate to: **fatemeh.geranmayeh00@imperial.ac.uk**
- Access is restricted to researchers at academic or clinical institutions.

---

## Acknowledgments
- Funding support was provided to G.S. (UK Research and Innovation [UKRI Centre for Doctoral Training in AI for Healthcare:EP/S023283/1]), F.G. (Medical Research Council P79100) and S.B. (EPSRC IAA -PSP415 and MRC IAA -PSP518)
- The authors would like to thank A. Coghlan, O. Burton and N. Parkinson for assistance with the IC3 study, and J. Friedland, K. Stephenson, G. Gvero, Z. Zaskorska and C. Leong for labelling the speech data. We are grateful to the patients, clinical teams, and the PLORAS team for their valuable contributions to the SONIVA database. Infrastructure support was provided by the NIHR Imperial Biomedical Research Centre and the NIHR Imperial Clinical Research Facility. The views expressed are those of the authors and not necessarily those of the NHS, the NIHR or the Department of Health and Social Care.

---

## 📞 Contact
For questions about this code or paper:
- **First Author:** gs2022@ic.ac.uk
- **Corresponding Author:** fatemeh.geranmayeh00@imperial.ac.uk
- **Issues:** Please use GitHub Issues for technical problems

---

**Last Updated:** 07/08/2025  


