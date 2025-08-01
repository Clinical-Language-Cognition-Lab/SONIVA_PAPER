# SONIVA Dataset Access

The SONIVA dataset is available in two complementary formats to support both acoustic classification and ASR experiments. All data is stored on OneDrive for controlled access.

---

## 📥 Quick Access

All SONIVA resources are available in a single OneDrive directory:  
**[🔗 Download SONIVA Data and Models](https://imperiallondon-my.sharepoint.com/shared?id=%2Fpersonal%2Ffg00%5Fic%5Fac%5Fuk%2FDocuments%2FSpeech%5FRecognition%5Fshared%2FSONIVA%2F%280%29%20SONIVA%5FFor%5FPublication%20%28GIULIA%29%2FSONIVA%5FPAPER&sortField=LinkFilename&isAscending=true)**

---

## 📋 Dataset Contents

### Acoustic Features (Classification)
- **File:** `acoustic_features_with_id.xlsx` (openSMILE features)
   - Columns: acoustic features, subject IDs, and labels (`Patient`/`Control`).

- **File:** `metadata.xlsx`
  - Demographics and clinical information divided by database source (IC3/PLORAS).

---

### ASR Data (Transcripts + Model)
- **File:** `whisper_finetuned`
  - Fine-tuned Whisper model for SONIVA transcripts.

- **Folder:** `transcripts_cha/`
  - `.cha` transcripts of patient and control speech.
---

## 🚀 How to Use
- For **acoustic classification**, download the acoustic features and place them under `acoustic_classification/data/`.
- For **ASR evaluation**, download the fine-tuned model then follow `asr_whisper_finetuning/README_asr.md`.
---

## 📋 Dataset Specifications

### Data Structure
- **Format:** Excel (.xlsx)
- **Groups:** Control and Patient participants
### Feature Categories
The dataset includes various acoustic measures extracted from openSMILE:
- **Fundamental frequency:** Mean, standard deviation, range
- **Formant frequencies:** F1, F2, F3 characteristics
- **Spectral features:** Centroid, bandwidth, rolloff
- **Voice quality:** Jitter, shimmer, harmonics-to-noise ratio
- **Prosodic measures:** Intensity, duration, pause characteristics
- **Cepstral features:** MFCC coefficients

### Data Quality
- All features are numeric (float values)
- Standardized subject ID format
- Validated label consistency

---
## 📝 Citation
When using any SONIVA dataset files, cite:
```bibtex
@article{sanguedolce2025soniva,
  title={SONIVA: Speech recOgNItion Validation in Aphasia},
  author={Sanguedolce, Giulia and Price, Cathy J. and Brook, Sophie and Gruia, Dragos C. and Parkinson, Niamh V. and Naylor, Patrick A. and Geranmayeh, Fatemeh},
  journal={[Journal Name]},
  year={2025},
  doi={To be assigned}
}
```

### Data Attribution
Please include the following acknowledgment in your work:
> "Acoustic feature data used in this study was obtained from the SONIVA project (Sanguedolce et al., 2025), available at: [GitHub repository link]."

---

## 🆘 Support

### Data Issues
If you encounter problems with the dataset:
- Check file integrity: Verify the download completed successfully.
- Validate format: Ensure Excel file opens correctly.
- Contact authors: For persistent issues or questions.

### Contact Information
- **Dataset questions:** gs2022@ic.ac.uk
- **Technical support:** fatemeh.geranmayeh00@imperial.ac.uk
- **GitHub issues:** Use repository issue tracker for code-related problems.

---

## 🔒 Data Governance

### Privacy and Ethics
- All data has been processed according to institutional guidelines.
- Participant privacy has been protected throughout.
- Data sharing approved by relevant ethics committees.

### Terms of Use
By downloading and using this dataset, you agree to:
- Use the data for research purposes only.
- Properly cite and acknowledge the source.
- Not redistribute the raw data without permission.
- Follow institutional data handling policies.

### Updates
- **Version:** 1.0 (Initial release)
- **Last updated:** July 29, 2025
- **Update notifications:** Watch this repository for announcements.

---

**Full Download Link:** [HERE](https://imperiallondon-my.sharepoint.com/shared?id=%2Fpersonal%2Ffg00%5Fic%5Fac%5Fuk%2FDocuments%2FSpeech%5FRecognition%5Fshared%2FSONIVA%2F%280%29%20SONIVA%5FFor%5FPublication%20%28GIULIA%29%2FSONIVA%5FPAPER&sortField=LinkFilename&isAscending=true) 

**Full Repository Link:** https://github.com/Clinical-Language-Cognition-Lab/SONIVA_PAPER
