# SONIVA Dataset Access

## 📥 Quick Access
The SONIVA acoustic features dataset is available for download via OneDrive:  
**🔗 [Download SONIVA Dataset](PUT_YOUR_ONEDRIVE_LINK_HERE)**

---

## What's Included

- **acoustic_features_with_id.xlsx** – Main dataset
  - Acoustic features extracted from speech samples
  - Subject IDs for group-based cross-validation
  - Control/Patient labels
  - Ready for use with the analysis code

- **metadata.xlsx** – Dataset documentation
  - Feature descriptions and units
  - Data collection methodology
  - Sample characteristics
  - Quality control information

- **README_data.txt** – Usage instructions
  - File format specifications
  - Loading instructions
  - Citation requirements

---

## 📋 Dataset Specifications

### Data Structure
- **Format:** Excel (.xlsx)
- **Subjects:** [NUMBER] unique participants
- **Samples:** [NUMBER] total speech samples
- **Features:** [NUMBER] acoustic features
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

## 🚀 Quick Start

1. Download the data from the OneDrive link above.
2. Extract to your project's `data/` directory.
3. Run the analysis:

```bash
# Test that data loads correctly
python main.py --data_path data/acoustic_features_with_id.xlsx --model SVM --cv_folds 3

# Full analysis with all models
python main.py --data_path data/acoustic_features_with_id.xlsx --model all
```

---

## 📝 Usage Requirements

### Citation
When using this dataset, please cite our paper:
```bibtex
@article{sanguedolce2025soniva,
  ...
}
```

### Data Attribution
Please include the following acknowledgment in your work:
> "Acoustic feature data used in this study was obtained from the SONIVA project (Sanguedolce et al., 2025), available at: [GitHub repository link]."

---

## 🔧 Technical Notes

### File Handling
```python
import pandas as pd

# Load the dataset
df = pd.read_excel('data/acoustic_features_with_id.xlsx')

# Check data structure
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Subjects: {df['ID'].nunique()}")
```

### Compatibility
- **Python:** 3.8+ (tested with 3.8, 3.9, 3.10)
- **Pandas:** ≥1.3.0
- **Excel readers:** openpyxl, xlrd

### Common Issues
- **File path:** Ensure the Excel file is in the correct directory.
- **Column names:** Feature names are case-sensitive.
- **Missing data:** Contact authors if download is incomplete.

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
- **Last updated:** July 28, 2025
- **Update notifications:** Watch this repository for announcements.

---

## 🌐 Alternative Access
If you cannot access the OneDrive link:
- Contact the authors directly using the email addresses above.
- Check repository issues for known access problems.

**Download Link:** [PUT_YOUR_ONEDRIVE_LINK_HERE]  
**Repository:** https://github.com/Clinical-Language-Cognition-Lab/SONIVA_PAPER
