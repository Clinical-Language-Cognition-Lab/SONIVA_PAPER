# SONIVA: Speech recOgNItion Validation in Aphasia

This repository contains the code for reproducing the results presented in **"SONIVA: Speech recOgNItion Validation in Aphasia"**. The code implements machine learning models (Support Vector Machine, Random Forest, and Neural Network) for classifying medical conditions based on acoustic features.

---

## 📄 Paper Information
**Title:** SONIVA: Speech recOgNItion Validation in Aphasia  
**Authors:** Giulia Sanguedolce, Cathy J. Price, Sophie Brook, Dragos C. Gruia, Niamh V. Parkinson, Patrick A. Naylor, and Fatemeh Geranmayeh  
**Journal/Conference:** –  
**DOI:** –  

---

## 📥 Dataset Access
The SONIVA acoustic features dataset is available for download via OneDrive:  
**[Download SONIVA Dataset](PUT_YOUR_ONEDRIVE_LINK_HERE)**  

For details on dataset structure, metadata, and usage instructions, see **[DATA_ACCESS.md](DATA_ACCESS.md)**.


---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required packages (see `requirements.txt`)

### Installation

**1. Clone this repository:**

```bash
    git clone https://github.com/Clinical-Language-Cognition-Lab/SONIVA_PAPER/acoustic-classification.git
    cd acoustic-classification
```

**2. Create a virtual environment (recommended):**

```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
```
**3. Install dependencies:**
```bash
    pip install -r requirements.txt
```

**4. Download the dataset from OneDrive and place it in the `data/` directory.**

---

## Basic Usage

**Run with a single model (SVM):**
```bash
    python main.py --data_path data/acoustic_features_with_id.xlsx --model SVM
```
**Run all models:**
```bash
    python main.py --data_path data/acoustic_features_with_id.xlsx --model all
```

## 📊 Data Format
The input data should be an Excel file with the following structure:

| ID   | Feature1 | Feature2 | ... | FeatureN | Label   |
|------|----------|----------|-----|----------|---------|
| S001 | 0.123    | 0.456    | ... | 0.789    | Control |
| S001 | 0.234    | 0.567    | ... | 0.890    | Control |
| S002 | 0.345    | 0.678    | ... | 0.901    | Patient |

See **[DATA_ACCESS.md](DATA_ACCESS.md)** for full specifications.


**Required columns:**
- **ID:** Subject identifier (for group-based cross-validation)
- **Label:** Class labels ('Control' or 'Patient')
- **Feature columns:** Numeric acoustic features

**Optional columns:**
- **filename:** Automatically removed if present.

---

## 🔧 Command Line Arguments
```bash
usage: main.py [-h] --data_path DATA_PATH [--model {SVM,RF,NN,all}]
                   [--output_dir OUTPUT_DIR] [--cv_folds CV_FOLDS]

Required arguments:
--data_path DATA_PATH` – Path to the Excel data file

Optional arguments:
--model {SVM,RF,NN,all}` – Model to train (default: SVM)
--output_dir OUTPUT_DIR` – Directory to save results (default: results)
--cv_folds CV_FOLDS` – Number of cross-validation folds (default: 9)

```

## 🧪 Methodology

### Cross-Validation Strategy
- **StratifiedGroupKFold:** Ensures subjects don't appear in both training and validation sets.
- **SMOTE:** Applied to training data to handle class imbalance.
- **9-fold cross-validation** (default).

### Models Implemented
1. **Support Vector Machine (SVM)**
   - RBF kernel
   - Probability estimates enabled
   - Hyperparameters optimized for acoustic data

2. **Random Forest (RF)**
   - 200 estimators
   - Balanced class weights
   - Feature importance analysis

3. **Neural Network (NN)**
   - Architecture: Input → 128 → 64 → 1
   - Batch normalization and layer normalization
   - Dropout regularization (0.4)
   - Focal loss with L1 regularization
   - AdamW optimizer with learning rate scheduling

### Feature Preprocessing
- **MinMaxScaler:** Scales features to [0,1] range
- Applied consistently across data splits

---

## 📊 Results and Outputs

Each run creates a timestamped directory in `results/` containing:
- `results_summary.json`: Complete results for all models
- `experimental_setup.json`: Configuration and data split information
- `*_confusion_matrix.png`: Confusion matrix visualizations
- `neural_network_model.pth`: Saved neural network model (if NN is trained)

**Example Results Structure:**
```bash
results/
└── run\_20241201\_143052/
├── results\_summary.json
├── experimental\_setup.json
├── support\_vector\_machine\_confusion\_matrix.png
├── random\_forest\_confusion\_matrix.png
├── neural\_network\_confusion\_matrix.png
└── neural\_network\_model.pth

```

## 🔬 Reproducibility

### Random Seed Control
All random operations use seed 42 for reproducibility:
- NumPy random operations
- PyTorch random operations
- Scikit-learn random operations
- CUDA operations (if GPU available)

### Environment Reproducibility
We recommend using the exact package versions specified in `requirements.txt`. The code has been tested with:
- Python 3.8, 3.9, 3.10
- CUDA 11.x (optional, for GPU acceleration)

### Hardware Requirements
- **Minimum:** 4GB RAM, 2-core CPU
- **Recommended:** 8GB RAM, 4-core CPU
- **GPU:** Optional (CUDA-compatible for neural networks)


## 📝 Example Usage

```bash

    # Train all models with 5-fold CV
    python main.py \
        --data_path data/acoustic_features_with_id.xlsx \
        --model all \
        --cv_folds 5 \
        --output_dir my_experiment

    # Train only SVM with custom output directory
    python main.py \
        --data_path data/acoustic_features_with_id.xlsx \
        --model SVM \
        --output_dir svm_results
```

## 🤛 Troubleshooting


### Data Issues
- Ensure all feature columns are numeric.
- Check for missing values in the dataset or NaNs.
- Verify that 'Label' column contains only 'Control' and 'Patient' values.


---

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

**Last Updated:** 28/07/2025  
**Version:** 1.0.0
