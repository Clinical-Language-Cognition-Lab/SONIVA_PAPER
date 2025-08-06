# Acoustic Classification – SONIVA Project

This folder contains the code and resources for **acoustic feature-based classification** of aphasic vs. control speech segments from the SONIVA dataset. It includes machine learning models (**Support Vector Machine, Random Forest, Neural Network**) trained on openSMILE-extracted features.

---

## 📥 Dataset Access
The acoustic features and metadata can be downloaded from the Drive:  
**[Download Acoustic Dataset](https://drive.google.com/drive/folders/1lqyKebne8jIBaTeD9MjTh6M2Kf5hsbVW?usp=sharing)**
  

Details about the dataset structure and usage are provided in **[DATA_ACCESS.md](../DATA_ACCESS.md)**.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

### Installation
1. Clone this repository:
```bash
   git clone https://github.com/Clinical-Language-Cognition-Lab/SONIVA_PAPER.git  
   cd SONIVA_PAPER/acoustic_classification
```
2. Create a virtual environment (recommended):
```bash
   python -m venv venv  
   source venv/bin/activate  # Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Place the dataset (`acoustic_features_with_id.xlsx`) in the `data/` directory.

---

## Running the Models

**Train a single model (e.g., SVM):**  
```bash
python main.py --data_path data/acoustic_features_with_id.xlsx --model SVM
```
**Train all models:**  
```bash
python main.py --data_path data/acoustic_features_with_id.xlsx --model all
```
---

## 📊 Data Format
Input data must be in Excel (.xlsx) format with the following structure:

```bash
| ID   | Feature1 | Feature2 | ... | FeatureN | Label   |
|------|----------|----------|-----|----------|---------|
| S001 | 0.123    | 0.456    | ... | 0.789    | Control |
| S002 | 0.234    | 0.567    | ... | 0.890    | Patient |
```

- **ID:** Unique subject identifier (used for group-based cross-validation).  
- **Label:** 'Control' or 'Patient'.  
- **Features:** Numeric acoustic features.

---

## 🔧 Command Line Arguments
```bash
usage: main.py [-h] --data_path DATA_PATH [--model {SVM,RF,NN,all}]  
               [--output_dir OUTPUT_DIR] [--cv_folds CV_FOLDS]
```
- `--data_path` : Path to the dataset Excel file.  
- `--model` : Model to train (SVM, RF, NN, or all).  
- `--output_dir` : Directory to save results (default: results/).  
- `--cv_folds` : Number of cross-validation folds (default: 9).

---

## 🧪 Methodology
- **Cross-Validation:** StratifiedGroupKFold ensures no subject overlap across folds.  
- **SMOTE:** Applied to handle class imbalance.  
- **Models:**  
  - SVM (RBF kernel, probability estimates).  
  - Random Forest (200 estimators, balanced class weights).  
  - Neural Network (Input → 128 → 64 → 1) with focal loss, dropout (0.4), and AdamW optimization.  
- **Preprocessing:** Features are scaled to `[0,1]` with MinMaxScaler.

---

## 📊 Results
Training results are saved in `results/`, including:
- results_summary.json
- experimental_setup.json
- Confusion matrices (e.g., *_confusion_matrix.png)
- Saved NN model (neural_network_model.pth)

---

## 🔬 Reproducibility
- Random seed fixed (42) across NumPy, PyTorch, and scikit-learn.
- Tested with Python 3.8–3.10.

---

## 📞 Contact
- **First Author:** gs2022@ic.ac.uk  
- **Corresponding Author:** fatemeh.geranmayeh00@imperial.ac.uk  
- **Issues:** Use GitHub Issues for technical questions.

