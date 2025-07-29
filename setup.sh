#!/bin/bash

# SONIVA: Speech recOgNItion Validation in Aphasia
# Environment Setup Script
# Authors: Giulia Sanguedolce, Fatemeh Geranmayeh et al.
# Last Updated: 29/07/2025

set -e  # Exit on any error

echo "==========================================="
echo "SONIVA Project - Environment Setup"
echo "==========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Create virtual environment
echo "Creating virtual environment (soniva_env)..."
python3 -m venv soniva_env

# Activate virtual environment
echo "Activating virtual environment..."
source soniva_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Create data directories if not exist
echo "Ensuring data directories exist..."
mkdir -p acoustic_classification/data
mkdir -p asr_whisper_finetuning/data

echo ""
echo "==========================================="
echo "Setup completed successfully!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Download the dataset and transcripts from OneDrive (see DATA_ACCESS.md)."
echo "   - Place acoustic features in: acoustic_classification/data/"
echo "   - Place .cha transcripts and Whisper model in: asr_whisper_finetuning/data/"
echo ""
echo "2. To activate the environment later, run:"
echo "   source soniva_env/bin/activate"
echo ""
echo "3. Example commands:"
echo "   (a) Acoustic classification:"
echo "       cd acoustic_classification"
echo "       python main.py --data_path data/acoustic_features_with_id.xlsx --model SVM"
echo ""
echo "   (b) ASR fine-tuning (see README_asr.md):"
echo "       cd asr_whisper_finetuning"
echo "       python train_asr.py --config configs/whisper_soniva.yaml"
echo ""
echo "For help with acoustic classification:"
echo "  python main.py --help"
echo ""

