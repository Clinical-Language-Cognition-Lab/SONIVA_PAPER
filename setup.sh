#!/bin/bash

# SONIVA: Speech recOgNItion Validation in Aphasia
# Environment Setup Script
# Authors: Giulia Sanguedolce, Fatemeh Geranmayeh et al.

set -e  # Exit on any error

echo "==========================================="
echo "SONIVA Project Environment Setup"
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
echo "Creating virtual environment..."
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

echo ""
echo "==========================================="
echo "Setup completed successfully!"
echo "==========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source soniva_env/bin/activate"
echo ""
echo "To run the code:"
echo "  python main.py --data_path your_data.xlsx --model SVM"
echo ""
echo "For help:"
echo "  python main.py --help"
echo ""
