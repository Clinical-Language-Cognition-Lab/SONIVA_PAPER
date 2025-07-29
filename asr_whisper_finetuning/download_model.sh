#!/bin/bash

# Script to download the fine-tuned Whisper model (medium)
# Author: Giulia Sanguedolce
# Date: 29/07/2025

set -e  # Exit if any command fails

echo "==========================================="
echo "Downloading SONIVA fine-tuned Whisper model"
echo "==========================================="

# Replace this with your actual OneDrive link
MODEL_LINK="PUT_YOUR_ONEDRIVE_LINK_HERE"

# Create models directory if it doesn't exist
mkdir -p models

# Download the model
echo "Downloading model from OneDrive..."
curl -L -o models/soniva_whisper_medium_finetuned.zip "$MODEL_LINK"

# Unzip the model
echo "Extracting model..."
unzip -o models/soniva_whisper_medium_finetuned.zip -d models/

echo "==========================================="
echo "Model downloaded and extracted successfully!"
echo "==========================================="

