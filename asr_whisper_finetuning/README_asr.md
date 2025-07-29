# SONIVA ASR Fine-Tuning – Whisper Medium

This folder contains resources and instructions for using the **fine-tuned Whisper Medium model** trained on the **SONIVA dataset**. The model has been fine-tuned by updating **all trainable parameters (encoder and decoder)** to improve transcription accuracy on aphasic speech.

---

## 📄 Model Details
- **Base model:** Whisper Medium (OpenAI).
- **Fine-tuning:** Full fine-tuning (encoder + decoder parameters).
- **Training data:** SONIVA transcripts (aphasic and control speech).
- **Goal:** Reduce word error rate (WER) for aphasic speech recognition.

---

## 📥 Download the Fine-Tuned Model
The fine-tuned Whisper model is hosted on OneDrive:

**🔗 [Download Whisper Medium – SONIVA Fine-Tuned Model](PUT_YOUR_ONEDRIVE_LINK_HERE)**

After downloading, place the model directory inside:
```bash
`asr_whisper_finetuning/models/whisper_soniva/`
```
---

## 🚀 Testing the Model
To test the fine-tuned model on your own `.wav` files, use the provided `test_asr.py` script.

### **Step 1: Prepare Your Audio Data**
- Ensure your test files are in `.wav` format.
- Recommended sample rate: **16 kHz** (the script automatically resamples).

### **Step 2: Run Transcription**
Run the script for a **directory of audio files**:
```bahs
`python test_asr.py --model_path asr_whisper_finetuning/models/whisper_soniva --audio_dir path/to/wavs/ --output_file my_transcriptions.json`
```

Or for a **single file**:
```bash
`python test_asr.py --model_path asr_whisper_finetuning/models/whisper_soniva --audio_path sample.wav`
```
---

## 🧩 Pipeline Overview
The `test_asr.py` script performs the following steps:

1. **Load the fine-tuned model**  
```python
model, processor = WhisperForConditionalGeneration, WhisperProcessor
```
   Loads both model weights and the processor (tokenizer + feature extractor).

3. **Create dataset**  
   Converts audio files into a Hugging Face `Dataset` with proper 16 kHz sampling.

4. **Generate input features**  
   Uses the processor to extract log-Mel spectrograms required by Whisper.

5. **Inference**  
   Runs `model.generate()` to produce token predictions.

6. **Decode**  
   Converts token predictions to readable text using `processor.batch_decode`.

7. **Save results**  
   Transcriptions are stored in a `.json` file.

---

## 📊 Word Error Rate (WER) Evaluation (Optional)
If you have **reference transcripts**, you can compute WER using the Hugging Face `evaluate` library.

### Example:
```python
import evaluate
import json

# Load the WER metric
wer = evaluate.load("wer")

# Load your predicted transcriptions
with open("my_transcriptions.json") as f:
    preds = json.load(f)

# Example reference transcripts (dict format: filename -> transcript)
references = {
    "file1.wav": "this is the correct transcript",
    "file2.wav": "another reference transcript"
}

# Align predictions and references
pred_texts = [preds[k] for k in references.keys()]
ref_texts = [references[k] for k in references.keys()]

# Compute WER
print("WER:", wer.compute(predictions=pred_texts, references=ref_texts))
```

## 📂 Folder Structure
```bash
asr_whisper_finetuning/
│
├── README_asr.md # Documentation for ASR model usage
├── test_asr.py # Script to test the fine-tuned model
├── download_model.sh # Helper script to download model from OneDrive
└── transcripts/ # .cha transcripts for benchmarking
```

## 📚 Citation
If you use this ASR model, please cite:
```bibtex
@article{sanguedolce2025soniva,
title={SONIVA: Speech recOgNItion Validation in Aphasia},
author={Sanguedolce, Giulia and Price, Cathy J. and Brook, Sophie and Gruia, Dragos C. and Parkinson, Niamh V. and Naylor, Patrick A. and Geranmayeh, Fatemeh},
journal={[Journal Name]},
year={2025},
doi={To be assigned}
}
```
---

## 📞 Contact
- **First Author:** gs2022@ic.ac.uk  
- **Corresponding Author:** fatemeh.geranmayeh00@imperial.ac.uk
