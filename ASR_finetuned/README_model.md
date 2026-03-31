# SONIVA ASR Fine-Tuning – Whisper Medium

This folder contains resources and instructions for using the fine-tuned Whisper Medium model trained on the SONIVA dataset. The model has been fine-tuned by updating all trainable parameters (encoder and decoder) to improve transcription accuracy on post-stroke speech.

---

## 📄 Model Details

* **Base model**: Whisper Medium (OpenAI)
* **Fine-tuning**: Full fine-tuning (encoder + decoder parameters)
* **Training data**: SONIVA transcripts (patients speech only)
* **Goal**: Reduce word error rate (WER) for aphasic speech recognition

---

## 📥 Accessing the Fine-Tuned Model

Due to GitHub file size limitations, the fine-tuned Whisper model is stored in this folder as split archive parts:

* `whisper-medium-finetuned.tar.gz.part-aa`
* `whisper-medium-finetuned.tar.gz.part-ab`

### ⚠️ Manual Step Required

Before using the model, reconstruct the archive from the split parts.

From inside `ASR_finetuned/`, run:

```bash
cat whisper-medium-finetuned.tar.gz.part-* > whisper-medium-finetuned.tar.gz
```

Then extract it:

```bash
tar -xzf whisper-medium-finetuned.tar.gz
```

After extraction, use the resulting folder as the path passed to `--model_path`.

---

## 🚀 Testing the Model

To test the fine-tuned model on your own `.wav` files, use the provided `test_asr.py` script.

### Step 1: Prepare Your Audio Data

* Ensure your test files are in `.wav` format
* Recommended sample rate: **16 kHz** (the script automatically resamples)

### Step 2: Run Transcription

For a directory of audio files:

```bash
python test_asr.py --model_path PATH_TO_EXTRACTED_MODEL --audio_dir path/to/wavs/ --output_file my_transcriptions.json
```

Or for a single file:

```bash
python test_asr.py --model_path PATH_TO_EXTRACTED_MODEL --audio_path sample.wav
```

---

## 🧩 Pipeline Overview

The `test_asr.py` script performs the following steps:

1. **Load the fine-tuned model**
   Loads both model weights and the processor (`WhisperForConditionalGeneration` and `WhisperProcessor`).

2. **Create dataset**
   Converts audio files into a Hugging Face Dataset with proper 16 kHz sampling.

3. **Generate input features**
   Uses the processor to extract log-Mel spectrograms required by Whisper.

4. **Inference**
   Runs `model.generate()` to produce token predictions.

5. **Decode**
   Converts token predictions to readable text using `processor.batch_decode`.

6. **Save results**
   Transcriptions are stored in a `.json` file.

---

## 📊 Word Error Rate (WER) Evaluation (Optional)

If you have reference transcripts, you can compute WER using your preferred method.

Before evaluation, ensure that you clean the transcriptions (e.g., remove empty strings, normalize casing and punctuation) to obtain accurate metrics.

Proceed with your normal WER calculation pipeline using the predicted `.json` file and your reference transcripts.

---

## 📂 Folder Structure

```text
ASR_finetuned/
│
├── README.md
├── test_asr.py
├── whisper-medium-finetuned.tar.gz.part-aa
└── whisper-medium-finetuned.tar.gz.part-ab
```

---

## 📚 Citation

If you use this ASR model, please cite:

```bibtex
@article{sanguedolce2025soniva,
...
}
```

---

## 📞 Contact

* First Author: [gs2022@ic.ac.uk](mailto:gs2022@ic.ac.uk)
* Corresponding Author: [fatemeh.geranmayeh00@imperial.ac.uk](mailto:fatemeh.geranmayeh00@imperial.ac.uk)
