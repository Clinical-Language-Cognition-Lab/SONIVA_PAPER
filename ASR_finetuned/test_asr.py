import os
import json
import torch
from datasets import Dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
import argparse

def load_model(model_path):
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor, device

def transcribe_audio(model, processor, device, audio_path):
    dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
    audio_array = dataset[0]["audio"]["array"]
    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def transcribe_directory(model, processor, device, audio_dir, output_file):
    wav_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
    transcriptions = {}
    for wav_file in tqdm(wav_files):
        filename = os.path.basename(wav_file)
        transcriptions[filename] = transcribe_audio(model, processor, device, wav_file)
    with open(output_file, "w") as f:
        json.dump(transcriptions, f, indent=2)
    print(f"Transcriptions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SONIVA fine-tuned Whisper model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned Whisper model")
    parser.add_argument("--audio_path", type=str, help="Path to a single .wav file")
    parser.add_argument("--audio_dir", type=str, help="Path to a directory of .wav files")
    parser.add_argument("--output_file", type=str, default="asr_transcriptions.json", help="Path to save transcriptions JSON")
    args = parser.parse_args()

    model, processor, device = load_model(args.model_path)

    if args.audio_path:
        transcription = transcribe_audio(model, processor, device, args.audio_path)
        print(f"Transcription for {args.audio_path}:\n{transcription}")
    elif args.audio_dir:
        transcribe_directory(model, processor, device, args.audio_dir, args.output_file)
    else:
        print("Please specify either --audio_path or --audio_dir.")
