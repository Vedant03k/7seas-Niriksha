"""
Niriksha — Fine-tune Wav2Vec2 for Audio Deepfake Detection
==========================================================

This script fine-tunes MelodyMachine/Deepfake-audio-detection-V2 on:
  1. WaveFake dataset (neural vocoder fakes — catches ElevenLabs-style TTS)
  2. Your own custom samples (optional — put them in data/real/ and data/fake/)

Usage:
  python train_model.py                          # Train on WaveFake only
  python train_model.py --add-custom              # Train on WaveFake + your custom samples
  python train_model.py --custom-only             # Train only on your custom samples
  python train_model.py --epochs 10               # More epochs
  python train_model.py --no-download             # Skip download if you already have data

The fine-tuned model is saved to  backend/models/niriksha-audio-v1/
The audio_detector.py will auto-detect and use it on next server restart.
"""

import os
import sys
import argparse
import random
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────

BASE_MODEL = "MelodyMachine/Deepfake-audio-detection-V2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models", "niriksha-audio-v1")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WAVEFAKE_DIR = os.path.join(DATA_DIR, "wavefake")
CUSTOM_REAL_DIR = os.path.join(DATA_DIR, "real")
CUSTOM_FAKE_DIR = os.path.join(DATA_DIR, "fake")

SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 10       # Truncate audio to 10 seconds max
MAX_AUDIO_LEN = SAMPLE_RATE * MAX_AUDIO_SEC


# ────────────────────────────────────────────────────────────────────────
# Audio loading (mirrors audio_detector.py — uses ffmpeg)
# ────────────────────────────────────────────────────────────────────────

def load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio file as 16kHz mono float32 numpy via ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", file_path,
             "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp.name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {file_path}: {result.stderr[:200]}")
        import soundfile as sf
        data, _ = sf.read(tmp.name, dtype="float32")
        return data
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


# ────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def collect_files(directory: str, label: int) -> list:
    """Recursively collect (filepath, label) pairs."""
    items = []
    directory = Path(directory)
    if not directory.exists():
        return items
    for f in directory.rglob("*"):
        if f.suffix.lower() in AUDIO_EXTS and f.is_file():
            items.append((str(f), label))
    return items


class AudioDataset(Dataset):
    """Lazy-loading audio dataset with on-the-fly ffmpeg decoding."""

    def __init__(self, file_list: list, processor: Wav2Vec2FeatureExtractor):
        self.file_list = file_list   # list of (path, label)
        self.processor = processor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        try:
            audio = load_audio(path)
        except Exception:
            # Return silence on failure — will be a small fraction of data
            audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

        # Truncate to MAX_AUDIO_LEN
        if len(audio) > MAX_AUDIO_LEN:
            # Random crop for augmentation during training
            start = random.randint(0, len(audio) - MAX_AUDIO_LEN)
            audio = audio[start : start + MAX_AUDIO_LEN]

        # Pad short clips
        if len(audio) < SAMPLE_RATE:  # minimum 1 second
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))

        inputs = self.processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False
        )
        input_values = inputs.input_values.squeeze(0)

        return {"input_values": input_values, "labels": torch.tensor(label, dtype=torch.long)}


# Data collator (pad to same length within batch)


class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = torch.stack([f["labels"] for f in features])

        # Pad to max length in batch
        max_len = max(v.shape[0] for v in input_values)
        padded = []
        for v in input_values:
            if v.shape[0] < max_len:
                v = torch.nn.functional.pad(v, (0, max_len - v.shape[0]))
            padded.append(v)

        return {
            "input_values": torch.stack(padded),
            "labels": labels,
        }


# ────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


# ────────────────────────────────────────────────────────────────────────
# HuggingFace dataset download helper
# ────────────────────────────────────────────────────────────────────────

def download_hf_dataset(data_dir: str, max_per_class: int = 0):
    """
    Download garystafford/deepfake-audio-detection from HuggingFace and organize
    into data_dir/real/ and data_dir/fake/ directories.
    
    Dataset info:
      - 1,866 FLAC samples (933 real, 933 fake)
      - Includes ElevenLabs, Amazon Polly, Speechify, Kokoro, Hume AI, Luvvoice
      - Label mapping in dataset: 0=real, 1=fake (alphabetical audiofolder)
      - We save into real/ and fake/ dirs; collect_files assigns model labels later
    """
    os.makedirs(data_dir, exist_ok=True)
    marker = os.path.join(data_dir, ".downloaded")
    if os.path.exists(marker):
        print("[Dataset] Already downloaded. Skipping. (Delete data/wavefake/.downloaded to re-download)")
        return

    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("Installing 'datasets' library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "soundfile"], check=True)
        from datasets import load_dataset, Audio

    import soundfile as sf

    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print("[Dataset] Loading 'garystafford/deepfake-audio-detection' from HuggingFace...")
    print("  1,866 samples (933 real + 933 fake), includes ElevenLabs/Polly/Speechify/etc.")

    ds = load_dataset("garystafford/deepfake-audio-detection", split="train")
    # Cast audio column using soundfile backend to avoid torchcodec issues
    ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    real_count = 0
    fake_count = 0
    total = len(ds)

    for i, sample in enumerate(ds):
        if i % 200 == 0:
            print(f"  Processing {i}/{total} ... (real: {real_count}, fake: {fake_count})")

        label = sample.get("label", None)
        audio_data = sample.get("audio", None)

        if label is None or audio_data is None:
            continue

        # decode=False gives us raw bytes; decode with soundfile manually
        audio_bytes = audio_data["bytes"]
        import io
        try:
            waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            continue

        # Convert stereo to mono if needed
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy.signal import resample
            num_samples = int(len(waveform) * 16000 / sr)
            waveform = resample(waveform, num_samples).astype(np.float32)

        # Dataset label: 0=real, 1=fake (alphabetical audiofolder convention)
        if label == 0:  # real
            if max_per_class > 0 and real_count >= max_per_class:
                continue
            out_path = os.path.join(real_dir, f"real_{real_count:06d}.wav")
            real_count += 1
        else:  # label == 1 → fake
            if max_per_class > 0 and fake_count >= max_per_class:
                continue
            out_path = os.path.join(fake_dir, f"fake_{fake_count:06d}.wav")
            fake_count += 1

        sf.write(out_path, waveform, 16000)

        # Early stop if we have enough of both classes
        if max_per_class > 0 and real_count >= max_per_class and fake_count >= max_per_class:
            break

    print(f"  Saved {real_count} real + {fake_count} fake samples")

    with open(marker, "w") as f:
        f.write("done")
    print("[Dataset] Download complete!")


#main training function with argparse for options

def main():
    parser = argparse.ArgumentParser(description="Fine-tune audio deepfake detection model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (reduce if OOM, default: 4)")
    parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate (default: 2e-6)")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples per class (0 = all, useful for quick test)")
    parser.add_argument("--add-custom", action="store_true", help="Also include data/real/ and data/fake/ custom samples")
    parser.add_argument("--custom-only", action="store_true", help="Only use data/real/ and data/fake/ (skip WaveFake)")
    parser.add_argument("--no-download", action="store_true", help="Skip WaveFake download (use existing data)")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Fraction of data for evaluation (default: 0.1)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow. A GPU is strongly recommended.")
        print("         Expect ~8-12 hours on CPU vs ~2-4 hours on GPU.\n")

    # ── Step 1: Prepare data ─────────────────────────────────────────

    all_files = []  # (path, label)   label: 0=fake, 1=real  (matches model config)

    if not args.custom_only:
        if not args.no_download:
            print("\n=== Downloading Deepfake Audio Dataset ===\n")
            download_hf_dataset(WAVEFAKE_DIR, max_per_class=args.max_samples if args.max_samples > 0 else 0)

        print("\n=== Collecting dataset files ===")
        wf_real = collect_files(os.path.join(WAVEFAKE_DIR, "real"), label=1)
        wf_fake = collect_files(os.path.join(WAVEFAKE_DIR, "fake"), label=0)
        print(f"  Dataset real:  {len(wf_real)} files")
        print(f"  Dataset fake:  {len(wf_fake)} files")
        all_files.extend(wf_real)
        all_files.extend(wf_fake)

    if args.add_custom or args.custom_only:
        print("\n=== Collecting custom samples ===")
        os.makedirs(CUSTOM_REAL_DIR, exist_ok=True)
        os.makedirs(CUSTOM_FAKE_DIR, exist_ok=True)
        c_real = collect_files(CUSTOM_REAL_DIR, label=1)
        c_fake = collect_files(CUSTOM_FAKE_DIR, label=0)
        print(f"  Custom real:  {len(c_real)} files")
        print(f"  Custom fake:  {len(c_fake)} files")
        if len(c_real) == 0 and len(c_fake) == 0:
            print("  WARNING: No custom samples found!")
            print(f"  Put real audio in:  {CUSTOM_REAL_DIR}")
            print(f"  Put fake audio in:  {CUSTOM_FAKE_DIR}")
            if args.custom_only:
                print("  Exiting because --custom-only was set but no custom data found.")
                return
        all_files.extend(c_real)
        all_files.extend(c_fake)

    if not all_files:
        print("\nERROR: No training data found. Run without --no-download or add custom samples.")
        return

    # Balance classes
    real_files = [f for f in all_files if f[1] == 1]
    fake_files = [f for f in all_files if f[1] == 0]
    min_class = min(len(real_files), len(fake_files))

    if args.max_samples > 0:
        min_class = min(min_class, args.max_samples)

    random.shuffle(real_files)
    random.shuffle(fake_files)
    balanced = real_files[:min_class] + fake_files[:min_class]
    random.shuffle(balanced)

    print(f"\n=== Dataset Summary ===")
    print(f"  Balanced: {min_class} real + {min_class} fake = {len(balanced)} total")

    # Train/eval split
    split_idx = int(len(balanced) * (1 - args.eval_split))
    train_files = balanced[:split_idx]
    eval_files = balanced[split_idx:]
    print(f"  Train: {len(train_files)}  |  Eval: {len(eval_files)}")

    # ── Step 2: Load model ───────────────────────────────────────────

    print(f"\n=== Loading base model: {BASE_MODEL} ===\n")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(BASE_MODEL)

    # Freeze only the CNN feature extractor — fine-tune all transformer layers + classifier
    model.freeze_feature_encoder()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ── Step 3: Build datasets ───────────────────────────────────────

    train_dataset = AudioDataset(train_files, processor)
    eval_dataset = AudioDataset(eval_files, processor)
    data_collator = DataCollator(processor)

    # ── Step 4: Training arguments ───────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        label_smoothing_factor=0.1,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2 if sys.platform != "win32" else 0,
        remove_unused_columns=False,
        report_to="none",
    )

    # ── Step 5: Train! ───────────────────────────────────────────────

    print(f"\n=== Starting Training ===")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} grad_accum = {args.batch_size * args.grad_accum} effective")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {OUTPUT_DIR}\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── Step 6: Save final model ─────────────────────────────────────

    print(f"\n=== Saving fine-tuned model to {OUTPUT_DIR} ===")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Evaluate
    results = trainer.evaluate()
    print(f"\n=== Final Evaluation ===")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")
    print(f"  Loss:     {results['eval_loss']:.4f}")

    print(f"\n=== Done! ===")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Restart the backend server and audio_detector.py will auto-load your fine-tuned model.")


if __name__ == "__main__":
    main()
