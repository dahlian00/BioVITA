# BioVITA

Official code for **BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment** (CVPR 2025).

BioVITA aligns audio, image, and text representations for zero-shot wildlife species retrieval using a 3-modal contrastive objective.

- **Model weights**: [risashinoda/BioVITA](https://huggingface.co/risashinoda/BioVITA)
- **Dataset**: [risashinoda/BioVITA](https://huggingface.co/datasets/risashinoda/BioVITA)

---

## Requirements

```bash
pip install torch torchaudio torchvision open_clip_torch transformers \
            librosa pandas tqdm huggingface_hub
```

---

## Data

Download the CSV from HuggingFace and fetch the audio files:

```bash
# 1. Download the CSV
from huggingface_hub import hf_hub_download
csv_path = hf_hub_download("risashinoda/BioVITA", "train/metadata.csv", repo_type="dataset")

# 2. Download audio files and populate file_name
python download_audio.py \
  --csv        path/to/train_metadata.csv \
  --out_dir    path/to/audio \
  --output_csv path/to/train_local.csv \
  --workers    8
```

The output CSV has the `file_name` column filled with local paths, ready for training.

---

## Training

### Stage 1 — Audio-Text Contrastive (ATC)

Aligns CLAP audio encoder to BioCLIP-2 text embedding space.

```bash
torchrun --nproc_per_node=8 train/stage1_audio_text.py \
  --train_csv path/to/train.csv \
  --test_csv  path/to/test.csv \
  --save_dir  path/to/stage1_output
```

### Stage 2 — VITA Joint Alignment

Phase 1 (warmup): ATC + AIC with text/image frozen.  
Phase 2 (full): ATC + λ(AIC + ITC) with text encoder unfrozen.

```bash
torchrun --nproc_per_node=8 train/stage2_vita.py \
  --train_csv  path/to/train.csv \
  --test_csv   path/to/test.csv \
  --img_root   path/to/images \
  --save_dir   path/to/stage2_output \
  --pretrained path/to/stage1_output/epoch030.pth
```

---

## Evaluation

### Step 1 — Extract features

```bash
torchrun --nproc_per_node=8 eval/extract_features.py \
  --ids_dir       path/to/benchmark/ids \
  --feat_root     path/to/features \
  --tag           biовita \
  --vita_model_id risashinoda/BioVITA \
  --modalities    audio,image,text
```

To use a custom checkpoint instead:

```bash
torchrun --nproc_per_node=8 eval/extract_features.py \
  --ids_dir    path/to/benchmark/ids \
  --feat_root  path/to/features \
  --tag        my_model \
  --ckpt_path  path/to/checkpoint.pth \
  --modalities audio,image,text
```

### Step 2 — Run benchmark

```bash
python eval/eval_benchmark.py \
  --base_dir    path/to/benchmark \
  --ids_dir     path/to/benchmark/ids \
  --feat_root   path/to/features \
  --tag         biовita \
  --out_json_dir path/to/results
```

---

## Citation

```bibtex
@inproceedings{shinoda2026biovita,
  title     = {BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment},
  author    = {Risa Shinoda and Kaede Shiohara and Nakamasa Inoue and Kuniaki Saito and Hiroaki Santo and Fumio Okura},
  booktitle = {CVPR},
  year      = {2026},
}
```
