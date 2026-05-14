#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Stage 1: Audio-Text Contrastive (ATC) pretraining
# CLAP (trainable) x BioCLIP2 text encoder (frozen)
# Risa Shinoda, CVL @ Osaka University

import argparse
import json
import os
import random
import warnings
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from time import perf_counter as now

import librosa
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchaudio.functional import resample
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
torch.backends.cudnn.benchmark = True

try:
    if "sox_io" in torchaudio.list_audio_backends():
        torchaudio.set_audio_backend("sox_io")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30),
                                init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def is_dist():
    return dist.is_initialized()


def is_main_process():
    return (not is_dist()) or (dist.get_rank() == 0)


def gather_embeddings(emb: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return emb
    world = dist.get_world_size()
    outs = [torch.zeros_like(emb) for _ in range(world)]
    dist.all_gather(outs, emb)
    outs[dist.get_rank()] = emb
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BioCLIP2xCLAP(nn.Module):
    """CLAP audio encoder aligned to BioCLIP2 text embedding space."""

    def __init__(self,
                 bioclip2_id: str = "imageomics/bioclip-2",
                 clap_id: str = "laion/clap-htsat-unfused"):
        super().__init__()

        self.oc_model, _, _ = open_clip.create_model_and_transforms(f"hf-hub:{bioclip2_id}")
        self.tokenize = open_clip.get_tokenizer(f"hf-hub:{bioclip2_id}")

        with torch.no_grad():
            _tok = self.tokenize(["dummy"])
            self.bio_dim = int(self.oc_model.encode_text(_tok).shape[-1])

        for p in self.oc_model.parameters():
            p.requires_grad = False
        self.oc_model.eval()

        self.clap_proc = ClapProcessor.from_pretrained(clap_id)
        self.clap_audio = ClapModel.from_pretrained(clap_id, use_safetensors=True)

        clap_dim = getattr(getattr(self.clap_audio, "config", object()), "projection_dim", 768)
        self.audio_adapter = (nn.Linear(clap_dim, self.bio_dim, bias=False)
                              if clap_dim != self.bio_dim else nn.Identity())

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

        try:
            fe = self.clap_proc.feature_extractor
            fe.do_resample = False
            fe.return_attention_mask = False
        except Exception:
            pass

    def train(self, mode: bool = True):
        super().train(mode)
        self.oc_model.eval()
        return self

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenize(texts).to(device)
        return self.oc_model.encode_text(tokens)

    def encode_audio(self, waveforms: torch.Tensor, sample_rate: int = 48000) -> torch.Tensor:
        audios = [a.detach().cpu().numpy() for a in waveforms]
        inputs = self.clap_proc(audios=audios, sampling_rate=sample_rate,
                                return_tensors="pt", padding=True)
        dev = next(self.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        return self.audio_adapter(self.clap_audio.get_audio_features(**inputs))

    def forward(self, waveforms, texts, device, sample_rate=48000):
        a = self.encode_audio(waveforms, sample_rate=sample_rate)
        t = self.encode_text(texts, device=device)
        return a, t, self.logit_scale.exp()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def atc_loss(audio_emb: torch.Tensor, text_emb: torch.Tensor,
             logit_scale: torch.Tensor) -> torch.Tensor:
    a = nn.functional.normalize(gather_embeddings(audio_emb), dim=-1)
    t = nn.functional.normalize(gather_embeddings(text_emb), dim=-1)
    logits = a @ t.T * logit_scale
    labels = torch.arange(len(logits), device=logits.device)
    return (nn.CrossEntropyLoss()(logits, labels) +
            nn.CrossEntropyLoss()(logits.T, labels)) / 2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnimalAudioDataset(Dataset):
    def __init__(self, csv_path: str, sample_rate: int = 48000, clip_len: float = 10.0,
                 cap_per_species: int = 20, split: str = "train", seed: int = 7777):
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for col in ["file_name", "scientific_name", "accepted_name", "class", "order",
                    "family", "genus", "com"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype(str).str.strip()

        df["species_name"] = df["accepted_name"].where(df["accepted_name"] != "",
                                                        df["scientific_name"])
        self.sample_rate = sample_rate
        self.n_samples = int(sample_rate * clip_len)
        self.split = split
        self.cap = cap_per_species
        self.seed = seed
        self.full_df = df.reset_index(drop=True)
        self.df_view = self.full_df
        if split == "train":
            self._make_epoch_view(0)
        else:
            self.df_view = df.head(1000).reset_index(drop=True)

    def _make_epoch_view(self, epoch: int):
        import hashlib
        epoch_seed = (self.seed + int(epoch)) % (2 ** 32)

        def _h32(s):
            return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

        picked = []
        for sci, g in self.full_df.groupby("scientific_name", sort=False):
            n = len(g)
            rs = _h32(f"{epoch_seed}|{sci}")
            if n >= self.cap:
                picked.append(g.sample(n=self.cap, replace=False, random_state=rs))
            elif n >= 5:
                picked.append(g.sample(n=n, replace=False, random_state=rs))
            elif n > 0:
                picked.append(g.sample(n=5, replace=True, random_state=rs))
        if picked:
            self.df_view = (pd.concat(picked, ignore_index=True)
                            .sample(frac=1.0, random_state=epoch_seed)
                            .reset_index(drop=True))

    def set_epoch(self, epoch: int):
        if self.split == "train":
            self._make_epoch_view(epoch)

    def _text_prompt(self, row) -> str:
        sci = row.get("scientific_name", "") or ""
        com = row.get("com", "") or sci or "unknown"
        cls = row.get("class", "") or ""
        ord_ = row.get("order", "") or ""
        fam = row.get("family", "") or ""
        gen = row.get("genus", "") or ""
        tax = " ".join(x for x in [cls, ord_, fam, gen, sci] if x)
        candidates = []
        if com: candidates.append("{com}")
        if sci: candidates.append("{sci}")
        if tax: candidates.append("{tax}")
        if sci and com: candidates.append("{sci} with common name {com}")
        if tax and com: candidates.append("{tax} with common name {com}")
        if not candidates:
            candidates = ["{com}"]
        return random.choice(candidates).format(com=com, sci=sci, tax=tax)

    def __len__(self):
        return len(self.df_view)

    def __getitem__(self, idx):
        row = self.df_view.iloc[idx]
        path = row["file_name"]
        waveform = self._load_audio(path)
        if self.split == "train":
            return waveform, self._text_prompt(row)
        else:
            com = (row.get("com", "") or "").strip() or "unknown"
            return waveform, com

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = None, None
        for loader in [self._load_torchaudio, self._load_librosa]:
            try:
                waveform, sr = loader(path)
                break
            except Exception:
                continue
        if waveform is None:
            return torch.zeros(self.n_samples, dtype=torch.float32)
        waveform = waveform.to(torch.float32)
        if sr and sr != self.sample_rate and waveform.numel() > 1:
            waveform = resample(waveform, sr, self.sample_rate)
        if waveform.numel() >= self.n_samples:
            return waveform[:self.n_samples]
        return torch.nn.functional.pad(waveform, (0, self.n_samples - waveform.numel()))

    def _load_torchaudio(self, path):
        si = torchaudio.info(path)
        if si.num_frames > self.n_samples:
            start = random.randint(0, si.num_frames - self.n_samples)
            wf, sr = torchaudio.load(path, frame_offset=start, num_frames=self.n_samples)
        else:
            wf, sr = torchaudio.load(path)
        return wf.mean(0) if wf.dim() == 2 else wf, sr

    def _load_librosa(self, path):
        y, sr = librosa.load(path, sr=None, mono=True)
        return torch.from_numpy(y), sr


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_topk(model, loader, class_texts, device, ks=(1, 5, 10)):
    m = model.module if isinstance(model, DDP) else model
    m.eval()
    text_feats = nn.functional.normalize(m.encode_text(class_texts, device=device), dim=-1)
    class_to_idx = {name: i for i, name in enumerate(class_texts)}
    ks = sorted(ks)
    ks_eff = [max(1, min(k, len(class_texts))) for k in ks]
    correct = {k: torch.tensor(0, device=device, dtype=torch.long) for k in ks}
    total = torch.tensor(0, device=device, dtype=torch.long)
    for audio, com in loader:
        audio = audio.to(device, non_blocking=True)
        labels = torch.tensor([class_to_idx.get(c, 0) for c in com],
                              device=device, dtype=torch.long)
        logits = nn.functional.normalize(m.encode_audio(audio), dim=-1) @ text_feats.t()
        _, topk_idx = torch.topk(logits, k=ks_eff[-1], dim=1)
        for k, k_eff in zip(ks, ks_eff):
            correct[k] += (topk_idx[:, :k_eff] == labels.unsqueeze(1)).any(1).long().sum()
        total += labels.numel()
    if is_dist():
        for k in ks:
            dist.all_reduce(correct[k], op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    m.train()
    return {k: (correct[k] / total.clamp(min=1)).item() for k in ks}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--bioclip2_id", default="imageomics/bioclip-2")
    p.add_argument("--clap_id", default="laion/clap-htsat-unfused")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cap_per_species", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    setup_distributed()

    seed = 1337 + (dist.get_rank() if is_dist() else 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        if torch.cuda.is_available() else "cpu"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log.json"

    model = BioCLIP2xCLAP(args.bioclip2_id, args.clap_id).to(device)
    if is_dist():
        dist.barrier()
        ddp_kw = dict(device_ids=[device.index], output_device=device.index) \
            if device.type == "cuda" else {}
        model = DDP(model, find_unused_parameters=True,
                    broadcast_buffers=False, gradient_as_bucket_view=True, **ddp_kw)
    model.train()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=0.05)

    loader_kw = dict(num_workers=args.num_workers,
                     pin_memory=(device.type == "cuda"),
                     persistent_workers=(args.num_workers > 0),
                     prefetch_factor=2 if args.num_workers > 0 else None)

    train_ds = AnimalAudioDataset(args.train_csv, cap_per_species=args.cap_per_species,
                                  split="train")
    test_ds = AnimalAudioDataset(args.test_csv, cap_per_species=args.cap_per_species,
                                 split="test")

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_dist() else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if is_dist() else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             sampler=test_sampler, shuffle=False, **loader_kw)

    class_texts = sorted({
        (c if c else "unknown")
        for c in test_ds.df_view["com"].fillna("").astype(str).str.strip()
    })

    logs = []

    # epoch 0 eval
    rec0 = evaluate_topk(model, test_loader, class_texts, device)
    if is_main_process():
        print(f"[epoch 0] top1={rec0[1]:.4f} top5={rec0[5]:.4f} top10={rec0[10]:.4f}")
        logs.append({"epoch": 0, "loss": None, **{f"top{k}": v for k, v in rec0.items()}})
        log_path.write_text(json.dumps(logs, indent=2))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, args.epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)

        total_loss = 0.0; steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main_process())

        for audio, txt in pbar:
            audio = audio.to(device, non_blocking=True)
            cm = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
            with cm:
                a_emb, t_emb, logit_scale = model(audio, list(txt), device=device)
                loss = atc_loss(a_emb, t_emb, logit_scale)

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu()); steps += 1
            if is_main_process():
                pbar.set_postfix(loss=f"{total_loss/steps:.4f}")

        epoch_loss_t = torch.tensor(total_loss / max(1, steps), device=device)
        if is_dist():
            dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.AVG)
        epoch_loss = float(epoch_loss_t)

        if is_main_process() and epoch % 5 == 0:
            ckpt = save_dir / f"epoch{epoch:03d}.pth"
            state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(state, ckpt)
            print(f"[CKPT] {ckpt}")

        log_entry = {"epoch": epoch, "loss": epoch_loss}
        if epoch % 10 == 0:
            if test_sampler:
                test_sampler.set_epoch(epoch)
            rec = evaluate_topk(model, test_loader, class_texts, device)
            log_entry.update({f"top{k}": v for k, v in rec.items()})
            if is_main_process():
                print(f"[epoch {epoch}] top1={rec[1]:.4f} top5={rec[5]:.4f} top10={rec[10]:.4f}")

        if is_main_process():
            logs.append(log_entry)
            log_path.write_text(json.dumps(logs, indent=2))

    if is_main_process():
        final = save_dir / "final.pth"
        state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save(state, final)
        print(f"[done] {final}")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
