#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Stage 2: VITA joint alignment
#   Phase 1 (warmup): ATC + AIC, image/text frozen
#   Phase 2 (full):   ATC + λ(AIC + ITC), text trainable, CLAP gradually unfrozen
# Risa Shinoda, CVL @ Osaka University

import argparse
import json
import math
import os
import random
import warnings
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import librosa
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchaudio.functional import resample
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
torch.backends.cudnn.benchmark = True

os.environ.setdefault("NCCL_DEBUG", "WARN")

try:
    backs = torchaudio.list_audio_backends()
    if "ffmpeg" in backs:
        torchaudio.set_audio_backend("ffmpeg")
    elif "sox_io" in backs:
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


def is_dist(): return dist.is_initialized()
def is_main_process(): return (not is_dist()) or (dist.get_rank() == 0)


def gather_embeddings(emb: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return emb
    world = dist.get_world_size()
    outs = [torch.zeros_like(emb) for _ in range(world)]
    dist.all_gather(outs, emb)
    outs[dist.get_rank()] = emb
    return torch.cat(outs, dim=0)


def _is_norm_or_bias(name: str) -> bool:
    return any(x in name for x in ("bias", "norm", "ln_", "layer_norm"))


# ---------------------------------------------------------------------------
# Text prompt
# ---------------------------------------------------------------------------

def _norm_sci(s: str) -> str:
    parts = " ".join(str(s or "").strip().split()).split()
    return " ".join(parts[:2]) if parts else ""


def _safe_dir_name(s: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9 _\-]+", "_", (s or "").strip().lower())
    return re.sub(r"_+", "_", re.sub(r"\s+", "_", s)).strip("_")[:120]


def _text_prompt(row: dict) -> str:
    sci = (row.get("scientific_name") or "").strip()
    com = (row.get("com") or sci or "unknown").strip()
    cls = (row.get("class") or "").strip()
    ord_ = (row.get("order") or "").strip()
    fam = (row.get("family") or "").strip()
    gen = (row.get("genus") or "").strip()
    tax = " ".join(x for x in [cls, ord_, fam, gen, sci] if x)
    cands = []
    if com: cands.append("{com}")
    if sci: cands.append("{sci}")
    if tax: cands.append("{tax}")
    if sci and com: cands.append("{sci} with common name {com}")
    if tax and com: cands.append("{tax} with common name {com}")
    return random.choice(cands or ["{com}"]).format(com=com, sci=sci, tax=tax)


# ---------------------------------------------------------------------------
# Image resolver
# ---------------------------------------------------------------------------

class LazyImageResolver:
    EXTS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")

    def __init__(self, img_root: str | Path, cap_per_species: int):
        self.img_root = Path(img_root)
        self.cap = cap_per_species
        self.cache: dict[str, list[str]] = {}

    def get(self, species: str) -> list[str]:
        if species in self.cache:
            return self.cache[species]
        d = self.img_root / _safe_dir_name(species)
        files = []
        try:
            if d.is_dir():
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_file() and e.name.lower().endswith(self.EXTS):
                            files.append(str(d / e.name))
                            if len(files) >= self.cap:
                                break
        except OSError:
            pass
        self.cache[species] = files
        return files


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AVTTripletDataset(Dataset):
    def __init__(self, audio_csv: str | Path, image_resolver: LazyImageResolver,
                 sample_rate: int = 48000, clip_len: float = 10.0,
                 cap_per_species: int = 20, seed: int = 7777,
                 split: str = "train", preprocess_image=None):
        df = pd.read_csv(audio_csv, dtype=str).fillna("")
        for col in ["file_name", "scientific_name", "accepted_name", "class", "order",
                    "family", "genus", "com"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype(str).str.strip()
        df["species_name"] = (
            df["accepted_name"].where(df["accepted_name"] != "", df["scientific_name"])
            .map(_norm_sci).str.lower()
        )
        self.sample_rate = sample_rate
        self.n_samples = int(sample_rate * clip_len)
        self.cap = cap_per_species
        self.seed = seed
        self.split = split
        self.image_resolver = image_resolver
        self.preprocess_image = preprocess_image or T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        pairs = self._build_pairs(df) if is_main_process() else None
        if is_dist():
            box = [pairs]; dist.broadcast_object_list(box, src=0); pairs = box[0]
        self.pairs = pairs or []
        self.order = np.arange(len(self.pairs), dtype=np.int64)

    def _build_pairs(self, df):
        rng = np.random.RandomState(self.seed)
        out = []
        for sp, g in df.groupby("species_name", sort=False):
            imgs = self.image_resolver.get(sp)
            audios = g["file_name"].tolist()
            if not imgs or not audios:
                continue
            txt = _text_prompt(g.iloc[0].to_dict())
            n = self.cap
            aud = list(rng.choice(audios, n, replace=(len(audios) < n)))
            img_idx = rng.choice(len(imgs), n, replace=(len(imgs) < n))
            for a, j in zip(aud, img_idx):
                out.append((a, imgs[int(j)], txt))
        rng.shuffle(out)
        if is_main_process():
            print(f"[dataset] triplets={len(out)}", flush=True)
        return out

    def shuffle_epoch(self, epoch: int):
        np.random.RandomState(self.seed + int(epoch)).shuffle(self.order)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        a_path, v_path, txt = self.pairs[int(self.order[idx])]
        return self._load_audio(a_path), self._load_image(v_path), txt

    def _load_audio(self, path: str) -> torch.Tensor:
        want = self.n_samples
        try:
            si = torchaudio.info(path)
            sr0, total = int(si.sample_rate), int(si.num_frames)
            want_orig = int(round(want * sr0 / self.sample_rate))
            start = random.randint(0, max(0, total - want_orig)) if total > want_orig else 0
            wf, sr_read = torchaudio.load(path, frame_offset=start, num_frames=want_orig)
            if wf.dim() == 2: wf = wf.mean(0)
            wf = wf.to(torch.float32)
            if sr_read != self.sample_rate and wf.numel() > 1:
                wf = resample(wf, sr_read, self.sample_rate)
        except Exception:
            try:
                y, sr1 = librosa.load(path, sr=None, mono=True)
                wf = torch.from_numpy(y).to(torch.float32)
                if sr1 != self.sample_rate and wf.numel() > 1:
                    wf = resample(wf, sr1, self.sample_rate)
            except Exception:
                return torch.zeros(want, dtype=torch.float32)
        if wf.numel() >= want:
            return wf[:want]
        return torch.nn.functional.pad(wf, (0, want - wf.numel()))

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            with Image.open(path) as im:
                return self.preprocess_image(im.convert("RGB"))
        except Exception:
            return torch.zeros(3, 224, 224, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BioCLIP2ImgText_x_CLAP(nn.Module):
    """BioCLIP2 image (always frozen) + text encoder x CLAP audio encoder."""

    def __init__(self, bioclip2_id: str = "imageomics/bioclip-2",
                 clap_id: str = "laion/clap-htsat-unfused"):
        super().__init__()
        self.oc_model, self.preprocess_train, self.preprocess_val = \
            open_clip.create_model_and_transforms(f"hf-hub:{bioclip2_id}")
        self.tokenize = open_clip.get_tokenizer(f"hf-hub:{bioclip2_id}")

        with torch.no_grad():
            img = self.preprocess_val(Image.new("RGB", (224, 224))).unsqueeze(0)
            self.bio_dim = int(self.oc_model.encode_image(img).shape[-1])

        # Image always frozen
        for p in self.oc_model.visual.parameters():
            p.requires_grad = False
        # Text frozen initially; unfrozen at Phase 2
        for p in self.oc_model.transformer.parameters():
            p.requires_grad = False
        if hasattr(self.oc_model, "token_embedding"):
            for p in self.oc_model.token_embedding.parameters():
                p.requires_grad = False

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
        self.oc_model.visual.eval()
        return self

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.oc_model.encode_image(images)

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        return self.oc_model.encode_text(self.tokenize(texts).to(device))

    def encode_audio(self, waveforms: torch.Tensor, sample_rate: int = 48000) -> torch.Tensor:
        audios = [a.detach().cpu().numpy() for a in waveforms]
        inputs = self.clap_proc(audios=audios, sampling_rate=sample_rate,
                                return_tensors="pt", padding=True)
        dev = next(self.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        return self.audio_adapter(self.clap_audio.get_audio_features(**inputs))

    def forward(self, waveforms, images, texts, device, sample_rate=48000):
        a = self.encode_audio(waveforms, sample_rate)
        v = self.encode_image(images)
        t = self.encode_text(texts, device)
        return a, v, t, self.logit_scale.exp()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _pair_loss(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    x = nn.functional.normalize(torch.nan_to_num(gather_embeddings(x)), dim=-1)
    y = nn.functional.normalize(torch.nan_to_num(gather_embeddings(y)), dim=-1)
    L = (x @ y.T) * s
    if not torch.isfinite(L).all():
        return x.sum() * 0.0
    labels = torch.arange(x.size(0), device=x.device)
    return (nn.CrossEntropyLoss()(L, labels) + nn.CrossEntropyLoss()(L.T, labels)) / 2


class AVGate:
    """Gates AIC loss when ATC loss rises too much above baseline."""

    def __init__(self, base_at_loss: float, tol_hi: float = 0.03,
                 tol_lo: float = 0.015, ema_alpha: float = 0.2, cool_steps: int = 20):
        self.thr_hi = base_at_loss * (1 + tol_hi)
        self.thr_lo = base_at_loss * (1 + tol_lo)
        self.alpha = ema_alpha
        self.cool_steps = cool_steps
        self.ema = base_at_loss
        self.av_enabled = True
        self._cool = 0

    def update_and_check(self, L_at: torch.Tensor) -> bool:
        lat = float(L_at.detach().cpu())
        self.ema = self.alpha * lat + (1 - self.alpha) * self.ema
        if self.av_enabled and self.ema > self.thr_hi:
            self.av_enabled = False; self._cool = 0
        elif not self.av_enabled:
            if self.ema <= self.thr_lo:
                self._cool += 1
                if self._cool >= self.cool_steps:
                    self.av_enabled = True
            else:
                self._cool = 0
        return self.av_enabled

    def thresholds(self):
        return self.thr_lo, self.thr_hi, self.ema


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, ks=(1, 5, 10), max_batches=5):
    m = model.module if isinstance(model, DDP) else model
    m.eval()
    ks = sorted(ks)
    tags = [("A", "V"), ("V", "A"), ("A", "T"), ("T", "A"), ("V", "T"), ("T", "V")]
    corr = {t: {k: torch.tensor(0, device=device, dtype=torch.long) for k in ks} for t in tags}
    total = torch.tensor(0, device=device, dtype=torch.long)

    for b_idx, (audio, images, texts) in enumerate(loader):
        if max_batches and b_idx >= max_batches:
            break
        audio = audio.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        a = nn.functional.normalize(m.encode_audio(audio), dim=-1)
        v = nn.functional.normalize(m.encode_image(images), dim=-1)
        t = nn.functional.normalize(m.encode_text(list(texts), device=device), dim=-1)
        s = m.logit_scale.exp()

        def _acc(x, y, tag):
            logits = (x @ y.t()) * s
            bsz = logits.size(0)
            _, topk = torch.topk(logits, k=min(ks[-1], bsz), dim=1)
            for k in ks:
                corr[tag][k] += (topk[:, :min(k, bsz)] ==
                                 torch.arange(bsz, device=device).unsqueeze(1)).any(1).long().sum()

        _acc(a, v, ("A", "V")); _acc(v, a, ("V", "A"))
        _acc(a, t, ("A", "T")); _acc(t, a, ("T", "A"))
        _acc(v, t, ("V", "T")); _acc(t, v, ("T", "V"))
        total += audio.size(0)

    if is_dist():
        for tag in tags:
            for k in ks: dist.all_reduce(corr[tag][k], op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    m.train()
    tot = total.float().clamp_min(1.0)

    def _avg(t1, t2):
        return {k: ((corr[t1][k] + corr[t2][k]).float() / 2 / tot).item() for k in ks}

    return {"A<->V": _avg(("A", "V"), ("V", "A")),
            "A<->T": _avg(("A", "T"), ("T", "A")),
            "V<->T": _avg(("V", "T"), ("T", "V"))}


# ---------------------------------------------------------------------------
# Warmstart
# ---------------------------------------------------------------------------

def load_warmstart(model, ckpt_path: str):
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    sd = {k.removeprefix("module."): v for k, v in raw.items()}
    ret = model.load_state_dict(sd, strict=False)
    if is_main_process():
        print(f"[warmstart] missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}",
              flush=True)
    ls_key = next((k for k in sd if "logit_scale" in k), None)
    if ls_key is not None:
        mm = model.module if isinstance(model, DDP) else model
        with torch.no_grad():
            val = float(sd[ls_key].view(()))
            mm.logit_scale.copy_(torch.tensor(val).clamp(np.log(1 / 100), np.log(100)))
        if is_main_process():
            print(f"[warmstart] logit_scale={val:.4f} (s={math.exp(val):.3f})", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: Warmup (ATC + AIC, text/image frozen)
# ---------------------------------------------------------------------------

def run_phase1(model, train_loader, test_loader, train_sampler, test_sampler,
               device, save_dir, log_path, logs, args):
    mm = model.module if isinstance(model, DDP) else model
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, mm.parameters()),
        lr=args.lr, weight_decay=0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, args.warmup_epochs + 1):
        if train_sampler: train_sampler.set_epoch(epoch)
        train_loader.dataset.shuffle_epoch(epoch)

        total_loss = 0.0; steps = 0
        pbar = tqdm(train_loader, desc=f"[Phase1] Epoch {epoch}", disable=not is_main_process())

        for audio, images, texts in pbar:
            audio = audio.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)

            cm = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
            with cm:
                a_emb, v_emb, t_emb, s = model(audio, images, list(texts), device=device)
                L_at = _pair_loss(a_emb, t_emb, s)
                L_av = _pair_loss(a_emb, v_emb, s)
                loss = 0.8 * L_at + 0.2 * L_av

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()

            with torch.no_grad():
                (model.module if isinstance(model, DDP) else model).logit_scale.clamp_(
                    min=np.log(1 / 100), max=np.log(100))

            total_loss += float(loss.detach().cpu()); steps += 1
            if is_main_process():
                pbar.set_postfix(loss=f"{total_loss / steps:.4f}")

        epoch_loss_t = torch.tensor(total_loss / max(1, steps), device=device)
        if is_dist(): dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.AVG)
        epoch_loss = float(epoch_loss_t)

        if is_main_process():
            ckpt = save_dir / f"phase1_epoch{epoch:03d}.pth"
            state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(state, ckpt)

        if test_sampler: test_sampler.set_epoch(epoch)
        rec = evaluate(model, test_loader, device)
        if is_main_process():
            print(f"[phase1 epoch {epoch}] loss={epoch_loss:.4f} {rec}", flush=True)
            logs.append({"phase": 1, "epoch": epoch, "loss": epoch_loss, **_flatten_rec(rec)})
            log_path.write_text(json.dumps(logs, indent=2))


# ---------------------------------------------------------------------------
# Phase 2: Full alignment (ATC + λ(AIC + ITC), text trainable)
# ---------------------------------------------------------------------------

def run_phase2(model, train_loader, test_loader, train_sampler, test_sampler,
               device, save_dir, log_path, logs, args):
    mm = model.module if isinstance(model, DDP) else model

    # Unfreeze text encoder
    for p in mm.oc_model.transformer.parameters():
        p.requires_grad = True
    if hasattr(mm.oc_model, "token_embedding"):
        for p in mm.oc_model.token_embedding.parameters():
            p.requires_grad = True

    # CLAP and logit_scale start frozen (unfrozen by step below)
    for p in mm.clap_audio.parameters():
        p.requires_grad = False
    mm.logit_scale.requires_grad = False
    with torch.no_grad():
        mm.logit_scale.fill_(np.log(1 / 0.2))

    optimizer = optim.AdamW([
        {"params": [p for n, p in mm.audio_adapter.named_parameters()
                    if p.requires_grad and not _is_norm_or_bias(n)],
         "lr": args.lr_adapter, "weight_decay": 0.05},
        {"params": [p for n, p in mm.audio_adapter.named_parameters()
                    if p.requires_grad and _is_norm_or_bias(n)],
         "lr": args.lr_adapter, "weight_decay": 0.0},
        {"params": [p for n, p in mm.oc_model.named_parameters()
                    if p.requires_grad and not _is_norm_or_bias(n)],
         "lr": args.lr_text, "weight_decay": 0.01},
        {"params": [p for n, p in mm.oc_model.named_parameters()
                    if p.requires_grad and _is_norm_or_bias(n)],
         "lr": args.lr_text, "weight_decay": 0.0},
    ], betas=(0.9, 0.98), eps=1e-8)

    baseline = _measure_at_baseline(model, train_loader, device)
    av_gate = AVGate(base_at_loss=baseline, tol_hi=0.03, tol_lo=0.015,
                     ema_alpha=0.2, cool_steps=20)
    if is_main_process():
        print(f"[Phase2 AT baseline] {baseline:.4f}", flush=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    global_step = 0
    clap_unfrozen = False
    ls_unfrozen = False

    for epoch in range(1, args.epochs + 1):
        if train_sampler: train_sampler.set_epoch(epoch)
        train_loader.dataset.shuffle_epoch(epoch)

        total_loss = 0.0; steps = 0
        pbar = tqdm(train_loader, desc=f"[Phase2] Epoch {epoch}", disable=not is_main_process())

        for audio, images, texts in pbar:
            mm_ = model.module if isinstance(model, DDP) else model

            if not clap_unfrozen and global_step >= args.unfreeze_clap_step:
                for n, p in mm_.clap_audio.named_parameters():
                    if any(k in n for k in ["proj", "final_layer", "layers.11",
                                            "encoder.layers.11"]):
                        p.requires_grad = True
                for grp in [
                    {"params": [p for n, p in mm_.clap_audio.named_parameters()
                                if p.requires_grad and not _is_norm_or_bias(n)],
                     "lr": args.lr_clap, "weight_decay": 0.02},
                    {"params": [p for n, p in mm_.clap_audio.named_parameters()
                                if p.requires_grad and _is_norm_or_bias(n)],
                     "lr": args.lr_clap, "weight_decay": 0.0},
                ]:
                    optimizer.add_param_group(grp)
                clap_unfrozen = True
                if is_main_process():
                    print(f"[step {global_step}] CLAP upper layers unfrozen", flush=True)

            if not ls_unfrozen and global_step >= args.unfreeze_ls_step:
                mm_.logit_scale.requires_grad = True
                ls_unfrozen = True

            audio = audio.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)

            cm = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
            with cm:
                a_emb, v_emb, t_emb, s = model(audio, images, list(texts), device=device)

            a_emb = torch.nan_to_num(a_emb).float()
            v_emb = torch.nan_to_num(v_emb).float()
            t_emb = torch.nan_to_num(t_emb).float()

            with torch.no_grad():
                ls = model.module.logit_scale if isinstance(model, DDP) else model.logit_scale
                ls.clamp_(min=np.log(1 / 100), max=np.log(100))
            s = (model.module.logit_scale if isinstance(model, DDP) else model.logit_scale).exp().float()

            L_at = _pair_loss(a_emb, t_emb, s)
            av_ok = av_gate.update_and_check(L_at)

            if global_step < args.start_av_step:
                lam_av = 0.0
            else:
                lam_av = args.lambda_av_max * min(1.0, (global_step - args.start_av_step) /
                                                  max(1, args.av_ramp_steps))
            L_av = _pair_loss(a_emb, v_emb, s) if (av_ok and lam_av > 0) else L_at.new_tensor(0.0)
            if not (av_ok and lam_av > 0): lam_av = 0.0

            if global_step < args.start_vt_step:
                lam_vt = 0.0; L_vt = L_at.new_tensor(0.0)
            else:
                lam_vt = args.lambda_vt_max * min(1.0, (global_step - args.start_vt_step) /
                                                  max(1, args.vt_ramp_steps))
                L_vt = (_pair_loss(v_emb, t_emb, s) if av_gate.av_enabled
                        else L_at.new_tensor(0.0))
                if not av_gate.av_enabled: lam_vt = 0.0

            _, thr_hi, _ = av_gate.thresholds()
            guard = torch.relu(L_at - torch.tensor(thr_hi, device=L_at.device)) * 2.0
            loss = L_at + lam_av * L_av + lam_vt * L_vt + guard

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad],
                    args.grad_clip, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad],
                    args.grad_clip, error_if_nonfinite=False)
                optimizer.step()

            with torch.no_grad():
                ls = model.module.logit_scale if isinstance(model, DDP) else model.logit_scale
                ls.clamp_(min=np.log(1 / 100), max=np.log(100))

            global_step += 1
            total_loss += float(loss.detach().cpu()); steps += 1
            if is_main_process():
                pbar.set_postfix(loss=f"{total_loss / steps:.4f}",
                                 AT=f"{float(L_at):.3f}",
                                 lam_av=f"{lam_av:.2f}", lam_vt=f"{lam_vt:.2f}")

        epoch_loss_t = torch.tensor(total_loss / max(1, steps), device=device)
        if is_dist(): dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.AVG)
        epoch_loss = float(epoch_loss_t)

        if is_main_process():
            ckpt = save_dir / f"epoch{epoch:03d}.pth"
            state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(state, ckpt)
            print(f"[CKPT] {ckpt}")

        if test_sampler: test_sampler.set_epoch(epoch)
        rec = evaluate(model, test_loader, device)
        if is_main_process():
            print(f"[phase2 epoch {epoch}] loss={epoch_loss:.4f} {rec}", flush=True)
            logs.append({"phase": 2, "epoch": epoch, "loss": epoch_loss, **_flatten_rec(rec)})
            log_path.write_text(json.dumps(logs, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",   required=True)
    p.add_argument("--test_csv",    required=True)
    p.add_argument("--img_root",    required=True)
    p.add_argument("--save_dir",    required=True)
    p.add_argument("--pretrained",  default="", help="Stage 1 checkpoint path")
    p.add_argument("--bioclip2_id", default="imageomics/bioclip-2")
    p.add_argument("--clap_id",     default="laion/clap-htsat-unfused")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--cap_per_species", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=4)
    # Phase 1 (warmup)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--lr",          type=float, default=1e-4, help="Phase 1 learning rate")
    # Phase 2 (VITA)
    p.add_argument("--epochs",      type=int, default=10)
    p.add_argument("--lr_adapter",  type=float, default=1e-4)
    p.add_argument("--lr_text",     type=float, default=5e-6)
    p.add_argument("--lr_clap",     type=float, default=1e-5)
    p.add_argument("--lambda_av_max",   type=float, default=0.1)
    p.add_argument("--lambda_vt_max",   type=float, default=0.1)
    p.add_argument("--start_av_step",   type=int,   default=0)
    p.add_argument("--av_ramp_steps",   type=int,   default=600)
    p.add_argument("--start_vt_step",   type=int,   default=0)
    p.add_argument("--vt_ramp_steps",   type=int,   default=300)
    p.add_argument("--unfreeze_clap_step", type=int, default=300)
    p.add_argument("--unfreeze_ls_step",   type=int, default=600)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_distributed()

    seed = 1337 + (dist.get_rank() if is_dist() else 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    device = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        if torch.cuda.is_available() else "cpu"
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log.json"

    model = BioCLIP2ImgText_x_CLAP(args.bioclip2_id, args.clap_id).to(device)

    if args.pretrained and os.path.isfile(args.pretrained):
        load_warmstart(model, args.pretrained)

    if is_dist():
        dist.barrier()
        ddp_kw = dict(device_ids=[device.index], output_device=device.index) \
            if device.type == "cuda" else {}
        model = DDP(model, find_unused_parameters=True,
                    broadcast_buffers=False, gradient_as_bucket_view=True, **ddp_kw)
    model.train()

    mm = model.module if isinstance(model, DDP) else model
    resolver = LazyImageResolver(args.img_root, args.cap_per_species)
    resolver_test = LazyImageResolver(args.img_root, args.cap_per_species)

    train_ds = AVTTripletDataset(args.train_csv, resolver,
                                 cap_per_species=args.cap_per_species,
                                 split="train", preprocess_image=mm.preprocess_val)
    test_ds = AVTTripletDataset(args.test_csv, resolver_test,
                                cap_per_species=args.cap_per_species,
                                split="test", preprocess_image=mm.preprocess_val)

    loader_kw = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                     persistent_workers=(args.num_workers > 0),
                     prefetch_factor=2 if args.num_workers > 0 else None)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_dist() else None
    test_sampler  = DistributedSampler(test_ds,  shuffle=False) if is_dist() else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None), **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              sampler=test_sampler,  shuffle=False, **loader_kw)

    logs = []
    rec0 = evaluate(model, test_loader, device)
    if is_main_process():
        print(f"[epoch 0] {rec0}", flush=True)
        logs.append({"epoch": 0, "loss": None, **_flatten_rec(rec0)})
        log_path.write_text(json.dumps(logs, indent=2))

    if is_main_process():
        print("=== Phase 1: Warmup (ATC + AIC, text/image frozen) ===", flush=True)
    run_phase1(model, train_loader, test_loader, train_sampler, test_sampler,
               device, save_dir, log_path, logs, args)

    if is_main_process():
        print("=== Phase 2: VITA (ATC + λ(AIC + ITC), text trainable) ===", flush=True)
    run_phase2(model, train_loader, test_loader, train_sampler, test_sampler,
               device, save_dir, log_path, logs, args)

    if is_main_process():
        final = save_dir / "final.pth"
        state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save(state, final)
        print(f"[done] {final}")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_rec(rec: dict) -> dict:
    return {f"{p}_{k}": v for p, d in rec.items() for k, v in d.items()}


@torch.no_grad()
def _measure_at_baseline(model, loader, device, max_batches=20) -> float:
    m = model.module if isinstance(model, DDP) else model
    m.eval()
    losses = []
    for b_idx, (audio, _, texts) in enumerate(loader):
        if b_idx >= max_batches:
            break
        audio = audio.to(device, non_blocking=True)
        a = torch.nan_to_num(m.encode_audio(audio)).float()
        t = torch.nan_to_num(m.encode_text(list(texts), device=device)).float()
        s = m.logit_scale.exp().float()
        L = _pair_loss(a, t, s)
        if torch.isfinite(L):
            losses.append(float(L))
    m.train()
    return sum(losses) / max(1, len(losses)) if losses else 3.0


if __name__ == "__main__":
    main()
