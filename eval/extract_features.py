#!/usr/bin/env python3
"""Extract audio/image/text features from index CSVs and save as .pt files.

Usage (BioVITA model — recommended):
  python extract_features.py \
    --ids_dir path/to/benchmark/ids \
    --feat_root path/to/output/features \
    --tag biовita \
    --vita_model_id risashinoda/BioVITA \
    --modalities audio,image,text

Usage (custom checkpoint):
  python extract_features.py \
    --ids_dir path/to/benchmark/ids \
    --feat_root path/to/output/features \
    --tag my_model \
    --bioclip2_id imageomics/bioclip-2 \
    --clap_id laion/clap-htsat-unfused \
    --ckpt_path path/to/checkpoint.pth \
    --modalities audio,image,text

Multi-GPU (torchrun):
  torchrun --nproc_per_node=8 extract_features.py ...
"""
import argparse
import inspect
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import List

import librosa
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample
from transformers import ClapModel, ClapProcessor

warnings.filterwarnings("ignore", category=UserWarning)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    bks = torchaudio.list_audio_backends()
    if "ffmpeg" in bks:
        torchaudio.set_audio_backend("ffmpeg")
    elif "sox_io" in bks:
        torchaudio.set_audio_backend("sox_io")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def get_dist_env():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local


def is_main():
    return int(os.environ.get("RANK", "0")) == 0


# ---------------------------------------------------------------------------
# Safe I/O (supports parallel multi-GPU extraction)
# ---------------------------------------------------------------------------

def _acquire_lock(p: Path, timeout=30.0, poll=0.05):
    lockp = str(p) + ".lock"
    start = time.time()
    while True:
        try:
            fd = os.open(lockp, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd, lockp
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout acquiring lock: {lockp}")
            time.sleep(poll)


def _release_lock(fd: int, lockp: str):
    try:
        os.close(fd)
    finally:
        try:
            os.unlink(lockp)
        except FileNotFoundError:
            pass


def _safe_load_torch(p: Path, retries=5, wait=0.1):
    last = None
    for _ in range(max(1, retries)):
        try:
            try:
                return torch.load(p, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(p, map_location="cpu")
        except (EOFError, RuntimeError) as e:
            last = e
            time.sleep(wait)
    raise last


def _atomic_torch_save(obj, p: Path):
    p = Path(p)
    with tempfile.NamedTemporaryFile(delete=False, dir=p.parent, suffix=".tmp") as f:
        tmp_path = Path(f.name)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, p)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)



# ---------------------------------------------------------------------------
# FeatureBank — stores per-ID embeddings as {id}.pt files
# ---------------------------------------------------------------------------

class FeatureBank:
    def __init__(self, root="features", tag=None):
        self.root = Path(root) / tag if tag else Path(root)
        self._cache = {"audio": {}, "image": {}, "text": {}}
        for m in ("audio", "image", "text"):
            (self.root / m).mkdir(parents=True, exist_ok=True)

    def _f(self, m, i):
        return self.root / m / f"{int(i)}.pt"

    def has(self, modality, i, sub=None):
        p = self._f(modality, i)
        if not p.exists():
            return False
        if sub is None:
            return True
        try:
            v = _safe_load_torch(p, retries=3, wait=0.05)
        except Exception:
            return False
        return isinstance(v, dict) and (sub in v)

    def put(self, modality, i, v, sub=None):
        p = self._f(modality, i)
        ensure_dir(p.parent)
        if sub is None:
            fd, lockp = _acquire_lock(p)
            try:
                _atomic_torch_save(v, p)
            finally:
                _release_lock(fd, lockp)
            return
        fd, lockp = _acquire_lock(p)
        try:
            cur = {}
            if p.exists():
                try:
                    cur = _safe_load_torch(p, retries=5, wait=0.1)
                except Exception:
                    cur = {}
            if not isinstance(cur, dict):
                cur = {"text": cur}
            cur[sub] = v
            _atomic_torch_save(cur, p)
        finally:
            _release_lock(fd, lockp)

    def get(self, modality, i, sub=None):
        key = (int(i), sub)
        cache = self._cache[modality]
        if key in cache:
            return cache[key]
        p = self._f(modality, i)
        if not p.exists():
            raise FileNotFoundError(f"feature not found: {p}")
        v = _safe_load_torch(p, retries=5, wait=0.1)
        if isinstance(v, dict):
            v = v[sub] if sub is not None else v.get("text", next(iter(v.values())))
        v = v.to(torch.float32)
        cache[key] = v
        return v

    def prefetch(self, modality, ids, sub=None):
        cache = self._cache[modality]
        for i in ids:
            key = (int(i), sub)
            if key in cache:
                continue
            p = self._f(modality, i)
            if not p.exists():
                continue
            try:
                v = _safe_load_torch(p, retries=3, wait=0.05)
            except Exception:
                continue
            if isinstance(v, dict):
                if sub is not None:
                    if sub not in v:
                        continue
                    v = v[sub]
                else:
                    v = v.get("text", next(iter(v.values())))
            cache[key] = v.to(torch.float32)

    def build_from_index(self, modality, index_csv, enc,
                         batch=512, sr=48000, seconds=10.0,
                         num_workers_image=8, num_workers_audio=8,
                         text_levels: List[str] | None = None):
        from tqdm import tqdm
        rank, world, _ = get_dist_env()
        df = pd.read_csv(index_csv)

        if modality == "audio":
            ids = df["id"].tolist()
            payloads = df["file_path"].astype(str).tolist()
            todo = [(i, p) for i, p in zip(ids, payloads) if not self.has("audio", i)]
            todo = todo[rank::world]
            if not todo:
                if is_main():
                    print(f"[audio] nothing to do on rank {rank}")
                return
            if is_main():
                print(f"[audio] {len(todo)} items → {self.root / 'audio'}")
            for s in tqdm(range(0, len(todo), batch), ncols=100, disable=not is_main()):
                chunk = todo[s:s + batch]
                ids_c = [i for i, _ in chunk]
                pls_c = [p for _, p in chunk]
                Z = enc.encode_audios(pls_c, sr=sr, seconds=seconds,
                                      batch_size=min(batch, 64),
                                      num_workers=num_workers_audio)
                for i, z in zip(ids_c, Z):
                    self.put("audio", i, z)
            return

        if modality == "image":
            ids = df["id"].tolist()
            payloads = df["file_path"].astype(str).tolist()
            todo = [(i, p) for i, p in zip(ids, payloads) if not self.has("image", i)]
            todo = todo[rank::world]
            if not todo:
                if is_main():
                    print(f"[image] nothing to do on rank {rank}")
                return
            if is_main():
                print(f"[image] {len(todo)} items → {self.root / 'image'}")
            for s in tqdm(range(0, len(todo), batch), ncols=100, disable=not is_main()):
                chunk = todo[s:s + batch]
                ids_c = [i for i, _ in chunk]
                pls_c = [p for _, p in chunk]
                Z = enc.encode_images(pls_c, batch_size=min(batch, 128),
                                      num_workers=num_workers_image)
                for i, z in zip(ids_c, Z):
                    self.put("image", i, z)
            return

        if modality == "text":
            base_cols = ["id", "species", "genus", "family"]
            extra_cols = [c for c in ["text", "common_name", "com_canon"] if c in df.columns]
            df = df[base_cols + extra_cols].copy()
            for col in ["species", "genus", "family"]:
                df[col] = df[col].astype(str)
            if "common_name" not in df.columns:
                df["common_name"] = df.get("com_canon", df.get("text", "")).astype(str).str.strip()

            ids = df["id"].tolist()
            levels = text_levels or ["species", "genus", "family", "common_name"]
            levels = [lv for lv in levels if lv in df.columns]

            for level in levels:
                level_ids = [i for i in ids if not self.has("text", i, sub=level)]
                level_ids = level_ids[rank::world]
                if not level_ids:
                    if is_main():
                        print(f"[text:{level}] nothing to do on rank {rank}")
                    continue
                if is_main():
                    print(f"[text:{level}] {len(level_ids)} items → {self.root / 'text'}")
                id2txt = dict(zip(df["id"], df[level]))
                level_ids = [i for i in level_ids
                             if isinstance(id2txt[i], str) and id2txt[i].strip()]
                for s in tqdm(range(0, len(level_ids), batch), ncols=100, disable=not is_main()):
                    ids_c = level_ids[s:s + batch]
                    txts_c = [id2txt[i] for i in ids_c]
                    Z = enc.encode_text(txts_c)
                    for i, z in zip(ids_c, Z):
                        self.put("text", i, z, sub=level)
            return

        raise ValueError(f"Unknown modality: {modality}")


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_onset_10s(path: str, n_samples: int, sr_target: int,
                         thr_ratio=0.01, preroll=0.2) -> torch.Tensor:
    wf = None
    sr0 = None
    try:
        w, s = torchaudio.load(path)
        if w.dim() == 2:
            w = w.mean(dim=0)
        wf, sr0 = w, int(s)
    except Exception:
        try:
            y_np, s = librosa.load(path, sr=None, mono=False)
            if y_np.ndim == 2:
                y_np = y_np.mean(axis=0)
            wf = torch.from_numpy(y_np)
            sr0 = int(s)
        except Exception:
            return torch.zeros(n_samples, dtype=torch.float32)

    wf = wf.to(torch.float32)
    if sr0 != sr_target and wf.numel() > 1:
        try:
            wf = resample(wf, sr0, sr_target)
        except Exception:
            return torch.zeros(n_samples, dtype=torch.float32)
    if wf.numel() <= 1:
        return torch.zeros(n_samples, dtype=torch.float32)

    energy = wf.abs()
    th = float(energy.max().item()) * thr_ratio
    nz = (energy > th).nonzero(as_tuple=True)[0]
    start0 = 0 if len(nz) == 0 else max(0, int(nz[0].item()) - int(preroll * sr_target))
    t = wf[start0: start0 + n_samples]
    if t.numel() < n_samples:
        t = F.pad(t, (0, n_samples - t.numel()))
    return t[:n_samples].to(torch.float32)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class Encoders:
    def __init__(self, bioclip2_id, clap_id, ckpt_path=None, device=None,
                 vita_model_id=None):
        _, _, local = get_dist_env()
        if device is None and torch.cuda.is_available():
            device = f"cuda:{local}"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # If vita_model_id is given, use it as the image/text encoder (overrides bioclip2_id)
        oc_id = vita_model_id if vita_model_id else bioclip2_id
        self.txt_img_model, self.img_pre, _ = open_clip.create_model_and_transforms(
            f"hf-hub:{oc_id}")
        self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{oc_id}")
        self.txt_img_model.eval().to(self.device)

        try:
            self.proc = ClapProcessor.from_pretrained(clap_id)
        except Exception:
            alt = str(clap_id).replace("fused", "unfused")
            print(f"[warn] ClapProcessor not found at {clap_id!r}, falling back to {alt!r}")
            self.proc = ClapProcessor.from_pretrained(alt)

        self.clap = ClapModel.from_pretrained(
            clap_id,
            low_cpu_mem_usage=True,
            torch_dtype=(torch.float16 if (torch.cuda.is_available()
                         and self.device.type == "cuda") else None),
        ).to(self.device)
        self.clap.eval()

        try:
            fe = self.proc.feature_extractor
            fe.do_resample = False
            fe.return_attention_mask = False
        except Exception:
            pass

        try:
            sig = inspect.signature(self.proc.__call__)
            self._audio_kw = "audio" if "audio" in sig.parameters else "audios"
        except Exception:
            self._audio_kw = "audio"

        self.adapter = torch.nn.Identity().to(self.device)
        self.target_dim = int(getattr(getattr(self.clap, "config", object()),
                                      "projection_dim", 768))

        if vita_model_id:
            # Load CLAP + adapter from clap_weights.pth in the same HF repo
            from huggingface_hub import hf_hub_download
            clap_path = hf_hub_download(vita_model_id, "clap_weights.pth")
            clap_sd = torch.load(clap_path, map_location="cpu")
            clap_sub = clap_sd.get("clap_audio", {})
            ret = self.clap.load_state_dict(clap_sub, strict=False)
            print(f"[vita] clap_audio: loaded={len(clap_sub)} "
                  f"missing={len(getattr(ret, 'missing_keys', []))}")
            ada_sub = clap_sd.get("audio_adapter", {})
            if "weight" in ada_sub:
                out_dim, in_dim = ada_sub["weight"].shape
                self.adapter = torch.nn.Linear(in_dim, out_dim, bias=False).to(self.device)
                self.adapter.load_state_dict(ada_sub, strict=False)
                self.target_dim = out_dim
                print(f"[vita] audio_adapter: out_dim={out_dim}, in_dim={in_dim}")

        elif ckpt_path:
            sd = torch.load(str(ckpt_path), map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

            clap_sub = {k[11:]: v for k, v in sd.items() if k.startswith("clap_audio.")}
            ret = self.clap.load_state_dict(clap_sub, strict=False)
            print(f"[ckpt] clap_audio: loaded={len(clap_sub)} "
                  f"missing={len(getattr(ret, 'missing_keys', []))} "
                  f"unexpected={len(getattr(ret, 'unexpected_keys', []))}")

            ada_sub = {k[14:]: v for k, v in sd.items() if k.startswith("audio_adapter.")}
            if "weight" in ada_sub:
                out_dim, in_dim = ada_sub["weight"].shape
                self.adapter = torch.nn.Linear(in_dim, out_dim, bias=False).to(self.device)
                self.adapter.load_state_dict(ada_sub, strict=False)
                self.target_dim = out_dim
                print(f"[ckpt] audio_adapter: out_dim={out_dim}, in_dim={in_dim}")

            oc_sub = {k[9:]: v for k, v in sd.items() if k.startswith("oc_model.")}
            if oc_sub:
                ret = self.txt_img_model.load_state_dict(oc_sub, strict=False)
                print(f"[ckpt] oc_model: loaded={len(oc_sub)} "
                      f"missing={len(getattr(ret, 'missing_keys', []))}")

        self._use_amp = (self.device.type == "cuda")
        self._amp_dtype = torch.float16

        for p in self.txt_img_model.parameters():
            p.requires_grad = False
        for p in self.clap.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(texts).to(self.device)
        with torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp):
            z = self.txt_img_model.encode_text(tok)
        return F.normalize(z, dim=-1).float().cpu()

    @torch.no_grad()
    def encode_images(self, paths: List[str], batch_size=128, num_workers=8) -> torch.Tensor:
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset

        class _ImageList(Dataset):
            def __init__(self, ps, pre):
                self.ps = ps
                self.pre = pre

            def __len__(self):
                return len(self.ps)

            def __getitem__(self, i):
                try:
                    img = Image.open(self.ps[i]).convert("RGB")
                    x = self.pre(img)
                except Exception:
                    x = self.pre(Image.new("RGB", (224, 224)))
                return i, x

        ds = _ImageList(paths, self.img_pre)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0),
                            prefetch_factor=2 if num_workers > 0 else None)
        zs = [None] * len(paths)
        for idxs, imgs in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                z = self.txt_img_model.encode_image(imgs)
            z = F.normalize(z, dim=-1).float().cpu()
            for i, zi in zip(idxs.tolist(), z):
                zs[i] = zi
        if not zs or zs[0] is None:
            return torch.zeros(0, 512)
        return torch.stack(zs, dim=0)

    @torch.no_grad()
    def encode_audios(self, paths: List[str], sr=48000, seconds=10.0,
                      batch_size=64, num_workers=8) -> torch.Tensor:
        import numpy as np
        from torch.utils.data import DataLoader, Dataset

        class _AudioList(Dataset):
            def __init__(self, ps, sr, sec):
                self.ps = ps
                self.sr = sr
                self.sec = sec

            def __len__(self):
                return len(self.ps)

            def __getitem__(self, i):
                n = int(self.sr * self.sec)
                w = load_audio_onset_10s(self.ps[i], n, self.sr)
                return i, w.numpy().astype(np.float32, copy=False)

        def _collate(batch):
            idxs, waves = zip(*batch)
            return list(idxs), list(waves)

        ds = _AudioList(paths, sr, seconds)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0),
                            prefetch_factor=2 if num_workers > 0 else None,
                            collate_fn=_collate)
        zs = [None] * len(paths)
        for idxs, wavs in loader:
            kwargs = {self._audio_kw: wavs, "sampling_rate": sr,
                      "return_tensors": "pt", "padding": True}
            try:
                a_in = self.proc(**kwargs)
            except TypeError:
                alt = "audios" if self._audio_kw == "audio" else "audio"
                kwargs.pop(self._audio_kw)
                kwargs[alt] = wavs
                a_in = self.proc(**kwargs)
            a_in = {k: v.to(self.device, non_blocking=True) for k, v in a_in.items()}
            model_dtype = next(self.clap.parameters()).dtype
            for k, v in a_in.items():
                if torch.is_floating_point(v):
                    a_in[k] = v.to(model_dtype)
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                a = self.clap.get_audio_features(**a_in)
            a = self.adapter(a)
            a = F.normalize(a, dim=-1).float().cpu()
            for i, zi in zip(idxs, a):
                zs[int(i)] = zi
        if not zs or zs[0] is None:
            return torch.zeros(0, self.target_dim)
        return torch.stack(zs, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids_dir",    required=True, help="Directory with audio/image/text_index.csv")
    ap.add_argument("--feat_root",  required=True, help="Output root for feature files")
    ap.add_argument("--tag",        default=None,  help="Subdirectory tag under feat_root")
    ap.add_argument("--modalities", default="audio,image,text")
    ap.add_argument("--vita_model_id", default=None,
                    help="HF model ID for BioVITA (e.g. risashinoda/BioVITA). "
                         "If set, loads image/text encoder and audio weights from this repo; "
                         "overrides --bioclip2_id and --ckpt_path.")
    ap.add_argument("--bioclip2_id", default="imageomics/bioclip-2",
                    help="HF ID for the image/text encoder (ignored if --vita_model_id is set)")
    ap.add_argument("--clap_id",     default="laion/clap-htsat-unfused")
    ap.add_argument("--ckpt_path",  default=None,
                    help="Trained model checkpoint (.pth) (ignored if --vita_model_id is set)")
    ap.add_argument("--batch_audio",  type=int, default=512)
    ap.add_argument("--batch_image",  type=int, default=256)
    ap.add_argument("--batch_text",   type=int, default=2048)
    ap.add_argument("--num_workers_image", type=int, default=8)
    ap.add_argument("--num_workers_audio", type=int, default=8)
    ap.add_argument("--sr",      type=int,   default=48000)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--text_levels", default=None,
                    help="Comma-separated text levels to encode, e.g. species,genus "
                         "(default: species,genus,family,common_name)")
    args = ap.parse_args()

    enc = Encoders(args.bioclip2_id, args.clap_id, args.ckpt_path,
                   vita_model_id=args.vita_model_id)
    bank = FeatureBank(args.feat_root, args.tag)

    paths = {
        "audio": f"{args.ids_dir}/audio_index.csv",
        "image": f"{args.ids_dir}/image_index.csv",
        "text":  f"{args.ids_dir}/text_index.csv",
    }
    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    text_levels = ([s.strip() for s in args.text_levels.split(",") if s.strip()]
                   if args.text_levels else None)

    with torch.inference_mode():
        for m in mods:
            bs = (args.batch_audio if m == "audio"
                  else args.batch_image if m == "image"
                  else args.batch_text)
            bank.build_from_index(
                m, paths[m], enc, batch=bs, sr=args.sr, seconds=args.seconds,
                num_workers_image=args.num_workers_image,
                num_workers_audio=args.num_workers_audio,
                text_levels=text_levels,
            )


if __name__ == "__main__":
    main()
