#!/usr/bin/env python3
"""Download audio files listed in a BioVITA CSV and populate the file_name column.

Usage:
  python download_audio.py \
    --csv        path/to/train.csv \
    --out_dir    path/to/audio \
    --output_csv path/to/train_local.csv \
    --workers    8
"""
import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


EXT_FROM_CONTENT_TYPE = {
    "audio/mpeg":  ".mp3",
    "audio/mp3":   ".mp3",
    "audio/mp4":   ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/wav":   ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg":   ".ogg",
    "audio/flac":  ".flac",
}


def _guess_ext(resp: requests.Response, url: str) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename=["\']?([^"\';\s]+)', cd)
    if m:
        ext = Path(m.group(1)).suffix
        if ext:
            return ext
    ct = resp.headers.get("Content-Type", "").split(";")[0].strip()
    if ct in EXT_FROM_CONTENT_TYPE:
        return EXT_FROM_CONTENT_TYPE[ct]
    url_path = url.split("?")[0]
    ext = Path(url_path).suffix
    if ext and len(ext) <= 5:
        return ext
    return ".mp3"


def download_one(row, out_dir: Path, session: requests.Session, timeout: int, retries: int):
    url = str(row.get("download_url", "") or "").strip()
    if not url:
        return None, "no_url"

    src = str(row.get("source", "")).strip()
    rid = str(row.get("recording_id", "")).strip()
    stem = f"{src}_{rid}" if rid else f"{src}_{row.name}"

    existing = list(out_dir.glob(f"{stem}.*"))
    if existing:
        return str(existing[0]), None

    last_err = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            ext = _guess_ext(resp, resp.url)
            out_path = out_dir / f"{stem}{ext}"
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            return str(out_path), None
        except Exception as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None, last_err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv",         required=True, help="Input CSV with download_url column")
    ap.add_argument("--out_dir",     required=True, help="Directory to save audio files")
    ap.add_argument("--output_csv",  required=True, help="Output CSV with file_name filled in")
    ap.add_argument("--workers",     type=int, default=8)
    ap.add_argument("--timeout",     type=int, default=30)
    ap.add_argument("--retries",     type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "file_name" not in df.columns:
        df["file_name"] = ""
    df["file_name"] = df["file_name"].fillna("").astype(str)

    todo_idx = df.index[df["file_name"].str.strip() == ""].tolist()
    print(f"Total rows: {len(df)}, to download: {len(todo_idx)}")

    fail_log = Path(args.output_csv).parent / "download_failures.txt"

    session = requests.Session()
    session.headers.update({"User-Agent": "BioVITA/1.0 (research)"})

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(download_one, df.loc[i], out_dir, session,
                      args.timeout, args.retries): i
            for i in todo_idx
        }
        with tqdm(total=len(futures), ncols=100) as pbar:
            for fut in as_completed(futures):
                idx = futures[fut]
                path, err = fut.result()
                if path:
                    df.at[idx, "file_name"] = path
                else:
                    failures.append((idx, str(df.at[idx, "download_url"]), err))
                pbar.update(1)

    df.to_csv(args.output_csv, index=False)
    print(f"[saved] {args.output_csv}")

    if failures:
        with open(fail_log, "w") as f:
            for idx, url, err in failures:
                f.write(f"{idx}\t{url}\t{err}\n")
        print(f"[failures] {len(failures)} → {fail_log}")


if __name__ == "__main__":
    main()
