#!/usr/bin/env python3
"""Evaluate a trained model on the BioVITA benchmark.

Requires pre-computed features from extract_features.py.

Usage:
  python eval_benchmark.py \
    --base_dir path/to/benchmark/species_genus_family_dirs \
    --ids_dir  path/to/benchmark/ids \
    --feat_root path/to/features \
    --tag my_model \
    --ks 1,5,10 \
    --out_json_dir path/to/results
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from extract_features import FeatureBank


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def get_task_csv_paths(base_path):
    task_paths = {}
    for level in ["species", "genus", "family"]:
        level_path = os.path.join(base_path, level)
        task_paths[level] = {
            "A2I": os.path.join(level_path, "test_audio_to_image.csv"),
            "I2A": os.path.join(level_path, "test_image_to_audio.csv"),
            "A2T": os.path.join(level_path, "test_audio_to_text.csv"),
            "T2A": os.path.join(level_path, "test_text_to_audio.csv"),
            "I2T": os.path.join(level_path, "test_image_to_text.csv"),
            "T2I": os.path.join(level_path, "test_text_to_image.csv"),
        }
    return task_paths


def evaluate_csv(csv_path, ids_dir, feat_root, tag, ks=(1, 5, 10),
                 class_column="taxon_class"):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {"n": 0, **{f"Top@{k}": 0.0 for k in ks}}

    task = df["task"].iloc[0]
    level = df["taxon_level"].iloc[0]

    if   task.startswith("A2I"): q_mod, g_mod = "audio", "image"
    elif task.startswith("I2A"): q_mod, g_mod = "image", "audio"
    elif task.startswith("A2T"): q_mod, g_mod = "audio", "text"
    elif task.startswith("T2A"): q_mod, g_mod = "text",  "audio"
    elif task.startswith("I2T"): q_mod, g_mod = "image", "text"
    elif task.startswith("T2I"): q_mod, g_mod = "text",  "image"
    else:
        raise ValueError(f"Unknown task: {task}")

    def subkey_for(mod):
        return level if mod == "text" else None

    # Build taxon→class mapping from text_index.csv
    tx = pd.read_csv(f"{ids_dir}/text_index.csv")
    for c in ["species", "genus", "family", "class"]:
        if c in tx.columns:
            tx[c] = tx[c].astype(str)
    taxon2class = {}
    if level in ("species", "genus", "family") and {level, "class"} <= set(tx.columns):
        g = tx.groupby(level)["class"].nunique()
        ok = set(g[g == 1].index)
        sub = tx[tx[level].isin(ok)].drop_duplicates(level)
        taxon2class.update(dict(zip(sub[level], sub["class"])))

    bank = FeatureBank(feat_root, tag)
    class_col_exists = class_column in df.columns

    # Prefetch all gallery IDs
    cand_ids_parsed = [json.loads(s) for s in df["candidates_target_ids"]]
    all_tgt_ids = set()
    for groups in cand_ids_parsed:
        for ids in groups:
            all_tgt_ids.update(int(i) for i in ids)
    bank.prefetch(g_mod, all_tgt_ids, sub=subkey_for(g_mod))

    # Load all gallery vectors into GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ids_sorted = sorted(all_tgt_ids)
    D_hint = None
    q_ids = set(df["query_payload_id"].astype(int).tolist())
    if q_ids:
        try:
            D_hint = bank.get(q_mod, next(iter(q_ids)), sub=subkey_for(q_mod)).numel()
        except Exception:
            pass

    cpu_vecs = []
    for i in tqdm(all_ids_sorted, desc=f"load {g_mod} features", ncols=100):
        if bank.has(g_mod, i, sub=subkey_for(g_mod)):
            cpu_vecs.append(bank.get(g_mod, i, sub=subkey_for(g_mod)))
        else:
            D = D_hint or (cpu_vecs[0].numel() if cpu_vecs else 512)
            cpu_vecs.append(torch.zeros(D, dtype=torch.float32))

    Z_all = torch.stack(cpu_vecs, dim=0)
    Z_all = Z_all.to(device=device,
                     dtype=(torch.float16 if device.type == "cuda" else torch.float32))

    # Prefetch query vectors
    bank.prefetch(q_mod, q_ids, sub=subkey_for(q_mod))
    D = Z_all.shape[1]
    q_cache = {}
    for qi in tqdm(sorted(q_ids), desc=f"load {q_mod} features", ncols=100):
        if bank.has(q_mod, int(qi), sub=subkey_for(q_mod)):
            z = bank.get(q_mod, int(qi), sub=subkey_for(q_mod))
            q_cache[int(qi)] = z.to(device=Z_all.device, dtype=Z_all.dtype, non_blocking=True)
        else:
            q_cache[int(qi)] = torch.zeros(D, device=Z_all.device, dtype=Z_all.dtype)

    id2row = {i: r for r, i in enumerate(all_ids_sorted)}
    cand_taxa_parsed = [json.loads(s) for s in df["candidates_taxa"]]

    hits = {int(k): 0 for k in ks}
    total = 0
    skipped_counts = {"missing_query_feature": 0, "missing_target_features": 0,
                      "gt_not_in_candidates": 0}
    per_taxon_hits = {}
    per_taxonomic_class_hits = {}

    for ridx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), ncols=100)):
        q_id = int(row["query_payload_id"])
        q = q_cache[q_id]

        groups = cand_ids_parsed[ridx]
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            s_all = torch.matmul(Z_all, q)

        scores_list = []
        for ids in groups:
            if not ids:
                scores_list.append(torch.tensor(float("-inf"),
                                                device=s_all.device, dtype=s_all.dtype))
                continue
            rows = [id2row.get(int(i)) for i in ids]
            rows = [r for r in rows if r is not None]
            if not rows:
                scores_list.append(torch.tensor(float("-inf"),
                                                device=s_all.device, dtype=s_all.dtype))
                continue
            idx = torch.tensor(rows, device=s_all.device, dtype=torch.long)
            scores_list.append(torch.amax(s_all.index_select(0, idx)))
        scores = torch.stack(scores_list, dim=0)

        cands = cand_taxa_parsed[ridx]
        gt = row["correct_taxon"]
        if gt not in cands:
            skipped_counts["gt_not_in_candidates"] += 1
            continue
        gt_idx = cands.index(gt)
        total += 1

        topk_idx = torch.topk(scores, k=max(ks), largest=True, sorted=True).indices.tolist()

        d_taxon = per_taxon_hits.setdefault(gt, {"n": 0})
        d_taxon["n"] += 1

        cls = row[class_column] if class_col_exists else taxon2class.get(gt)
        d_cls = None
        if cls and str(cls).lower() not in ("", "nan"):
            d_cls = per_taxonomic_class_hits.setdefault(str(cls), {"n": 0})
            d_cls["n"] += 1

        for k in ks:
            if gt_idx in topk_idx[:k]:
                hits[int(k)] += 1
                d_taxon[int(k)] = d_taxon.get(int(k), 0) + 1
                if d_cls is not None:
                    d_cls[int(k)] = d_cls.get(int(k), 0) + 1

    per_taxon_acc = {
        tax: {**{f"Top@{k}": d.get(int(k), 0) / max(1, d["n"]) for k in ks}, "n": d["n"]}
        for tax, d in per_taxon_hits.items()
    }
    per_class_acc = {
        cls: {**{f"Top@{k}": d.get(int(k), 0) / max(1, d["n"]) for k in ks}, "n": d["n"]}
        for cls, d in per_taxonomic_class_hits.items()
    }

    return {
        "task": task,
        "level": level,
        "n": total,
        **{f"Top@{k}": hits[int(k)] / max(1, total) for k in ks},
        "skipped_counts": skipped_counts,
        "per_taxon": per_taxon_acc,
        "per_taxonomic_class": per_class_acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir",  required=True,
                    help="Benchmark directory with species/genus/family subdirs")
    ap.add_argument("--ids_dir",   required=True,
                    help="Directory with audio/image/text_index.csv")
    ap.add_argument("--feat_root", required=True,
                    help="Feature root directory (output of extract_features.py)")
    ap.add_argument("--tag",       default=None,
                    help="Subdirectory tag under feat_root (same as used in extract_features.py)")
    ap.add_argument("--ks",        default="1,5,10")
    ap.add_argument("--out_json_dir", default=None)
    ap.add_argument("--class_column", default="taxon_class")
    args = ap.parse_args()

    ks = tuple(int(x) for x in args.ks.split(","))
    task_csv_paths = get_task_csv_paths(args.base_dir)

    all_results = {}
    for level, tasks in task_csv_paths.items():
        for task_name, csv_path in tasks.items():
            print(f"\nEvaluating {task_name} @ {level} ...")
            if not os.path.exists(csv_path):
                print(f"  [skip] not found: {csv_path}")
                continue
            res = evaluate_csv(
                csv_path=csv_path,
                ids_dir=args.ids_dir,
                feat_root=args.feat_root,
                tag=args.tag,
                ks=ks,
                class_column=args.class_column,
            )
            summary = {k: v for k, v in res.items()
                       if k not in ("per_taxon", "per_taxonomic_class", "skipped_counts")}
            print(json.dumps(summary, indent=2))
            all_results[f"{task_name}_{level}"] = res

            if args.out_json_dir:
                Path(args.out_json_dir).mkdir(parents=True, exist_ok=True)
                outp = os.path.join(args.out_json_dir, f"{task_name}_{level}.json")
                with open(outp, "w") as f:
                    json.dump(res, f, indent=2)

    if args.out_json_dir:
        with open(os.path.join(args.out_json_dir, "_summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[saved] {args.out_json_dir}/_summary.json")


if __name__ == "__main__":
    main()
