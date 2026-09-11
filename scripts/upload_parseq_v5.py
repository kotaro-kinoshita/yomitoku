"""Validate and register v5 recognition weights on the Hugging Face Hub.

Example (prepare locally first):
    python scripts/upload_parseq_v5.py --size middle --checkpoint checkpoint.pth \
        --save-dir /tmp/parseq-middle-v5
Add --push to upload. Authentication uses the cached HF login or HF_TOKEN.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi
from omegaconf import OmegaConf

from yomitoku.configs import (
    TextRecognizerPARSeqMiddleDynwV5Config,
    TextRecognizerPARSeqTinyDynwV5Config,
)
from yomitoku.models import PARSeq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["tiny", "middle"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--repo", help="Defaults to the model config's HF repository")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    config = (
        TextRecognizerPARSeqTinyDynwV5Config
        if args.size == "tiny"
        else TextRecognizerPARSeqMiddleDynwV5Config
    )
    cfg = OmegaConf.structured(config)
    repo = args.repo or cfg.hf_hub_repo
    model = PARSeq(cfg)
    # Training checkpoints include OmegaConf metadata; use trusted local checkpoints.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.save_pretrained(args.save_dir)
    with args.checkpoint.open("rb") as source:
        checkpoint_hash = hashlib.file_digest(source, "sha256").hexdigest()
    manifest = {
        "repo": repo,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": checkpoint_hash,
        "num_tokens": cfg.num_tokens,
        "data": OmegaConf.to_container(cfg.data),
        "nfkc_normalize": cfg.nfkc_normalize,
    }
    (args.save_dir / "v5_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    card = (
        "---\ntags:\n- image-to-text\n- yomitoku\n- parseq\n- safetensors\n---\n\n"
        f"# PARSeq {args.size} v5\n\n"
        "Japanese text recognition weights for YomiToku.\n\n"
        f"Input canvas: {cfg.data.img_size[0]} x {cfg.data.img_size[1]}; "
        "dynamic-width batching with aspect-preserving fit resize.\n\n"
        "Uses charset v3 and the character_post_expand_table_v3 replacement table; "
        "uniform NFKC normalization is disabled. Architecture, charset, and "
        "preprocessing are defined by the v5 model config in YomiToku.\n\n"
        f"Source checkpoint: `{args.checkpoint.name}`.\n\n"
        f"SHA-256: `{checkpoint_hash}`.\n\n"
        "See `v5_manifest.json` for preprocessing settings and provenance.\n"
    )
    (args.save_dir / "README.md").write_text(card, encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if args.push:
        api = HfApi()
        api.create_repo(repo_id=repo, private=args.private, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo,
            folder_path=args.save_dir,
            allow_patterns=["model.safetensors", "README.md", "v5_manifest.json"],
            commit_message=f"Register PARSeq {args.size} v5",
        )
        print(commit.commit_url)


if __name__ == "__main__":
    main()
