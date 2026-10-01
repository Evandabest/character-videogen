"""Pin and locally load the umT5 tokenizer without downloading model weights."""

from __future__ import annotations

import argparse

from app.weights import MODEL_ROOT


REPO = "google/umt5-xxl"
REVISION = "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
FILES = (
    "config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)
DIRECTORY = MODEL_ROOT / "tokenizer" / "umt5-xxl"


def download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPO,
        revision=REVISION,
        allow_patterns=list(FILES),
        local_dir=DIRECTORY,
    )
    print(f"Downloaded tokenizer from {REPO}@{REVISION} to {DIRECTORY}")


def load():
    from transformers import AutoTokenizer

    missing = [name for name in FILES if not (DIRECTORY / name).is_file()]
    if missing:
        raise RuntimeError(f"umT5 tokenizer is incomplete: {missing}")
    return AutoTokenizer.from_pretrained(DIRECTORY, local_files_only=True, fix_mistral_regex=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "probe"))
    args = parser.parse_args()
    if args.action == "download":
        download()
    else:
        tokenizer = load()
        print(tokenizer("A person waves hello", return_tensors="np")["input_ids"].tolist())


if __name__ == "__main__":
    main()
