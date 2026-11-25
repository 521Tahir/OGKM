"""
Download and cache an LLM model from HuggingFace Hub.

Usage:
    # Download Llama 3 8B (requires HF token with Meta license approved):
    python scripts/download_model.py --model llama3-8b --token hf_xxx

    # Download Qwen 2.5 7B (fully open, no token needed):
    python scripts/download_model.py --model qwen2.5-7b

    # Verify a model loads correctly on GPU:
    python scripts/download_model.py --model llama3-8b --verify

HOW TO GET LLAMA 3 ACCESS:
  1. Create a HuggingFace account at https://huggingface.co
  2. Go to: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
  3. Click "Access repository" and accept Meta's license
  4. Create an access token at: https://huggingface.co/settings/tokens
  5. Set HF_TOKEN=your_token in .env or pass --token flag
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.local_client import MODEL_REGISTRY
from src.utils.logging_utils import setup_logging, get_logger

setup_logging(log_level="INFO")
logger = get_logger("download_model")


def parse_args():
    p = argparse.ArgumentParser(description="Download HuggingFace LLM model")
    p.add_argument(
        "--model", default="llama3.1-8b",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model shorthand key",
    )
    p.add_argument("--token", default=None, help="HuggingFace access token")
    p.add_argument("--cache-dir", default=None, help="Custom cache directory")
    p.add_argument("--verify", action="store_true", help="Test inference after download")
    p.add_argument("--gpu", type=int, default=5, help="GPU index for verify test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    entry = MODEL_REGISTRY[args.model]
    hf_id = entry["hf_id"]

    logger.info(f"Model    : {hf_id}")
    logger.info(f"Gated    : {entry['requires_token']}")
    logger.info(f"Family   : {entry['family']}")

    if entry["requires_token"] and not args.token:
        import os
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("\nERROR: This model requires a HuggingFace token.")
            print("  1. Accept license at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
            print("  2. Generate token at: https://huggingface.co/settings/tokens")
            print("  3. Run: python scripts/download_model.py --model llama3-8b --token hf_YOUR_TOKEN")
            print("\nAlternatively, use the fully open Qwen model (no token needed):")
            print("  python scripts/download_model.py --model qwen2.5-7b")
            sys.exit(1)
    else:
        token = args.token

    # Download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    logger.info("Downloading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, token=token, cache_dir=args.cache_dir, trust_remote_code=True
    )
    logger.info("Tokenizer downloaded.")

    logger.info("Downloading model weights (this may take several minutes) ...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        token=token,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    logger.info(f"Model downloaded. Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    if args.verify:
        logger.info(f"Running verify test on GPU {args.gpu} ...")
        device = f"cuda:{args.gpu}"
        model = model.to(device)
        from src.llm.local_client import _format_prompt
        prompt = _format_prompt(
            "Name three famous beaches in Bali.", "", entry["family"], tokenizer
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        result = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        print(f"\n--- VERIFY TEST OUTPUT ---\n{result}\n--------------------------")
        logger.info("Verify test passed.")

    print(f"\n✓ Model '{args.model}' ({hf_id}) is ready.")
    print("  Set in .env: LOCAL_MODEL_KEY=" + args.model)


if __name__ == "__main__":
    main()
