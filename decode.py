import argparse
import os
import numpy as np
import torch
import timeit
import json

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.adamw import AdamW
from cse599o_basics.train_utils import decode

def main():
    parser = argparse.ArgumentParser(description="Report file to read")
    parser.add_argument("--report_file", type=str)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--prompt", type=str, default="Once upon a time")

    args = parser.parse_args()
    report_file: str = os.path.join("reports", args.report_file)
    max_tokens: int = args.max_tokens
    temperature: float = args.temperature
    top_p: float = args.top_p
    prompt: str = args.prompt

    print(f"Reading report from {report_file}...")
    with open(report_file, "r") as f:
        report = json.load(f)
    model_args = report["model_args"]
    optim_args = report["optim_args"]
    ckpt_file = report["ckpt_file"]

    print("Initializing tokenizer...")
    tokenizer = BPETokenizer(vocab={}, merges=[])

    print("Initializing model and optimizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = TransformerLM(**model_args).to(device)
    optim = AdamW(model.parameters(), **optim_args)

    generated = decode(model, tokenizer, optim, max_tokens, temperature, top_p, 
                       prompt, ckpt_file)
    print("generated:", generated)

if __name__ == "__main__":
    main()