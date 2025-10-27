import argparse
import os
import torch
import json
import numpy as np
import timeit
import torch.cuda.nvtx as nvtx

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.adamw import AdamW
from cse599o_basics.train_utils import prep_datasets, train
from cse599o_basics.model_utils import cross_entropy

def main():
    parser = argparse.ArgumentParser(description="Hyperparameters for training")
    # transformerLM hyperparameters
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # get_batch parameters
    parser.add_argument("--batch_size", type=int, default=4)

    # now we just choose small or large model: so either parse --small or --large
    parser.add_argument("--model_size", type=str, choices=["small", "large"], default="small")

    args = parser.parse_args()
    context_length: int = args.context_length
    rope_theta: float = args.rope_theta
    batch_size: int = args.batch_size

    if args.model_size == "small":
        d_model = 768
        d_ff = 3072
        num_layers = 12
        num_heads = 12
    else:  # large
        d_model = 1280
        d_ff = 5120
        num_layers = 36
        num_heads = 20

    # print("Initializing tokenizer to get vocab size")
    tokenizer = BPETokenizer(vocab={}, merges=[])
    vocab_size = tokenizer.tokenizer.n_vocab

    print("Initializing model and optimizer...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_args = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "num_layers": num_layers,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": rope_theta,
        "profile": True
    }
    optim_args = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    model = TransformerLM(**model_args).to(device)
    optim = AdamW(model.parameters(), **optim_args)

    print("Starting training loop...")
    time_train(model, optim,
               batch_size, context_length, w=5, n=10, device=device)

def get_dummy_batch(batch_size, context_length, vocab_size, device):
    inputs = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long).to(device)
    labels = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long).to(device)
    return inputs, labels

def time_train(model: TransformerLM, optim: AdamW, 
          batch_size, context_length, w, n, device):
    for step in range(1, w + n + 1):
        track = step - w >= 1
        model.train()

        inputs, labels = get_dummy_batch(batch_size, context_length, model.vocab_size, device)
        optim.zero_grad()

        # also track a complete training step with another nvtx range
        if track: # start step
            nvtx.range_push("train_step")

        if track: # start forward
            nvtx.range_push("forward")

        outputs = model(inputs)
        loss = cross_entropy(outputs, labels)

        if track: # end forward, start backward
            nvtx.range_pop()
            nvtx.range_push("backward")

        loss.backward()

        if track: # end backward
            nvtx.range_pop()
        
        optim.step()

        if track: # end step
            nvtx.range_pop()

if __name__ == "__main__":
    main()