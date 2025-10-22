# Now, it’s finally time to put all of the components you implemented together into your main training
#  script. It will pay off to make it easy to start training runs with different hyperparameters (e.g., by taking
#  them as command-line arguments), since you will be doing these many times later to study how different
#  choices impact training.
# Deliverable: Write a script that runs a training loop to train your model on user-provided input. In
#  particular, we recommend that your training script allow for (at least) the following:
#  • Ability to configure and control the various model and optimizer hyperparameters.
#  • Memory-efficient loading of large training and validation datasets with np.memmap.
#  • Serializing checkpoints to a user-provided path.
#  • Periodically logging training and validation performance (e.g., to console and/or an external
#  service like Weights & Biases)a.

import argparse
import os
import numpy as np
import torch

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.adamw import AdamW
from cse599o_basics.train_utils import get_batch, save_checkpoint, load_checkpoint
from cse599o_basics.model_utils import cross_entropy, lr_scheduler, gradient_clipping

def main():
    train_txt, valid_txt = "data/TinyStoriesV2-GPT4-train.txt", "data/TinyStoriesV2-GPT4-valid.txt"
    parser = argparse.ArgumentParser(description="Hyperparameters for training")
    # transformerLM hyperparameters
    # parser.add_argument("--vocab_size", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # get_batch parameters
    parser.add_argument("--batch_size", type=int, default=32)

    # training parameters
    parser.add_argument("--num_steps", type=int, default=5000)

    # steps:
    # first, parse args
    args = parser.parse_args()
    context_length: int = args.context_length
    num_layers: int = args.num_layers
    d_model: int = args.d_model
    num_heads: int = args.num_heads
    d_ff: int = args.d_ff
    rope_theta: float = args.rope_theta
    batch_size: int = args.batch_size
    num_steps: int = args.num_steps

    # we have txt files, and a tokenizer. We need to read the txt files and tokenize them.
    # then we can create a memmap dataset from the tokenized data, and use that for get_batch.
    tokenizer = BPETokenizer(vocab={}, merges=[]) # should just load tiktoken.get_encoding("gpt2")
    # get vocab size from tokenizer, add to args
    vocab_size = tokenizer.tokenizer.n_vocab

    # create memmap datasets
    if not os.path.exists("data/train_memmap.dat"):
        print("Tokenizing training data...")
        with open(train_txt, "r", encoding="utf-8") as f:
            train_data = f.read()
        train_tokens = tokenizer.encode(train_data)
        print("Creating training memmap dataset...")
        train_dataset = np.memmap("data/train_memmap.dat", dtype=np.uint16, mode="w+", shape=(len(train_tokens),))
        train_dataset[:] = np.array(train_tokens, dtype=np.uint16)
    if not os.path.exists("data/valid_memmap.dat"):
        print("Repeating for validation data...")
        with open(valid_txt, "r", encoding="utf-8") as f:
            valid_data = f.read()
        valid_tokens = tokenizer.encode(valid_data)
        valid_dataset = np.memmap("data/valid_memmap.dat", dtype=np.uint16, mode="w+", shape=(len(valid_tokens),))
        valid_dataset[:] = np.array(valid_tokens, dtype=np.uint16)

    train_dataset = np.memmap("data/train_memmap.dat", dtype=np.uint16, mode="r")
    valid_dataset = np.memmap("data/valid_memmap.dat", dtype=np.uint16, mode="r")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing model and optimizer...")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta).to(device)
    optim = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    print("Starting training loop...")

    for step in range(1, num_steps + 1):
        model.train()
        inputs, labels = get_batch(train_dataset, batch_size, context_length, device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = cross_entropy(outputs, labels)
        loss.backward()
        # gradient_clipping(model, max_norm=1.0)
        # lr = lr_scheduler(step, args.num_steps, initial_lr=1e-4, final_lr=1e-5)
        # for param_group in optim.param_groups:
        #     param_group['lr'] = lr
        optim.step()

        if num_steps < 50 or step % (num_steps // 50) == 0:
            print(f"Step {step}, Training Loss: {loss.item():.4f}")

        if (num_steps < 10 and step % 2 == 0) or (num_steps > 10 and step % (num_steps // 10)) == 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_labels = get_batch(valid_dataset, args.batch_size, args.context_length, device)
                val_outputs = model(val_inputs)
                val_loss = cross_entropy(val_outputs, val_labels)
                print(f"Step {step}, Validation Loss: {val_loss.item():.4f}")

            # use directory, actually
            save_checkpoint(model, optim, step, os.path.join("checkpoints", f"checkpoint_step_{step}.pt"))

        # How to run:
# python train.py --num_steps 5000
# on slurm:
# 


if __name__ == "__main__":
    main()